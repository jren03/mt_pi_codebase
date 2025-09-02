"""
The utility functions in this file can be used for online eval.
Specifically, they are used to visualize the predicted trajectories
of the robot gripper in 2D and 3D, but can also optionally send commands
to the robot consistent with the predictions.
"""

from abc import ABC
from ast import literal_eval
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyrallis
import torch
from dataset.image_transforms import (
    get_resize_to_original_image_fn,
    resize_and_normalize_image,
)
from dataset.process_robot_dataset import (
    calculate_offsets_with_gripper,
    extract_robot_actions_2d,
    get_eef_offsets_and_adjustments,
)
from dataset.utils import (
    ROBOT_POINT_TO_IDX,
    extract_point_cloud,
    unnormalize_data,
)
from envs.franka_env import FrankaEnv
from line_profiler import profile
from models.diffusion_policy import DiffusionPolicy
from scripts.train_mtpi import TrainConfig

from utils.stopwatch import Stopwatch
from utils.transformations import (
    triangulate_points,
    update_gripper_transform,
)
from utils.visualizations import get_track_colors, o3d_to_img_array


@dataclass
class EvalConfig(ABC):
    """Abstract config class for online eval"""

    log_dir: str
    steps_to_solve: int = 4
    show_camera: bool = False
    freq: float = 10
    max_vis_frames: int = 100
    vis_file_type: str = "mp4"  # ['gif', 'mp4']
    device: str = "cuda"
    ckpt: str = "best"
    consec_grasp_thresh: int = 1
    viewpoint_jsons: List[str] = field(
        default_factory=lambda: [
            "vis_assets/front_view.json",
            "vis_assets/left_back.json",
        ]
    )
    train_cfg: List[Any] = field(default_factory=list)

    def __post_init__(self):
        assert self.vis_file_type in ["gif", "mp4"]

        if "[" in self.log_dir:
            self.log_dir = literal_eval(self.log_dir)
        else:
            self.log_dir = [self.log_dir]

        self.train_cfg = []
        for log_dir in self.log_dir:
            cfg = pyrallis.load(TrainConfig, open(f"{log_dir}/config.yaml", "r"))
            cfg.device = self.device
            cfg.policy_cfg.use_ddpm = False  # use ddim during eval
            self.train_cfg.append(cfg)


@profile
def detect_and_interpolate_outliers(
    tracks: np.ndarray, mean_point: np.ndarray, radius: float = 100.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Detect outliers in tracks using a fixed radius from the mean point and interpolate new points to replace them.
    Interpolation is done along the action_horizon dimension (axis=0).
    Only non-outlier points are used for interpolation.

    :param tracks: numpy array of shape (action_horizon, num_points, 2)
    :param mean_point: numpy array of shape (num_points, 2) representing the center point
    :param radius: radius threshold for outlier detection
    :return: numpy array of tracks with outliers replaced by interpolated points, mask of outliers, center point, radius
    """
    action_horizon, num_points, _ = tracks.shape

    # Calculate distances from each point to the mean point
    distances = np.linalg.norm(tracks - mean_point, axis=2)

    # Create a mask for outliers
    outlier_mask = distances > radius

    # Create a copy of the original tracks
    new_tracks = tracks.copy()

    for i in range(num_points):
        # Find all non-outlier points for this trajectory
        valid_indices = np.where(~outlier_mask[:, i])[0]

        if len(valid_indices) < 2:
            # If there are less than 2 valid points, we can't interpolate
            continue

        for t in range(action_horizon):
            if outlier_mask[t, i]:
                # Find the nearest non-outlier points before and after in time
                before = valid_indices[valid_indices < t]
                after = valid_indices[valid_indices > t]

                if len(before) > 0 and len(after) > 0:
                    # Interpolate between the nearest non-outlier points
                    before_idx = before[-1]
                    after_idx = after[0]
                    for coord in [0, 1]:  # 0 for u, 1 for v
                        new_tracks[t, i, coord] = np.interp(
                            t,
                            [before_idx, after_idx],
                            [
                                new_tracks[before_idx, i, coord],
                                new_tracks[after_idx, i, coord],
                            ],
                        )
                elif len(before) > 0:
                    # If no non-outlier point after, use the last valid point
                    new_tracks[t, i] = new_tracks[before[-1], i]
                elif len(after) > 0:
                    # If no non-outlier point before, use the next valid point
                    new_tracks[t, i] = new_tracks[after[0], i]

    # Recalculate center point using the cleaned data
    center_point = [mean_point[:, 0].mean(), mean_point[:, 1].mean()]
    return new_tracks, outlier_mask, center_point, radius


def check_grasp_logic(
    num_grasps_preds,
    num_non_grasps_preds,
    consec_grasp_thresh,
    info,
    curr_gripper_open,
    viewpoints=[1, 2],
):
    # gripper_change_idx = {vp: float("inf") for vp in viewpoints}
    grasp_set = {vp: num_grasps_preds[vp] == 0 for vp in viewpoints}
    for vp in viewpoints:
        # Initialize counters
        grasps_list = info[f"agent{vp}"]["grasp"][: info["solves_done"]]
        # num_grasps = sum(grasps_list < 0.5)
        # if num_grasps >= consec_grasp_thresh:
        #     num_grasps_preds[vp] += 1
        #     num_non_grasps_preds[vp] = 0
        # else:
        #     num_non_grasps = len(grasps_list) - num_grasps
        #     num_non_grasps_preds[vp] += 1
        #     num_grasps_preds[vp] = 0

        # Iterate through grasp values
        # for value in grasps_list:
        #     if value < 0.5:
        #         num_grasps += 1
        #         num_grasps_preds[vp] += 1
        #         num_non_grasps_preds[vp] = 0
        #     else:
        #         num_non_grasps += 1
        #         num_non_grasps_preds[vp] += 1
        #         num_grasps_preds[vp] = 0
        last_grasp = grasps_list[-1]
        if last_grasp < 0.5:
            num_grasps_preds[vp] += 1
            num_non_grasps_preds[vp] = 0
        else:
            num_non_grasps_preds[vp] += 1
            num_grasps_preds[vp] = 0

        # Check thresholds after each iteration
        print(
            f"num_grasps: {num_grasps_preds[vp]}, num_non_grasps: {num_non_grasps_preds[vp]}"
        )
        if num_grasps_preds[vp] >= consec_grasp_thresh:
            grasp_set[vp] = True
        elif num_non_grasps_preds[vp] >= consec_grasp_thresh:
            grasp_set[vp] = False
        else:
            print(f"Pass on vp {vp}")

    # Requires both viewpoints to predict the same grasp state to change
    print(f"grasp_set: {grasp_set}, curr_gripper_open: {curr_gripper_open}")
    if curr_gripper_open == 0:
        # gripper currently closed, stay closed as long as one viewpoint predicts a grasp
        if any(grasp_set.values()):
            gripper_open = 0
        else:
            gripper_open = 1
            num_grasps_preds[vp] = 0
    else:
        # gripper currently open, stay open as long as one viewpoint predicts a non-grasp
        if not all(grasp_set.values()):
            gripper_open = 1
        else:
            gripper_open = 0
            num_non_grasps_preds[vp] = 0
    return num_grasps_preds, num_non_grasps_preds, gripper_open


@profile
def triangulate_tracks(
    cfg: EvalConfig,
    obs: List[Dict[str, np.ndarray]],
    policy: DiffusionPolicy,
    stats: Dict[str, np.ndarray],
    steps_to_solve: int,
    action_horizon: int,
    intrinsics: Dict[str, np.ndarray],
    transforms: Dict[str, np.ndarray],
    device: str = "cuda",
    add_grasp_info_to_tracks: bool = False,
    stopwatch: Stopwatch = None,
) -> np.ndarray:
    """
    Generates visualizations for predicted tracks using 3D triangulation and saves them as a GIF.
    """
    track_colors, grasp_colors, _ = get_track_colors(action_horizon)
    offsets, euler_adjustment = get_eef_offsets_and_adjustments()
    dims_per_point = 2
    viewpoints = np.concatenate(
        [cfg.train_cfg[i].dataset_cfg.viewpoints for i in range(len(cfg.train_cfg))]
    )

    single_train_cfg: TrainConfig = cfg.train_cfg[0]
    img_dim = single_train_cfg.dataset_cfg.image_dim
    kpt_keys = single_train_cfg.dataset_cfg.kpt_keys
    kpt_idxs = [ROBOT_POINT_TO_IDX[key] for key in kpt_keys]

    proprio = obs[-1]["proprio"]
    offsets_with_gripper = calculate_offsets_with_gripper(offsets, proprio)
    # offsets_with_gripper = offsets

    info = defaultdict(lambda: defaultdict(np.ndarray))
    depth_key = "grayscale" if cfg.train_cfg[0].add_depth_gs else "colored"
    for vp in viewpoints:
        images = [o[f"agent{vp}_image"] for o in obs]  # List of (H, W, C)
        depth_imgs = [
            o.get(f"agent{vp}_depth_{depth_key}", np.zeros_like(images[0])) for o in obs
        ]
        # wrist_imgs = [o["wrist_image"] for o in obs]
        state_conds = []
        for o in obs:
            state_cond_full = extract_robot_actions_2d(
                agent_intrinsics=intrinsics[f"agent{vp}"],
                agent_transforms=transforms[f"agent{vp}"]["trc"],
                proprio=o["proprio"],
                euler_adjustment=euler_adjustment,
                offsets=offsets_with_gripper,
            )  # (num_points, 2)
            state_cond_filtered = state_cond_full[kpt_idxs]
            state_conds.append(state_cond_filtered)

        # Since point clouds are in original image size, need to upsize predicted keypoints
        resize_to_original_image = get_resize_to_original_image_fn(images[0])

        if stopwatch is not None:
            resize_context = stopwatch.time("resize_and_normalize_image")
        else:
            resize_context = nullcontext()

        with resize_context:
            processed_images = []
            processed_depths = []
            processed_state_conds = []
            resize_kwargs = dict(
                target_size=img_dim,
                action_stats=stats[vp]["action"],
                add_grasp_info_to_tracks=add_grasp_info_to_tracks,
                normalize_image=single_train_cfg.dataset_cfg.normalize_image,
            )
            for image, depth, state_cond in zip(images, depth_imgs, state_conds):
                # for image, state_cond in zip(images, state_conds):
                image, state_cond = resize_and_normalize_image(
                    image, state_cond, image_stats=stats[vp]["obs"], **resize_kwargs
                )
                depth, _ = resize_and_normalize_image(
                    depth, None, image_stats=stats[vp]["depth"], **resize_kwargs
                )
                processed_images.append(image)
                processed_depths.append(depth)
                processed_state_conds.append(state_cond)
            stacked_images = (
                torch.from_numpy(np.stack(processed_images))
                .to(device)
                .to(torch.float32)
            )  # (obs_horizon, C, H, W)
            stacked_depths = (
                torch.from_numpy(np.stack(processed_depths))
                .to(device)
                .to(torch.float32)
            )
            stacked_conds = (
                torch.from_numpy(np.stack(processed_state_conds))
                .to(device)
                .to(torch.float32)
            )  # (obs_horizon, num_points * dims_per_point)

        obs_dict = {
            "main": stacked_images,
            "depth": stacked_depths,
        }
        if stopwatch:
            act_context = stopwatch.time("policy_act")
        else:
            act_context = nullcontext()
        with act_context:
            denoised_action = policy.act(
                obs_dict,
                stacked_conds,
                vp=vp,
                first_action_abs=stacked_conds[0],
            )  # (action_horizon, num_points * dims_per_point)

        denoised_action = denoised_action.cpu().numpy()
        denoised_action = unnormalize_data(denoised_action, stats=stats[vp]["action"])
        denoised_action, terminal = denoised_action[:, :-1], denoised_action[:, -1]
        if add_grasp_info_to_tracks:
            denoised_action, grasp = denoised_action[:, :-1], denoised_action[:, -1]
            info[f"agent{vp}"]["grasp"] = grasp

        # Take the most recent img
        cropped_img_np = processed_images[-1].transpose(1, 2, 0)
        _, denoised_action = resize_to_original_image(cropped_img_np, denoised_action)
        denoised_action = denoised_action.reshape((action_horizon, -1, dims_per_point))

        # Detect and interpolate outliers
        new_tracks, outlier_mask, mean_point, radius = detect_and_interpolate_outliers(
            denoised_action, state_conds[-1]
        )

        info[f"agent{vp}"]["act"] = new_tracks
        info[f"agent{vp}"]["terminal"] = terminal

    points_3d_dt = triangulate_points(
        info["agent1"]["act"], info["agent2"]["act"], intrinsics, transforms
    )  # (action_horizon, num_points, 3)

    current_gripper_transform, future_gripper_transform, solves_done = (
        update_gripper_transform(
            proprio, points_3d_dt, steps_to_solve, min_z=cfg.env_cfg.min_z_coordinate
        )
    )
    info["solves_done"] = solves_done
    info["points_3d_dt"] = points_3d_dt
    info["current_gripper_transform"] = current_gripper_transform
    info["future_gripper_transform"] = future_gripper_transform

    # Extract point cloud for background from most recent image
    points, colors = extract_point_cloud(obs[-1], intrinsics, transforms, 20000)
    info["background_points"] = points
    info["background_colors"] = colors

    # Prepare data for trajectory visualization
    trajectory_data = []
    for t, points_3d in enumerate(points_3d_dt):
        if add_grasp_info_to_tracks:
            is_grasp = (
                info["agent1"]["grasp"][t] <= 0.5 or info["agent2"]["grasp"][t] <= 0.5
            )
            color = grasp_colors[t] if is_grasp else track_colors[t]
        else:
            color = track_colors[t]
        color = list(map(float, color[:3]))  # Convert to float and remove alpha
        trajectory_data.append((points_3d, color))

    info["trajectory_data"] = trajectory_data
    info["add_grasp_info_to_tracks"] = add_grasp_info_to_tracks

    info = dict(info)  # for npz pickling purposes
    return info


def triangulate_ground_truth_tracks(
    vis, viewpoint_jsons, obs, agent1_actions, agent2_actions, intrinsics, transforms
):
    # Load gripper mesh
    current_gripper_mesh = o3d.io.read_triangle_mesh("vis_assets/robotiq.obj")
    future_gripper_mesh = o3d.io.read_triangle_mesh("vis_assets/robotiq.obj")
    current_gripper_mesh.paint_uniform_color([0, 1, 0])  # Green for current position
    future_gripper_mesh.paint_uniform_color([1, 0, 0])  # Red for future position
    vis.add_geometry(current_gripper_mesh)
    vis.add_geometry(future_gripper_mesh)

    proprio = obs["proprio"]
    points_3d_dt = triangulate_points(
        agent1_actions, agent2_actions, intrinsics, transforms
    )

    current_gripper_transform, future_gripper_transform, solves_done = (
        update_gripper_transform(proprio, points_3d_dt, steps_to_solve=4, min_z=0.0)
    )

    # Visualize the 3D points and grippers
    vis.clear_geometries()
    vis.add_geometry(current_gripper_mesh)
    vis.add_geometry(future_gripper_mesh)

    # Extract point cloud for background
    points, colors = extract_point_cloud(obs, intrinsics, transforms, 20000)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    if len(viewpoint_jsons) > 0:
        images = []
        for viewpoint_json in viewpoint_jsons:
            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters(viewpoint_json)
            ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
            vis.poll_events()
            vis.update_renderer()
            img_arr = o3d_to_img_array(
                vis, crop_percentage=0.3, resize=(320 * len(viewpoint_jsons), 240)
            )
            images.append(img_arr)
        image = np.hstack(images)
    else:
        vis.poll_events()
        vis.update_renderer()
        image = o3d_to_img_array(vis, crop_percentage=0.3, resize=(320, 240))

    # Reset gripper meshes to their original state since all transformations are absolute
    current_gripper_mesh.transform(np.linalg.inv(current_gripper_transform))
    future_gripper_mesh.transform(np.linalg.inv(future_gripper_transform))
    # vis.clear_geometries()
    return image, future_gripper_transform


def image_tracks(
    cfg: EvalConfig,
    policy: DiffusionPolicy,
    env: FrankaEnv,
    stats: Dict[str, np.ndarray],
    action_horizon: int,
    intrinsics: Dict[str, np.ndarray],
    transforms: Dict[str, np.ndarray],
    device: str = "cuda",
    num_eps: int = 1,
    add_grasp_info_to_tracks: bool = False,
) -> np.ndarray:
    """
    Generates visualizations for predicted tracks on images and saves them as a GIF.
    """
    track_colors, grasp_colors, _ = get_track_colors(action_horizon)
    offsets, euler_adjustment = get_eef_offsets_and_adjustments()
    consec_grasp_thresh = cfg.consec_grasp_thresh

    viewpoints = np.concatenate(
        [cfg.train_cfg[i].dataset_cfg.viewpoints for i in range(len(cfg.train_cfg))]
    )

    single_train_cfg = cfg.train_cfg[0]
    img_dim = single_train_cfg.dataset_cfg.image_dim
    obs_horizon = single_train_cfg.dataset_cfg.obs_horizon
    kpt_keys = single_train_cfg.dataset_cfg.kpt_keys
    kpt_idxs = [ROBOT_POINT_TO_IDX[key] for key in kpt_keys]

    num_grasps_preds = {vp: 0 for vp in viewpoints}
    num_non_grasps_preds = {vp: 0 for vp in viewpoints}
    grasp_set = {vp: False for vp in viewpoints}

    all_mse_losses = []
    unstacked_frames = []
    obs_history = deque(maxlen=obs_horizon)
    gt_tracks_per_vp = {vp: deque(maxlen=action_horizon) for vp in viewpoints}
    curr_episode = 0

    depth_key = "grayscale" if cfg.train_cfg[0].add_depth_gs else "colored"
    while curr_episode < num_eps:
        obs = env.observe()
        obs_history.append(obs)
        while len(obs_history) < obs_horizon:
            obs_history.append(obs)

        proprio = obs["proprio"]
        offsets_with_gripper = calculate_offsets_with_gripper(offsets, proprio)

        vis_images = []
        for vp in viewpoints:
            images = [o[f"agent{vp}_image"] for o in obs_history]  # List of (H, W, C)
            depths = [o[f"agent{vp}_depth_{depth_key}"] for o in obs_history]
            # wrist_imgs = [o["wrist_image"] for o in obs_history]
            state_conds = []
            for o in obs_history:
                state_cond_full = extract_robot_actions_2d(
                    agent_intrinsics=intrinsics[f"agent{vp}"],
                    agent_transforms=transforms[f"agent{vp}"]["trc"],
                    proprio=o["proprio"],
                    euler_adjustment=euler_adjustment,
                    offsets=offsets_with_gripper,
                )  # (num_points * 2)
                state_cond_filtered = state_cond_full[kpt_idxs]
                state_conds.append(state_cond_filtered)
            gt_tracks_per_vp[vp].append((state_conds[-1], o["proprio"][-1]))

            resize_to_original_image = get_resize_to_original_image_fn(images[0])

            processed_images = []
            processed_depths = []
            processed_state_conds = []
            resize_kwargs = dict(
                target_size=img_dim,
                action_stats=stats[vp]["action"],
                add_grasp_info_to_tracks=add_grasp_info_to_tracks,
                normalize_image=single_train_cfg.dataset_cfg.normalize_image,
            )
            for image, depth, state_cond in zip(images, depths, state_conds):
                # for image, state_cond in zip(images, state_conds):
                image, state_cond = resize_and_normalize_image(
                    image, state_cond, image_stats=stats[vp]["obs"], **resize_kwargs
                )
                if "depth" in stats[vp]:
                    depth, _ = resize_and_normalize_image(
                        depth, None, image_stats=stats[vp]["depth"], **resize_kwargs
                    )
                else:
                    depth = np.zeros_like(image)
                processed_images.append(image)
                processed_depths.append(depth)
                processed_state_conds.append(state_cond)
            stacked_images = (
                torch.from_numpy(np.stack(processed_images))
                .to(device)
                .to(torch.float32)
            )  # (obs_horizon, C, H, W)
            stacked_depths = (
                torch.from_numpy(np.stack(processed_depths))
                .to(device)
                .to(torch.float32)
            )
            stacked_conds = (
                torch.from_numpy(np.stack(processed_state_conds))
                .to(device)
                .to(torch.float32)
            )  # (obs_horizon, num_points * dims_per_point)

            obs_dict = {
                "main": stacked_images,
                "depth": stacked_depths,
                # "wrist": stacked_wrists,
            }
            denoised_action = policy.act(
                obs_dict, stacked_conds, vp=vp, first_action_abs=stacked_conds[0]
            )  # (action_horizon, num_points * dims_per_point)
            denoised_action = denoised_action.cpu().numpy()
            denoised_action = unnormalize_data(
                denoised_action, stats=stats[vp]["action"]
            )
            denoised_action, terminal = denoised_action[:, :-1], denoised_action[:, -1]
            if add_grasp_info_to_tracks:
                denoised_action, grasp = denoised_action[:, :-1], denoised_action[:, -1]

            # Take the most recent img
            cropped_img_np = processed_images[-1].transpose(1, 2, 0)
            _, denoised_action = resize_to_original_image(
                cropped_img_np, denoised_action
            )
            denoised_action = denoised_action.reshape((action_horizon, -1, 2))

            # Detect and interpolate outliers
            new_tracks, outlier_mask, mean_point, radius = (
                detect_and_interpolate_outliers(denoised_action, state_conds[-1])
            )

            # Visualize
            vis = images[-1].copy()  # take last image in obs_horizon
            # for timestep, action_t in enumerate(denoised_action):
            for timestep, (original_action_t, new_action_t) in enumerate(
                zip(denoised_action, new_tracks)
            ):
                if add_grasp_info_to_tracks and grasp[timestep] <= 0.5:
                    color = grasp_colors[timestep]
                    is_grasp = True
                else:
                    color = track_colors[timestep]
                    is_grasp = False

                if timestep == 0:
                    # Takes consec_grasp_thresh number of consecutive grasps to set grasp_set to True
                    # and vice versa for non-grasps
                    if is_grasp:
                        num_grasps_preds[vp] += 1
                        num_non_grasps_preds[vp] = 0
                    else:
                        num_non_grasps_preds[vp] += 1
                        num_grasps_preds[vp] = 0

                    if num_grasps_preds[vp] >= consec_grasp_thresh:
                        grasp_set[vp] = True
                    elif num_non_grasps_preds[vp] >= consec_grasp_thresh:
                        grasp_set[vp] = False

                    if grasp_set[vp]:
                        cv2.rectangle(
                            vis,
                            (0, vis.shape[0] - 20),
                            (vis.shape[1], vis.shape[0]),
                            (160, 189, 151),
                            -1,
                        )

                    # Add text annotation for the number of consecutive grasps and non-grasps
                    text_annotation = f"Grasps: {num_grasps_preds[vp]}, Non-grasps: {num_non_grasps_preds[vp]}"
                    cv2.putText(
                        vis,
                        text_annotation,
                        (10, vis.shape[0] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # color = list(map(int, color * 255))[:3]
                # for u, v in action_t:
                #     cv2.circle(vis, (int(u), int(v)), 5, color, -1)

                color = list(map(int, color * 255))[:3]
                for k, (orig_point, new_point) in enumerate(
                    zip(original_action_t, new_action_t)
                ):
                    u, v = orig_point
                    new_u, new_v = new_point
                    if outlier_mask[timestep, k]:
                        # Mark original outlier point
                        cv2.circle(
                            vis, (int(u), int(v)), 3, (234, 7, 242), -1
                        )  # Pink for outlier
                        # Mark new interpolated point
                        cv2.circle(
                            vis, (int(new_u), int(new_v)), 6, (0, 255, 0), -1
                        )  # Green for interpolated
                    else:
                        cv2.circle(vis, (int(new_u), int(new_v)), 5, color, -1)

            vis_images.append(vis)

            # Retroactively add gt tracks
            if len(unstacked_frames) >= action_horizon - 1:
                target_idx = len(unstacked_frames) - action_horizon + 1
                # temp = unstacked_frames[target_idx][vp - 1].copy()
                # cv2.imwrite("temp.png", cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))
                gt_tracks = np.array(
                    [gt_track for gt_track, _ in reversed(gt_tracks_per_vp[vp])]
                )
                action_l2 = ((denoised_action - gt_tracks) ** 2).mean()
                all_mse_losses.append(action_l2)

                for timestep, (gt_track, gripper_open) in enumerate(
                    gt_tracks_per_vp[vp]
                ):
                    color = (
                        track_colors[timestep]
                        if gripper_open
                        else grasp_colors[timestep]
                    )
                    # Slightly lighter color for gt tracks
                    color = list(map(int, color * 155))[:3]
                    for u, v in gt_track:
                        unstacked_frames[target_idx][vp - 1] = cv2.circle(
                            unstacked_frames[target_idx][vp - 1],
                            (int(u), int(v)),
                            3,
                            color,
                            1,
                        )

        unstacked_frames.append(vis_images)
        # stacked_image = np.hstack(vis_images)
        # frames.append(stacked_image)
        episode_done = env.step(denoised_action[0])

        if episode_done:
            print(
                f"Episode {curr_episode + 1} done, average action_l2 {np.mean(all_mse_losses)}"
            )
            curr_episode += 1
            if curr_episode < num_eps:
                env.reset()
            else:
                break

    frames = [np.hstack(unstacked_frame) for unstacked_frame in unstacked_frames]
    return np.array(frames)
