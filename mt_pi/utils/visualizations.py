from typing import Dict, Tuple

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch.nn as nn
from dataset_utils.common import unnormalize_data
from models.train_utils import process_namedtuple_batch
from torch.utils.data import DataLoader
from tqdm import tqdm


def process_cv2_color(color):
    return list(map(int, (color * 255)))[:3]


def o3d_to_img_array(vis, crop_percentage=0.3, resize=None):
    """
    Convert an Open3D visualizer to an image array, with pixel values
    as uint8 between [0, 255]

    Arguments:
        vis (open3d.visualization.Visualizer): The Open3D visualizer.
        crop_percentage (float): The percentage of the image to crop.
            e.g.: 0.3 means take the center 40% of the image.
        resize: The size to resize the image to.
    """
    o3d_img = vis.capture_screen_float_buffer(False)
    img_arr = (np.array(o3d_img) * 255).astype(np.uint8)
    h, w = img_arr.shape[:2]
    if crop_percentage > 0.0:
        left = int(w * (crop_percentage))
        right = int(w * (1 - crop_percentage))
        top = int(h * (crop_percentage))
        bottom = int(h * (1 - crop_percentage))
        img_arr = img_arr[top:bottom, left:right]
    if resize:
        img_arr = cv2.resize(img_arr, resize)
    return img_arr


def get_track_colors(
    action_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, plt.cm.ScalarMappable]]:
    """
    Initialize common visualization parameters.

    Args:
        action_horizon (int): The number of steps in the action sequence.
        add_grasp_info_to_tracks (bool): Whether to add grasp information to points.

    Returns:
        Tuple containing:
        - track_colors (np.ndarray): Colors for tracking visualization.
        - grasp_colors (np.ndarray): Colors for grasping visualization.
        - cmap_dict (Dict[str, plt.cm.ScalarMappable]): Dictionary of color maps.
    """
    track_cmap = plt.get_cmap("autumn")
    grasp_cmap = plt.get_cmap("winter")
    values = np.linspace(0, 1, action_horizon)
    track_colors = track_cmap(values)
    grasp_colors = grasp_cmap(values)

    cmap_dict = {"track": track_cmap, "grasp": grasp_cmap}

    return track_colors, grasp_colors, cmap_dict


def visualize_image_tracks_2d(
    policy: nn.Module,
    dataloader: DataLoader,
    stats: Dict[str, np.ndarray],
    action_horizon: int,
    device: str = "cuda",
    max_frames: int = 100,
    add_grasp_info_to_tracks: bool = False,
) -> np.ndarray:
    """
    Generates visualizations for predicted tracks on pointclouds and saves them as a GIF.
    """
    # Set up color maps for tracks and grasps
    track_cmap = plt.get_cmap("autumn_r")
    grasp_cmap = plt.get_cmap("winter")
    values = np.linspace(0, 1, action_horizon)
    track_colors = track_cmap(values)
    grasp_colors = grasp_cmap(values)

    frames = []
    dims_per_point = 2
    for idx, batch in tqdm(
        enumerate(iter(dataloader)),
        total=len(dataloader),
        desc="Generating Visualizations",
    ):
        batch = process_namedtuple_batch(batch, device)
        # Remove the batch dim -- denoised_action.shape == (action_horizon, num_points*2)
        image, state_cond, first_action = (
            batch.obs[0],  # (1, C, H, W)
            batch.state_cond[0],  # (1, num_points*2)
            batch.action[0, 0],  # (num_points*2) -- should be same values as state_cond
        )
        denoised_action = policy.act(image, state_cond, first_action_abs=first_action)
        denoised_action = denoised_action.cpu().numpy()
        # Remove obs_horizon dim -- last_image.shape == (H, W, C)
        last_image = image[-1].cpu().numpy().transpose(1, 2, 0)
        last_image = unnormalize_data(last_image, stats=stats["obs"]).astype(np.uint8)
        action = unnormalize_data(denoised_action, stats=stats["action"])
        action, terminal = action[:, :-1], action[:, -1]

        vis = last_image.copy()
        if add_grasp_info_to_tracks:
            action, grasp = action[:, :-1], action[:, -1]
        action = action.reshape((action_horizon, -1, dims_per_point))
        for timestep, action_t in enumerate(action):
            if terminal[timestep] > 0.5:
                color = (0, 0, 0)
            elif add_grasp_info_to_tracks:
                # Select colors based on grasp values
                color = (
                    grasp_colors[timestep]
                    if grasp[timestep] <= 0.5
                    else track_colors[timestep]
                )
            else:
                color = track_colors[timestep]
            color = process_cv2_color(color)
            for u, v in action_t:
                cv2.circle(vis, (int(u), int(v)), 2, color, -1)

        # added as (H, W, C)
        frames.append(vis)
        if len(frames) > max_frames:
            break
    return np.array(frames)


def visualize_3d(vis_data, cfg):
    frames = []
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    consec_grasp_thresh = cfg.consec_grasp_thresh

    # Load gripper mesh
    current_gripper_mesh = o3d.io.read_triangle_mesh("vis_assets/robotiq.obj")
    future_gripper_mesh = o3d.io.read_triangle_mesh("vis_assets/robotiq.obj")
    current_gripper_mesh.paint_uniform_color([0, 1, 0])  # Green for current position
    future_gripper_mesh.paint_uniform_color([1, 0, 0])  # Red for future position
    vis.add_geometry(current_gripper_mesh)
    vis.add_geometry(future_gripper_mesh)

    for frame_data in tqdm(vis_data, desc="Generating 3D Visualizations"):
        vis.clear_geometries()
        vis.add_geometry(current_gripper_mesh)
        vis.add_geometry(future_gripper_mesh)

        # Add background point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_data["background_points"])
        pcd.colors = o3d.utility.Vector3dVector(frame_data["background_colors"])
        vis.add_geometry(pcd)

        # Visualize predicted trajectories
        for points_3d, color in frame_data["trajectory_data"]:
            for point_3d in points_3d:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                sphere.translate(point_3d)
                sphere.paint_uniform_color(color)
                vis.add_geometry(sphere)

        # Update gripper positions
        current_gripper_mesh.transform(frame_data["current_gripper_transform"])
        future_gripper_mesh.transform(frame_data["future_gripper_transform"])

        if len(cfg.viewpoint_jsons) > 0:
            images = []
            num_grasps_preds = 0
            for vp_idx, viewpoint_json in enumerate(cfg.viewpoint_jsons):
                ctr = vis.get_view_control()
                param = o3d.io.read_pinhole_camera_parameters(viewpoint_json)
                ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
                vis.poll_events()
                vis.update_renderer()
                img_arr = o3d_to_img_array(
                    vis,
                    crop_percentage=0.3,
                    resize=(320 * len(cfg.viewpoint_jsons), 240),
                )

                # img_annotation = f"Terminal: {frame_data['agent1']['terminal'][-1]:.5f}"
                img_annotation = ""
                if frame_data["add_grasp_info_to_tracks"]:
                    # Separate grasp signals per viewpoint
                    # This currently assumes the o3d vps are in order of camera vps
                    is_grasp = frame_data[f"agent{vp_idx + 1}"]["grasp"][-1] < 0.5

                    if is_grasp:
                        num_grasps_preds += 1
                    else:
                        num_grasps_preds = 0
                    img_annotation = f"Num Consecutive Grasps: {num_grasps_preds}."

                    if num_grasps_preds >= consec_grasp_thresh:
                        # add green rectangle to bottom if consecutive grasps
                        cv2.rectangle(
                            img_arr,
                            (0, img_arr.shape[0] - 20),
                            (img_arr.shape[1], img_arr.shape[0]),
                            (160, 189, 151),
                            -1,
                        )

                cv2.putText(
                    img_arr,
                    img_annotation,
                    (10, 20),
                    fontScale=1.0,
                    color=(0, 0, 255),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                )
                images.append(img_arr)
            image = np.hstack(images)
        else:
            vis.poll_events()
            vis.update_renderer()
            image = o3d_to_img_array(vis, crop_percentage=0.3, resize=(320, 240))

        frames.append(image)

        # Reset gripper meshes to their original state
        current_gripper_mesh.transform(
            np.linalg.inv(frame_data["current_gripper_transform"])
        )
        future_gripper_mesh.transform(
            np.linalg.inv(frame_data["future_gripper_transform"])
        )

    vis.destroy_window()
    return np.array(frames)


def save_depth_frames(
    grayscale_frames,
    colored_frames,
    file_name: str,
    file_extension_type: str = "mp4",
    fps: int = 10,
    max_frames: int = 250,
):
    gs_frames = np.array(
        [
            cv2.cvtColor(grayscale_frames[i], cv2.COLOR_GRAY2BGR)
            for i in range(min(max_frames, len(grayscale_frames)))
        ]
    )
    stacked_depth_frames = np.hstack([gs_frames, colored_frames[:max_frames]])
    save_frames(stacked_depth_frames, file_extension_type, file_name, fps=fps)


def save_frames(
    frames: np.ndarray, file_extension_type: str, file_name: str, fps: int = 10
):
    """
    Given a set of frames, use imageio to save them as a file_extension_type
    """

    file_name = file_name.removesuffix(f".{file_extension_type}")

    if file_extension_type == "gif":
        imageio.mimsave(f"{file_name}.gif", frames, fps=fps, loop=0)
    elif file_extension_type == "mp4":
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(
            f"{file_name}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (width, height),
        )
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
    else:
        raise ValueError(f"Unsupported file extension type: {file_extension_type}")

    print(f"Visualization saved as '{file_name}.{file_extension_type}'")
