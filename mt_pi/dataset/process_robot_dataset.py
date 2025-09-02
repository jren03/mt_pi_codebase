import argparse
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm, trange
from utils.constants import CALIB_DIR
from utils.pointclouds import project
from utils.visualizations import get_track_colors, save_depth_frames, save_frames

from dataset.depth_agent import DepthAnythingAgent
from dataset.utils import (
    calculate_point_offset,
    create_colorbar_image,
)


def visualize_2d_robot_points(
    image: np.ndarray,
    points_dt: np.ndarray,
    gripper_open: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """
    Visualizes 2D points on an image.

    Parameters:
    - image (np.ndarray): The image to visualize the points on.
    - points_dt (np.ndarray): The 2D points to visualize, of length AGENT_DT
    - gripper_open (np.ndarray): The gripper state at each timestep.
    - colors (np.ndarray): The colors of the points.

    Returns:
    - np.ndarray: The image with the points visualized.
    """

    vis = image.copy()
    frames = []
    for timestep, points in enumerate(points_dt):
        color = colors[timestep]
        color = list(map(int, color * 255))[:3]
        for i, point in enumerate(points):
            x, y = point
            vis = cv2.circle(vis, (int(x), int(y)), radius=5, color=color, thickness=-1)
            frames.append(vis.copy())
            points[i] = [int(x), int(y)]

        # Draw lines for the first timestep (For Debugging)
        # if timestep == 0:
        #     if len(points) > 4:
        #         vis = cv2.line(
        #             vis,
        #             tuple(map(int, points[1])),
        #             tuple(map(int, points[3])),
        #             color,
        #             2,
        #         )
        #         vis = cv2.line(
        #             vis,
        #             tuple(map(int, points[2])),
        #             tuple(map(int, points[4])),
        #             color,
        #             2,
        #         )

        #         midpoint = [
        #             int((points[1][0] + points[2][0]) // 2),
        #             int((points[1][1] + points[2][1]) // 2),
        #         ]
        #         vis = cv2.line(
        #             vis,
        #             tuple(map(int, points[0])),
        #             tuple(midpoint),
        #             color,
        #             2,
        #         )
        #         vis = cv2.line(
        #             vis,
        #             tuple(midpoint),
        #             tuple(map(int, points[1])),
        #             color,
        #             2,
        #         )
        #         vis = cv2.line(
        #             vis,
        #             tuple(midpoint),
        #             tuple(map(int, points[2])),
        #             color,
        #             2,
        #         )

        if int(gripper_open) == 0:
            vis = cv2.rectangle(
                vis, (0, 0), (vis.shape[1], 50), color=(0, 255, 0), thickness=-1
            )
    return vis


def visualize_3d_points(
    ax: plt.Axes, points_dt: np.ndarray, colors: np.ndarray
) -> None:
    """
    Visualizes 3D points in a 3D plot.

    Parameters:
    - ax (plt.Axes): The 3D plot to visualize the points on.
    - points (np.ndarray): The 3D points to visualize.
    - colors (np.ndarray): The colors of the points.
    """
    for timestep, points in enumerate(points_dt):
        color = colors[timestep]
        for i, point in enumerate(points):
            x, y, z = point
            ax.scatter(x, y, z, c=[color], s=2.5, alpha=1.0)


def get_eef_offsets_and_adjustments():
    """Returns the offsets and adjustments for the end-effector and points."""

    x_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    y_offset = np.array([0.0, 0.05, -0.05, 0.05, -0.05])
    z_offset = np.array([0.0, -0.075, -0.075, -0.15, -0.15])
    euler_adjustment = np.pi * np.array([-1.0, 0.0, 0.0])
    offsets = np.array([x_offset, y_offset, z_offset]).T
    return offsets, euler_adjustment


def extract_robot_actions_2d(
    agent_intrinsics: np.ndarray,
    agent_transforms: np.ndarray,
    proprio: np.ndarray,
    euler_adjustment: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """
    Calculate the offsets based on current proprio state.

    Parameters:
    agent_intrinsics (np.ndarray): The agent camera intrinsics.
    agent_transforms (np.ndarray): The agent camera transforms.
    proprio (np.ndarray): The proprioceptive input array.
    euler_adjustment (np.ndarray): The adjustment to apply to the Euler angles.
    offset (np.ndarray): The home offset for the agent.
    """
    curr_fingertip_pos = [
        proprio[:3]
        + calculate_point_offset(proprio[3:6], point_offset, euler_adjustment)
        for point_offset in offsets
    ]
    curr_2d_action = [
        project(pos, agent_intrinsics, agent_transforms) for pos in curr_fingertip_pos
    ]
    curr_2d_action = [pos.astype(int) for pos in curr_2d_action]
    return np.array(curr_2d_action)


def calculate_offsets_with_gripper(
    original_offsets: np.ndarray, proprio: np.ndarray
) -> np.ndarray:
    """
    Adjusts the offsets for points on the robot's fingers based on the gripper state.

    This function modifies the offsets for points that are on the robot's fingers by adding an offset
    opposite of the original sign. The adjustment is based on the current state of the gripper, which
    is determined from the proprioceptive input. Points on the base of the robot are not adjusted.

    Parameters:
    - original_offsets (np.ndarray): The original offsets for each point on the robot. It is assumed
        that the last column indicates whether a point is on the base (0.0) or on a finger (non-zero).
    - proprio (np.ndarray): The proprioceptive input array, where the last element indicates the
        state of the gripper (open=1, close=0).

    Returns:
    - np.ndarray: The adjusted offsets, with modifications made to points on the robot's fingers
        based on the gripper state.
    """
    sign = lambda x: x / abs(x)  # noqa: E731
    if proprio[-1] < 0.9:
        curr_gripper_state = 0
    else:
        curr_gripper_state = 1

    gripper_state_inverse = 1 - curr_gripper_state
    offsets_with_gripper = deepcopy(original_offsets)

    for j in range(1, len(original_offsets)):
        # Don't adjust the base point
        offsets_with_gripper[j][1] = original_offsets[j][1] - (
            sign(original_offsets[j][1]) * gripper_state_inverse / 40
        )
    return offsets_with_gripper

    # y_offset = np.array([0.0, 0.1, -0.1, 0.1, -0.1])
    # z_offset = np.array([0.0, -0.075, -0.075, -0.15, -0.15])


def process_data_split(
    depth_model: DepthAnythingAgent,
    fns: List[str],
    demo_dir: Path,
    output_data_dir: Path,
    vis_total: int,
) -> None:
    """
    Processes a split of data, extracting point clouds and saving them.

    Parameters:
    - fns (List[str]): Filenames of the data to process.
    - demo_dir (Path): Directory containing the demo data files.
    - output_data_dir (Path): Directory to save the processed point cloud data.
    - vis_total (int): Number of trajectories to visualize
    """
    if len(fns) == 0:
        return

    intrinsics = {
        "agent1": np.load(f"{CALIB_DIR}/agent1_intrinsics.npy"),
        "agent2": np.load(f"{CALIB_DIR}/agent2_intrinsics.npy"),
    }
    example_data = np.load(os.path.join(demo_dir, fns[0]), allow_pickle=True)
    if "episode" in example_data:
        transforms = example_data["transforms"].item()
        if "intrinsics" in example_data:
            intrinsics = example_data["intrinsics"].item()
        data_key = "episode"
    else:
        transforms = np.load(
            f"{CALIB_DIR}/transforms_both.npy", allow_pickle=True
        ).item()
        data_key = "arr_0"

    vis_dir_2d = Path(output_data_dir, "vis_2d")
    vis_dir_2d.mkdir(exist_ok=True, parents=True)

    # 2D information
    agent1_2d_img = []
    agent2_2d_img = []
    wrist_img = []
    agent1_depth_gs = []
    agent1_depth_colored = []
    agent2_depth_gs = []
    agent2_depth_colored = []

    agent1_2d_kpts = []
    agent2_2d_kpts = []
    robot_actions = []

    proprio_data = []
    gripper_open = []
    episode_ends = []

    # Define configs
    data_idx = 0
    last_episode_end = 0
    AGENT_DT = 16

    # Define offsets for the 5 points on the gripper
    offsets, euler_adjustment = get_eef_offsets_and_adjustments()
    num_points = len(offsets)

    # Create colorbar image
    track_color, grasp_color, cmap_dict = get_track_colors(AGENT_DT)
    colorbar_image = create_colorbar_image(cmap_dict["track"], AGENT_DT)
    colorbar_image_pil = Image.fromarray(colorbar_image)
    colorbar_image_pil = colorbar_image_pil.resize(
        (colorbar_image_pil.width // 2, 480), Image.LANCZOS
    )
    spacer = Image.new("RGB", (10, 480), (255, 255, 255))

    for traj_num, fn in tqdm(
        enumerate(fns),
        total=len(fns),
        leave=False,
        desc=f"Processing {output_data_dir.stem} data",
    ):
        data = np.load(os.path.join(demo_dir, fn), allow_pickle=True)[data_key]
        timesteps = list(range(len(data)))
        for timestep in timesteps:
            obs = data[timestep]["obs"]
            proprio = obs["proprio"]
            gripper_offsets = calculate_offsets_with_gripper(offsets, proprio)

            # Process real robot actions
            robot_actions.append(data[timestep]["action"])

            # Process 2D Points
            shared_kwargs = dict(
                proprio=proprio,
                euler_adjustment=euler_adjustment,
                offsets=gripper_offsets,
            )
            agent1_2d_action = extract_robot_actions_2d(
                agent_intrinsics=intrinsics["agent1"],
                agent_transforms=transforms["agent1"]["trc"],
                **shared_kwargs,
            )
            agent2_2d_action = extract_robot_actions_2d(
                agent_intrinsics=intrinsics["agent2"],
                agent_transforms=transforms["agent2"]["trc"],
                **shared_kwargs,
            )

            # Add depth
            a1_depth_gs, a1_depth_colored = depth_model.run_inference(
                obs["agent1_image"]
            )
            a2_depth_gs, a2_depth_colored = depth_model.run_inference(
                obs["agent2_image"]
            )

            agent1_2d_img.append(obs["agent1_image"])
            agent2_2d_img.append(obs["agent2_image"])
            wrist_img.append(obs.get("wrist_image", np.zeros_like(obs["agent1_image"])))
            agent1_depth_gs.append(a1_depth_gs)
            agent1_depth_colored.append(a1_depth_colored)
            agent2_depth_gs.append(a2_depth_gs)
            agent2_depth_colored.append(a2_depth_colored)

            agent1_2d_kpts.append(agent1_2d_action)
            agent2_2d_kpts.append(agent2_2d_action)
            proprio_data.append(proprio)
            if obs["gripper_open"][0] < 0.9:
                gripper_open.append(0)
            else:
                gripper_open.append(1)
            # gripper_open.append(int(obs["gripper_open"][0]))

            data_idx += 1

            if args.debug and timestep > 5 * AGENT_DT:
                break

        episode_ends.append(data_idx)

        # Visualizations
        if vis_total <= 0:
            continue
        frames_2d = []
        for i in trange(
            last_episode_end,
            data_idx,
            total=data_idx - last_episode_end,
            leave=False,
            desc="Visualizing",
        ):
            range_end = min(i + AGENT_DT, len(agent1_2d_kpts))

            # Visualize 2D keypoints
            next_agent1_2d_kpts = np.zeros((AGENT_DT, num_points, 2))
            next_agent2_2d_kpts = np.zeros((AGENT_DT, num_points, 2))
            next_agent1_2d_kpts[: range_end - i] = agent1_2d_kpts[i:range_end]
            next_agent2_2d_kpts[: range_end - i] = agent2_2d_kpts[i:range_end]
            agent1_2d_img_vis = visualize_2d_robot_points(
                agent1_2d_img[i].copy(),
                next_agent1_2d_kpts,
                gripper_open[i],
                track_color,
            )
            agent2_2d_img_vis = visualize_2d_robot_points(
                agent2_2d_img[i].copy(),
                next_agent2_2d_kpts,
                gripper_open[i],
                track_color,
            )
            wrist = wrist_img[i].copy()
            combined_image = np.hstack([agent1_2d_img_vis, agent2_2d_img_vis, wrist])
            combined_image_pil = Image.fromarray(combined_image)
            final_image = Image.new(
                "RGB",
                (
                    combined_image_pil.width + spacer.width + colorbar_image_pil.width,
                    combined_image_pil.height,
                ),
            )
            final_image.paste(combined_image_pil, (0, 0))
            final_image.paste(spacer, (combined_image_pil.width, 0))
            final_image.paste(
                colorbar_image_pil, (combined_image_pil.width + spacer.width, 0)
            )
            frames_2d.append(np.array(final_image))

        # Save depth images
        save_depth_frames(
            agent1_depth_gs,
            agent1_depth_colored,
            file_extension_type="mp4",
            file_name=f"{vis_dir_2d}/depth1",
        )
        save_depth_frames(
            agent2_depth_gs,
            agent2_depth_colored,
            file_extension_type="mp4",
            file_name=f"{vis_dir_2d}/depth2",
        )

        # Save GIFs after processing all frames for this trajectory
        save_frames(
            frames_2d,
            "mp4",
            f"{vis_dir_2d}/trajectory_{traj_num}",
        )
        last_episode_end = data_idx
        vis_total -= 1

        if args.debug:
            break

    # Save data to zarr, follow same naming conventions as image and pc
    data_path = "%s/dataset.zarr" % output_data_dir
    zarr_store = zarr.open(data_path, mode="w")
    data_group = zarr_store.create_group("data")

    H, W, C = agent1_2d_img[0].shape
    data_group.create_dataset(
        "img1", data=np.array(agent1_2d_img), chunks=(10, H, W, C)
    )
    data_group.create_dataset(
        "img2", data=np.array(agent2_2d_img), chunks=(10, H, W, C)
    )
    data_group.create_dataset(
        "depth1_gs", data=np.array(agent1_depth_gs), chunks=(10, H, W, 1)
    )
    data_group.create_dataset(
        "depth1_colored",
        data=np.array(agent1_depth_colored),
        chunks=(10, H, W, C),
    )
    data_group.create_dataset(
        "depth2_gs", data=np.array(agent2_depth_gs), chunks=(10, H, W, 1)
    )
    data_group.create_dataset(
        "depth2_colored",
        data=np.array(agent2_depth_colored),
        chunks=(10, H, W, C),
    )
    data_group.create_dataset(
        "wrist_img", data=np.array(wrist_img), chunks=(10, H, W, C)
    )
    data_group.create_dataset(
        "track1", data=np.array(agent1_2d_kpts), chunks=(10, num_points, 2)
    )
    data_group.create_dataset(
        "track2", data=np.array(agent2_2d_kpts), chunks=(10, num_points, 2)
    )
    data_group.create_dataset(
        "robot_action",
        data=np.array(robot_actions),
        chunks=(10, *robot_actions[0].shape),
    )
    data_group.create_dataset(
        "proprio", data=proprio_data, chunks=(10, len(proprio_data[0]))
    )
    data_group.create_dataset("gripper_open", data=np.array(gripper_open), chunks=(10,))

    # Create meta group and arrays
    meta_group = zarr_store.create_group("meta")
    meta_group.create_dataset(
        "episode_ends", data=episode_ends, chunks=(10,)
    )  # Chunk size 10 for the first dimension

    print(f"Sample dataset created and saved to {str(data_path)} successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--demo_dir",
        type=str,
        required=True,
        help="Directory containing the demo data files.",
    )
    parser.add_argument(
        "-vt", "--vis_total", type=int, default=2, help="Number of visualizations."
    )
    parser.add_argument(
        "--process_3d",
        action="store_true",
        help="Process 3D data in addition to 2D data.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()

    demo_dir = args.demo_dir
    if "data" not in demo_dir:
        demo_dir = f"data/{demo_dir}"
    fns = list(sorted([fn for fn in os.listdir(demo_dir) if "npz" in fn]))
    random.shuffle(fns)
    split_idx = min(int(len(fns) * 0.85), len(fns) - 1)

    base_data_dir = f"{demo_dir}_tracks"
    depth_model = DepthAnythingAgent(encoder="vits")
    process_data_split(
        depth_model,
        fns[:split_idx],
        demo_dir,
        Path(base_data_dir, "train"),
        args.vis_total,
    )
    process_data_split(
        depth_model,
        fns[split_idx:],
        demo_dir,
        Path(base_data_dir, "val"),
        args.vis_total,
    )
