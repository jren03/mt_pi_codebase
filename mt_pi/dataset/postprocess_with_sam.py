import argparse
from collections import deque
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from line_profiler import profile
from sam2.build_sam import build_sam2_video_predictor
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# two for thumb (3, 4); one per other fingertip (8, 12, 16, 20)
FINGERTIP_POINTS = [3, 4, 8, 12, 16, 20]
CONSEC_GRASP_THRESH = 5

# FINGER_DISTANCE_THRESHOLD = 10
FINGER_DISTANCE_THRESHOLD = 15
THUMB_DISTANCE_THRESHOLD = 15


def is_point_near_mask(point, mask, img_w, img_h, distance_threshold=10):
    """Check if a point is within a certain distance from any point in the mask."""
    # Ensure the mask is boolean
    bool_mask = mask.astype(bool)
    # Compute distance transform on the inverse of the boolean mask
    distance_map = distance_transform_edt(~bool_mask)
    x, y = int(point[0]), int(point[1])
    x = min(max(x, 0), img_w - 1)
    y = min(max(y, 0), img_h - 1)
    return distance_map[y, x] <= distance_threshold


def get_finger_color(index: int):
    color_index = 0

    if index == 0:
        color_index = 5  # Palm
    elif index <= 4:
        color_index = 0  # Thumb
    elif index <= 8:
        color_index = 1  # Index
    elif index <= 12:
        color_index = 2  # Middle
    elif index <= 16:
        color_index = 3  # Ring
    elif index <= 20:
        color_index = 4  # Pinky

    finger_colors = [
        (255, 0, 0),  # Thumb: Red
        (0, 255, 0),  # Index: Green
        (0, 0, 255),  # Middle: Blue
        (255, 255, 0),  # Ring: Yellow
        (255, 0, 255),  # Pinky: Magenta
        (0, 255, 255),  # Palm: Cyan
    ]
    return finger_colors[color_index]


@profile
def plot_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
) -> np.ndarray:
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    # Display the image
    ax.imshow(image)

    # Overlay masks and plot bounding boxes with labels
    for mask in masks:
        # Overlay the mask
        if mask is not None:
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
            mask_rgba[:, :, 0] = 1  # Red channel
            mask_rgba[:, :, 3] = mask * 0.5  # Alpha channel
            ax.imshow(mask_rgba)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Convert the Matplotlib figure to a numpy array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape((height, width, 3))

    # Convert the numpy array to a cv2 image
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return img_cv2  # Return the cv2 image


@profile
def process_with_sam(zarr_path, points_path, vis_dir, max_frames):
    # Load the points
    points_data = np.load(points_path)
    # (num_episodes, num_points, 2)
    selected_points = points_data["points"]
    # (num_episodes, 2) --> (viewpoint, episode_idx)
    indices = points_data["indices"]
    # sort indices and selected points by (viewpoint, episode_idx)
    indices, selected_points = zip(
        *sorted(zip(indices, selected_points), key=lambda x: (x[0][0], x[0][1]))
    )

    # Open the zarr file
    store = zarr.open(zarr_path, mode="r+")
    data_group = store["data"]
    meta = store["meta"]

    # Initialize SAM
    checkpoint = "third_party/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    # Check if 'gripper_open' group already exists
    if "gripper_open" in store:
        print("'gripper_open' group already exists. Overwriting...")
        del store["gripper_open"]

    # Create a new group for gripper state
    gripper_group = store.create_group("gripper_open")

    # Read from zarr file
    vp_to_info = {
        1: {
            "images": np.array(data_group["img1"]),
            "finger_points": np.array(data_group["track1"]),
            "episode_ends": np.array(meta["episode_ends1"]),
        },
        2: {
            "images": np.array(data_group["img2"]),
            "finger_points": np.array(data_group["track2"]),
            "episode_ends": np.array(meta["episode_ends2"]),
        },
    }

    frames_per_vp = {1: [], 2: []}
    gripper_states_per_vp = {1: [], 2: []}
    count = 0
    for (vp, ep_idx), points in tqdm(zip(indices, selected_points), total=len(indices)):
        images = vp_to_info[vp]["images"]
        finger_points = vp_to_info[vp]["finger_points"]
        episode_ends = vp_to_info[vp]["episode_ends"]

        # Get the images for this episode
        start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        end_idx = episode_ends[ep_idx]
        ep_imgs = images[start_idx:end_idx]
        ep_finger_points = finger_points[start_idx:end_idx]

        # Process with SAM
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(frames=ep_imgs)
            predictor.reset_state(state)

            for obj_id, point in enumerate(points):
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=np.array([point]),
                    labels=np.array([1], np.int32),
                )

            video_segments = {}
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        grasp_deque = deque(maxlen=CONSEC_GRASP_THRESH)
        temp_gripper_states = []

        # Process gripper states and visualize
        starting_frame = len(frames_per_vp[vp])
        for frame_idx, (img, keypoints, obj_to_mask) in enumerate(
            zip(ep_imgs, ep_finger_points, video_segments.values())
        ):
            masks = np.array(list(obj_to_mask.values())).astype(np.float32)
            masks = masks.transpose(0, 2, 3, 1)
            h, w = img.shape[:2]

            # Resize masks to original image size
            resized_masks = np.zeros((masks.shape[0], h, w), dtype=np.float32)
            for i in range(masks.shape[0]):
                resized_masks[i] = cv2.resize(
                    masks[i], (w, h), interpolation=cv2.INTER_NEAREST
                )
            masks = resized_masks.astype(bool)

            vis = img.copy()
            thumb_near, other_fingers_near = False, 0
            for j, (x, y) in enumerate(keypoints):
                if j not in FINGERTIP_POINTS:
                    continue

                base_color = get_finger_color(j)
                for mask in masks:
                    is_thumb = j in [3, 4]
                    is_near = is_point_near_mask(
                        (x, y),
                        mask,
                        img.shape[1],
                        img.shape[0],
                        distance_threshold=THUMB_DISTANCE_THRESHOLD
                        if is_thumb
                        else FINGER_DISTANCE_THRESHOLD,
                    )

                    if is_near:
                        if is_thumb:
                            thumb_near = True
                        else:
                            other_fingers_near += 1
                        # Brighten the color for points near the mask
                        color = tuple(min(c + 100, 255) for c in base_color)
                        cv2.circle(
                            vis,
                            (int(x), int(y)),
                            radius=5,
                            color=(255, 255, 255),
                            thickness=1,
                        )
                    else:
                        color = base_color

                cv2.circle(
                    vis,
                    (int(x), int(y)),
                    radius=3,
                    color=color,
                    thickness=-1,
                )

            cv2.putText(
                vis,
                f"Thumb near: {'Yes' if thumb_near else 'No'}, Other fingers near: {other_fingers_near}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            all_near = thumb_near and other_fingers_near >= 1
            grasp_deque.append(all_near)

            # Grasp only registered if the last consec_grasp_thresh frames are all near
            is_grasp = len(grasp_deque) == CONSEC_GRASP_THRESH and all(grasp_deque)
            temp_gripper_states.append(int(not is_grasp))
            if is_grasp:
                for j in range(max(0, frame_idx - CONSEC_GRASP_THRESH), frame_idx + 1):
                    temp_gripper_states[j] = 0
                    frame_index = starting_frame + j
                    if frame_index < len(frames_per_vp[vp]):
                        frames_per_vp[vp][frame_index] = cv2.rectangle(
                            frames_per_vp[vp][frame_index],
                            (0, 0),
                            (frames_per_vp[vp][frame_index].shape[1], 50),
                            color=(0, 255, 255),
                            thickness=-1,
                        )

            vis = plot_masks(vis, masks)
            if max_frames == -1 or len(frames_per_vp[vp]) < max_frames:
                frames_per_vp[vp].append(vis)

        gripper_states_per_vp[vp].extend(temp_gripper_states)
        count += 1

    # Save visualizations
    for vp, frames in frames_per_vp.items():
        # Save the gripper states to the new group
        gripper_group.create_dataset(
            f"vp{vp}", data=np.array(gripper_states_per_vp[vp]), chunks=(1000,)
        )
        if frames:
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(
                f"{vis_dir}/sam2_grasps_viewpoint_{vp}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (width, height),
            )
            for frame in frames:
                out.write(frame)
            out.release()

    print(
        f"Processing complete. Results saved to {zarr_path} and visualizations to {vis_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add gripper state and visualize from zarr file"
    )
    parser.add_argument(
        "-d", "--data_path", nargs="+", help="Path to the data directory"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process",
    )
    args = parser.parse_args()

    for data_path in args.data_path:
        if "data/" not in data_path:
            data_path = f"data/{data_path}"

        partition_path = f"{data_path}/train"
        points_path = f"{partition_path}/first_frames_points.npz"
        zarr_path = f"{partition_path}/dataset.zarr"
        vis_dir = f"{partition_path}/vis_outs"

        print(f"Processing {zarr_path}")
        process_with_sam(zarr_path, points_path, vis_dir, args.max_frames)
