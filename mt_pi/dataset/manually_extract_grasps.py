import argparse
import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm
import cv2


def extract_all_frames(zarr_path, debug=False):
    print(f"Extracting all frames from {zarr_path}")
    store = zarr.open(zarr_path, mode="r")
    data_group = store["data"]
    meta = store["meta"]

    all_frames = {1: [], 2: []}
    episode_indices = {1: [], 2: []}

    for vp in [1, 2]:
        images = np.array(data_group[f"img{vp}"])
        episode_ends = np.array(meta[f"episode_ends{vp}"])
        start_idx = 0
        num_episodes = len(episode_ends)

        if debug:
            num_episodes = min(num_episodes, 3)  # Limit to 3 episodes per viewpoint

        for i, end_idx in tqdm(
            enumerate(episode_ends[:num_episodes]), total=num_episodes
        ):
            episode_frames = images[start_idx:end_idx]
            all_frames[vp].extend(episode_frames)
            episode_indices[vp].extend([(vp, i)] * len(episode_frames))
            start_idx = end_idx

    return all_frames, episode_indices


def annotate_gripper_states(frames, episode_indices):
    gripper_states = []
    current_state = 1  # Assume gripper starts open
    frame_index = 0
    total_frames = len(frames[1]) + len(frames[2])

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 800, 600)

    while frame_index < total_frames:
        if frame_index < len(frames[1]):
            vp = 1
            actual_frame_index = frame_index
        else:
            vp = 2
            actual_frame_index = frame_index - len(frames[1])

        frame = frames[vp][actual_frame_index]
        ep_info = episode_indices[vp][actual_frame_index]

        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"VP: {vp}, Episode: {ep_info[1]}, Frame: {frame_index}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"Gripper State: {'Open' if current_state == 1 else 'Closed'}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Frame", display_frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("a"):
            if frame_index > 0:
                frame_index -= 1
                gripper_states.pop()  # Remove the last appended state
        elif key == ord("d"):
            frame_index = min(total_frames - 1, frame_index + 1)
        elif key == ord("s"):
            current_state = 0 if current_state == 1 else 1  # Toggle state

        if frame_index >= len(gripper_states):
            gripper_states.append(current_state)

    cv2.destroyAllWindows()
    return gripper_states


def save_gripper_states(zarr_path, gripper_states):
    store = zarr.open(zarr_path, mode="r+")

    if "gripper_open" in store:
        print("'gripper_open' group already exists. Overwriting...")
        del store["gripper_open"]

    gripper_group = store.create_group("gripper_open")

    split_index = len(store["data"]["img1"])
    gripper_group.create_dataset(
        "vp1", data=np.array(gripper_states[:split_index]), chunks=(1000,)
    )
    gripper_group.create_dataset(
        "vp2", data=np.array(gripper_states[split_index:]), chunks=(1000,)
    )

    print(f"Saved gripper states to {zarr_path}")


def extract_gripper_states(zarr_path):
    store = zarr.open(zarr_path, mode="r")
    gripper_states = []

    for vp in [1, 2]:
        gripper_states.extend(store["gripper_open"][f"vp{vp}"])

    return gripper_states


def visualize_annotations(frames, gripper_states, output_path):
    print("Creating visualization video...")
    total_frames = len(frames)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for frame_index in tqdm(range(total_frames)):
        frame = frames[frame_index]
        state = gripper_states[frame_index]

        vis_frame = frame.copy()
        if state == 0:  # Closed
            cv2.rectangle(
                vis_frame, (0, 0), (vis_frame.shape[1], 50), (0, 255, 255), -1
            )

        cv2.putText(
            vis_frame,
            f"Gripper: {'Open' if state == 1 else 'Closed'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        out.write(vis_frame)

    out.release()
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manually annotate gripper states in zarr file and visualize results"
    )
    parser.add_argument("-d", "--data_dir", help="Path to the data directory")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with a limited number of episodes for debugging",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if "data/" not in data_dir:
        data_dir = "data/" + data_dir
    zarr_path = Path(data_dir, "train", "dataset.zarr")
    vis_dir = Path(data_dir, "train", "vis_outs")
    vis_dir.mkdir(exist_ok=True, parents=True)

    frames, episode_indices = extract_all_frames(zarr_path, debug=args.debug)
    gripper_states = annotate_gripper_states(frames, episode_indices)
    save_gripper_states(zarr_path, gripper_states)
    gripper_states = extract_gripper_states(zarr_path)
    for vp in [1, 2]:
        output_path = vis_dir / f"manual_grasps_viewpoint_{vp}.mp4"
        visualize_annotations(
            frames[vp],
            gripper_states[: len(frames[vp])]
            if vp == 1
            else gripper_states[len(frames[1]) :],
            str(output_path),
        )
