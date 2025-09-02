"""
This file processes 2D points using HaMeR or MediaPipe for human hand data. In particular,
since HaMeR does not always detect a hand in every frame, we process different viewpoints
separately, meaning there may be more frames from one viewpoint than another for any
given trajectory.

Example Usage:
python dataset/process_hand_dataset.py -d hand_pick
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, List

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
import zarr
from PIL import Image
from tqdm import tqdm, trange
from utils.visualizations import save_depth_frames, save_frames

from dataset.depth_agent import DepthAnythingAgent
from dataset.hamer_agent import HandTrackAgent
from dataset.utils import HAMER_KEYPOINTS, create_colorbar_image


def linear_interpolation(start_kpts, end_kpts, frames_to_interpolate):
    # Ensure keypoints are numpy arrays
    start_kpts = np.array(start_kpts)
    end_kpts = np.array(end_kpts)

    # Number of frames to interpolate
    num_frames = len(frames_to_interpolate)

    # Generate interpolated frames
    interpolated_frames = np.zeros((num_frames, *start_kpts.shape))

    if num_frames == 1:
        # If only one frame to interpolate, return the midpoint
        interpolated_frames[0] = (start_kpts + end_kpts) / 2
    else:
        for i, frame in enumerate(frames_to_interpolate):
            alpha = (frame - frames_to_interpolate[0]) / (
                frames_to_interpolate[-1] - frames_to_interpolate[0]
            )  # Linear interpolation factor
            interpolated_frames[i] = (1 - alpha) * start_kpts + alpha * end_kpts

    return interpolated_frames


def plot_hand_skeleton(image, hand_landmarks, output_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    cv2.imwrite(output_path, cv2.flip(annotated_image, 1))


def expand_list(lst: List[Any], idx: int, filler: Any) -> List[Any]:
    """Expands a list to a given index."""
    return lst + [filler for _ in range(idx - len(lst) + 1)]


def plot_avg_confidence_scores(
    avg_confidence_scores: List[float], output_data_dir: Path
) -> None:
    """Plots the confidence scores over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(avg_confidence_scores, marker="o", linestyle="-", color="b")
    plt.title("Confidence Scores Over Time")
    plt.savefig(Path(output_data_dir, "avg_confidence_scores.png"), format="png")
    plt.close()


def plot_frames_dropped(frames_dropped_2d: List[int], output_data_dir: Path) -> None:
    """Plots the frames dropped over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    num_frames = len(frames_dropped_2d)
    ax1.hist(
        frames_dropped_2d,
        bins=num_frames,
        range=(0, num_frames),
        align="left",
        rwidth=0.8,
    )
    ax1.set_title("Frames Dropped over Time 2D")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Count")
    ax1.set_xlim(0, num_frames)
    ax1.set_yscale("log")

    plt.savefig(Path(output_data_dir, "frames_dropped.png"), format="png")
    plt.close()


def visualize_2d_hand_points(
    image: np.ndarray, points_dt: np.ndarray, colors
) -> np.ndarray:
    """
    Visualizes 2D points on an image.

    Parameters:
    - image (np.ndarray): The image to visualize the points on.
    - points_dt (np.ndarray): The 2D points to visualize, of length AGENT_DT

    Returns:
    - np.ndarray: The image with the points visualized.
    """
    vis = image.copy()

    for timestep, points in enumerate(points_dt):
        color = colors[timestep]
        color = list(map(int, color * 255))[:3]
        for i, point in enumerate(points):
            x, y = point
            if i in HAMER_KEYPOINTS:
                vis = cv2.circle(
                    vis, (int(x), int(y)), radius=5, color=color, thickness=-1
                )
    return vis


@torch.no_grad()
def process_data_split(
    predictor,
    depth_model,
    fns: List[str],
    demo_dir: str,
    output_data_dir: str,
    vis_total: int,
    predictor_type: str = "hamer",
) -> None:
    example_data = np.load(os.path.join(demo_dir, fns[0]), allow_pickle=True)
    if "episode" in example_data:
        example_episode = example_data["episode"]
        data_key = "episode"
    else:
        example_episode = example_data["arr_0"]
        data_key = "arr_0"
    num_frames = len(example_episode)

    vis_dir_2d = Path(output_data_dir, "vis_2d")
    vis_dir_2d.mkdir(exist_ok=True, parents=True)
    vis_out_dir = Path(output_data_dir, "vis_outs")
    vis_out_dir.mkdir(exist_ok=True, parents=True)
    dropped_frames_dir = Path(output_data_dir, "dropped_frames")
    dropped_frames_dir.mkdir(exist_ok=True)

    # 2D information
    viewpoints = [1, 2]
    img_action_dict = defaultdict(list)
    new_data_counter_per_viewpoint = defaultdict(int)

    # Meta information
    avg_confidence_scores = [
        (0, 0) for _ in range(num_frames)
    ]  # tuple of (running_mean, count)
    frames_dropped_2d = [0 for _ in range(num_frames)]

    # Define configs
    AGENT_DT = 8
    last_episode_end_2d = defaultdict(int)

    # Create colorbar image
    cmap = plt.get_cmap("coolwarm")
    colorbar = cmap(np.linspace(0, 1, AGENT_DT + 1))
    colorbar_image = create_colorbar_image(cmap, AGENT_DT)
    colorbar_image_pil = Image.fromarray(colorbar_image)
    colorbar_image_pil = colorbar_image_pil.resize(
        (colorbar_image_pil.width // 2, 480), Image.LANCZOS
    )
    spacer = Image.new("RGB", (10, 480), (255, 255, 255))

    errors = []
    all_interpolated_frames = []
    hand_disagreements = []

    for traj_idx, fn in tqdm(
        enumerate(fns),
        total=len(fns),
        leave=True,
        desc=f"Processing {output_data_dir.stem} data",
    ):
        try:
            data = np.load(os.path.join(demo_dir, fn), allow_pickle=True)[data_key]
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            errors.append(f"Error processing {fn}: {e}")
            continue
        for vp in viewpoints:
            frames_to_interpolate = []
            traj_is_right_hand = False
            initial_found = False
            interpolated = False
            last_detected_frame = None
            last_detected_kpts = None
            starting_frame = -1
            existing_frames = len(img_action_dict[f"img{vp}"])

            # Debug purposes
            disagreement_count = 0
            total_checked_frames = 0

            for timestep in trange(
                len(data), desc=f"Processing episode {traj_idx}", leave=False
            ):
                if timestep >= len(frames_dropped_2d):
                    frames_dropped_2d = expand_list(
                        frames_dropped_2d, timestep, filler=0
                    )
                    avg_confidence_scores = expand_list(
                        avg_confidence_scores, timestep, filler=(0, 0)
                    )

                # Process 2D Points
                obs = data[timestep]["obs"]
                image = obs[f"agent{vp}_image"]
                depth_gs, depth_colored = depth_model.run_inference(image)

                if predictor_type == "hamer":
                    kpts = predictor.step(image)
                    if kpts is None:
                        if initial_found:
                            frames_to_interpolate.append(timestep)
                            img_action_dict[f"img{vp}"].append(image)
                            img_action_dict[f"depth{vp}_gs"].append(depth_gs)
                            img_action_dict[f"depth{vp}_colored"].append(depth_colored)
                            img_action_dict[f"action{vp}"].append(None)  # placeholder
                            img_action_dict[f"is_right{vp}"].append(traj_is_right_hand)
                            frames_dropped_2d[timestep] += 1
                        continue
                    curr_hand_kpts = kpts["2d"][0].cpu().numpy()
                    confidence = kpts["confidence"]
                    is_right = kpts["is_right"][0]
                else:  # mediapipe
                    results = predictor.process(image)
                    if results.multi_hand_landmarks is None:
                        if initial_found:
                            frames_to_interpolate.append(timestep)
                            img_action_dict[f"img{vp}"].append(image)
                            img_action_dict[f"depth{vp}_gs"].append(depth_gs)
                            img_action_dict[f"depth{vp}_colored"].append(depth_colored)
                            img_action_dict[f"action{vp}"].append(None)  # placeholder
                            img_action_dict[f"is_right{vp}"].append(traj_is_right_hand)
                            frames_dropped_2d[timestep] += 1
                        continue
                    hand_landmarks = results.multi_hand_landmarks[0]
                    curr_hand_kpts = np.array(
                        [
                            [kpt.x * image.shape[1], kpt.y * image.shape[0]]
                            for kpt in hand_landmarks.landmark
                        ]
                    )
                    confidence = results.multi_handedness[0].classification[0].score
                    is_right = (
                        results.multi_handedness[0].classification[0].label == "Right"
                    )

                if not initial_found:
                    # first frame with hand detected
                    starting_frame = timestep
                    traj_is_right_hand = is_right

                else:
                    # Check for disagreement with the initial prediction
                    total_checked_frames += 1
                    if is_right != traj_is_right_hand:
                        disagreement_count += 1

                # Interpolate previous frames if needed
                if frames_to_interpolate:
                    interpolated = True
                    if last_detected_frame is not None:
                        interpolated_kpts = linear_interpolation(
                            last_detected_kpts, curr_hand_kpts, frames_to_interpolate
                        )
                        for i, frame in enumerate(frames_to_interpolate):
                            idx_to_replace = frame - starting_frame + existing_frames
                            img_action_dict[f"action{vp}"][idx_to_replace] = (
                                interpolated_kpts[i]
                            )
                            new_data_counter_per_viewpoint[vp] += 1
                            annotated_img = img_action_dict[f"img{vp}"][
                                idx_to_replace
                            ].copy()
                            for kpt in interpolated_kpts[i]:
                                cv2.circle(
                                    annotated_img,
                                    (int(kpt[0]), int(kpt[1])),
                                    3,
                                    (0, 255, 0),
                                    -1,
                                )
                            for kpt in last_detected_kpts:
                                cv2.circle(
                                    annotated_img,
                                    (int(kpt[0]), int(kpt[1])),
                                    1,
                                    (255, 0, 0),
                                    -1,
                                )
                            for kpt in curr_hand_kpts:
                                cv2.circle(
                                    annotated_img,
                                    (int(kpt[0]), int(kpt[1])),
                                    2,
                                    (0, 0, 255),
                                    -1,
                                )
                            all_interpolated_frames.append(np.array(annotated_img))

                    frames_to_interpolate = []

                new_data_counter_per_viewpoint[vp] += 1
                img_action_dict[f"img{vp}"].append(image)
                img_action_dict[f"depth{vp}_gs"].append(depth_gs)
                img_action_dict[f"depth{vp}_colored"].append(depth_colored)
                img_action_dict[f"action{vp}"].append(curr_hand_kpts)
                img_action_dict[f"is_right{vp}"].append(traj_is_right_hand)

                conf_mean, conf_count = avg_confidence_scores[timestep]
                avg_confidence_scores[timestep] = (
                    (conf_mean * conf_count + confidence) / (conf_count + 1),
                    conf_count + 1,
                )

                last_detected_frame = timestep
                last_detected_kpts = curr_hand_kpts

            # Remove any remaining frames to interpolate
            if frames_to_interpolate:
                for key in ["img", "action", "is_right"]:
                    img_action_dict[f"{key}{vp}"] = img_action_dict[f"{key}{vp}"][
                        : -len(frames_to_interpolate)
                    ]
                img_action_dict[f"depth{vp}_gs"] = img_action_dict[f"depth{vp}_gs"][
                    : -len(frames_to_interpolate)
                ]
                img_action_dict[f"depth{vp}_colored"] = img_action_dict[
                    f"depth{vp}_colored"
                ][: -len(frames_to_interpolate)]

            # Calculate the average disagreement
            if total_checked_frames > 0:
                avg_disagreement = disagreement_count / total_checked_frames
            else:
                avg_disagreement = 0

            hand_disagreements.append(avg_disagreement)

            img_action_dict[f"episode_ends{vp}"].append(
                new_data_counter_per_viewpoint[vp]
            )

            if not args.debug and (
                len(fns) - traj_idx > vis_total and not interpolated
            ):
                # Prioritize visualizing interpolated trajs
                continue

            vis_total -= 1
            new_data_added = new_data_counter_per_viewpoint[vp]
            frames_2d = []
            for i in trange(
                last_episode_end_2d[vp],
                new_data_added,
                total=new_data_added - last_episode_end_2d[vp],
                leave=False,
                desc=f"Visualizing 2D viewpoints episode {traj_idx}",
            ):
                img = img_action_dict[f"img{vp}"]
                kpts = img_action_dict[f"action{vp}"]
                range_end = min(i + AGENT_DT, len(kpts))
                next_kpts = np.zeros((AGENT_DT, 21, 2))
                next_kpts[: range_end - i] = kpts[i:range_end]
                img_vis = visualize_2d_hand_points(img[i].copy(), next_kpts, colorbar)
                agent2_2d_pil = Image.fromarray(img_vis)
                final_image = Image.new(
                    "RGB",
                    (
                        agent2_2d_pil.width + spacer.width + colorbar_image_pil.width,
                        agent2_2d_pil.height,
                    ),
                )
                final_image.paste(agent2_2d_pil, (0, 0))
                final_image.paste(spacer, (agent2_2d_pil.width, 0))
                final_image.paste(
                    colorbar_image_pil, (agent2_2d_pil.width + spacer.width, 0)
                )
                frames_2d.append(np.array(final_image))
            if frames_2d == []:
                print(f"No frames to visualize for viewpoint {vp}")
                continue
            save_frames(
                frames_2d,
                "mp4",
                f"{vis_dir_2d}/trajectory_{traj_idx}_2d_vp{vp}",
            )
            last_episode_end_2d[vp] = new_data_counter_per_viewpoint[vp]

        if args.debug:
            break

    if all_interpolated_frames:
        save_frames(
            all_interpolated_frames,
            "mp4",
            f"{vis_out_dir}/interpolated_frames",
        )

    # Save depth images
    save_depth_frames(
        img_action_dict["depth1_gs"],
        img_action_dict["depth1_colored"],
        file_extension_type="mp4",
        file_name=f"{vis_out_dir}/depth1",
    )
    save_depth_frames(
        img_action_dict["depth2_gs"],
        img_action_dict["depth2_colored"],
        file_extension_type="mp4",
        file_name=f"{vis_out_dir}/depth2",
    )

    # Create visualization plots
    plot_avg_confidence_scores(avg_confidence_scores, vis_out_dir)
    plot_frames_dropped(frames_dropped_2d, vis_out_dir)

    # Save errors
    if len(errors) > 0:
        with open(Path(output_data_dir, "errors.txt"), "w") as f:
            f.write("\n".join(errors))

    # Save data to zarr, follow same naming conventions as image and pc
    data_path = "%s/dataset.zarr" % output_data_dir
    zarr_store = zarr.open(data_path, mode="w")
    data_group = zarr_store.create_group("data")
    meta_group = zarr_store.create_group("meta")

    H, W, C = img_action_dict["img1"][0].shape
    for vp in viewpoints:
        print(
            f"Total number of viewpoint {vp} data: {len(img_action_dict[f'img{vp}'])}"
        )
        print(f"Total frames droped: {sum(frames_dropped_2d)} for viewpoint {vp} data")
        data_group.create_dataset(
            f"img{vp}", data=img_action_dict[f"img{vp}"], chunks=(10, H, W, C)
        )
        data_group.create_dataset(
            f"depth{vp}_gs", data=img_action_dict[f"depth{vp}_gs"], chunks=(10, H, W, 1)
        )
        data_group.create_dataset(
            f"depth{vp}_colored",
            data=img_action_dict[f"depth{vp}_colored"],
            chunks=(10, H, W, C),
        )
        data_group.create_dataset(
            f"track{vp}", data=img_action_dict[f"action{vp}"], chunks=(10, 21, 2)
        )
        data_group.create_dataset(
            f"is_right{vp}", data=img_action_dict[f"is_right{vp}"], chunks=(10,)
        )
        meta_group.create_dataset(
            f"episode_ends{vp}", data=img_action_dict[f"episode_ends{vp}"], chunks=(10,)
        )

    print(f"Sample dataset created and saved to {str(data_path)} successfully!")
    print(
        f"{np.min(hand_disagreements)=}, {np.max(hand_disagreements)=}, {np.mean(hand_disagreements)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--demo_dirs",
        type=str,
        nargs="+",
        default=["data/dev1"],
        help="List of directories containing the demo data files.",
    )
    parser.add_argument(
        "-vt", "--vis_total", type=int, default=6, help="Number of visualizations."
    )
    parser.add_argument(
        "--predictor",
        type=str,
        choices=["hamer", "mp"],
        default="hamer",
        help="Choose the hand tracking predictor.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_model = DepthAnythingAgent(encoder="vits")

    if args.predictor == "hamer":
        predictor = HandTrackAgent(device)
    else:
        mp_hands = mp.solutions.hands
        predictor = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
        )

    demo_dirs = args.demo_dirs
    for demo_dir in demo_dirs:
        demo_dir = demo_dir.removesuffix("/")
        if "data/" not in demo_dir:
            demo_dir = f"data/{demo_dir}"
        fns = list(sorted([fn for fn in os.listdir(demo_dir) if "npz" in fn]))
        base_data_dir = f"{demo_dir}_tracks"
        process_data_split(
            predictor,
            depth_model,
            fns,
            demo_dir,
            Path(base_data_dir, "train"),
            args.vis_total,
            args.predictor,
        )

    if args.predictor == "mp":
        predictor.close()
