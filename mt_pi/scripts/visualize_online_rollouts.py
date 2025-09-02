from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from common_utils.visualizations import get_track_colors, save_frames
from tqdm import tqdm


def visualize_2d_predictions(rollout_dir, vis_type, no_tracks):
    viewpoints = [1, 2]
    npzs = [f for f in rollout_dir.iterdir() if f.suffix == ".npz"]
    npzs.sort()
    frames = []
    track_colors, grasp_colors, _ = get_track_colors(action_horizon=16)
    for traj_idx, npz in tqdm(enumerate(npzs), total=len(npzs)):
        data = np.load(npz, allow_pickle=True)
        for step in data["episode"]:
            views = []

            for vp in viewpoints:
                img = step["raw_obs"][f"agent{vp}_image"]
                if not no_tracks:
                    action = step[f"agent{vp}"]["act"]
                    grasp = step[f"agent{vp}"]["grasp"]
                    for timestep, action_t in enumerate(action):
                        if timestep == 0:
                            color = (0, 255, 0)
                        elif grasp[timestep] < 0.5:
                            color = grasp_colors[timestep]
                            color = list(map(int, color * 255))[:3]
                        else:
                            color = track_colors[timestep]
                            color = list(map(int, color * 255))[:3]
                        for u, v in action_t:
                            cv2.circle(img, (int(u), int(v)), 4, color, -1)
                views.append(img)

            combined = np.hstack(views)
            cv2.putText(
                combined,
                f"Trajectory {traj_idx}",
                (10, views[0].shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            frames.append(combined)

    if frames:
        save_frames(
            frames, vis_type, file_name=f"online_rollouts_2d_{rollout_dir.parent.name}"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--rollout_dir", type=Path, required=True)
    parser.add_argument("--vis_type", type=str, choices=["gif", "mp4"], default="mp4")
    parser.add_argument("--no_tracks", action="store_true", default=False)
    args = parser.parse_args()

    rollouts_dir = args.rollout_dir
    if "real_rollouts" not in rollouts_dir.stem:
        rollouts_dir = Path(rollouts_dir, "real_rollouts")

    assert rollouts_dir.exists(), f"Rollout directory {rollouts_dir} does not exist"
    visualize_2d_predictions(rollouts_dir, args.vis_type, args.no_tracks)
