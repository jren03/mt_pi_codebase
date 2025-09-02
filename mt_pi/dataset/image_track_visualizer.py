from pathlib import Path

import cv2
import numpy as np
import pyrallis
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.visualizations import get_track_colors, save_frames

from dataset.image_track_dataset import ImageTrackDataset, ImageTrackDatasetConfig
from dataset.utils import unnormalize_data


class ImageTrackDatasetVisualizer:
    def __init__(self, dataset, pred_hor, vis_name="visualizations"):
        self.dataset = dataset
        self.stats = dataset.stats
        self.pred_hor = pred_hor
        self.track_color, self.grasp_color, _ = get_track_colors(pred_hor)

        self.output_name = f"visualizations/{vis_name}"
        Path("visualizations").mkdir(exist_ok=True)

    def visualize_sequence(self, max_frames, fps=10):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        frames = []

        for batch in tqdm(dataloader, desc="Visualizing sequences"):
            # (B, obs_hor, C, H, W) -> (H, W, C)
            image = batch.obs.squeeze(0)[-1].numpy().transpose(1, 2, 0)
            image = unnormalize_data(image, stats=self.stats["obs"]).astype(np.uint8)

            # (B, action_horizon, num_points*2 + 1) -> (action_horizon, num_points, 2)
            actions = batch.action.squeeze(0).numpy()
            actions = unnormalize_data(actions, stats=self.stats["action"])
            actions, gripper_opens, terminals = (
                actions[:, :-2],
                actions[:, -2],
                actions[:, -1],
            )
            actions = actions.reshape((self.pred_hor, -1, 2))

            vis_image = self.plot_2d_tracks(image, actions, gripper_opens)
            frames.append(vis_image)

            if max_frames != -1 and len(frames) >= max_frames:
                break

        # Save frames as MP4
        save_frames(frames, "mp4", self.output_name, fps=fps)

    def plot_2d_tracks(
        self,
        image: np.ndarray,
        points_dt: np.ndarray,
        gripper_opens: np.ndarray,
    ) -> np.ndarray:
        """
        Visualizes 2D points on an image.

        Parameters:
        - image (np.ndarray): The image to visualize the points on.
        - points_dt (np.ndarray): (AGENT_DT, num_points*2)
        - gripper_open (np.ndarray): (AGENT_DT, )

        Returns:
        - np.ndarray: The image with the points visualized.
        """

        vis = image.copy()
        for timestep, points in enumerate(points_dt):
            gripper_open = gripper_opens[timestep] > 0.5
            if gripper_open:
                color = self.track_color[timestep]
            else:
                color = self.grasp_color[timestep]
            color = list(map(int, color * 255))[:3]
            if timestep == 0:
                for i, point in enumerate(points):
                    x, y = point
                    vis = cv2.circle(vis, (int(x), int(y)), 1, color, -1)
                    # vis = cv2.putText(
                    #     vis,
                    #     str(i),
                    #     (int(x), int(y)),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.25,
                    #     color,
                    #     1,
                    #     cv2.LINE_AA,
                    # )
            # else:
            #     for i, point in enumerate(points):
            #         x, y = point
            #         vis = cv2.circle(vis, (int(x), int(y)), 1, color, -1)
            if timestep == 0 and not gripper_open:
                cv2.rectangle(
                    vis,
                    (0, vis.shape[0] - 20),
                    (vis.shape[1], vis.shape[0]),
                    (160, 189, 151),
                    -1,
                )
        return vis


@pyrallis.wrap()
def main(cfg: ImageTrackDatasetConfig):
    dataset = ImageTrackDataset(cfg)
    cfg.normalize_image = False

    data_paths = []
    for dpath in cfg.data_paths:
        # e.g. data/robot_pick_tracks/train/dataset.zarr
        dpath = Path(dpath)
        partition = dpath.parent.stem
        data_name = dpath.parent.parent.stem
        data_paths.append(f"{data_name}_{partition}")
    vis_name = "_".join(data_paths)

    visualizer = ImageTrackDatasetVisualizer(dataset, cfg.pred_horizon, vis_name)

    max_frames = -1
    visualizer.visualize_sequence(max_frames=max_frames)


if __name__ == "__main__":
    main()
