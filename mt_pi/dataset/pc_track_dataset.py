from typing import List

import numpy as np
import torch
import zarr

from dataset.utils import (
    create_sample_indices,
    downsample_pc,
    get_data_stats,
    normalize_data,
    sample_sequence,
)


def remove_padding_from_point_cloud(
    point_cloud: np.ndarray, rgb: np.ndarray
) -> np.ndarray:
    """
    This function removes the padding from the point cloud. The padding is added by the add_padding_to_point_cloud function.

    Parameters:
    - point_cloud (np.ndarray): The point cloud to remove padding from.
    - rgb (np.ndarray): The rgb of point cloud to remove padding from.

    Returns:
    - np.ndarray: The point cloud with padding removed.
    """
    special_point = point_cloud[-1]
    if special_point[0] != -121 or special_point[1] != -141:
        # No padding was added
        return point_cloud, rgb
    original_length = int(special_point[2])
    return point_cloud[:original_length], rgb[:original_length]


class PointCloudTrackDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_paths: List[str],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        num_points: int,
        add_grasp_info_to_tracks: bool = False,
        additional_kwargs: dict = {},
    ):
        """
        The current design of the function loads all datasets, computes the normalization
        stats over the aggregate, and then normalizes the data.
        """
        self.indices = []
        self.stats = {}
        self.normalized_train_data = {}

        # First pass: load data and compute global statistics
        all_data = {
            key: []
            for key in ["agent_pos", "action", "point_cloud", "point_cloud_full"]
        }
        cumulative_length = 0

        for data_path in data_paths:
            dataset = self._load_single_dataset(
                data_path,
                pred_horizon,
                obs_horizon,
                action_horizon,
                num_points,
                add_grasp_info_to_tracks,
                additional_kwargs,
            )
            # Adjust indices for the current dataset
            dataset_indices = [
                (
                    start + cumulative_length,
                    end + cumulative_length,
                    sample_start,
                    sample_end,
                )
                for start, end, sample_start, sample_end in dataset["indices"]
            ]
            self.indices.extend(dataset_indices)
            # Collect all data for global statistics
            for key in all_data:
                all_data[key].append(dataset["train_data"][key])
            cumulative_length += len(dataset["indices"])

        # Concatenate all data, compute global statistics, and normalize
        for key in all_data:
            all_data[key] = np.concatenate(all_data[key], axis=0)
            self.stats[key] = get_data_stats(all_data[key])
            self.normalized_train_data[key] = normalize_data(
                all_data[key], self.stats[key]
            )

        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self._num_points = num_points
        self._add_grad_info_to_points = add_grasp_info_to_tracks

    def _load_single_dataset(
        self,
        data_path,
        pred_horizon,
        obs_horizon,
        action_horizon,
        num_points,
        add_grasp_info_to_tracks,
        additional_kwargs,
    ):
        assert data_path.exists(), f"Dataset path {data_path} does not exist"
        dataset_root = zarr.open(data_path, "r")

        points = dataset_root["data"]["points"][:]
        colors = dataset_root["data"]["colors"][:]
        pc_subsample_size = min(
            additional_kwargs["pc_subsample_size"],
            points.shape[1],
        )
        points, colors = downsample_pc(points, colors, pc_subsample_size)
        points_colors = np.concatenate([points, colors], axis=-1)
        points_colors_full = np.concatenate(
            [dataset_root["data"]["points"][:], dataset_root["data"]["colors"][:]],
            axis=-1,
        )
        train_data = {
            "agent_pos": dataset_root["data"]["proprio"][:].astype(np.float32),
            "action": dataset_root["data"]["action3d"][:].astype(np.float32),
            "point_cloud": points_colors.astype(np.float32),
            "point_cloud_full": points_colors_full.astype(np.float32),
        }
        train_data["action"] = self._reshape_action_data(
            train_data, num_points, add_grasp_info_to_tracks
        )
        if train_data["agent_pos"].shape[1] == 1:
            # NOTE: hard coded for now
            train_data["agent_pos"] = np.zeros((train_data["agent_pos"].shape[0], 7))
        episode_ends = dataset_root["meta"]["episode_ends"][:]

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        return {
            "indices": indices,
            "train_data": train_data,  # 'train_data' is a dictionary containing 'agent_pos', 'action', 'point_cloud', 'point_cloud_full'
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["point_cloud"] = nsample["point_cloud"][: self.obs_horizon, ...]
        nsample["agent_pos"] = nsample["agent_pos"][: self.obs_horizon, ...]
        return nsample

    def _reshape_action_data(self, train_data, num_points, add_grasp_info_to_tracks):
        action, proprio = train_data["action"], train_data["agent_pos"]

        if add_grasp_info_to_tracks:
            # (N, total_points, 3) -> (N, total_points, 4)
            raise NotImplementedError(
                "Grasp info should be added to the end of the action data, this was refactored on 07-10 in image_track_dataset.py, but not replicated in pc_track_dataset.py"
            )
            gripper_pos = proprio[:, -1:]
            gripper_pos = np.expand_dims(gripper_pos, axis=1)
            gripper_pos = np.repeat(gripper_pos, action.shape[1], axis=1)
            action = np.concatenate([action, gripper_pos], axis=-1)

        # (N, total_points, 3) -> (N, total_points*3)
        action = action.reshape(action.shape[0], -1)
        # (N, total_points*4) -> (N, num_points*3)
        action = action[:, : num_points * (3 + int(add_grasp_info_to_tracks))]
        return action
