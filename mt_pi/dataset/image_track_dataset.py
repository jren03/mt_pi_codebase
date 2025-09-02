from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import zarr
from termcolor import cprint
from tqdm import tqdm, trange

from dataset.image_transforms import (
    get_augmentation_transform_by_name,
    image_preprocess_transform,
    transform_image_and_action_kpts,
)
from dataset.utils import (
    KPT_TYPE_TO_KEYS,
    LH_POINT_TO_IDX,
    RH_POINT_TO_IDX,
    ROBOT_POINT_TO_IDX,
    create_sample_indices,
    get_data_stats,
    normalize_data,
    sample_sequence,
)

ACTION_TYPE_ROBOT = "robot"
ACTION_TYPE_TRACKS = "tracks"
ACTION_TYPES = [ACTION_TYPE_ROBOT, ACTION_TYPE_TRACKS]

Batch = namedtuple("Batch", ["obs", "action", "lang", "state_cond", "depth", "label"])


@dataclass
class ImageTrackDatasetConfig:
    # Image settings
    image_dim: int = 128
    normalize_image: bool = True
    transforms_name: str = "medium-geometric"

    # Horizon settings
    pred_horizon: int = 16
    action_horizon: int = 14
    obs_horizon: int = field(default_factory=lambda: 1)

    # Action settings
    action_pred_type: str = ACTION_TYPE_TRACKS
    kpt_type: str = "all_low"  # ['all', 'rh', 'lh', 'both3', 'both4']
    jitter_kpts: bool = False
    fixed_kpt_dims: List[str] = field(default_factory=lambda: ["wrist_low"])

    # Language settings
    lang_instruction: str = ""

    # Grasp information
    add_grasp_info_to_tracks: bool = True

    # Data paths and keys
    data_paths: List[Path] = field(default_factory=list)
    viewpoints: List[int] = field(default_factory=lambda: [1, 2])

    # Additional settings
    add_in_wrist: bool = False
    add_depth_gs: bool = False
    add_depth_colored: bool = False
    hand_train_percentage: float = 1.0
    robot_train_percentage: float = 1.0

    def __post_init__(self):
        assert self.action_pred_type in ACTION_TYPES, (
            f"Invalid action prediction type {self.action_pred_type}. "
        )
        assert self.kpt_type in KPT_TYPE_TO_KEYS.keys(), (
            f"Invalid keypoint subset {self.kpt_type}."
        )

        if self.hand_train_percentage < 1.0:
            cprint(
                f"[WARNING] Using only {self.hand_train_percentage} of human data",
                "yellow",
                attrs=["bold"],
            )
        if self.robot_train_percentage < 1.0:
            cprint(
                f"[WARNING] Using only {self.robot_train_percentage} of robot data",
                "yellow",
                attrs=["bold"],
            )

    @property
    def kpt_keys(self) -> List[str]:
        return KPT_TYPE_TO_KEYS[self.kpt_type]

    @property
    def num_points(self) -> int:
        return len(self.kpt_keys)


class ImageTrackDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: ImageTrackDatasetConfig):
        self.indices = {"hand": [], "robot": []}
        self.stats = {}
        self.pred_horizon = cfg.pred_horizon
        self.action_horizon = cfg.action_horizon
        self.obs_horizon = cfg.obs_horizon
        self.action_pred_type = cfg.action_pred_type
        self.lang_instruction = cfg.lang_instruction
        self._image_dim = cfg.image_dim
        self._add_grasp_info_to_tracks = cfg.add_grasp_info_to_tracks
        self._grasp_and_term_dim = 1 + int(cfg.add_grasp_info_to_tracks)
        self._hand_train_percentage = cfg.hand_train_percentage
        self._robot_train_percentage = cfg.robot_train_percentage
        self._kpt_keys = cfg.kpt_keys
        self._num_points = cfg.num_points
        self.cfg = cfg

        self.preprocess_tf = image_preprocess_transform(cfg.image_dim)
        train_data = {
            key: {"hand": [], "robot": []}
            # for key in ["obs", "wrist", "action", "state_cond"]
            for key in ["obs", "action", "state_cond", "depth"]
        }
        cumulative_length = {"hand": 0, "robot": 0}

        pbar = tqdm(cfg.data_paths, desc="Loading datasets", leave=False)
        for data_path in pbar:
            pbar.set_description(f"Loading {data_path.parent.parent}")
            dataset = self._load_single_dataset(data_path, cfg.viewpoints)
            dataset_type = "hand" if "hand" in str(data_path) else "robot"

            dataset_indices = [
                (
                    start + cumulative_length[dataset_type],
                    end + cumulative_length[dataset_type],
                    sample_start,
                    sample_end,
                )
                for start, end, sample_start, sample_end in dataset["indices"]
            ]
            self.indices[dataset_type].extend(dataset_indices)

            for key in train_data:
                train_data[key][dataset_type].append(dataset["train_data"][key])
            cumulative_length[dataset_type] += len(dataset["indices"])

        for key in train_data:
            for dtype in ["hand", "robot"]:
                if train_data[key][dtype]:
                    train_data[key][dtype] = np.concatenate(
                        train_data[key][dtype], axis=0
                    )

        all_data = {}
        for key in train_data:
            if len(train_data[key]["hand"]) > 0 and len(train_data[key]["robot"]) > 0:
                all_data[key] = np.concatenate(
                    [train_data[key]["hand"], train_data[key]["robot"]], axis=0
                )
            elif len(train_data[key]["hand"]) > 0:
                all_data[key] = np.array(train_data[key]["hand"])
            else:
                all_data[key] = np.array(train_data[key]["robot"])

        for key in all_data:
            self.stats[key] = get_data_stats(all_data[key])

        self.obs_augment_tf = get_augmentation_transform_by_name(
            cfg.transforms_name,
            obs_stats=self.stats["obs"],
            normalize_image=cfg.normalize_image,
        )
        self.depth_augment_tf = get_augmentation_transform_by_name(
            name="none",
            obs_stats=self.stats["depth"],
            normalize_image=cfg.normalize_image,
        )
        self.train_data = train_data
        self.all_data = all_data

        _lengths = {k: len(v) for k, v in self.indices.items()}
        cprint(
            f"Loaded dataset with lengths: {_lengths}", color="green", attrs=["bold"]
        )

    @property
    def action_dim(self):
        return self.stats["action"]["max"].shape[0]

    @property
    def state_cond_dim(self):
        return self.stats["state_cond"]["max"].shape[0]

    def __len__(self) -> int:
        # Return the maximum number of samples possible
        # Use modulo is wrap around in the smaller dataset
        if len(self.indices["hand"]) == 0:
            return len(self.indices["robot"])
        elif len(self.indices["robot"]) == 0:
            return len(self.indices["hand"])
        else:
            return max(len(self.indices["hand"]), len(self.indices["robot"])) * 2

    def __getitem__(self, idx: int) -> Batch:
        # Determine which dataset to sample from
        dataset_type = "hand" if idx % 2 == 0 else "robot"

        # If one dataset is empty, always sample from the other
        if not self.indices["hand"]:
            dataset_type = "robot"
            adjusted_idx = idx
        elif not self.indices["robot"]:
            dataset_type = "hand"
            adjusted_idx = idx
        else:
            adjusted_idx = idx // 2 % len(self.indices[dataset_type])

        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[dataset_type][adjusted_idx]
        )

        nsample = sample_sequence(
            train_data={
                key: self.train_data[key][dataset_type] for key in self.train_data
            },
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # Rest of the method remains the same
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        nsample["depth"] = nsample["depth"][: self.obs_horizon, :]

        raw_action = nsample["action"][:, : -self._grasp_and_term_dim]
        nsample["obs"], transformed_action = transform_image_and_action_kpts(
            nsample["obs"], raw_action, self.obs_augment_tf
        )
        nsample["depth"], _ = transform_image_and_action_kpts(
            nsample["depth"], None, self.depth_augment_tf
        )

        if self.cfg.jitter_kpts and np.random.rand() < 0.5:
            transformed_action = self._jitter_keypoints(
                transformed_action, self.cfg.fixed_kpt_dims
            )

        nsample["action"] = np.concatenate(
            [transformed_action, nsample["action"][:, -self._grasp_and_term_dim :]],
            axis=1,
        )
        nsample["action"] = normalize_data(nsample["action"], self.stats["action"])

        if self.action_pred_type == ACTION_TYPE_TRACKS:
            state_cond = nsample["action"][: self.obs_horizon, : self._num_points * 2]
            nsample["state_cond"] = deepcopy(state_cond)
        else:
            nsample["state_cond"] = normalize_data(
                nsample["state_cond"][: self.obs_horizon, :], self.stats["state_cond"]
            )

        nsample["lang"] = self.lang_instruction
        nsample["label"] = 0 if dataset_type == "hand" else 1

        nsample = Batch(**nsample)
        return nsample

    def _load_single_dataset(self, data_path: Path, viewpoints: List[int]):
        assert data_path.exists(), f"Dataset path {data_path} does not exist"
        dataset_root = zarr.open(data_path, "r")
        is_hand_data = "hand" in str(data_path)

        # Take only self._train_percentage of the data
        if "val" in str(data_path):
            # Validation set is always 100%
            percentage = 1.0
        elif is_hand_data:
            percentage = self._hand_train_percentage
        else:
            percentage = self._robot_train_percentage

        action_data = []
        image_data = []
        depth_data = []
        grasp_data = []
        state_cond_data = []
        episode_ends = []
        num_samples_per_vp = {}
        prev_episode_end = 0

        for viewpoint in viewpoints:
            # Process image + action
            if self.action_pred_type == ACTION_TYPE_TRACKS:
                action_key = f"track{viewpoint}"
                resize_actions = True
            else:
                action_key = "robot_action"
                resize_actions = False
            image_key = f"img{viewpoint}"

            # Calculate initial num_samples
            initial_num_samples = int(
                percentage * len(dataset_root["data"][action_key][:])
            )

            # Get episode_ends for this viewpoint
            if is_hand_data:
                current_episode_ends = dataset_root["meta"][f"episode_ends{viewpoint}"][
                    :
                ]
            else:
                current_episode_ends = dataset_root["meta"]["episode_ends"][:]

            # Find the nearest episode end to initial_num_samples
            nearest_episode_end = min(
                current_episode_ends, key=lambda x: abs(x - initial_num_samples)
            )
            num_samples = nearest_episode_end

            # Truncate episode_ends
            episode_end_index = (
                current_episode_ends.tolist().index(nearest_episode_end) + 1
            )
            truncated_episode_ends = current_episode_ends[:episode_end_index]

            num_samples_per_vp[viewpoint] = num_samples

            # Truncate and process data
            original_image = dataset_root["data"][image_key][:num_samples]
            original_action = dataset_root["data"][action_key][:num_samples]
            resized_image, resized_action = self._preprocess_transforms(
                original_image,
                original_action,
                resize_actions=resize_actions,
            )
            image_data.append(resized_image)
            action_data.append(resized_action)

            # Process possible depth_data
            if f"depth{viewpoint}_gs" in dataset_root["data"]:
                if self.cfg.add_depth_colored:
                    resized_depth, _ = self._preprocess_transforms(
                        dataset_root["data"][f"depth{viewpoint}_colored"][:num_samples]
                    )
                else:
                    depth_gs = dataset_root["data"][f"depth{viewpoint}_gs"][
                        :num_samples
                    ]
                    if len(depth_gs.shape) == 3:
                        depth_gs = depth_gs[:, :, :, np.newaxis]
                    resized_depth, _ = self._preprocess_transforms(depth_gs)
                depth_data.append(resized_depth)
            elif self.cfg.add_depth_colored or self.cfg.add_depth_gs:
                raise ValueError("Depth data not found in the dataset")
            else:
                depth_data.append(np.zeros_like(resized_image))

            # Process state_cond for robot actions
            if self.action_pred_type == ACTION_TYPE_ROBOT:
                state_cond = dataset_root["data"]["proprio"][:num_samples]
                state_cond_data.append(state_cond)

            # Process grasp info
            if is_hand_data:
                grasp_data.append(
                    dataset_root["gripper_open"][f"vp{viewpoint}"][:num_samples]
                )
            else:
                grasp_data.append(dataset_root["data"]["gripper_open"][:num_samples])

            # Process episode ends
            episode_ends.append(truncated_episode_ends + prev_episode_end)
            prev_episode_end = episode_ends[-1][-1]

        image_data = np.concatenate(image_data, axis=0)
        depth_data = np.concatenate(depth_data, axis=0)
        action_data = np.concatenate(action_data, axis=0)
        grasp_data = np.concatenate(grasp_data, axis=0) if grasp_data != [] else []
        episode_ends = np.concatenate(episode_ends, axis=0)

        # Assumes same hand is present throughout the episode. Robot data defaults to left hand.
        if is_hand_data:
            is_right = np.concatenate(
                [
                    dataset_root["data"][f"is_right{vp}"][: num_samples_per_vp[vp]]
                    for vp in viewpoints
                ],
                axis=0,
            )
            is_right = np.logical_and(is_hand_data, is_right)
        else:
            is_right = None

        if self.action_pred_type == ACTION_TYPE_TRACKS:
            action_data = self._reshape_action_data(
                action_data, self._num_points, is_hand_data, is_right=is_right
            )
            state_cond_data = deepcopy(action_data)
        else:
            state_cond_data = np.concatenate(state_cond_data, axis=0)

        # action_data is (N, num_points*2 + grasp_info + terminal_info)
        action_data = np.concatenate(
            [action_data, np.zeros((action_data.shape[0], self._grasp_and_term_dim))],
            axis=1,
        )

        # Set the set of indices of action_horizon before each episode_ends is a terminal index
        terminal_flags = np.zeros((action_data.shape[0],))
        # NOTE: This is not used for now
        # terminal_indices = [
        #     max(0, idx - ahor)
        #     for idx in episode_ends
        #     for ahor in range(1, self.action_horizon + 1)
        # ]
        # terminal_flags[terminal_indices] = 1
        action_data[:, -1] = terminal_flags
        if self._add_grasp_info_to_tracks:
            action_data[:, -2] = grasp_data

        train_data = {
            "obs": image_data.astype(np.uint8),
            "depth": depth_data.astype(np.uint8),
            "action": action_data.astype(np.float32),
            "state_cond": state_cond_data.astype(np.float32),
        }
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=self.pred_horizon,
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1,
        )
        return {
            "indices": indices,
            "train_data": train_data,
        }

    def _preprocess_transforms(
        self,
        original_image,
        original_action=None,
        resize_actions=True,
    ):
        """Preprocess image (and optionally action) by resizing."""
        resize_actions = resize_actions and original_action is not None

        resized_imgs, resized_actions = [], []
        for i in trange(len(original_image), desc="Process Transforms", leave=False):
            image = original_image[i]
            if resize_actions:
                action = original_action[i]
                action = action.reshape((-1))
                image, action = transform_image_and_action_kpts(
                    image, action, self.preprocess_tf
                )
                action = action.reshape((-1, 2))
                resized_imgs.append(image)
                resized_actions.append(action)
            else:
                image, _ = transform_image_and_action_kpts(
                    image, None, self.preprocess_tf
                )
                resized_imgs.append(image)

        resized_imgs = np.array(resized_imgs)
        resized_actions = (
            np.array(resized_actions) if resize_actions else original_action
        )

        return resized_imgs, resized_actions

    def _reshape_action_data(
        self,
        action: np.ndarray,
        num_points: int,
        is_hand_data: bool,
        is_right: List[bool] = None,
    ) -> np.ndarray:
        """
        Reshapes actions from (N, total_points, 2) to (N, num_points*2)
        """
        assert len(action.shape) == 3, (
            f"Action data should be (N, total_points, 2), got {action.shape}"
        )

        if is_hand_data:
            assert is_right is not None, "is_right must be provided for hand data"
            assert len(is_right) == action.shape[0], (
                f"Length of is_right ({len(is_right)}) must match number of samples in action ({action.shape[0]})"
            )

        # Prepare the output array
        output = np.zeros((action.shape[0], num_points, 2), dtype=action.dtype)

        if is_hand_data:
            for i, right_hand in enumerate(is_right):
                point_to_idx = RH_POINT_TO_IDX if right_hand else LH_POINT_TO_IDX
                kpt_idxs = [point_to_idx[key] for key in self._kpt_keys[:num_points]]
                output[i] = action[i, kpt_idxs, :]
        else:
            # For robot data, use ROBOT_POINT_TO_IDX for all samples
            kpt_idxs = [ROBOT_POINT_TO_IDX[key] for key in self._kpt_keys[:num_points]]
            output = action[:, kpt_idxs, :]

        # (N, num_points, 2) -> (N, num_points*2)
        output = output.reshape(output.shape[0], -1)
        return output.astype(np.float32)

    def _jitter_keypoints(
        self,
        keypoints: Union[np.ndarray, torch.Tensor],
        fixed_kpt_dims: List[int] = None,
        batched=False,
        post_normalization=False,
    ) -> np.ndarray:
        """
        Assumes keypoints is of shape (B, obs_horizon, num_points*2)
        and will return the same shape.
        """
        is_numpy = isinstance(keypoints, np.ndarray)
        reshape = np.reshape if is_numpy else torch.reshape
        # normal = np.random.normal if is_numpy else torch.normal
        noise = (
            np.random.uniform
            if is_numpy
            else lambda _, scale, size: torch.rand(size) * scale
        )
        array = np.array if is_numpy else torch.tensor
        zeros = np.zeros if is_numpy else torch.zeros
        _float_type = np.float32 if is_numpy else torch.float32
        _to_type = lambda x: x.astype(_float_type) if is_numpy else x.to(_float_type)

        # Create jitter of (transformed_action.shape[1]//2, 2) for each track
        if not batched:
            keypoints = reshape(keypoints, (1, -1, self.cfg.num_points * 2))

        # (B, obs_hor, num_points*2,)
        jitter = noise(0, 8, keypoints.shape)
        jitter = _to_type(jitter)
        if post_normalization:
            # keypoints already normalized to [-1, 1]
            jitter = normalize_data(jitter, self.stats["state_cond"])

        if fixed_kpt_dims:
            fixed_idx = array([self._kpt_keys.index(k) for k in fixed_kpt_dims])
            for idx in fixed_idx:
                jitter[..., idx : idx + 2] = zeros(2, dtype=_float_type)
        keypoints_jittered = keypoints + jitter

        # keypoints = reshape(keypoints, (*keypoints.shape[:2], -1, 2))
        # keypoints_jittered = keypoints + jitter
        # keypoints_jittered = reshape(
        #     keypoints_jittered, (*keypoints_jittered.shape[:2], -1)
        # )

        if not batched:
            keypoints_jittered = keypoints_jittered[0]
        return keypoints_jittered


def create_dataloader(
    partition: str,
    log_dir: str,
    batch_size: int,
    num_workers: int,
    data_paths: List[str],
    dataset_cfg: ImageTrackDatasetConfig,
    dataloader_overrides: dict = None,
    no_transforms: bool = False,
    human_only_train: bool = False,
):
    dataset_cfg.data_paths = []
    for i, d in enumerate(data_paths):
        zarr_path = Path(f"data/{d}/{partition}/dataset.zarr")

        if zarr_path.exists():
            if partition == "train" and human_only_train and "robot" in str(zarr_path):
                continue

            dataset_cfg.data_paths.append(zarr_path)
        else:
            cprint(
                f"[WARNING]: Skipping {zarr_path} as it does not exist", color="yellow"
            )

    if len(dataset_cfg.data_paths) == 0:
        cprint(f"[ERROR]: No valid data paths found for {partition}", color="red")
        return None, None

    if no_transforms:
        original_transforms = dataset_cfg.transforms_name
        dataset_cfg.transforms_name = "none"
    dataset = ImageTrackDataset(dataset_cfg)
    if no_transforms:
        dataset_cfg.transforms_name = original_transforms

    dset_str = ",".join([path.parent.parent.name for path in dataset_cfg.data_paths])
    cprint(
        f" Created {partition.capitalize()} dataset with {len(dataset)} samples from [{dset_str}]",
        color="cyan",
        attrs=["bold"],
    )
    stats = dataset.stats
    Path(log_dir, "dataset_stats").mkdir(parents=True, exist_ok=True)
    np.save(f"{log_dir}/dataset_stats/{partition}.npy", stats)

    base_dataloader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    if dataloader_overrides is not None:
        base_dataloader_kwargs.update(dataloader_overrides)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **base_dataloader_kwargs,
    )

    return dataloader, stats


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image

    from dataset.utils import unnormalize_data

    data_paths = [
        Path("data/robot_pick_tracks/val/dataset.zarr"),
    ]
    obs_horizon = 3
    pred_horizon = 16
    action_horizon = 5
    num_points = 5
    img_dim = 128
    transforms_name = "medium-affine"
    # action_pred_type = ACTION_TYPE_ROBOT
    action_pred_type = ACTION_TYPE_TRACKS
    add_grasp_info_to_tracks = True
    dataset = ImageTrackDataset(
        ImageTrackDatasetConfig(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            transforms_name=transforms_name,
            action_pred_type=action_pred_type,
            image_dim=img_dim,
            add_grasp_info_to_tracks=add_grasp_info_to_tracks,
            viewpoints=[2, 1],
            data_paths=data_paths,
        )
    )
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    batch: Batch = next(iter(dataloader))

    extra_action_dims = 1 + int(add_grasp_info_to_tracks)
    assert batch.obs.shape == (
        batch_size,
        obs_horizon,
        3,
        img_dim,
        img_dim,
    ), f"{batch.obs.shape=}"
    if action_pred_type == ACTION_TYPE_ROBOT:
        assert batch.action.shape == (
            batch_size,
            pred_horizon,
            7 + 1,  # 7 for robot actions + 1 for terminal flag
        ), f"{batch.action.shape=}"
        assert batch.state_cond.shape == (
            batch_size,
            obs_horizon,
            7,  # robot proprioception,
        ), f"{batch.state_cond.shape=}"
    elif action_pred_type == ACTION_TYPE_TRACKS:
        assert batch.action.shape == (
            batch_size,
            pred_horizon,
            num_points * 2 + extra_action_dims,
        ), f"{batch.action.shape=}"
        assert batch.state_cond.shape == (
            batch_size,
            obs_horizon,
            num_points * 2,
        ), f"{batch.state_cond.shape=}"

        one_image, one_action = batch.obs[0][0], batch.action[0]
        one_image = one_image.permute(1, 2, 0).numpy()
        one_image = unnormalize_data(one_image, dataset.stats["obs"]).astype(np.uint8)
        one_image_vis = one_image.copy()
        one_action = unnormalize_data(one_action, dataset.stats["action"])
        one_action = one_action[:, :-extra_action_dims]  # remove terminal flag

        cmap = plt.get_cmap("coolwarm")
        colors = cmap(np.linspace(0, 1, pred_horizon + 1))
        num_circles = 0
        for timestep, action in enumerate(one_action):
            color = colors[timestep]
            color = list(map(int, color * 255))[:3]
            for u, v in action.reshape(-1, 2):
                cv2.circle(one_image_vis, (int(u), int(v)), 2, color, -1)
        img_pil = Image.fromarray(one_image_vis)
        img_pil.save("test_image_track_dataset.png")
        print(
            "Check test_image_track_dataset.png to see if it is RGB and tracks are labelled properly"
        )
        one_image_state_cond = one_image.copy()
        state_cond = unnormalize_data(batch.state_cond[0], dataset.stats["state_cond"])
        for timestep, action in enumerate(state_cond):
            color = colors[timestep * 2]
            color = list(map(int, color * 255))[:3]
            for u, v in action.reshape(-1, 2):
                cv2.circle(one_image_state_cond, (int(u), int(v)), 2, color, -1)
        img_pil = Image.fromarray(one_image_state_cond)
        img_pil.save("test_image_track_dataset_state_cond.png")
        print(
            "Check test_image_track_dataset_state_cond.png to see if it is RGB and conditioned points are labelled properly"
        )
