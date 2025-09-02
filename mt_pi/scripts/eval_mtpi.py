"""
Example usage:
python scripts/eval_mtpi.py --env_cfg_path envs/fr2.yaml --show_camera False --log_dir experiment_logs/tracks-image-unet/115719_a36186
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pyrallis
import torch
from dataset.image_track_dataset import ACTION_TYPE_TRACKS
from envs.franka_env import FrankaEnv, FrankaEnvConfig
from models.diffusion_policy import DiffusionPolicy
from models.multiview_wrapper import MultiViewWrapper
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils.checkpointer import Checkpointer
from utils.constants import CALIB_DIR, FRANKA_ENV_CONFIG_PATH
from utils.evals import (
    EvalConfig,
    check_for_interrupt,
    check_grasp_logic,
    triangulate_tracks,
)
from utils.freq_guard import FreqGuard
from utils.recorders import StreamRecorder
from utils.stopwatch import Stopwatch
from utils.visualizations import save_frames, visualize_3d


@dataclass
class EvalOnlineConfig(EvalConfig):
    max_len: int = 100
    num_episodes: int = 10
    env_cfg_path: str = FRANKA_ENV_CONFIG_PATH
    record: bool = True

    @property
    def env_cfg(self):
        return pyrallis.load(FrankaEnvConfig, open(self.env_cfg_path, "r"))


def reset_to_home(
    env: FrankaEnv,
    recorder: StreamRecorder = None,
    stopwatch: Stopwatch = None,
):
    """Reset the environment to its home position."""
    if recorder:
        recorder.end_episode(save=True, save_gif=False)
    if stopwatch:
        stopwatch.summary(reset=True)
    env.reset()
    print("Environment reset to home position.")


def run_episode_tracks(
    cfg: EvalOnlineConfig,
    multi_view_policy: MultiViewWrapper,
    env: FrankaEnv,
    stats_per_vp: Dict[str, Dict[str, np.ndarray]],
    intrinsics: Dict[str, np.ndarray],
    transforms: Dict[str, np.ndarray],
    stopwatch: Stopwatch,
    recorder: StreamRecorder = None,
):
    train_cfg = cfg.train_cfg[0]
    dataset_cfg = train_cfg.dataset_cfg

    with stopwatch.time("reset"):
        env.reset()

    vis_data = []
    obs_history = deque(maxlen=dataset_cfg.obs_horizon)

    viewpoints = sorted(
        set(vp for cfg in cfg.train_cfg for vp in cfg.dataset_cfg.viewpoints)
    )
    num_grasps_preds = {vp: 0 for vp in viewpoints}
    num_non_grasps_preds = {vp: 0 for vp in viewpoints}

    print("Starting rollout. Press Enter at any time to interrupt and reset.")

    for _ in tqdm(range(cfg.max_len), desc="Running Episode"):
        if check_for_interrupt():
            print("Rollout interrupted by user.")
            reset_to_home(env, recorder, stopwatch)
            return

        with FreqGuard(cfg.freq):
            with stopwatch.time("observe"):
                obs = env.observe()
        obs_history.append(obs)
        while len(obs_history) < dataset_cfg.obs_horizon:
            obs_history.append(obs)

        info = triangulate_tracks(
            cfg=cfg,
            obs=list(obs_history),
            policy=multi_view_policy,
            stats=stats_per_vp,
            steps_to_solve=cfg.steps_to_solve,
            action_horizon=dataset_cfg.action_horizon,
            intrinsics=intrinsics,
            transforms=transforms,
            add_grasp_info_to_tracks=dataset_cfg.add_grasp_info_to_tracks,
            stopwatch=stopwatch,
        )
        vis_data.append(info)
        if recorder:
            info["raw_obs"] = obs
            recorder.record(obs=info)

        print(f"Steps done: {info['solves_done']}")

        curr_gripper_open = 0 if obs["proprio"][-1] < 0.9 else 1
        num_grasps_preds, num_non_grasps_preds, gripper_open = check_grasp_logic(
            num_grasps_preds,
            num_non_grasps_preds,
            cfg.consec_grasp_thresh,
            info,
            curr_gripper_open=curr_gripper_open,
            viewpoints=viewpoints,
        )

        # gripper_open = 0 if all(grasp_set.values()) else 1
        terminate = all(info[f"agent{vp}"]["terminal"][0] > 0.5 for vp in viewpoints)

        # If gripper action is different from current, move position, then move gripper
        # future_gripper_transform = info["future_gripper_transform"]
        # target_ee_pos = future_gripper_transform[:3, 3]
        # target_ee_rot = future_gripper_transform[:3, :3]
        # target_ee_euler = R.from_matrix(target_ee_rot).as_euler("xyz")
        # env.move_via_waypoints(
        #     target_ee_pos, target_ee_euler, gripper_open, control_freq=cfg.freq
        # )
        # if gripper_open != curr_gripper_open:
        #     env.apply_action(
        #         ee_pos=np.array([0, 0, 0]),
        #         ee_euler=np.array([0, 0, 0]),
        #         gripper_open=gripper_open,
        #         is_delta=True,
        #     )

        # curr_proprio = obs["proprio"]
        # Don't send action if gripper close predicted for the first time
        # if gripper_open == 1 or curr_proprio[-1] < 0.9:
        #   future_gripper_transform = info["future_gripper_transform"]
        #   target_ee_pos = future_gripper_transform[:3, 3]
        #   target_ee_rot = future_gripper_transform[:3, :3]
        #   target_ee_euler = R.from_matrix(target_ee_rot).as_euler("xyz")
        #   env.move_via_waypoints(
        #       target_ee_pos, target_ee_euler, gripper_open, control_freq=cfg.freq
        #   )
        # else:
        #    target_ee_pos = obs["proprio"][:3]
        #    target_ee_euler = obs["proprio"][3:6]

        future_gripper_transform = info["future_gripper_transform"]
        target_ee_pos = future_gripper_transform[:3, 3]
        target_ee_rot = future_gripper_transform[:3, :3]
        target_ee_euler = R.from_matrix(target_ee_rot).as_euler("xyz")
        env.move_via_waypoints(
            target_ee_pos,
            target_ee_euler,
            gripper_open,
            control_freq=cfg.freq,
            gen_ori_separately=False,
        )

        # Force sending gripper close a few times, if grasp is predicted
        if not gripper_open:
            for i in range(25):
                env.apply_action(np.zeros(3), np.zeros(3), gripper_open=gripper_open)

        if terminate:
            break

    frames = visualize_3d(vis_data, cfg)
    if len(cfg.viewpoint_jsons) > 1:
        viewpoint_str = "_and_".join(Path(vp).stem for vp in cfg.viewpoint_jsons)
        ckpt_stems = "_".join(Path(log_dir).stem for log_dir in cfg.log_dir)
        save_frames(
            frames,
            cfg.vis_file_type,
            file_name=f"onpolicy_{viewpoint_str}_{ckpt_stems}",
        )
    if recorder:
        recorder.end_episode(save=cfg.record, save_gif=cfg.record)


def main(cfg: EvalOnlineConfig):
    # Load intrinsics and transforms
    intrinsics = {
        "agent1": np.load(f"{CALIB_DIR}/agent1_intrinsics.npy"),
        "agent2": np.load(f"{CALIB_DIR}/agent2_intrinsics.npy"),
    }
    transforms = np.load(f"{CALIB_DIR}/transforms_both.npy", allow_pickle=True).item()

    # Initialize environment
    env = FrankaEnv(cfg.env_cfg)
    env.reset()

    if "depth" in cfg.train_cfg[0].policy_cfg.cam_names:
        env.add_depth_anything_to_obs()

    for _ in range(int(cfg.freq) * 5):
        with FreqGuard(cfg.freq):
            env.observe()

    if cfg.record:
        recorder_dir = Path(Path(cfg.log_dir[0]), "real_rollouts")
        recorder_dir.mkdir(parents=False, exist_ok=True)
        additional_info = {
            "loaded_checkpoint": [log_dir for log_dir in cfg.log_dir],
            "transforms": transforms,
        }
        recorder = StreamRecorder(str(recorder_dir), additional_info)
    else:
        recorder = None

    policies = []
    stats_per_vp = defaultdict(dict)
    for log_dir, train_cfg in zip(cfg.log_dir, cfg.train_cfg):
        dataset_cfg = train_cfg.dataset_cfg
        policy_cfg = train_cfg.policy_cfg

        # Initialize policy and load checkpoint
        policy_kwargs = dict(
            action_dim=train_cfg.action_dim,
            state_cond_dim=train_cfg.state_cond_dim,
            num_points=dataset_cfg.num_points,
            obs_horizon=dataset_cfg.obs_horizon,
            pred_horizon=dataset_cfg.pred_horizon,
            action_horizon=dataset_cfg.action_horizon,
            cfg=policy_cfg,
        )
        policy = DiffusionPolicy(**policy_kwargs)
        policy = policy.to(cfg.device)
        policy = torch.compile(policy)
        checkpointer = Checkpointer(Path(log_dir))
        checkpointer.update_checkpoint_from_dir(Path(log_dir, "checkpoints"))
        policy, _ = Checkpointer.load_checkpoint(
            policy, checkpointer, checkpoint_type=cfg.ckpt
        )
        policy.eval()

        # Get the viewpoint for this policy
        for viewpoint in train_cfg.dataset_cfg.viewpoints:
            policies.append((viewpoint, policy))
            stats_per_vp[viewpoint] = np.load(
                f"{log_dir}/dataset_stats/train.npy", allow_pickle=True
            ).item()

    # Wrap policies with MultiViewWrapper
    multi_view_policy = MultiViewWrapper(policies)

    stopwatch = Stopwatch()
    for eps_num in range(cfg.num_episodes):
        print(f"Running episode {eps_num}")
        if cfg.train_cfg[0].action_pred_type == ACTION_TYPE_TRACKS:
            run_episode_tracks(
                cfg=cfg,
                multi_view_policy=multi_view_policy,
                env=env,
                stats_per_vp=stats_per_vp,
                intrinsics=intrinsics,
                transforms=transforms,
                stopwatch=stopwatch,
                recorder=recorder,
            )
        else:
            raise NotImplementedError("Robot action type not implemented yet")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = pyrallis.parse(config_class=EvalOnlineConfig)
    main(cfg)
