"""
This script trains a diffusion-policy like architecture whose actions are the next `pred_horizon`
points that the end-effector will visit in either 2D pixel-space or 3D space.

Example usage:
python train_mtpi.py --policy unet --input_type pc --data_dir /path/to/data
"""

import os
import warnings


def filter_torch_warning(message, category, filename, lineno, file=None, line=None):
    if (
        category is UserWarning
        and "torch.utils._pytree._register_pytree_node is deprecated" in str(message)
    ):
        return None  # Suppress this specific warning
    return True  # Show other warnings


warnings.showwarning = filter_torch_warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import datetime
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import models.train_utils as train_utils
import numpy as np
import pyrallis
import torch
from dataset.image_track_dataset import (
    ACTION_TYPE_TRACKS,
    ACTION_TYPES,
    ImageTrackDatasetConfig,
    create_dataloader,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from models.diffusion_policy import DiffusionPolicy, DiffusionPolicyConfig
from termcolor import cprint
from tqdm import tqdm, trange
from utils.checkpointer import Checkpointer
from utils.logger import Logger
from utils.visualizations import visualize_image_tracks_2d


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    grad_clip: float = 5
    weight_decay: float = 0
    cosine_schedule: bool = True
    lr_warm_up_steps: int = 0


@dataclass
class TrainConfig:
    data_paths: List[str]
    debug: bool = False

    # Checkpoint Params
    resume_training_from: Optional[str] = None
    resume_ckpt_epoch: bool = False
    lr: Optional[float] = None

    # Logging Params
    log_every: int = 100
    exp: str = ""
    project_name: str = "motion_tracks"
    experiment_name: Optional[str] = None
    log_dir: Optional[Path] = None
    use_tb: bool = False
    use_wandb: bool = True
    print_to_stdout: bool = False
    generate_visualizations: bool = False

    # Training params
    action_dim: int = -1
    state_cond_dim: int = -1
    action_pred_type: str = ACTION_TYPE_TRACKS
    add_grasp_info_to_tracks: bool = True
    seed: int = 123
    epochs: int = 800
    batch_size: int = 128
    num_workers: int = 8
    val_every: int = 20
    use_ema: bool = True
    ema_tau: float = 0.01
    torch_compile: bool = True
    max_vis_frames: int = 100
    device = "cuda"

    # Auxiliary params
    human_only_train: bool = False
    add_in_wrist: bool = False
    add_depth_gs: bool = False
    add_depth_colored: bool = False

    # Additional cfgs
    optim_cfg: OptimizerConfig = field(default_factory=lambda: OptimizerConfig())
    policy_cfg: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)
    dataset_cfg: ImageTrackDatasetConfig = field(
        default_factory=lambda: ImageTrackDatasetConfig()
    )

    def __post_init__(self):
        assert self.action_pred_type in ACTION_TYPES, (
            f"Invalid action_pred_type: {self.action_pred_type}. "
            f"Must be one of {ACTION_TYPES}"
        )

        self.policy_cfg.action_pred_type = self.action_pred_type
        self.dataset_cfg.action_pred_type = self.action_pred_type

        self.policy_cfg.add_grasp_info_to_tracks = self.add_grasp_info_to_tracks
        self.dataset_cfg.add_grasp_info_to_tracks = self.add_grasp_info_to_tracks

        self.policy_cfg.add_in_wrist = self.add_in_wrist
        self.dataset_cfg.add_in_wrist = self.add_in_wrist

        assert not (self.add_depth_gs and self.add_depth_colored), (
            "Cannot add both depth_gs and depth_colored"
        )

        if (
            self.add_depth_gs or self.add_depth_colored
        ) and "depth" not in self.policy_cfg.cam_names:
            self.policy_cfg.cam_names.append("depth")

        self.policy_cfg.add_depth_gs = self.add_depth_gs
        self.policy_cfg.add_depth_colored = self.add_depth_colored
        self.dataset_cfg.add_depth_gs = self.add_depth_gs
        self.dataset_cfg.add_depth_colored = self.add_depth_colored

        if self.lr:
            cprint(f"[NOTE] Overriding learning rate to {self.lr}", color="blue")
            self.optim_cfg.lr = self.lr

        if self.add_grasp_info_to_tracks:
            assert self.action_pred_type == ACTION_TYPE_TRACKS, (
                "add_grasp_info_to_tracks can only be used with ACTION_TYPE_TRACKS"
            )

        # Don't normalize images if using Voltron
        if "v-" in self.policy_cfg.encoder or "vit" in self.policy_cfg.encoder:
            self.dataset_cfg.normalize_image = False

        # Register logging information
        if not self.log_dir:
            pc = self.policy_cfg
            dc = self.dataset_cfg
            run_id = str(uuid.uuid4())[:6]
            log_dir_base = (
                f"experiment_logs/{dc.action_pred_type}-{pc.input_type}-{pc.backbone}"
            )
            if self.experiment_name:
                log_dir = f"{log_dir_base}/%m%d-{self.experiment_name}/%H%M%S_{run_id}"
            else:
                log_dir = f"{log_dir_base}/%m%d/%H%M%S_{run_id}"
            log_dir = datetime.datetime.now().strftime(log_dir)
            self.log_dir = Path(f"{log_dir}_{self.exp}") if self.exp else Path(log_dir)

        if self.debug:
            self.num_workers = 1
            self.epochs = 2
            self.val_every = 1
            self.max_vis_frames: int = 20
            self.log_dir = Path("experiment_logs/debug")
            self.dataset_cfg.robot_train_percentage = 0.1
            self.dataset_cfg.hand_train_percentage = 0.1


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("trace_" + str(p.step_num) + ".json")


@pyrallis.wrap()
def main(cfg: TrainConfig):
    train_utils.set_seed(cfg.seed)

    # ---------- Data -----------
    dataloader_kwargs = dict(
        log_dir=cfg.log_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        data_paths=cfg.data_paths,
        dataset_cfg=cfg.dataset_cfg,
        human_only_train=cfg.human_only_train,
    )
    train_loader, _ = create_dataloader("train", **dataloader_kwargs)
    val_loader, _ = create_dataloader("val", no_transforms=True, **dataloader_kwargs)

    # ---------- Model ----------
    cfg.action_dim = train_loader.dataset.action_dim
    cfg.state_cond_dim = train_loader.dataset.state_cond_dim
    policy_kwargs = dict(
        action_dim=cfg.action_dim,
        state_cond_dim=cfg.state_cond_dim,
        num_points=cfg.dataset_cfg.num_points,
        obs_horizon=cfg.dataset_cfg.obs_horizon,
        pred_horizon=cfg.dataset_cfg.pred_horizon,
        action_horizon=cfg.dataset_cfg.action_horizon,
        cfg=cfg.policy_cfg,
    )
    policy: DiffusionPolicy = DiffusionPolicy(**policy_kwargs)
    ema_policy: DiffusionPolicy = DiffusionPolicy(**policy_kwargs)
    policy = policy.to(cfg.device)
    ema_policy = ema_policy.to(cfg.device)

    optim = torch.optim.Adam(
        policy.parameters(),
        lr=cfg.optim_cfg.lr,
        weight_decay=cfg.optim_cfg.weight_decay,
    )

    start_epoch = 0
    if cfg.resume_training_from:
        checkpointer = Checkpointer(Path(cfg.resume_training_from))
        checkpointer.update_checkpoint_from_dir(
            Path(cfg.resume_training_from, "checkpoints")
        )
        policy, loaded_checkpoint = Checkpointer.load_checkpoint(
            policy, checkpointer, checkpoint_type="best"
        )
        if cfg.resume_ckpt_epoch:
            start_epoch = loaded_checkpoint["epoch"] + 1

    if cfg.torch_compile:
        policy = torch.compile(policy)
        ema_policy = torch.compile(ema_policy)

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optim,
        num_warmup_steps=cfg.optim_cfg.lr_warm_up_steps,
        num_training_steps=len(train_loader) * (cfg.epochs - start_epoch),
    )
    ema = EMAModel(parameters=policy.parameters(), power=0.75)

    # --------- Logging ---------
    log_dir = cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    pyrallis.dump(cfg, open(log_dir / "config.yaml", "w"))
    cprint(f"Logging to {log_dir}", color="cyan", attrs=["bold"])
    dset_str = ",".join(
        [path.parent.parent.name for path in cfg.dataset_cfg.data_paths]
    )
    wandb_kwargs = {
        "project": cfg.project_name,
        "group": cfg.experiment_name or "default",
        "name": log_dir.name,
        "job_type": dset_str,
        "config": train_utils.flatten_dict(asdict(cfg), no_parent_key=False),
    }
    logger = Logger(
        log_dir=log_dir,
        step=0,
        print_to_stdout=cfg.print_to_stdout,
        use_tensorboard=cfg.use_tb,
        use_wandb=cfg.use_wandb,
        wandb_kwargs=wandb_kwargs,
    )

    # --------- Training ---------
    checkpointer = Checkpointer(save_dir=log_dir / "checkpoints")
    policy.train()
    total_samples = len(train_loader.dataset)
    for epoch_idx in trange(cfg.epochs, desc="Epoch"):
        train_epoch_loss, grad_norms = [], []
        train_epoch_dict_accum = defaultdict(list)
        epoch_start_time = time.time()

        # Train step
        for batch in tqdm(train_loader, desc="Train Loop", leave=False):
            batch = train_utils.process_namedtuple_batch(batch, cfg.device)
            loss, loss_dict = policy.loss(batch)
            optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(  # type: ignore
                policy.parameters(), max_norm=cfg.optim_cfg.grad_clip
            )
            optim.step()
            lr_scheduler.step()
            ema.step(policy.parameters())
            train_epoch_loss.append(loss.item())
            grad_norms.append(grad_norm.item())
            for k, v in loss_dict.items():
                if hasattr(v, "item"):
                    train_epoch_dict_accum[k].append(v.item())
                else:
                    train_epoch_dict_accum[k].append(v)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        fps = total_samples / epoch_duration
        logger.scalar("train_loss", np.mean(train_epoch_loss))
        logger.scalar("train_grad_norm", np.mean(grad_norms))
        logger.scalar("profile/train_fps", fps)
        for k, v in train_epoch_dict_accum.items():
            logger.scalar(f"train_loss_dict_{k}", np.mean(v))

        # Validation step
        if epoch_idx != 0 and epoch_idx % cfg.val_every == 0:
            ema_val_epoch_loss = []
            action_l2_epoch_loss = []
            ema_epoch_dict_accum = defaultdict(list)

            ema_state_dict = {k: v.clone() for k, v in policy.state_dict().items()}
            try:
                ema.copy_to(ema_state_dict.values())
            except Exception as e:
                # use original weights for eval for now (mainly issue for R3M)
                breakpoint()
                cprint(f"Failed to copy to EMA model: {e}", color="red")

            ema_policy.load_state_dict(ema_state_dict)
            ema_policy.eval()
            for batch in tqdm(val_loader, desc="Val Loop", leave=False):
                batch = train_utils.process_namedtuple_batch(batch, cfg.device)
                with torch.no_grad():
                    ema_loss, ema_loss_dict = ema_policy.loss(batch, action_mse=True)
                ema_val_epoch_loss.append(ema_loss.item())
                action_l2_epoch_loss.append(ema_loss_dict["action_l2"].item())
                for k, v in ema_loss_dict.items():
                    if hasattr(v, "item"):
                        ema_epoch_dict_accum[k].append(v.item())
                    else:
                        ema_epoch_dict_accum[k].append(v)
            mean_val_loss = np.mean(ema_val_epoch_loss)
            mean_action_l2 = np.mean(action_l2_epoch_loss)
            logger.scalar("ema_val_loss", mean_val_loss)
            for k, v in ema_epoch_dict_accum.items():
                logger.scalar(f"val_{k}", np.mean(v))
            checkpointer.save(policy, optim, epoch_idx, mean_action_l2)

        logger.log_metrics(fps=False, step=epoch_idx)

    ema.copy_to(policy.parameters())

    # ---------- Load Best Checkpoint ----------
    cfg.policy_cfg.use_ddpm = False
    policy = DiffusionPolicy(**policy_kwargs)
    policy = policy.to(cfg.device)
    policy, _ = Checkpointer.load_checkpoint(policy, checkpointer, "best", optim)

    # ---------- Visualizations ----------
    if cfg.action_pred_type == ACTION_TYPE_TRACKS and cfg.generate_visualizations:
        dataloader_kwargs["batch_size"] = 1
        dataloader_kwargs["dataloader_overrides"] = {"shuffle": False}
        val_loader, val_stats = create_dataloader(
            "val", no_transforms=True, **dataloader_kwargs
        )
        policy.eval()
        vis_frames = visualize_image_tracks_2d(
            policy=policy,
            dataloader=val_loader,
            stats=val_stats,
            max_frames=cfg.max_vis_frames,
            action_horizon=cfg.dataset_cfg.action_horizon,
            add_grasp_info_to_tracks=cfg.policy_cfg.add_grasp_info_to_tracks,
        )
        logger.gif("vis_2d", vis_frames)
    logger.log_metrics(fps=False, step=epoch_idx)
    logger.close()
    cprint(f"Saved all logs to {cfg.log_dir}", color="cyan", attrs=["bold"])

    # --------- Save Sweep Info ----------
    # if cfg.experiment_name:
    #     train_utils.save_sweep_to_csv(ema_val_epoch_loss, log_dir, cfg)


if __name__ == "__main__":
    main()
