"""
This class takes data paths as inputs and tries to
predict the original spacing of the keypoints.
"""

import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import models.train_utils as train_utils
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from dataset.image_track_dataset import ImageTrackDatasetConfig, create_dataloader
from models.keypoint_map_predictor import KeypointMapPredictor
from termcolor import cprint
from tqdm import tqdm, trange
from utils.checkpointer import Checkpointer
from utils.logger import Logger

delta = False


@pyrallis.wrap()
def main(cfg: ImageTrackDatasetConfig):
    cfg.jitter_kpts = True

    run_id = str(uuid.uuid4())[:6]
    log_dir = Path(
        f"experiment_logs/keypoint_predictor/{datetime.now().strftime('%m%d')}/{run_id}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = Checkpointer(save_dir=log_dir / "checkpoints")
    cprint(f"Logging to {log_dir}", "green", attrs=["bold"])

    # Create dataloader
    data_paths = cfg.data_paths.copy()
    dataloader_kwargs = dict(
        log_dir=log_dir,
        batch_size=64,
        num_workers=8,
        data_paths=data_paths,
        dataset_cfg=cfg,
        no_transforms=True,
    )
    train_dataloader, _ = create_dataloader("train", **dataloader_kwargs)
    val_dataloader, _ = create_dataloader("val", **dataloader_kwargs)

    # Create keypoint predictor
    keypoint_predictor = KeypointMapPredictor(num_keypoints=len(cfg.kpt_keys))
    optimizer = torch.optim.Adam(keypoint_predictor.parameters(), lr=1e-3)

    # Create Logger
    dset_str = ",".join([path.parent.parent.name for path in cfg.data_paths])
    wandb_kwargs = {
        "project": "keypoint_predictor",
        "group": "default",
        "name": log_dir.name,
        "job_type": dset_str,
        "config": train_utils.flatten_dict(asdict(cfg), no_parent_key=False),
    }
    logger = Logger(
        log_dir=log_dir,
        step=0,
        print_to_stdout=False,
        use_tensorboard=False,
        use_wandb=True,
        wandb_kwargs=wandb_kwargs,
    )

    # Train keypoint predictor
    iters_per_epoch = len(train_dataloader)
    if val_dataloader is None:
        # If no validation dataset, use 85% of the data for training
        train_iters = int(0.85 * iters_per_epoch)
    else:
        train_iters = iters_per_epoch

    epochs = int(5e3)
    for epoch in trange(int(epochs)):
        train_loss = []
        val_loss = []
        nearest_dist_targets = []
        furthest_dist_targets = []
        nearest_dist_preds = []
        furthest_dist_preds = []
        for i, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc="Train",
            leave=False,
        ):
            target_keypoints = batch.state_cond

            nearest_dist_targ, furthest_dist_targ = get_min_max_distance(
                target_keypoints
            )
            nearest_dist_targets.append(nearest_dist_targ.item())
            furthest_dist_targets.append(furthest_dist_targ.item())

            jittered_keypoints = train_dataloader.dataset._jitter_keypoints(
                target_keypoints,
                cfg.fixed_kpt_dims,
                batched=True,
                post_normalization=True,
            )

            if delta:
                predicted_deltas = keypoint_predictor(jittered_keypoints)
                predicted_keypoints = jittered_keypoints + predicted_deltas
            else:
                predicted_keypoints = keypoint_predictor(jittered_keypoints)

            nearest_dist_pred, furthest_dist_pred = get_min_max_distance(
                predicted_keypoints
            )
            nearest_dist_preds.append(nearest_dist_pred.item())
            furthest_dist_preds.append(furthest_dist_pred.item())

            if i <= train_iters:
                optimizer.zero_grad()
                loss = F.mse_loss(predicted_keypoints, target_keypoints)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            elif epoch % 10 == 0:
                loss = F.mse_loss(predicted_keypoints, target_keypoints)
                val_loss.append(loss.item())

        logger.scalar("train_loss", np.mean(train_loss))
        logger.scalar("nearest_dist_target", np.mean(nearest_dist_targets))
        logger.scalar("furthest_dist_target", np.mean(furthest_dist_targets))
        logger.scalar("nearest_dist_pred", np.mean(nearest_dist_preds))
        logger.scalar("furthest_dist_pred", np.mean(furthest_dist_preds))

        if epoch % 10 == 0:
            if val_dataloader is not None:
                for batch in tqdm(
                    val_dataloader, total=len(val_dataloader), desc="Val", leave=False
                ):
                    target_keypoints = batch.state_cond
                    jittered_keypoints = train_dataloader.dataset._jitter_keypoints(
                        target_keypoints, cfg.fixed_kpt_dims, batched=True
                    )
                    predicted_keypoints = keypoint_predictor(jittered_keypoints)

                    loss = F.mse_loss(predicted_keypoints, target_keypoints)
                    val_loss.append(loss.item())

            checkpointer.save(keypoint_predictor, optimizer, epoch, np.mean(val_loss))
            logger.scalar("val_loss", np.mean(val_loss))

        logger.log_metrics(step=epoch)


def get_min_max_distance(keypoints):
    """
    Given a set of keypoints, return the min and max pairwise distances

    Input is of shape (B, obs_horizon, num_points*2)
    Output is of shape (B, 1) and (B, 1)
    """
    # Take the first obs_horizon points
    keypoints = keypoints[:, 0, :]
    # (B, num_points * 2) -> (B, num_points, 2)
    keypoints = keypoints.view(-1, keypoints.size(1) // 2, 2)
    pairwise_distances = torch.cdist(keypoints, keypoints)
    # all points normalized, so the max distance is sqrt(2)
    inf_mask = torch.eye(pairwise_distances.size(1)) * 10
    pairwise_distances += inf_mask
    # take nearest neighbor along the second dimension
    nearest_distances, _ = torch.min(
        pairwise_distances, dim=1
    )  # (B*obs_horizon, num_points)
    furthest_distances, _ = torch.max(
        pairwise_distances, dim=1
    )  # (B*obs_horizon, num_points)
    nearest_distances = torch.mean(nearest_distances)
    furthest_distances = torch.mean(furthest_distances)
    return nearest_distances, furthest_distances


if __name__ == "__main__":
    main()  # type: ignore
