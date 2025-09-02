from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from dataset.image_track_dataset import ACTION_TYPE_TRACKS, ACTION_TYPES, Batch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from line_profiler import profile
from termcolor import cprint

from models.encoders import SUPPORTED_ENCODERS, get_encoder_by_name
from models.feature_compressor import FeatureCompressor
from models.keypoint_map_predictor import load_keypoint_map_predictor
from models.noise_predictor import NoisePredictor
from models.policy_nets import ConditionalUnet1D, TransformerForDiffusion
from models.train_utils import absolute_to_delta_action, delta_to_absolute_action


@dataclass
class DDPMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    prediction_type: str = "epsilon"


@dataclass
class DDIMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 16
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    prediction_type: str = "epsilon"


@dataclass
class DiffusionPolicyConfig:
    # arch
    encoder: str = "resnet50"
    input_type: str = "image"
    backbone: str = "unet"
    add_depth_gs: bool = False
    add_depth_colored: bool = False
    cam_names: List[str] = field(default_factory=lambda: ["main"])

    # finetuning
    freeze_encoder: bool = False
    pretrained_encoder: bool = True
    num_compress_layers: int = 1
    upsample_prop: bool = False
    layers_to_freeze: List[str] = field(default_factory=lambda: [])

    # auxiliary losses
    learn_prop_fusion: bool = False
    balance_grasp_loss: bool = True
    adv_domain_loss: bool = False
    kpt_loss_weight: float = 0.0
    kl_loss_weight: float = 0.0
    meanvar_loss_weight: float = 0.0
    img_dropout: float = 0.0

    # keypoint mapping
    noise_only_state_cond: bool = True
    keypoint_map_from: Optional[str] = None

    # noise scheduler
    use_ddpm: bool = False
    ddpm: DDPMConfig = field(default_factory=lambda: DDPMConfig())
    ddim: DDIMConfig = field(default_factory=lambda: DDIMConfig())

    # track prediction
    action_pred_type: str = field(default=ACTION_TYPE_TRACKS)
    add_cond_vector: bool = True  # whether to condition on eef or proprio state info
    add_grasp_info_to_tracks: bool = False  # add additional dimension for grasp
    delta_actions: bool = True  # predict delta points instead of absolute

    # legacy
    prop_bias_estimator: bool = False
    train_only_bias: bool = False
    add_in_wrist: bool = False
    domain_loss_weight: float = 0.0

    def __post_init__(self):
        assert self.action_pred_type in ACTION_TYPES
        assert self.input_type in [
            "image",
            "pc",
        ], f"Unknown input type: {self.input_type}"
        assert self.encoder in SUPPORTED_ENCODERS, f"Unknown encoder: {self.encoder}"
        assert self.backbone in ["unet"], f"Unknown backbone: {self.backbone}"

        if (
            self.add_depth_gs or self.add_depth_colored
        ) and "depth" not in self.cam_names:
            self.cam_names = self.cam_names.append("depth")


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        num_points: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        action_horizon: int,
        state_cond_dim: int,
        cfg: DiffusionPolicyConfig,
    ):
        super().__init__()

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.num_points = num_points
        self.action_dim = action_dim
        self.cfg = cfg

        if cfg.domain_loss_weight > 0.0:
            self.adv_domain_loss = True

        self.noise_predictor: NoisePredictor = self._setup_noise_predictor(
            cfg, action_dim, obs_horizon, pred_horizon, state_cond_dim
        )

        if cfg.use_ddpm:
            scheduler_kwargs = asdict(cfg.ddpm)
            self.num_inference_timesteps = scheduler_kwargs.pop(
                "num_inference_timesteps"
            )
            self.num_train_timesteps = scheduler_kwargs["num_train_timesteps"]
            self.noise_scheduler = DDPMScheduler(**scheduler_kwargs)
            cprint("Using DDPM", "green", attrs=["bold"])
        else:
            scheduler_kwargs = asdict(cfg.ddim)
            self.num_inference_timesteps = scheduler_kwargs.pop(
                "num_inference_timesteps"
            )
            self.num_train_timesteps = scheduler_kwargs["num_train_timesteps"]
            self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
            cprint("Using DDIM", "green", attrs=["bold"])

    def _setup_noise_predictor(
        self,
        cfg: DiffusionPolicyConfig,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        state_cond_dim: int,
    ) -> nn.Module:
        input_channels = []
        for cam_name in cfg.cam_names:
            if cam_name == "main":
                input_channels.append(3)
            elif cam_name == "depth":
                input_channels.append(1 if cfg.add_depth_gs else 3)
            else:
                raise NotImplementedError(f"Unknown camera name: {cam_name}")
        encoder = get_encoder_by_name(
            cfg.encoder,
            cfg.freeze_encoder,
            cfg.pretrained_encoder,
            cfg.cam_names,
            input_channels=input_channels,
        )

        self.keypoint_map_predictor = (
            load_keypoint_map_predictor(cfg.keypoint_map_from, state_cond_dim // 2)
            if cfg.keypoint_map_from
            else None
        )

        embed_in_dim = encoder.embed_dim
        embed_hid_dim = int(encoder.embed_dim // 2)
        embed_out_dim = int(encoder.embed_dim // 4)
        feat_mlp = FeatureCompressor(
            in_dim=embed_in_dim,
            hid_dim=embed_hid_dim,
            out_dim=embed_out_dim,
            prop_dim=state_cond_dim,
            obs_horizon=obs_horizon,
            num_layers=cfg.num_compress_layers,
            learn_prop_fusion=cfg.learn_prop_fusion,
            regress_kpts=cfg.kpt_loss_weight > 0,
            adapt_domain=cfg.adv_domain_loss,
            img_dropout=cfg.img_dropout,
        )

        cond_dim = embed_out_dim + int(cfg.add_cond_vector) * state_cond_dim
        if cfg.backbone == "unet":
            noise_pred_net = ConditionalUnet1D(
                input_dim=action_dim,
                obs_horizon=obs_horizon,
                cond_dim=cond_dim,
            )
        elif cfg.backbone == "transformer":
            noise_pred_net = TransformerForDiffusion(
                input_dim=action_dim,
                output_dim=action_dim,
                horizon=pred_horizon,
                n_obs_steps=obs_horizon,
                n_layer=8,
                n_head=4,
                n_emb=256,
                p_drop_emb=0.0,
                p_drop_attn=0.3,
                n_cond_layers=0,
                causal_attn=True,
                time_as_cond=True,
                obs_as_cond=True,
                cond_dim=cond_dim,
            )
        else:
            raise NotImplementedError(f"Missing backbone: {cfg.backbone}")

        noise_predictor = NoisePredictor(
            encoder, feat_mlp, noise_pred_net, cfg.layers_to_freeze
        )
        return noise_predictor

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        state_cond: torch.Tensor,
        *,
        first_action_abs: torch.Tensor = None,
        return_full_action: bool = False,
        cpu: bool = True,
    ) -> torch.Tensor:
        """Generate action from noise"""
        assert not self.training
        assert not (self.cfg.delta_actions and first_action_abs is None), (
            "first_action_abs is required for delta_actions"
        )

        batched = True
        if state_cond is not None and state_cond.dim() == 2:
            # state_cond: (obs_horizon, cond_dim)
            batched = False
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
            state_cond = state_cond.unsqueeze(0)
            first_action_abs = first_action_abs.unsqueeze(0)

        if self.keypoint_map_predictor:
            state_cond = self.keypoint_map_predictor(state_cond)

        bsize = obs["main"].size(0)
        device = obs["main"].device

        # pure noise input to begin with
        noisy_action = torch.randn(
            (bsize, self.pred_horizon, self.action_dim), device=device
        )
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

        cache_dict = {}
        for k in self.noise_scheduler.timesteps:
            noise_pred, obs_emb, predicted_keypoints, domain_logits = (
                self.noise_predictor.predict_noise(
                    obs,
                    noisy_action,
                    k,
                    additional_cond=state_cond,
                    cache_dict=cache_dict,
                )
            )
            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action,  # type: ignore
            ).prev_sample.detach()  # type: ignore
            cache_dict = {
                "obs_emb": obs_emb,
                "predicted_keypoints": predicted_keypoints,
                "domain_logits": domain_logits,
            }

        action = noisy_action
        if self.cfg.delta_actions and self.cfg.action_pred_type == ACTION_TYPE_TRACKS:
            # (B, action_dim) -> (B, 1, action_dim) + (B, action_horizon - 1, action_dim)
            first_action_abs = first_action_abs.unsqueeze(1)
            action = delta_to_absolute_action(
                action,
                first_action_abs,
                add_grasp_info_to_tracks=self.cfg.add_grasp_info_to_tracks,
            )

        # when obs_horizon=2, the model was trained as
        # o_0, o_1,
        # a_0, a_1, a_2, a_3, ..., a_{h-1}  -> action_horizon number of predictions
        # so we DO NOT use the first prediction at test time
        offset_start = self.obs_horizon - 1
        if not return_full_action:
            action = action[:, offset_start : self.action_horizon + offset_start]

        if not batched:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    @profile
    def loss(
        self,
        batch: Batch,
        avg: bool = True,
        action_mse: bool = False,
        progress: int = 0,
    ) -> torch.Tensor:
        """
        Calculate loss for the diffusion model.

        If `action_mse` is true, the loss is calculated as the MSE between the predicted
        actions and the ground truth actions. This is used for monitoring only and is not
        used in optimization.
        """
        actions = batch.action
        state_cond = batch.state_cond
        bsize = actions.size(0)
        obs_dict = {
            "main": batch.obs,
            "depth": batch.depth,
            # "wrist": batch.wrist,
        }

        if self.keypoint_map_predictor:
            if not self.cfg.noise_only_state_cond:
                # Pass both actions and state_cond through the keypoint map predictor
                # (B, pred_horizon, 2 * num_points) --> (B * phor, 2 * num_points)
                actions_no_grasp = actions[:, :, : self.num_points * 2]
                actions_no_grasp = actions_no_grasp.view(bsize * self.pred_horizon, -1)
                actions_no_grasp_mapped = self.keypoint_map_predictor(actions_no_grasp)
                actions_no_grasp_mapped = actions_no_grasp_mapped.view(
                    bsize, self.pred_horizon, -1
                )
                actions = torch.cat(
                    [actions_no_grasp_mapped, actions[:, :, self.num_points * 2 :]],
                    dim=2,
                )

            state_cond = self.keypoint_map_predictor(state_cond)

        if self.cfg.delta_actions and self.cfg.action_pred_type == ACTION_TYPE_TRACKS:
            actions = absolute_to_delta_action(
                actions,
                add_grasp_info_to_tracks=self.cfg.add_grasp_info_to_tracks,
            )

        noise = torch.randn(actions.shape, device=actions.device)
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config["num_train_timesteps"],
            size=(bsize,),
            device=actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)  # type: ignore

        noise_pred, obs_emb, predicted_keypoints, domain_logits = (
            self.noise_predictor.predict_noise(
                obs_dict,
                noisy_actions,
                timesteps,
                lang=batch.lang,
                additional_cond=state_cond if self.cfg.add_cond_vector else None,
            )
        )

        loss_dict = {}

        # Noise prediction loss
        noise_loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        action_noise_loss = noise_loss[:, :, : self.num_points * 2]
        grasp_and_term_noise_loss = noise_loss[:, :, self.num_points * 2 :]

        if self.cfg.balance_grasp_loss:
            action_dims = self.num_points * 2
            grasp_and_term_dims = noise_loss.shape[2] - action_dims
            normalized_action_loss = action_noise_loss.sum(dim=2) / action_dims
            normalized_grasp_and_term_loss = (
                grasp_and_term_noise_loss.sum(dim=2) / grasp_and_term_dims
            )
            action_loss = (normalized_action_loss + normalized_grasp_and_term_loss) / 2
        else:
            action_loss = noise_loss.sum(dim=2)

        loss_dict["total/action_noise"] = action_noise_loss.mean()
        loss_dict["total/grasp_noise"] = grasp_and_term_noise_loss.mean()

        # KL Divergence Loss:
        if self.cfg.kl_loss_weight > 0:
            # Separate robot from human embeddings
            human_idx = torch.where(batch.label == 0)
            robot_idx = torch.where(batch.label == 1)
            human_embs = obs_emb[human_idx]
            robot_embs = obs_emb[robot_idx]
            kl_loss = self._kl_embedding_loss(human_embs, robot_embs)

            # warmup_steps = 0.1  # 10% of the training
            # if progress < warmup_steps:
            #     kl_loss_weight = self.cfg.kl_loss_weight * (progress / warmup_steps)
            # else:
            #     kl_loss_weight = self.cfg.kl_loss_weight
            kl_loss_weight = self.cfg.kl_loss_weight
            kl_loss = kl_loss_weight * kl_loss

            loss_dict["kl_loss"] = kl_loss
            loss_dict["kl_loss_weight"] = kl_loss_weight

        # Mean Var Loss:
        if self.cfg.meanvar_loss_weight > 0:
            # Separate robot from human embeddings
            human_idx = torch.where(batch.label == 0)
            robot_idx = torch.where(batch.label == 1)
            human_embs = obs_emb[human_idx]
            robot_embs = obs_emb[robot_idx]
            meanvar_loss = self._mean_variance_alignment_loss(human_embs, robot_embs)
            loss_dict["meanvar_loss"] = meanvar_loss

        # Keypoint prediction Loss
        if self.cfg.kpt_loss_weight > 0:
            kp_loss = nn.functional.mse_loss(
                predicted_keypoints, batch.state_cond, reduction="mean"
            )
            loss_dict["kp_loss"] = kp_loss

        # Domain-Adversarial Loss
        domain_loss_weight = 0.0
        if self.cfg.adv_domain_loss:
            # Schedule from original Domain Adaptation by Backpropgation
            domain_loss_weight = (
                2.0 / (1.0 + np.exp(-10 * progress)) - 1
                if self.cfg.adv_domain_loss
                else 1.0
            )
            domain_loss_weight = domain_loss_weight * self.cfg.domain_loss_weight
            domain_loss = nn.functional.binary_cross_entropy_with_logits(
                domain_logits.squeeze(-1), batch.label, reduction="mean"
            )
            loss_dict["domain_loss"] = domain_loss
            loss_dict["domain_loss_weight"] = domain_loss_weight

        # Action prediction MSE (for monitoring only, not used in optimization)
        if action_mse:
            first_action_abs = batch.action[:, 0] if self.cfg.delta_actions else None
            predicted_actions = self.act(
                obs=obs_dict,
                state_cond=batch.state_cond if self.cfg.add_cond_vector else None,
                first_action_abs=first_action_abs,
                return_full_action=True,  # Don't trim for action_horizon
                cpu=False,  # Keep tensor on the same device
            )

            # Ensure predicted_actions and actions have the same shape
            predicted_actions = predicted_actions[:, : actions.shape[1]]

            l2_loss = nn.functional.mse_loss(
                predicted_actions, actions, reduction="none"
            )
            action_l2_loss = l2_loss[:, :, : self.num_points * 2]
            grasp_and_term_l2_loss = l2_loss[:, :, self.num_points * 2 :]

            loss_dict["action_l2"] = l2_loss.mean()
            loss_dict["total/action_l2"] = action_l2_loss.mean()
            loss_dict["total/grasp_l2"] = grasp_and_term_l2_loss.mean()

        if avg:
            # Reduction from official DP repo
            action_loss = reduce(action_loss, "b ... -> b (...)", "mean")
            action_loss = action_loss.mean()
        else:
            action_loss = action_loss

        final_loss = (
            action_loss
            + self.cfg.kpt_loss_weight * loss_dict.get("kp_loss", 0)
            + self.cfg.kl_loss_weight * loss_dict.get("kl_loss", 0)
            + self.cfg.meanvar_loss_weight * loss_dict.get("meanvar_loss", 0)
            + domain_loss_weight * loss_dict.get("domain_loss", 0)
        )

        return final_loss, loss_dict

    def _symm_kl_embedding_loss(self, emb1, emb2):
        # Ensure emb1 and emb2 have the same batch size
        min_batch = min(emb1.size(0), emb2.size(0))
        emb1 = emb1[:min_batch]
        emb2 = emb2[:min_batch]

        mu1, sigma1 = torch.mean(emb1, dim=1), torch.std(emb1, dim=1)  # (B, emb_dim)
        mu2, sigma2 = torch.mean(emb2, dim=1), torch.std(emb2, dim=1)  # (B, emb_dim)

        # KL Loss from emb1 to emb2
        term1 = torch.log(sigma2 / sigma1)
        term2 = (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
        kl_loss_1to2 = term1 + term2 - 0.5

        # KL Loss from emb2 to emb1
        term1 = torch.log(sigma1 / sigma2)
        term2 = (sigma2**2 + (mu2 - mu1) ** 2) / (2 * sigma1**2)
        kl_loss_2to1 = term1 + term2 - 0.5

        kl_loss = (kl_loss_1to2 + kl_loss_2to1) / 2.0

        return kl_loss.sum()

    def _kl_embedding_loss(self, emb1, emb2):
        # Ensure emb1 and emb2 have the same batch size
        min_batch = min(emb1.size(0), emb2.size(0))
        emb1 = emb1[:min_batch]
        emb2 = emb2[:min_batch]

        mu1, sigma1 = torch.mean(emb1, dim=1), torch.std(emb1, dim=1)  # (B, emb_dim)
        mu2, sigma2 = torch.mean(emb2, dim=1), torch.std(emb2, dim=1)  # (B, emb_dim)

        # KL Loss -> (min(B1, B2))
        term1 = torch.log(sigma2 / sigma1)
        term2 = (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
        kl_loss = term1 + term2 - 0.5

        return kl_loss.sum()

    def _mean_variance_alignment_loss(self, emb1, emb2):
        min_batch = min(emb1.size(0), emb2.size(0))
        emb1 = emb1[:min_batch]
        emb2 = emb2[:min_batch]

        mu1, sigma1 = torch.mean(emb1, dim=1), torch.std(emb1, dim=1)
        mu2, sigma2 = torch.mean(emb2, dim=1), torch.std(emb2, dim=1)

        mean_loss = torch.sum((mu1 - mu2) ** 2)
        variance_loss = torch.sum((sigma1 - sigma2) ** 2)

        return mean_loss + variance_loss


def test():
    print("Testing Diffusion Policy")
    cprint(
        "[WARNING], addin grasp info to points is not implemented/tested",
        "yellow",
        attrs=["bold", "blink"],
    )
    # Define parameters
    num_points = 5
    action_dim = 10
    obs_horizon = 2
    pred_horizon = 16
    action_horizon = 8
    state_cond_dim = 10
    add_grasp_info_to_tracks = True

    # Create config
    cfg = DiffusionPolicyConfig(
        input_type="image",
        encoder="resnet",
        backbone="unet",
        use_ddpm=True,
        add_cond_vector=True,
        add_grasp_info_to_tracks=add_grasp_info_to_tracks,
        delta_actions=True,
    )

    # Instantiate DiffusionPolicy
    policy = DiffusionPolicy(
        num_points=num_points,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
        state_cond_dim=state_cond_dim,
        cfg=cfg,
    ).cuda()

    # Set to evaluation mode
    policy.eval()

    # Create dummy input tensors
    batch_size = 2
    obs = torch.rand(
        batch_size, obs_horizon, 3, 96, 96
    ).cuda()  # Assuming image input of shape (3, 96, 96)
    state_cond = torch.rand(batch_size, obs_horizon, action_dim).cuda()

    # Test act method
    action = policy.act(
        obs,
        state_cond,
        first_action_abs=torch.rand(batch_size, action_dim).cuda(),
    )
    print(f"Output action shape: {action.shape}")
    expected_shape = (
        batch_size,
        action_horizon,
        action_dim + 1 + int(add_grasp_info_to_tracks),
    )
    assert action.shape == expected_shape, (
        f"Expected shape {expected_shape}, but got {action.shape}"
    )

    # Test act method without batch
    action = policy.act(
        obs[0], state_cond[0], first_action_abs=torch.rand(action_dim).cuda()
    )
    print(f"Output action shape: {action.shape}")
    expected_shape = (action_horizon, action_dim)
    assert action.shape == expected_shape, (
        f"Expected shape {expected_shape}, but got {action.shape}"
    )

    # Test loss calculation
    policy.train()
    batch = Batch(
        obs=obs,
        action=torch.rand(batch_size, pred_horizon, action_dim).cuda(),
        state_cond=state_cond,
    )
    loss = policy.loss(batch)
    print(f"Loss: {loss.item()}")

    print("Test passed successfully!")


if __name__ == "__main__":
    test()
