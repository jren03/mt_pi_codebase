# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from line_profiler import profile
from termcolor import cprint
from torchvision import models
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from torchvision.models.vision_transformer import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
)
from voltron import instantiate_extractor, load

SUPPORTED_ENCODERS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "v-cond",
    "vitb-16",
    "vitb-32",
]


def _make_norm(norm_cfg):
    if norm_cfg["name"] == "batch_norm":
        return nn.BatchNorm2d
    if norm_cfg["name"] == "group_norm":
        num_groups = norm_cfg["num_groups"]
        return lambda num_channels: nn.GroupNorm(num_groups, num_channels)
    if norm_cfg["name"] == "diffusion_policy":
        return lambda num_channels: nn.GroupNorm(num_channels // 16, num_channels)
    raise NotImplementedError(f"Missing norm layer: {norm_cfg['name']}")


def _replace_norm_layers(model, norm_layer):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.GroupNorm):
            num_channels = (
                module.num_features
                if isinstance(module, nn.BatchNorm2d)
                else module.num_channels
            )
            new_norm = norm_layer(num_channels)
            setattr(model, name, new_norm)
        else:
            _replace_norm_layers(module, norm_layer)


def _construct_resnet(size, norm, pretrained=False, input_channels=3):
    if size == 18:
        w = models.ResNet18_Weights.DEFAULT
        m = models.resnet18(weights=w if pretrained else None)
    elif size == 34:
        w = models.ResNet34_Weights.DEFAULT
        m = models.resnet34(weights=w if pretrained else None)
    elif size == 50:
        w = models.ResNet50_Weights.DEFAULT
        m = models.resnet50(weights=w if pretrained else None)
    else:
        raise NotImplementedError(f"Missing size: {size}")

    # Modify the first convolutional layer to accept the specified number of input channels
    if input_channels != 3:
        m.conv1 = nn.Conv2d(
            input_channels,
            m.conv1.out_channels,
            kernel_size=m.conv1.kernel_size,
            stride=m.conv1.stride,
            padding=m.conv1.padding,
            bias=m.conv1.bias,
        )

    # Replace the norm layers
    _replace_norm_layers(m, norm)

    return m


class ResNet(nn.Module):
    def __init__(
        self, size, norm_cfg, pretrained=False, input_channels=3, *, restore_path=""
    ):
        super().__init__()
        norm_layer = _make_norm(norm_cfg)
        model = _construct_resnet(size, norm_layer, pretrained, input_channels)
        model.fc = nn.Identity()

        if restore_path:
            print("Restoring model from", restore_path)
            state_dict = torch.load(restore_path, map_location="cpu")
            state_dict = (
                state_dict["features"]
                if "features" in state_dict
                else state_dict["model"]
            )
            self.load_state_dict(state_dict)

        self._model = model
        self._size = size

    def forward(self, x, has_batch_dim: bool = True):
        if has_batch_dim:
            # Flatten obs_horizon into batch dimension
            # (B, obs_horizon, C, H, W) -> (B * obs_horizon, C, H, W)
            bsize, oh = x.shape[0], x.shape[1]
            x = x.flatten(end_dim=1)
        x_emb = self._model(x)
        if has_batch_dim:
            # Reshape features back
            x_emb = x_emb.reshape(bsize, oh, -1)
        return x_emb

    @property
    def embed_dim(self):
        return {18: 512, 34: 512, 50: 2048}[self._size]


class MultiViewEncoderWrapper(nn.Module):
    """Handles n>=1 cameras and their respective encoders"""

    def __init__(
        self,
        encoders: List[nn.Module],
        cam_names: List[str],
    ):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.cam_names = cam_names
        cprint(f"MultiViewEncoderWrapper: {cam_names}", "magenta", attrs=["bold"])

    @profile
    def forward(
        self, obs: dict[str, torch.Tensor], has_batch_dim: bool = True
    ) -> List[torch.Tensor]:
        feats = []
        for i, camera in enumerate(self.cam_names):
            x = obs[camera]
            # if has_batch_dim and len(x.shape) == 4:
            #     # (B, obs_horizon, H, W) -> (B, obs_horizon, 1, H, W)
            #     x = x.unsqueeze(3)
            f = self.encoders[i](x, has_batch_dim)
            feats.append(f)
        return feats

    @property
    def embed_dim(self):
        return sum([encoder.embed_dim for encoder in self.encoders])


class LangInputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img, has_batch_dim, lang):
        return self.model(img, has_batch_dim)

    @property
    def embed_dim(self):
        return self.model.embed_dim


class ViT(nn.Module):
    def __init__(self, size, pretrained=False, restore_path=""):
        super().__init__()

        _, self.preprocess = load("v-cond", device="cuda", freeze=False)
        if size == "b-16":
            self._model = vit_b_16(
                weights=ViT_B_16_Weights.DEFAULT if pretrained else None
            )
            self._embed_dim = 768
        elif size == "b-32":
            self._model = vit_b_32(
                weights=ViT_B_32_Weights.DEFAULT if pretrained else None
            )
            self._embed_dim = 768
        elif size == "l-16":
            self._model = vit_l_16(
                weights=ViT_L_16_Weights.DEFAULT if pretrained else None
            )
            self._embed_dim = 1024
        elif size == "l-32":
            self._model = vit_l_32(
                weights=ViT_L_32_Weights.DEFAULT if pretrained else None
            )
            self._embed_dim = 1024
        else:
            raise ValueError(f"Unknown ViT size: {size}")

        self._model.heads = nn.Identity()

        if restore_path:
            print("Restoring model from", restore_path)
            state_dict = torch.load(restore_path, map_location="cpu")
            state_dict = (
                state_dict["features"]
                if "features" in state_dict
                else state_dict["model"]
            )
            self.load_state_dict(state_dict)

        self._size = size

    def forward(self, x):
        x = self.preprocess(x)
        return self._model(x)

    @property
    def embed_dim(self):
        return self._embed_dim


class Voltron(nn.Module):
    def __init__(self, model, freeze=False):
        super().__init__()
        assert model in ["v-cond", "v-dual", "v-gen"], f"Unknown Voltron model: {model}"
        self.vcond, self.preprocess = load(model, device="cuda", freeze=freeze)
        self.vector_extractor = instantiate_extractor(self.vcond)()

    def forward(self, img, lang=None):
        # (B, C, H, W) -> (B, C, 224, 224)
        img = self.preprocess(img)
        # (B, E, 384), where E = 196 if mode="visual", 196+20 if "multimodal"
        embedding = self.vcond(img, lang, mode="visual")
        # (B, 384)
        representation = self.vector_extractor(embedding)
        return representation

    @property
    def embed_dim(self):
        return 384


if __name__ == "__main__":
    print("Testing ResNet encoder")
    size = 50
    norm_cfg = {"name": "diffusion_policy"}
    encoder = ResNet(size, norm_cfg)
    assert encoder.embed_dim == 2048

    size = 18
    norm_cfg = {"name": "diffusion_policy"}
    encoder = ResNet(size, norm_cfg)
    assert encoder.embed_dim == 512

    print("Testing Voltron v-cond")
    img = torch.randn(1, 3, 128, 128).cuda()
    lang = ["pick eraser"]
    voltron = Voltron("v-cond")
    voltron.cuda()
    voltron(img, lang)

    # print("Testing Voltron v-gen")
    # voltron = Voltron("v-gen")
    # voltron.cuda()
    # voltron(img, lang)

    print("Tests pass!")
