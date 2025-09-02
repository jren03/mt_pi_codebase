from .image_encoders import (
    ResNet,
    R3M,
    ViT,
    Voltron,
    LangInputWrapper,
    MultiViewEncoderWrapper,
)
from .image_encoders import SUPPORTED_ENCODERS

ALL_CAM_NAMES = ["main", "depth"]


def get_encoder_by_name(
    name: str,
    freeze_encoder: bool = False,
    pretrained: bool = False,
    cam_names: list[str] = ALL_CAM_NAMES,
    input_channels: list[int] = [3],
):
    assert len(cam_names) == len(input_channels)
    if "resnet" in name:
        size = int(name[6:])
        norm_cfg = {"name": "diffusion_policy"}
        encoders = [
            ResNet(size, norm_cfg, pretrained, input_channels[i])
            for i in range(len(cam_names))
        ]
        encoder = MultiViewEncoderWrapper(encoders, cam_names)

    elif "vit" in name:
        size = name.replace("vit", "")
        encoder = ViT(size, pretrained)

    elif "r3m" in name:
        size = int(name[3:])
        encoder = R3M(size)

    elif "v-" in name:
        encoder = Voltron(name, freeze_encoder)

    else:
        raise NotImplementedError(f"Unknown encoder: {name}")

    if "v-" not in name:
        # Wrap model to accept language input
        encoder = LangInputWrapper(encoder)

    if freeze_encoder:
        encoder.eval()
    return encoder
