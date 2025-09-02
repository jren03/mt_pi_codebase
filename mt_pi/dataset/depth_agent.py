import matplotlib
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


class DepthAnythingAgent:
    def __init__(self, encoder="vits"):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(
            torch.load(
                f"third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        self.model = self.model.to(self.device).eval()
        self.cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    def run_inference(self, image):
        depth = self.model.infer_image(image)

        lower_thresh, upper_thresh = np.quantile(depth, [0.1, 0.75])
        depth = np.clip(depth, lower_thresh, upper_thresh)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = depth.astype(np.uint8)

        depth_grayscale = depth[:, :, None]
        depth_colored = (self.cmap(depth)[:, :, :3] * 255).astype(np.uint8)
        return depth_grayscale, depth_colored
