from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from envs.minrobot.camera import ParallelCameras, SequentialCameras


@dataclass
class CameraConfig:
    name: str
    type: str  # "realsense" or "zed"
    serial_number: str
    width: int
    height: int
    use_depth: bool
    record_width: int = 640
    record_height: int = 480
    camera_upsidedown: bool = False  # if camera mounted upside down


@dataclass
class CameraEnvConfig:
    cameras: List[CameraConfig] = field(default_factory=list)
    parallel_camera: bool = False
    show_camera: bool = False


class CameraEnv:
    def __init__(self, cfg: CameraEnvConfig):
        self.cfg = cfg
        camera_args_list = [
            self._camera_config_to_dict(cam_cfg) for cam_cfg in cfg.cameras
        ]

        if cfg.parallel_camera:
            self.camera = ParallelCameras(camera_args_list)
        else:
            self.camera = SequentialCameras(camera_args_list)

    def _camera_config_to_dict(self, cam_cfg: CameraConfig) -> dict:
        return {
            "name": cam_cfg.name,
            "type": cam_cfg.type,
            "serial_number": cam_cfg.serial_number,
            "width": cam_cfg.width,
            "height": cam_cfg.height,
            "use_depth": cam_cfg.use_depth,
            "record_width": cam_cfg.record_width,
            "record_height": cam_cfg.record_height,
            "camera_upsidedown": cam_cfg.camera_upsidedown,
        }

    def reset(self):
        pass

    def observe(self):
        obs = {}
        rgb_images = []  # for rendering

        cam_frames = self.camera.get_frames()
        for name, frames in cam_frames.items():
            # depth = frames["depth"]
            # depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # rgb_images.append(depth)

            rgb_images.append(frames["image"])
            for k, v in frames.items():
                obs[f"{name}_{k}"] = v

        if self.cfg.show_camera:
            image = np.hstack(rgb_images)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", image)
            cv2.waitKey(1)

        return obs

    def __del__(self):
        del self.camera
