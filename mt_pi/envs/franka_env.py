from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import pyrallis
from termcolor import cprint
from utils.freq_guard import FreqGuard
from utils.recorders import ActMode, DatasetRecorder
from utils.robot import get_ori, get_waypoint
from utils.stopwatch import Stopwatch

from envs.minrobot.camera import ParallelCameras, SequentialCameras
from envs.minrobot.controller import Controller, ControllerConfig


@dataclass
class FrankaEnvConfig:
    cameras: List[dict] = field(default_factory=list)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    randomize_init: int = 0
    show_camera: int = 0
    channel_first: int = 0
    parallel_camera: int = 0
    min_z_coordinate: float = 0.170  # minimum z coordinate for the end effector
    z_safety_buffer: float = 0.005  # safety buffer for the z coordinate
    add_depth_anything: int = 0  # add depth image to the observation


class FrankaEnv:
    def __init__(self, cfg: FrankaEnvConfig):
        self.cfg = cfg
        self.controller = Controller(cfg.controller)
        self.controller.reset(randomize=False)
        proprio = self.controller.get_proprio()
        self.home_pos = proprio.eef_pos
        self.home_euler = proprio.eef_euler

        if cfg.parallel_camera and not cfg.add_depth_anything:
            self.camera = ParallelCameras(cfg.cameras)
        else:
            self.camera = SequentialCameras(cfg.cameras)

        self.depth_agent = None

        self.min_z_coordinate = cfg.min_z_coordinate
        self.z_safety_buffer = cfg.z_safety_buffer

    def add_depth_anything_to_obs(self):
        from dataset.depth_agent import DepthAnythingAgent

        cprint("Adding depth anything to the observation")
        self.depth_agent = DepthAnythingAgent(encoder="vits")

    def reset(self):
        self.move_to(self.home_pos, self.home_euler, gripper_open=1)
        self.controller.reset(bool(self.cfg.randomize_init))

    def _cornell_reset(self):
        """FR3 defines gripper rotation as -pi/4 in the urdf's, so just manually turning it at the start"""
        new_home_euler = self.home_euler.copy()
        new_home_euler[2] -= np.pi / 4
        gen_ori = get_ori(self.home_euler, new_home_euler, num_steps=10)
        for i in range(1, 11):
            next_ee_euler = gen_ori(i)
            with FreqGuard(10):
                self.controller.position_control(
                    self.home_pos, next_ee_euler, gripper_open=1
                )

    def observe_camera(self):
        obs = {}
        rgb_images = []  # for rendering

        cam_frames = self.camera.get_frames()
        for name, frames in cam_frames.items():
            rgb_images.append(frames["image"])

            for k, v in frames.items():
                if self.cfg.channel_first:
                    v = v.transpose(2, 0, 1)
                obs[f"{name}_{k}"] = v

                if "agent" in name and self.depth_agent is not None:
                    depth_grayscale, depth_colored = self.depth_agent.run_inference(
                        obs[f"{name}_image"]
                    )
                    obs[f"{name}_depth_grayscale"] = depth_grayscale
                    obs[f"{name}_depth_colored"] = depth_colored

        if self.cfg.show_camera:
            image = np.hstack(rgb_images)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", image)
            cv2.waitKey(1)

        return obs

    def observe_proprio(self):
        return self.controller.get_proprio()

    def observe(self):
        obs = self.observe_camera()

        proprio = self.controller.get_proprio()
        obs["ee_pos"] = proprio.eef_pos
        obs["ee_euler"] = proprio.eef_euler
        obs["ee_quat"] = proprio.eef_quat
        obs["gripper_open"] = np.array([proprio.gripper_open])
        obs["proprio"] = proprio.eef_pos_euler

        return obs

    def adjust_z_coordinate(self, ee_pos: np.ndarray) -> np.ndarray:
        if ee_pos[2] <= self.min_z_coordinate:
            ee_pos[2] = self.min_z_coordinate + self.z_safety_buffer
        return ee_pos

    def apply_action(
        self,
        ee_pos: np.ndarray,
        ee_euler: np.ndarray,
        gripper_open: float,
        is_delta=True,
    ):
        if is_delta:
            new_ee_pos = self.controller.get_proprio().eef_pos + ee_pos
        else:
            new_ee_pos = ee_pos

        # Check and adjust the z-coordinate
        new_ee_pos = self.adjust_z_coordinate(new_ee_pos)

        if is_delta:
            adjusted_delta = new_ee_pos - self.controller.get_proprio().eef_pos
            self.controller.delta_control(adjusted_delta, ee_euler, gripper_open)
        else:
            self.controller.position_control(new_ee_pos, ee_euler, gripper_open)

    def move_to(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: float,
        max_delta=0.013,
        control_freq: float = 10,
        recorder: Optional[DatasetRecorder] = None,
    ):
        proprio = self.controller.get_proprio()
        original_gripper_open = proprio.gripper_open > 0.9

        # Adjust the target position before moving
        adjusted_target_pos = self.adjust_z_coordinate(target_pos)

        self.move_via_waypoints(
            adjusted_target_pos,
            target_euler,
            original_gripper_open,
            max_delta,
            control_freq,
            recorder,
        )

        # then, apply gripper
        self.controller.set_gripper(gripper_open=(gripper_open > 0))

    def move_via_waypoints(
        self,
        target_pos,
        target_euler,
        gripper_open,
        max_delta=0.01,
        control_freq=10,
        recorder=None,
        gen_ori_separately=True,
    ):
        proprio = self.controller.get_proprio()

        gen_waypoint, num_steps = get_waypoint(
            (proprio.eef_pos, proprio.eef_euler),
            (target_pos, target_euler),
            max_linear_delta=max_delta,
            max_angular_delta=max_delta,
        )
        if gen_ori_separately:
            gen_ori = get_ori(proprio.eef_euler, target_euler, num_steps=num_steps)

        num_steps = max(num_steps, 1)
        for i in range(1, num_steps + 1):
            next_pose = gen_waypoint(i)
            next_ee_pos = next_pose[:3]
            if gen_ori_separately:
                next_ee_euler = gen_ori(i)
            else:
                next_ee_euler = next_pose[3:]

            # Adjust the z-coordinate of each waypoint
            next_ee_pos = self.adjust_z_coordinate(next_ee_pos)

            with FreqGuard(control_freq):
                if recorder is not None:
                    obs = self.observe()
                    delta_pos, delta_euler = self.controller.position_to_delta(
                        next_ee_pos, next_ee_euler
                    )
                    action = np.concatenate(
                        [delta_pos, delta_euler, [gripper_open]]
                    ).astype(np.float32)
                    recorder.record(ActMode.Interpolate, obs, action)

                self.controller.position_control(
                    next_ee_pos, next_ee_euler, gripper_open
                )

    def __del__(self):
        del self.camera


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=FrankaEnvConfig)  # type: ignore
    # cfg.show_camera = 1
    env = FrankaEnv(cfg)
    env.reset()

    # warm up
    stopwatch = Stopwatch()
    for _ in range(100):
        env.observe_camera()
    # stopwatch.summary()

    for i in range(500):
        with FreqGuard(10):
            with stopwatch.time("observe"):
                env.observe_camera()

            with stopwatch.time("step"):
                ee_pos = np.array([0.01, 0.01, 0])
                ee_pos = ee_pos * (((i // 10) % 2) * 2 - 1)
                ee_euler = np.array([0.00, 0.00, 0])
                env.apply_action(ee_pos, ee_euler, 0)

    stopwatch.summary()
    del env
