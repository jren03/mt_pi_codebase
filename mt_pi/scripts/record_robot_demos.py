"""
Example command:

python scripts/record_robot_demos.py  --data_folder data/robot_dev
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os

import msgpack
import numpy as np
import pyrallis
import websockets
from envs.franka_env import FrankaEnv, FrankaEnvConfig
from scipy.spatial.transform import Rotation as R
from utils.constants import CALIB_DIR, FRANKA_ENV_CONFIG_PATH, SPACEMOUSE_SPECS
from utils.freq_guard import FreqGuard
from utils.helper import wrap_ruler
from utils.pointclouds import crop, deproject
from utils.recorders import ActMode, DatasetRecorder
from utils.spacemouse import SpaceMouseInterface
from utils.stopwatch import Stopwatch

from scripts.interface_assets.serve import http_server


class InteractiveBot:
    def __init__(
        self,
        num_point,
        stream_freq,
        control_freq,
        data_folder,
        robot_cfg: FrankaEnvConfig,
    ):
        self.num_point = num_point
        self.stream_freq = stream_freq
        self.control_freq = control_freq
        self.data_folder = data_folder
        self.env = FrankaEnv(robot_cfg)

        # self.transforms = None
        assert os.path.exists(f"{CALIB_DIR}/transforms_both.npy"), (
            f"Transforms file not found in {CALIB_DIR}/transforms_both.npy. Did you run python scripts/calibrate.py?"
        )
        self.transforms = np.load(
            f"{CALIB_DIR}/transforms_both.npy", allow_pickle=True
        ).item()
        self.agent1_intrinsics = self.env.camera.get_intrinsics("agent1")["matrix"]
        self.agent2_intrinsics = self.env.camera.get_intrinsics("agent2")["matrix"]
        self.view_to_intrinsics = {
            "agent1": self.agent1_intrinsics,
            "agent2": self.agent2_intrinsics,
        }

        spacemouse_args = SPACEMOUSE_SPECS
        spacemouse_args["pos_sensitivity"] = 5.0
        spacemouse_args["rot_sensitivity"] = 18.0
        self.teleop_interface = SpaceMouseInterface(**spacemouse_args)
        self.teleop_interface.start_control()

    def reset(self):
        # First open gripper
        proprio = self.env.observe_proprio()
        ee_pos = proprio.eef_pos
        ee_euler = proprio.eef_euler
        self.env.move_to(
            ee_pos,
            ee_euler,
            1.0,
            control_freq=self.control_freq,
            recorder=None,
        )
        # Then reset
        self.env.reset()
        self.initial_euler = self.env.observe_proprio().eef_euler
        print(f"Initial Euler: {self.initial_euler}")

    def take_point_cloud(self, obs):
        points_list = []
        colors_list = []

        # for view in ["agent1"]:
        for view in ["agent1", "agent2"]:
            rgb_frame = obs[f"{view}_image"]
            depth_frame = obs[f"{view}_depth"]
            depth_frame = depth_frame.squeeze()
            tf = self.transforms[view]["tcr"]
            points = deproject(depth_frame, self.view_to_intrinsics[view], tf)
            colors = rgb_frame.reshape(points.shape) / 255.0
            points_list.append(points)
            colors_list.append(colors)

        merged_points = np.vstack(points_list)
        merged_colors = np.vstack(colors_list)

        idxs = crop(
            merged_points, min_bound=[0.2, -0.24, 0.0], max_bound=[0.7, 0.24, 0.3]
        )
        merged_points = merged_points[idxs]
        merged_colors = merged_colors[idxs]
        return merged_points, merged_colors

    def transform_robotframe_to_uiframe(self, waypoints):
        waypoints = np.array(waypoints)
        waypoints += np.array([-0.4, 0, 0])
        waypoints_ui = np.zeros_like(np.array(waypoints))
        transf = R.from_euler("x", -90, degrees=True)
        waypoints_ui = transf.apply(waypoints)
        rescale_amt = 10
        waypoints_ui *= rescale_amt
        return waypoints_ui

    def transform_uiframe_to_robotframe(self, waypoints):
        waypoints_rob = waypoints.copy()
        waypoints_rob /= 10
        transf = R.from_euler("x", 90, degrees=True)
        waypoints_rob = transf.apply(waypoints_rob)
        waypoints_rob += np.array([0.4, 0, 0])
        return waypoints_rob

    def prepare_point_cloud(self, obs, ravel=True):
        points, colors = self.take_point_cloud(obs)
        idxs = np.random.choice(
            np.arange(len(points)), min(len(points), self.num_point), replace=False
        )

        points_ui = self.transform_robotframe_to_uiframe(points[idxs])
        colors = colors[idxs]

        if ravel:
            points_ui = points_ui.ravel()
            colors = colors.ravel()

        points_ui = list(points_ui)
        colors = list(colors)
        return points_ui, colors

    def calculate_fingertip_offset(self, ee_euler: np.ndarray) -> np.ndarray:
        home_fingertip_offset = np.array([0, 0, -0.145])
        ee_euler_adjustment = ee_euler.copy() - np.pi * np.array([-1.0, 0, 0.0])

        fingertip_offset = (
            R.from_euler("xyz", ee_euler_adjustment).as_matrix() @ home_fingertip_offset
        )
        return fingertip_offset

    def init_webcontent(self):
        proprio = self.env.observe_proprio()
        fingertip_pos = proprio.eef_pos + self.calculate_fingertip_offset(
            proprio.eef_euler
        )

        # Initialize point cloud and gripper in scene
        fingertip_pos_ui = self.transform_robotframe_to_uiframe(
            fingertip_pos.reshape(1, 3)
        ).squeeze()
        ee_euler_ui = np.zeros(3)

        fingertip_pos_code = "new THREE.Vector3(%.2f, %.2f, %.2f);\n" % (
            fingertip_pos_ui[0],
            fingertip_pos_ui[1],
            fingertip_pos_ui[2],
        )
        ee_euler_code = "new THREE.Euler(%.2f, %.2f, %.2f);\n" % (
            ee_euler_ui[0],
            ee_euler_ui[1],
            ee_euler_ui[2],
        )

        with open("scripts/interface_assets/template_demo.html") as f:
            html_content = f.read()

        html_content = html_content % (
            self.num_point,
            fingertip_pos_code,
            ee_euler_code,
        )

        with open("scripts/interface_assets/index.html", "w") as f:
            f.write(html_content)

        # Start Server
        # this starts the localhost which we visit on browser
        # hold it in self to prevent destruction
        self.webserver_proc = mp.Process(target=http_server)
        self.webserver_proc.start()

    def init_ui_listen_process(self):
        ui_queue = mp.Queue(maxsize=1)

        async def listen_ui(websocket):
            async for message in websocket:
                message = json.loads(message)
                if not len(message):
                    continue

                if not ui_queue.empty():
                    print(
                        "WARNING: the ui_queue is not empty, dropping new UI command. "
                        "This should not happen"
                    )
                    continue

                data = message[-1]  # Retrieve the last waypoint in the UI
                click_ui_pos = [
                    data["click"]["x"],
                    data["click"]["y"],
                    data["click"]["z"],
                ]
                fingertip_ui_pos = [
                    data["position"]["x"],
                    data["position"]["y"],
                    data["position"]["z"],
                ]
                rotation = [
                    data["orientation"]["x"],
                    data["orientation"]["y"],
                    data["orientation"]["z"],
                ]

                info = {
                    "click_ui_pos": click_ui_pos,
                    "fingertip_ui_pos": fingertip_ui_pos,
                    "rotation": rotation,
                    "gripper_open": float(
                        data.get("url") == "http://localhost:8080/robotiq.obj"
                    ),
                    "done": data["done"],
                }
                print("from ui: gripper open?", info["gripper_open"])
                # block=True should take no extra time as the queue should be empty
                ui_queue.put(info, block=True)

        async def listen_ui_main():
            async with websockets.serve(listen_ui, "localhost", 8766):
                await asyncio.Future()

        self.listen_process = mp.Process(target=asyncio.run, args=(listen_ui_main(),))
        self.listen_process.start()

        return ui_queue

    def init_ui_update_process(self):
        ui_update_queue = mp.Queue(maxsize=1)

        async def send_data_to_web(server):
            while True:
                to_send = ui_update_queue.get()
                obs, update_ui = to_send["obs"], to_send["update_ui"]

                points, colors = self.prepare_point_cloud(obs)

                ee_pos = obs["ee_pos"]
                ee_euler = obs["ee_euler"]
                gripper_open = obs["gripper_open"]

                # Transform to UI frame
                fingertip_pos = ee_pos + self.calculate_fingertip_offset(ee_euler)
                fingertip_pos_ui = (
                    self.transform_robotframe_to_uiframe(fingertip_pos.reshape(1, 3))
                    .squeeze()
                    .tolist()
                )

                ee_euler_ui = [
                    ee_euler[0] + np.pi,
                    ee_euler[1],
                    ee_euler[2],
                ]

                data = {
                    "positions": points,
                    "colors": colors,
                    "fingertip_pos_ui": fingertip_pos_ui,
                    "ee_euler_ui": ee_euler_ui,
                    "gripper_action": [1 - int(gripper_open[0])],
                    "update_ui": update_ui,
                }

                msg = msgpack.packb(data)
                await server.send(msg)

        async def send_data_to_web_main():
            async with websockets.serve(send_data_to_web, "localhost", 8765):
                await asyncio.Future()

        self.send_process = mp.Process(
            target=asyncio.run, args=(send_data_to_web_main(),)
        )
        self.send_process.start()
        return ui_update_queue

    def apply_waypoint_mode(self, ui_cmd: dict, recorder: DatasetRecorder):
        click_pos = np.array(ui_cmd["click_ui_pos"])
        click_pos = self.transform_uiframe_to_robotframe(
            click_pos.reshape(1, 3)
        ).squeeze()

        fingertip_pos_cmd = np.array(ui_cmd["fingertip_ui_pos"])

        ee_euler_cmd = np.array(
            [
                ui_cmd["rotation"][0] + self.initial_euler[0],
                -ui_cmd["rotation"][2] + self.initial_euler[1],
                ui_cmd["rotation"][1] + self.initial_euler[2],
            ]
        )

        ee_pos_cmd = self.transform_uiframe_to_robotframe(
            fingertip_pos_cmd.reshape(1, 3)
        ).squeeze()
        ee_pos_cmd -= self.calculate_fingertip_offset(ee_euler_cmd)
        gripper_open_cmd = ui_cmd["gripper_open"]

        obs = self.env.observe()
        action = np.concatenate([ee_pos_cmd, ee_euler_cmd, [gripper_open_cmd]]).astype(
            np.float32
        )
        recorder.record(ActMode.Waypoint, obs, action, click_pos=click_pos)

        self.env.move_to(
            ee_pos_cmd,
            ee_euler_cmd,
            gripper_open_cmd,
            control_freq=self.control_freq,
            recorder=recorder,
        )

    def maybe_apply_dense_mode(
        self, send_queue: mp.Queue, recorder: DatasetRecorder, stopwatch
    ):
        # Step env with spacemouse actions
        data = self.teleop_interface.get_controller_state()
        dpos = data["dpos"]
        drot = data["raw_drotation"]
        hold = int(data["hold"])
        end_demo = data["lock"] == 1
        gripper_open = int(1 - float(data["grasp"]))  # binary
        assert isinstance(dpos, np.ndarray) and isinstance(drot, np.ndarray)

        dense_mode_triggered = np.linalg.norm(dpos) or np.linalg.norm(drot) or hold
        if not dense_mode_triggered:
            return False, end_demo

        # print("dense mode triggered")
        stopwatch.record_for_freq("dense")
        with FreqGuard(self.control_freq):
            with stopwatch.time("dense"):
                obs = self.env.observe()
                action = np.concatenate([dpos, drot, [gripper_open]]).astype(np.float32)
                recorder.record(ActMode.Dense, obs, action=action)
                self.env.apply_action(dpos, drot, gripper_open=gripper_open)

                # vis = np.hstack([obs["agent"], obs["agent2"]])
                # cv2.imshow("img", vis)
                # cv2.waitKey(1)

                # NOTE: technically the observation is 1-step off, but we re-use it to save time
                # one call of self.env.observe() takes about 55ms
                with stopwatch.time("dense.send"):
                    send_queue.put({"obs": obs, "update_ui": True})
        return True, end_demo

    def record_demo(self):
        self.init_webcontent()
        ui_queue = self.init_ui_listen_process()
        send_queue = self.init_ui_update_process()
        recorder = DatasetRecorder(self.data_folder)

        # self.env._cornell_reset()

        def one_episode():
            stopwatch = Stopwatch()

            while True:
                # waypoint mode
                if not ui_queue.empty():
                    ui_cmd = ui_queue.get()
                    if ui_cmd["done"]:
                        break

                    self.apply_waypoint_mode(ui_cmd, recorder)
                    # we can afford to call observe again here in waypoint mode
                    send_queue.put({"obs": self.env.observe(), "update_ui": True})
                    # await self.send_data_to_web(server, self.env.observe(), True, stopwatch)
                    continue

                # dense mode
                sent, end_demo = self.maybe_apply_dense_mode(
                    send_queue, recorder, stopwatch
                )
                need_update = sent

                if end_demo:
                    break

                if not sent:
                    with stopwatch.time("stream"):
                        # streaming mode
                        with FreqGuard(self.stream_freq):
                            with stopwatch.time("stream.observe"):
                                obs = self.env.observe()
                            with stopwatch.time("stream.send"):
                                send_queue.put({"obs": obs, "update_ui": need_update})
                                need_update = False

                if stopwatch.count("stream") >= 50 or stopwatch.count("dense") >= 50:
                    stopwatch.summary(reset=True)

            # By default, save npz but not gif
            save_demo, save_gif = True, False
            # if input("Save demo?") in ["n", "N"]:
            #     save_demo = False
            # if save_demo and input("Save gif?") in ["y", "Y"]:
            #     save_gif = True
            additional_info = {
                "transforms": self.transforms,
                "intrinsics": self.view_to_intrinsics,
            }
            recorder.end_episode(
                save=save_demo, save_gif=save_gif, additional_info=additional_info
            )

        while True:
            print(wrap_ruler("beginning new demo, resetting"))
            self.reset()
            self.teleop_interface.gripper_is_closed = False

            print("reset done, sending first frame")
            send_queue.put({"obs": self.env.observe(), "update_ui": True})

            print("episode start!")
            one_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_point", type=int, default=2500)
    parser.add_argument("--stream_freq", type=int, default=20)
    parser.add_argument("--control_freq", type=int, default=10)
    parser.add_argument("--data_folder", type=str, default="data/robot_dev")
    args = parser.parse_args()

    np.set_printoptions(precision=4, linewidth=100, suppress=True)

    cfg_path = FRANKA_ENV_CONFIG_PATH
    robot_config = pyrallis.load(FrankaEnvConfig, open(cfg_path, "r"))
    robot = InteractiveBot(
        args.num_point,
        args.stream_freq,
        args.control_freq,
        args.data_folder,
        robot_config,
    )

    robot.reset()
    robot.record_demo()
