"""
Example command:

python scripts/record_human_demos.py --camera_cfg_path envs/camera.yaml --freq 10
"""

import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import pyrallis
from envs.camera_env import CameraEnv, CameraEnvConfig
from utils.constants import CAMERA_CONFIG_PATH, SPACEMOUSE_SPECS
from utils.freq_guard import FreqGuard
from utils.recorders import ActMode, DatasetRecorder
from utils.spacemouse import SpaceMouseInterface
from utils.stopwatch import Stopwatch


@dataclass
class RecordConfig:
    camera_cfg_path: str = CAMERA_CONFIG_PATH
    data_folder: str = "data/hand_dev"
    show_camera: int = 1
    freq: float = 10
    max_len: int = 500
    num_demos: int = 50

    @property
    def env_cfg(self):
        return pyrallis.load(CameraEnvConfig, open(self.camera_cfg_path))


class EnterKeyListener:
    def __init__(self):
        self.enter_pressed = False
        self.thread = threading.Thread(target=self._listen_for_enter)
        self.thread.daemon = True
        self.thread.start()

    def _listen_for_enter(self):
        input()
        self.enter_pressed = True

    def reset(self):
        self.enter_pressed = False


class SpaceMouseListener:
    def __init__(self):
        self.lock_pressed = False
        self.interface = SpaceMouseInterface(**SPACEMOUSE_SPECS)
        self.interface.start_control()
        self.thread = threading.Thread(target=self._listen_for_lock)
        self.thread.daemon = True
        self.thread.start()

    def _listen_for_lock(self):
        while True:
            data = self.interface.get_controller_state()
            if data["lock"] == 1:
                self.lock_pressed = True
                break
            time.sleep(0.01)

    def reset(self):
        self.lock_pressed = False


def flush_input():
    try:
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        import sys
        import termios  # for linux/unix

        termios.tcflush(sys.stdin, termios.TCIOFLUSH)


def get_user_input(prompt, spacemouse_interface):
    print(prompt)
    flush_input()

    keyboard_input = threading.Event()
    keyboard_result = [None]

    def keyboard_listener():
        keyboard_result[0] = input().lower().strip()
        keyboard_input.set()

    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()

    last_button_states = spacemouse_interface.get_button_states()

    while not keyboard_input.is_set():
        current_button_states = spacemouse_interface.get_button_states()
        # print(current_button_states)

        # Check if the right button (lock) was just pressed
        if current_button_states["right"] == 1 and last_button_states["right"] == 0:
            return "r"

        # Check if the left button (grasp) was just pressed
        if current_button_states["left"] == 1 and last_button_states["left"] == 0:
            return "l"

        last_button_states = current_button_states
        time.sleep(0.01)

    return keyboard_result[0]


def record_episode(
    env: CameraEnv,
    recorder: DatasetRecorder,
    freq: float,
    stopwatch: Stopwatch,
    show_camera: int,
):
    with stopwatch.time("reset"):
        env.reset()

    assert os.path.exists("calib/transforms_both.npy")
    transforms = np.load("calib/transforms_both.npy", allow_pickle=True).item()

    spacemouse_listener = SpaceMouseListener()
    print("Right click spacemouse to stop recording")

    st = time.time()
    while not spacemouse_listener.lock_pressed:
        with FreqGuard(freq):
            with stopwatch.time("observe"):
                obs = env.observe()
                obs.pop("wrist_image", None)
                action = np.zeros(7)
                recorder.record(ActMode.Dense, obs, action)  # Record obs and action

    print(f"Recording took {time.time() - st:.2f} seconds.")
    stopwatch.summary()

    # By default, save npz but not gif
    save_demo, save_gif = True, False

    spacemouse_interface = spacemouse_listener.interface
    seperator = "-" * 50
    print(f"\n{seperator}")
    print(
        "Pressing Enter or Right Click on Spacemouse will use the default option in brackets.",
    )
    save_input = get_user_input("Save demo? ([y]/n): ", spacemouse_interface)
    if save_input.lower().strip() in ["n", "l"]:
        save_demo = False

    if save_demo:
        gif_input = get_user_input("Save gif? (y/[n]): ", spacemouse_interface)
        if gif_input.lower().strip() in ["y", "l"]:
            save_gif = True

    additional_info = {"transforms": transforms}
    recorder.end_episode(
        save=save_demo, save_gif=save_gif, additional_info=additional_info
    )
    print(f"{seperator}\n")


def main(cfg: RecordConfig):
    env = CameraEnv(cfg.env_cfg)

    # warm up camera for 10s
    print("Warm up cameras...")
    for _ in range(int(cfg.freq) * 5):
        with FreqGuard(cfg.freq):
            env.observe()

    stopwatch = Stopwatch()
    recorder = DatasetRecorder(cfg.data_folder)

    input("Press Enter to start recording.")
    for _ in range(cfg.num_demos):
        record_episode(env, recorder, cfg.freq, stopwatch, cfg.show_camera)


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=RecordConfig)  # type: ignore
    main(cfg)
