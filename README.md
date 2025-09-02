<h1 align="center" id="top">Motion Tracks Policy (MT-&pi;)</h1>

<p align="center">
      <img src="https://img.shields.io/badge/python-3.10-blue" />
      <a href= "https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>


This repository contains implementations of the algorithms presented in [Motion Tracks: A Unified Representation for Human-Robot Transfer in Few-Shot Imitation Learning](https://arxiv.org/abs/2501.06994). For videos and key insights, please check out our [website](https://portal.cs.cornell.edu/motion_track_policy/).

## Table of Contents
- [Installation](#installation-hammer_and_wrench)
- [Camera Calibration](#camera-calibration-camera)
- [Recording Demonstrations](#recording-demonstrations-clapper)
  - [Recording Robot Demos](#recording-robot-demos)
  - [Recording Human Demos](#recording-human-demos)
- [Post-processing Demonstrations](#post-processing-demonstrations-computer)
  - [Processing Robot Demos](#processing-robot-demos)
  - [Processing Human Demos](#processing-human-demos)
  - [Visualizing Data](#visualizing-data)
- [Training](#training-mortar_board)
  - [Keypoint Predictor](#keypoint-predictor)
  - [MT-π](#mt-π)
- [Testing](#testing-test_tube)
- [Acknowledgments](#acknowledgements-pencil)
- [Citation](#citation)

This repository is structured as follows:
```
├── mt_pi/                      # Main package directory
│   ├── dataset/                # Data processing and visualization utilities
│   │   ├── process_robot_dataset.py    # Robot demo processing
│   │   ├── process_hand_dataset.py     # Human demo processing
│   ├── envs/                   # Robot environment interfaces
│   ├── models/                 # Model architectures and training utilities
│   │   ├── diffusion_policy.py         # Main diffusion policy model
│   │   ├── keypoint_map_predictor.py   # Keypoint prediction model
│   ├── scripts/                # Training and evaluation scripts
│   │   ├── train_mtpi.py               # Main training script
│   │   ├── eval_mtpi.py                # Evaluation script
│   │   ├── record_robot_demos.py       # Robot demo recording
│   │   ├── record_human_demos.py       # Human demo recording
│   │   └── ...                         # Additional scripts
│   ├── utils/                  # Utility functions and helpers
│   └── third_party/            # Third-party dependencies
├── install_deps.sh            # Dependency installation script
└── set_env.sh                 # Environment setup script
```


## Installation :hammer_and_wrench:

Please run
```
source install_deps.sh
``` 
to set up all dependencies needed for this project. 

Note this script only needs to be run once. For future uses, please run `source set_env.sh` once per shell before running any script from this repo. 

> [!NOTE] 
> Our real-robot setup follows that of [SPHINX](https://github.com/priyasundaresan/sphinx/blob/main/README.md). Specifically, please find a walkthrough on how to collect data using the point cloud interface on this [Google Doc](https://docs.google.com/document/d/1mpHAVoCbp7k2y1qc_WS0c4HW7EAOpovFvp6tYPP46hI/edit?usp=sharing), and [this link](https://github.com/priyasundaresan/sphinx/blob/main/README.md#starting-up-the-robot--gripper-on-nuc) on how to set up the interface between the robot and workstation.


## Camera Calibration :camera:

Well-calibrated cameras are crucial for accurate triangulation used by MT-π. We provide a script for camera calibration with a Franka Panda. 

>[!IMPORTANT]
> You may have to adjust the `self.home` variable in `envs/minrobot/controller.py` depending on the version of your robot.

We will use a yellow-colored cube for calibration. You may change the color as you wish, but in general, try picking a color with maximum contrast against your background. 

1. Run `python scripts/calibrate.py`
2. Follow the instructions on the terminal to place the cube as close to the gripper root as possible, then press Enter to close.
3. After initial calibration, the associated error should be on the order of `1e-3`. If not, it's possible that a) one of the cameras detected an object in the background instead of the block itself, or b) one or both of the cameras are flipped.
4. Continue following the instructions on the terminal to finish calibration and alignment of the two point clouds.


## Recording Demonstrations :clapper:

### Recording Robot Demos
To record robot demonstrations, run
```
python scripts/record_robot_demos.py --data_folder data/robot_<task_name>
```
For robot data collection, we use a combination of waypoint interpolation via a UI and fine-grained control via a spacemouse, as shown in [SPHINX](https://github.com/priyasundaresan/sphinx/blob/main/README.md). For information on how to collect data using this interface, please refer to this [Google Doc](https://docs.google.com/document/d/1mpHAVoCbp7k2y1qc_WS0c4HW7EAOpovFvp6tYPP46hI/edit?usp=sharing).

>[!TIP]
> The right button on the spacemouse is currently mapped to "End Demo". You can choose to use that, or the button on the interface at the end of each trajectory.

### Recording Human Demos
To record human demonstrations, run 
```
python scripts/record_human_demos.py --data_folder data/hand_<task_name>
```

The cameras will first warm up, then a prompt will appear on the terminal for you to press Enter to start recording. To speed up human demonstration collection, we have added two prompts at the end of each trajectory:
1. The first asks if you want to save the demo
2. The second asks if you would like to save the gif visualization

By default (triggered when only the Enter key is pressed), the first is True and the second is False (we save the human demonstration but skip the gif-generation process). In other words, if you are happy with your trajectory, pressing Enter twice will start the next recording.

>[!TIP]
>Processing human demonstrations, as described below, is a multi-step process. Since the post-processing step involves passing the demonstrations through multiple pre-trained models (HaMeR, SAM2), we can improve efficiency by saving chunks of the demonstration data to sequential directories, i.e. `hand_<task_name>_1`, `hand_<task_name>_2`, etc.

## Post-processing Demonstrations :computer:

### Processing Robot Demos
To process the robot demonstrations, simply run
```
python dataset/process_robot_dataset.py --demo_dir data/robot_<task_name>
```

This will create a corresponding directory called `data/robot_<task_name>_tracks` which contains both train and validation directories.

### Processing Human Demos
Human demonstrations are processed in a multi-step fashion:
1. Extraction of human hand poses
2. Extraction of bounding boxes around objects of interest
3. Extraction of grasp information

First, to extract the hand pose using HaMeR, run 
```
python dataset/process_human_datasets.py --demo_dirs data/hand_<task_name>
```
Like the robot demonstrations, this will create a corresponding directory called `data/hand_<task_name>_tracks`.

Next, we need to produce a segmentation of the object in order to extract the grasp pose. Begin by extracting the first frame of each episode via
```
python dataset/extract_first_frames.py --data_dirs data/hand_<task_name>_tracks
```

In most cases, running [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) is sufficient to provide a good enough bounding box for SAM2. To get those, run
```
python dataset/select_points.py --npz_path `data/hand_<task_name>_tracks/train/first_frames.npz` --use_dino --object_name carrot
```

Otherwise, there is also the option to manually select points of the manipulated object using the same script without the `--use_dino` flag. In the cv2 window that pops up, to undo a click, press `u`, and to go to the next image, press `n`.

At the end of the point selection process, you should end up with a new file at `data/hand_<task_name>_tracks/train/first_frames_points.npz`. Finally, we extract the grasp information using
```
python dataset/postprocess_with_sam.py -d data/hand_<task_name>_tracks 
```

### Visualizing Data
We provide additional scripts to help visualize the results of this post-processing. It is recommended to run these before training at least once to check that calibration is done properly for robot demonstrations and that tracks/grasp information is correct for human demonstrations.
```
python dataset/image_track_visualizer.py --data_paths [data/robot_<task_name>_tracks/val/dataset.zarr, data/human_<task_name>_tracks/val/dataset.zarr]
```


## Training :mortar_board:

### Keypoint Predictor

The keypoint predictor is a small MLP that maps from noise to human keypoints to bridge the representation gap between robot and human end-effector poses. Train this first via
```
python scripts/train_keypoint_predictor.py --data_path [hand_<task_name>_tracks]
```
Then, update the path to the keypoint predictor in `models/diffusion_policy.py` under the variable `keypoint_map_from` to the final checkpoint.

### MT-π
To train the main policy, run 
```
python scripts/train_mtpi.py --data_path [<insert_data_paths>]
```

## Testing :test_tube:
To test a trained policy on the real robot, first make sure that the log directory containing `config.yaml`, `dataset_stats/`, and `checkpoints/` is on the workstation. Note that if this policy is trained with the keypoint predictor, that checkpoint will also need to be copied to the same relative path as during training (or updated in `config.yaml`).

Then run
```
python scripts/eval_mtpi.py --log_dir experiments_logs/<path_to_parent_directory_of_ckpt>
```

>[!TIP]
> At any point during rollout, you can press Enter to end the trajectory and reset the arm back to its starting state. By default, all rollouts are recorded. You can turn this off via `record=False`. All recordings are saved under a `real_rollouts/` subdirectory where the checkpoint resides (e.g. `experiment_logs/MMDD/<timestamp>_<uuid>/real_rollouts`).

To visualize the predicted tracks for on-policy rollouts, run 
```
python scripts/visualize_online_rollouts.py -d experiment_logs/MMDD/<timestamp>_<uuid>/real_rollouts
```
directly on the workstation.


## Acknowledgements :pencil:

This codebase is built on many files from [SPHINX](https://github.com/priyasundaresan/sphinx/tree/main), and real robot experiments done on top of [Monometis](https://github.com/hengyuan-hu/monometis). We also use a number of third-party repositories in our work, namely [HaMeR](https://github.com/geopavlakos/hamer.git), [SAM2](https://github.com/facebookresearch/segment-anything-2.git), and [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2.git). Additionally, we thank Zi-ang Cao for his wrappers and utility functions interfacing with HaMeR. 


## Citation

If you found this repository useful in your research, please consider citing our paper.
```bibtex
@article{ren2025motion,
  title={Motion tracks: A unified representation for human-robot transfer in few-shot imitation learning},
  author={Ren, Juntao and Sundaresan, Priya and Sadigh, Dorsa and Choudhury, Sanjiban and Bohg, Jeannette},
  journal={arXiv preprint arXiv:2501.06994},
  year={2025}
}
```

For any questions regarding the paper or issues with the codebase, please feel free to contact juntao [dot] ren [at] stanford [dot] edu.

***
[Back to top](#top)