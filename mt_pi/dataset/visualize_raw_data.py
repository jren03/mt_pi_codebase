"""
This script creates videos for the raw training images and images with ground-truth tracks.
"""

import cv2
import zarr
import numpy as np

from argparse import ArgumentParser
from pathlib import Path


def visualize_raw_images(data_dir: Path):
    zarr_dir = Path(data_dir, "dataset.zarr")
    if not zarr_dir.exists():
        print(f"Skipping {zarr_dir} as it does not exist")
        return
    dataset_root = zarr.open(str(zarr_dir), "r")
    save_dir = Path(data_dir, "vis_outs")
    save_dir.mkdir(exist_ok=True, parents=True)

    img_dir_names = [
        d.name
        for d in Path(zarr_dir, "data").iterdir()
        if d.is_dir() and "img" in d.stem
    ]
    for img_dir_name in img_dir_names:
        print(f"Processing raw images for {img_dir_name}")
        # float32, [0,1], (N,H,W,3)
        train_image_data = dataset_root["data"][img_dir_name][:]
        train_image_data = (train_image_data).astype(np.uint8)

        video_name = Path(save_dir, f"{img_dir_name}.mp4")
        height, width, _ = train_image_data[0].shape
        out = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
        )
        for img in train_image_data:
            img_rgb = img[..., ::-1]  # reverse the channel order
            out.write(img_rgb)
        out.release()
        print(f"Saved video to {video_name}")


def visualize_preprocessed_images(data_dir: Path):
    vis_dirs = [
        Path(data_dir, "vis"),
        Path(data_dir, "vis_2d"),
        Path(data_dir, "vis_3d"),
    ]
    save_dir = Path(data_dir, "vis_outs")
    save_dir.mkdir(exist_ok=True, parents=True)

    for vis_dir in vis_dirs:
        if not vis_dir.exists():
            continue
        frames = sorted([f for f in vis_dir.glob("*png")])
        if not frames:
            print(f"No frames found in {vis_dir}")
            continue
        img = cv2.imread(str(frames[0]))
        height, width, _ = img.shape
        if "_" in vis_dir.stem:
            video_name = Path(
                save_dir, f"preprocessed_{vis_dir.stem.split('_')[-1]}.mp4"
            )
        else:
            video_name = Path(save_dir, "preprocessed.mp4")
        out = cv2.VideoWriter(
            str(video_name), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
        )
        for frame in frames:
            img = cv2.imread(str(frame))
            out.write(img)
        out.release()
        print(f"Saved preprocessed video to {video_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=Path, default="data/multi_track_dset")
    args = parser.parse_args()

    for split in ["train", "val"]:
        data_dir = Path(args.data_dir, split)
        if "data" not in str(data_dir):
            data_dir = Path("data", data_dir)

        if not data_dir.exists():
            print(f"Skipping {data_dir} as it does not exist")
            continue
        visualize_raw_images(data_dir)
        visualize_preprocessed_images(data_dir)
