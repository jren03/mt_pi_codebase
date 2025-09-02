import argparse
from pathlib import Path

import cv2
import numpy as np
import zarr
from tqdm import tqdm

from dataset.depth_agent import DepthAnythingAgent


def process_with_depth(zarr_path, vis_dir, max_frames=-1):
    # Open the zarr file
    store = zarr.open(zarr_path, mode="r+")
    data_group = store["data"]

    # Initialize DepthAnythingV2 model
    depth_model = DepthAnythingAgent(encoder="vitb")

    # Check if 'depth' group already exists
    if "depth" in store:
        print("'depth' group already exists. Overwriting...")
        del store["depth"]

    # Create a new group for depth maps
    depth_group = store.create_group("depth")

    # Process images for each viewpoint
    for vp in [1, 2]:
        images = np.array(data_group[f"img{vp}"])

        # Create a dataset for depth maps
        depth_map_gs = depth_group.create_dataset(
            f"depth{vp}_gs",
            shape=(len(images), *images.shape[1:3], 1),
            chunks=(10, *images.shape[1:3], 1),
            dtype=np.uint8,
        )
        depth_map_colored = depth_group.create_dataset(
            f"depth{vp}_colored",
            shape=(len(images), *images.shape[1:4]),
            chunks=(10, *images.shape[1:4]),
            dtype=np.uint8,
        )

        # Process images and generate depth maps
        for i, img in tqdm(
            enumerate(images), total=len(images), desc=f"Processing viewpoint {vp}"
        ):
            depth_grayscale, depth_colored = depth_model.run_inference(img)
            depth_map_gs[i] = depth_grayscale
            depth_map_colored[i] = depth_colored

        print(f"Processed {len(images)} images for viewpoint {vp}, saving video...")

        max_frames = max_frames if max_frames > 0 else len(images)
        depth_map_gs_bgr = np.array(
            [
                cv2.cvtColor(depth_map_gs[i], cv2.COLOR_GRAY2BGR)
                for i in range(len(depth_map_gs))
            ]
        )
        frames = np.hstack(
            [depth_map_gs_bgr[:max_frames], depth_map_colored[:max_frames]]
        )
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(
            f"{vis_dir}/depth_{vp}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (width, height),
        )
        for frame in frames:
            out.write(frame)
        out.release()

    print(f"Processing complete. Results saved to {zarr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add depth modality to zarr file")
    parser.add_argument(
        "-d", "--data_path", nargs="+", help="Path to the data directory"
    )
    parser.add_argument(
        "--max_frames", type=int, default=-1, help="Maximum number of frames to process"
    )
    args = parser.parse_args()

    for data_path in args.data_path:
        if "data/" not in data_path:
            data_path = f"data/{data_path}"

        for partition in ["train", "val"]:
            partition_path = f"{data_path}/{partition}"

            if Path(f"{partition_path}/dataset.zarr").exists():
                zarr_path = f"{partition_path}/dataset.zarr"
                vis_dir = f"{partition_path}/vis_outs"
                if not Path(vis_dir).exists():
                    Path(vis_dir).mkdir(parents=True, exist_ok=True)

                print(f"Processing {zarr_path}")
                process_with_depth(zarr_path, vis_dir, args.max_frames)
