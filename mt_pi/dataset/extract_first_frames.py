import argparse
import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm


def extract_first_frames(zarr_path):
    print(f"Extracting first frames from {zarr_path}")
    output_path = Path(Path(zarr_path).parent, "first_frames.npz")
    # Open the zarr file
    store = zarr.open(zarr_path, mode="r")
    data_group = store["data"]
    meta = store["meta"]

    first_frames = []
    episode_indices = []

    # Process each viewpoint
    for vp in [1, 2]:
        images = np.array(data_group[f"img{vp}"])
        episode_ends = np.array(meta[f"episode_ends{vp}"])

        start_idx = 0
        for i, end_idx in tqdm(enumerate(episode_ends), total=len(episode_ends)):
            first_frame = images[start_idx]
            first_frames.append(first_frame)
            episode_indices.append((vp, i))
            start_idx = end_idx

    # Save the first frames and their indices
    np.savez(
        output_path, frames=np.array(first_frames), indices=np.array(episode_indices)
    )

    print(f"Saved {len(first_frames)} first frames to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first frames from zarr file")
    parser.add_argument("-d", "--data_dirs", nargs="+", help="Paths to the zarr files")
    args = parser.parse_args()

    for data_dir in args.data_dirs:
        if "data/" not in data_dir:
            data_dir = "data/" + data_dir
        zarr_path = Path(data_dir, "train", "dataset.zarr")
        extract_first_frames(zarr_path)
