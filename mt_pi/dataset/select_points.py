import argparse
import numpy as np
import cv2
from tqdm import tqdm
from transformers import pipeline
from PIL import Image

from dataclasses import dataclass
from typing import Dict, List, Optional

points = []
current_image = None
original_image = None


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


def click_event(event, x, y, flags, param):
    global points, current_image, original_image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        current_image = original_image.copy()
        for point in points:
            cv2.circle(current_image, point, 5, (0, 255, 0), -1)
        cv2.imshow("image", cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))


def get_highest_box(results):
    return max(results, key=lambda r: r.score)


def select_points_dino(frame, object_name, detector):
    img_pil = Image.fromarray(frame)
    labels = [f"{object_name}."]
    results = detector(img_pil, candidate_labels=labels, threshold=0.4)
    results = [DetectionResult.from_dict(result) for result in results]
    if not results:
        return None
    highest_box = get_highest_box(results).box.xyxy
    x_min, y_min, x_max, y_max = highest_box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return [(int(center_x), int(center_y))]


def select_points(npz_path, output_path, use_dino, object_name):
    global points, current_image, original_image
    data = np.load(npz_path)
    frames = data["frames"]
    indices = data["indices"]
    all_points = []

    if use_dino:
        detector_id = "IDEA-Research/grounding-dino-tiny"
        object_detector = pipeline(
            model=detector_id,
            task="zero-shot-object-detection",
            device="cuda",
        )

    for i, (frame, (vp, ep_idx)) in enumerate(
        tqdm(zip(frames, indices), total=len(frames))
    ):
        original_image = frame.copy()
        current_image = original_image.copy()

        if use_dino:
            points = select_points_dino(frame, object_name, object_detector)
            if points is None:
                print(
                    f"No object detected for viewpoint {vp}, episode {ep_idx}. Skipping..."
                )
                all_points.append(np.array([]))
                continue
        else:
            cv2.imshow("image", cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
            cv2.setMouseCallback("image", click_event)
            points = []
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):  # Press 'n' to move to the next image
                    break
                elif key == ord("u"):  # Press 'u' to undo the last point
                    if points:
                        points.pop()
                        current_image = original_image.copy()
                        for point in points:
                            cv2.circle(current_image, point, 5, (0, 255, 0), -1)
                        cv2.imshow(
                            "image", cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
                        )

        all_points.append(np.array(points))
        print(f"Selected {len(points)} points for viewpoint {vp}, episode {ep_idx}")

    if not use_dino:
        cv2.destroyAllWindows()

    np.savez(output_path, points=np.array(all_points), indices=indices)
    print(f"Saved selected points to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive point selection")
    parser.add_argument(
        "-d",
        "--npz_path",
        default="first_frames.npz",
        help="Path to the input NPZ file",
    )
    parser.add_argument(
        "--use_dino",
        action="store_true",
        help="Use DINO-v2 for automatic point selection",
    )
    parser.add_argument(
        "-o",
        "--object_name",
        type=str,
        default="object",
        help="Name of the object to detect when using DINO-v2",
    )
    args = parser.parse_args()

    npz_path = args.npz_path
    if npz_path.endswith("_tracks"):
        npz_path = f"{npz_path}/train/first_frames.npz"
    if "data/" not in npz_path:
        npz_path = f"data/{npz_path}"

    print(f"Parsing {npz_path}")

    output_path = npz_path.replace(".npz", "_points.npz")
    select_points(npz_path, output_path, args.use_dino, args.object_name)
