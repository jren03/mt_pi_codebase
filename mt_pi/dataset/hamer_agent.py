"""
HaMeR agent for tracking hands in images. Provide HamerDebuger to visualize the hand keypoints on 2D images.
Author: Zi-ang-Cao
Date: Jun 28, 2024
"""

import os
from copy import deepcopy as dcp
from pathlib import Path

import cv2
import hamer
import numpy as np
import requests
import torch
import utils.hamer as hamer_vis
from detectron2.config import LazyConfig
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, HAMER
from hamer.utils import recursive_to
from hamer.utils.renderer import cam_crop_to_full
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from vitpose_model import ViTPoseModel


def get_parent_folder_of_package(package_name):
    # Import the package
    package = __import__(package_name)

    # Get the absolute path of the imported package
    package_path = os.path.abspath(package.__file__)

    # Get the parent directory of the package directory
    return os.path.dirname(os.path.dirname(package_path))


def load_hamer(checkpoint_path, root_dir=None):
    """Override `load_hamer` from hamer.models.__init__.py to update the model path"""
    from pathlib import Path

    from hamer.configs import get_config

    model_cfg = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
    model_cfg = get_config(model_cfg, update_cachedir=True)
    # update model and params path
    if root_dir:
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = os.path.join(root_dir, model_cfg.MANO.DATA_DIR)
        model_cfg.MANO.MODEL_PATH = os.path.join(
            root_dir, model_cfg.MANO.MODEL_PATH.replace("./", "")
        )
        model_cfg.MANO.MEAN_PARAMS = os.path.join(
            root_dir, model_cfg.MANO.MEAN_PARAMS.replace("./", "")
        )
        model_cfg.freeze()

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and (
        "BBOX_SHAPE" not in model_cfg.MODEL
    ):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, (
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        )
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


def download_detectron_ckpt(root_dir, ckpt_path):
    url = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    save_path = Path(root_dir, ckpt_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


class HamerDebuger(object):
    def __init__(self):
        self._init_hamer()
        print("HamerDebuger is initialized!")

    def _init_hamer(self, device_str="cuda"):
        # initialize hand tracker on cuda:1
        hamer_device = (
            torch.device(device_str)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hand_tracker = HandTrackAgent(None, hamer_device)

    def step(self, left_img, right_img=None, t=101):
        return self.hand_tracker.step(left_img, vis=True, t=t)


class HandTrackAgent(object):
    def __init__(
        self,
        device=torch.device("cuda:0"),
    ):
        """
        Pre-trained HaMeR model, for inference

        args:
        batch_size: type=int, default=2, help='Batch size for inference/fitting'
        rescale_factor: type=float, default=2.0, help='Factor for padding the bbox'
        """

        self.ROOT_DIR = get_parent_folder_of_package(
            "hamer"
        )  # auto detect the root directory of the project
        self.batch_size = 2
        self.rescale_factor = 2.0

        self.device = device
        print(">>> HandTrackAgent.device: ", self.device)

        # Initialize HaMeR model
        self._auto_init()

    def _auto_init(self):
        # Download and load checkpoints
        # download_models(CACHE_DIR_HAMER)
        ckpt_dir = Path(self.ROOT_DIR, DEFAULT_CHECKPOINT)
        self.hamer, self.hamer_cfg = load_hamer(ckpt_dir, self.ROOT_DIR)

        # Calculation specific to intel realsense d435i: https://github.com/geopavlakos/hamer/issues/18
        # EXTRA.FOCAL_LENGTH = real_focal_length / max(img_size.W, img_size.H) * 256
        # self.hamer_cfg.defrost()
        # self.hamer_cfg.EXTRA.FOCAL_LENGTH = 606.5 / 640 * 256  # 242.6
        # self.hamer_cfg.freeze()

        # Setup HaMeR model
        self.hamer = self.hamer.to(self.device)
        self.hamer.eval()

        # Load detector
        cfg_path = (
            Path(hamer.__file__).parent
            / "configs"
            / "cascade_mask_rcnn_vitdet_h_75ep.py"
        )
        detectron2_cfg = LazyConfig.load(str(cfg_path))

        detectron2_cfg.train.init_checkpoint = os.path.join(
            self.ROOT_DIR, "_DATA/detectron_ckpts/model_final_f05665.pkl"
        )
        if not os.path.exists(detectron2_cfg.train.init_checkpoint):
            download_detectron_ckpt(
                self.ROOT_DIR, "_DATA/detectron_ckpts/model_final_f05665.pkl"
            )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detectron2 = DefaultPredictor_Lazy(detectron2_cfg)

        # Keypoint detector
        # TODO: need to update the ROOT_DIR
        self.cpm = ViTPoseModel(self.device, root_dir=self.ROOT_DIR)

        print("Loaded HaMeR model")

    def debug_step(self, rgb_img, vis=False, t=None, verbose=False):
        rgb_img = np.ascontiguousarray(rgb_img)
        kpts_HaMeR_cam_Dict = self._step(self._preprocess(rgb_img), verbose=verbose)

        annotated_img = dcp(rgb_img)

        for kpts_ix, kpts in enumerate(kpts_HaMeR_cam_Dict["2d"]):
            color = tuple([255 if i == kpts_ix else 0 for i in range(3)])
            annotated_img = hamer_vis.label_2D_kpt_on_img(
                kpts_2d=kpts_HaMeR_cam_Dict["2d"][kpts_ix],
                img_cv2=annotated_img,
                color=color,
            )
        return rgb_img, annotated_img, kpts_HaMeR_cam_Dict

    def step(
        self, rgb_img, vis=False, t=None, verbose=False, put_result_on_cuda_tensor=False
    ):
        """
        Return keypoints in camera coordinate
        """
        rgb_img = np.ascontiguousarray(rgb_img)
        kpts_HaMeR_cam_Dict = self._step(
            self._preprocess(rgb_img),
            verbose=verbose,
            put_result_on_cuda_tensor=put_result_on_cuda_tensor,
        )

        if vis:
            for kpts_ix, kpts in enumerate(kpts_HaMeR_cam_Dict["2d"]):
                color = tuple([255 if i == kpts_ix else 0 for i in range(3)])
                annotated_img = hamer_vis.label_2D_kpt_on_img(
                    kpts_2d=kpts_HaMeR_cam_Dict["2d"][kpts_ix],
                    img_cv2=rgb_img,
                    color=color,
                )
            print("annotated_img.shape: ", annotated_img.shape)
            # save the annotated_img to a file
            parent_dir = os.path.join(os.getcwd(), "hamer_vis")
            os.makedirs(parent_dir, exist_ok=True)
            cv2.imwrite(
                f"{parent_dir}/labeled_img_{t}.jpg", annotated_img[..., ::-1].copy()
            )
            print("Saved debug img to ", f"{parent_dir}/labeled_img_{t}.jpg")
            breakpoint()
            input("Press Enter to continue...")

        return kpts_HaMeR_cam_Dict

    def _filter_boxes(self, boxes, right):
        # Initialize variables to store the index of the smallest box for right and not-right
        right_index = None
        not_right_index = None
        right_min_area = float("inf")
        not_right_min_area = float("inf")

        for i in range(len(boxes)):
            # Calculate the area of the current box
            box = boxes[i]
            area = (box[2] - box[0]) * (box[3] - box[1])

            # Check if the current box is for right or not-right and update the indices and min areas accordingly
            if right[i]:  # Right hand
                if area < right_min_area:
                    right_min_area = area
                    right_index = i
            else:  # Not-right hand
                if area < not_right_min_area:
                    not_right_min_area = area
                    not_right_index = i

        # Filter the boxes and right arrays to include only the boxes with the smallest areas for right and not-right
        filtered_boxes = []
        filtered_right = []
        if right_index is not None:
            filtered_boxes.append(boxes[right_index])
            filtered_right.append(right[right_index])
        if not_right_index is not None:
            filtered_boxes.append(boxes[not_right_index])
            filtered_right.append(right[not_right_index])

        if len(filtered_boxes) < 1:
            import pdb

            pdb.set_trace()
        return np.array(filtered_boxes), np.array(filtered_right)

    def _preprocess(self, rgb_img, debug=False):
        # Assume the H and W of rgb_img are multiples of 32
        # img_h = rgb_img.shape[0] - rgb_img.shape[0] % 32
        # rgb_img = rgb_img[:img_h]

        # print("[IN HAMER] rgb_img.shape: ", rgb_img.shape)

        ### ============= [1] VitPose detection ================== ###
        # Return the keypoint for whole body!!
        det_out = self.detectron2(rgb_img)

        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            rgb_img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        ### ============= [2] Hand area box ================== ###
        bboxes = []
        is_right = []
        idx_to_confidence = []

        # Use hands based on hand keypoint detections
        idx = 0
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            # Rejecting not confident detections
            num_left_valid_pts = sum(left_hand_keyp[:, 2] > 0.5)
            num_right_valid_pts = sum(right_hand_keyp[:, 2] > 0.5)
            num_valid_pts_threshold = 3
            if (
                num_left_valid_pts < num_valid_pts_threshold
                and num_right_valid_pts < num_valid_pts_threshold
            ):
                break
            if num_left_valid_pts > num_valid_pts_threshold:
                keyp = left_hand_keyp
                valid = keyp[:, 2] > 0.5
                bbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max(),
                ]
                bboxes.append(bbox)
                is_right.append(0)
                idx_to_confidence.append((idx, sum(keyp[valid, 2])))
                idx += 1
            if num_right_valid_pts > num_valid_pts_threshold:
                keyp = right_hand_keyp  # [21,3], pixel coords
                valid = keyp[:, 2] > 0.5
                bbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max(),
                ]
                bboxes.append(bbox)
                is_right.append(1)
                idx_to_confidence.append((idx, sum(keyp[valid, 2])))
                idx += 1

        if len(bboxes) == 0:
            # print("No hands detected")
            return None

        # Sort the boxes by confidence
        idx_to_confidence = sorted(idx_to_confidence, key=lambda x: x[1], reverse=True)

        top_idx, top_confidence = idx_to_confidence[0]
        bboxes_top = np.array([bboxes[top_idx]])
        right_top = np.array([is_right[top_idx]])

        boxes = np.stack(bboxes_top)
        right = np.stack(right_top)

        # If there are multiple left or right boxes, filter larger boxes.
        filtered_boxes, filtered_right = self._filter_boxes(boxes, right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(
            self.hamer_cfg,
            rgb_img,
            filtered_boxes,
            filtered_right,
            rescale_factor=self.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Save the boxes for debuging:
        if debug:
            self.latest_boxes = boxes
            self.latest_right = right
            self.filtered_boxes = filtered_boxes
            self.filtered_right = filtered_right
            self.dataset = dataset
        try:
            for batch in dataloader:
                batch = recursive_to(batch, self.device)
        except Exception as e:
            print("Error in _preprocess: ", e)
            import pdb

            pdb.set_trace()

        return dataloader, top_confidence

    def _step(
        self,
        preprocess_tup,
        verbose=False,
        debug=False,
        put_result_on_cuda_tensor=False,
    ):
        if preprocess_tup is None:
            if debug:
                import pdb

                pdb.set_trace()
            else:
                # assert False, "No hands detected"
                return None

        dataloader, top_confidence = preprocess_tup
        kpts_HaMeR_cam_Dict = {
            "is_right": [],
            "2d": [],
            "3d": [],
            "camera_translation": [],
            "verts": [],
            "confidence": top_confidence,
        }

        ### ============= [3] HaMeR ================== ###
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.hamer(batch)

            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            # NOTE: FOR HaMeR, they are using the img_size as (W, H)
            W_H_shapes = batch[
                "img_size"
            ].float()  # (2, 2): full image size (2208, 1242)

            multiplier = 2 * batch["right"] - 1
            scaled_focal_length = (
                self.hamer_cfg.EXTRA.FOCAL_LENGTH
                / self.hamer_cfg.MODEL.IMAGE_SIZE
                * W_H_shapes.max()
            )

            # Get cam_t to full image (instead of bbox)
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, W_H_shapes, scaled_focal_length
            )

            batch_size = pred_cam_t_full.shape[0]
            for n in range(batch_size):
                kpts_3d = out["pred_keypoints_3d"][n].detach().cpu().numpy()  # [21, 3]
                verts = out["pred_vertices"][n].detach().cpu().numpy()  # [778, 3]

                is_right = batch["right"][n].cpu().numpy()
                # NOTE: For the left hand, must flip the x-axis of the kpts_3d_H_world!!!
                kpts_3d[:, 0] = (2 * is_right - 1) * kpts_3d[:, 0]

                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]

                if put_result_on_cuda_tensor:
                    kpts_3d = torch.tensor(kpts_3d).to(self.device)
                    verts = torch.tensor(verts).to(self.device)
                    pred_cam_t_full = pred_cam_t_full.to(self.device)
                else:
                    kpts_3d = torch.tensor(kpts_3d).cpu()
                    verts = torch.tensor(verts).cpu()
                    pred_cam_t_full = pred_cam_t_full.cpu()

                camera_translation = pred_cam_t_full[n].unsqueeze(0)

                img_W = W_H_shapes[n][0]
                img_H = W_H_shapes[n][1]

                # we are hoping to get H, W here!!
                kpts_2d_HaMeR_cam = hamer_vis.project_hamer_3D_to_2D(
                    kpts_3d, img_W, img_H, camera_translation, scaled_focal_length
                )
                if verbose:
                    print(
                        "finish vision_module.utils.hamer_utils.project_hamer_3D_to_2D"
                    )

                kpts_3d_HaMeR_cam = kpts_3d + camera_translation
                verts_HaMeR_cam = verts + camera_translation

                kpts_HaMeR_cam_Dict["is_right"].append(is_right)

                kpts_HaMeR_cam_Dict["2d"].append(kpts_2d_HaMeR_cam)
                kpts_HaMeR_cam_Dict["3d"].append(kpts_3d_HaMeR_cam)
                kpts_HaMeR_cam_Dict["camera_translation"].append(camera_translation)

                kpts_HaMeR_cam_Dict["verts"].append(verts_HaMeR_cam)

        kpts_HaMeR_cam_Dict["pred_mano_params"] = out["pred_mano_params"]
        return kpts_HaMeR_cam_Dict


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hand_detector = HandTrackAgent(device)
