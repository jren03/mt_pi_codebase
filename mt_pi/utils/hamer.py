"""
Utils for HaMeR
Author: Zi-ang-Cao
Date: Jun 28, 2024
"""

from typing import Optional

import cv2
import numpy as np
import torch
from hamer.utils.geometry import perspective_projection
from hamer.utils.render_openpose import render_openpose

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
LIGHT_GREEN = (0.839, 0.929, 0.824)
PURE_RED = (1, 0, 0)
PURE_PURPLE = (1, 0, 1)
LIGHT_BLUE_INT = tuple(round(value * 255) for value in LIGHT_BLUE)
LIGHT_GREEN_INT = tuple(round(value * 255) for value in LIGHT_GREEN)
PURE_RED_INT = tuple(round(value * 255) for value in PURE_RED)
PURE_PURPLE_INT = tuple(round(value * 255) for value in PURE_PURPLE)


def eval_error(correspond_3d_TRI_cam, kpts_3d_HaMeR_cam, verbose=False):
    # Assume TRI_cam and HaMeR_cam are sharing the same origin
    # Compute the relative difference between TRI_cam and HaMeR_cam
    dist_TRI = torch.norm(correspond_3d_TRI_cam, dim=1)
    dist_HaMeR = torch.norm(kpts_3d_HaMeR_cam, dim=1)
    ratio = dist_TRI / dist_HaMeR
    delta_3d = correspond_3d_TRI_cam - kpts_3d_HaMeR_cam

    depth_ratio = correspond_3d_TRI_cam[:2] / kpts_3d_HaMeR_cam[:2]

    if verbose:
        print("depth_ratio: ", depth_ratio)
        print(">>>> avg depth_ratio: ", depth_ratio.mean())

        print("delta_3d: ", delta_3d)
        print(">>>> avg offset: ", delta_3d.mean(dim=0))

        print("norm ratio (dist_HaMeR / dist_TRI): ", ratio)
        print(">>>> avg norm ratio: ", ratio.mean())

    return delta_3d, ratio, depth_ratio


def calibrate_3d_kpt(
    clear_idx, correspond_3d_TRI_cam, kpts_3d_HaMeR_cam, verbose=False
):
    raw_kpt_3d_HaMeR_cam = kpts_3d_HaMeR_cam.clone()[clear_idx]
    raw_kpt_3d_TRI_cam = correspond_3d_TRI_cam.clone()[clear_idx]

    raw_error = raw_kpt_3d_TRI_cam - raw_kpt_3d_HaMeR_cam
    if verbose:
        print("raw_error: ", raw_error)
        print("eval the error before calibration")
    delta_3d, norm_ratio, depth_ratio = eval_error(
        correspond_3d_TRI_cam[clear_idx], kpts_3d_HaMeR_cam[clear_idx], verbose
    )
    return_scale = 1
    return_offset = 0
    return_mode = ""

    prev_error_drop = 0
    for mode in ["offset", "norm", "depth"]:
        # try to apply offset, ratio, and depth_ratio to the kpts_3d_HaMeR_cam to reduce the error
        offset = 0
        ratio = 1
        if mode == "offset":
            offset = delta_3d.mean(dim=0)
        elif mode == "norm":
            ratio = norm_ratio.mean()
        elif mode == "depth":
            ratio = depth_ratio.mean()

        error = raw_kpt_3d_TRI_cam - (raw_kpt_3d_HaMeR_cam + offset) * ratio

        error_drop = (raw_error - error).norm(dim=1).mean()
        if verbose:
            print(f"[{mode}] lead error_drop={error_drop} & updated error to : {error}")

        if error_drop > prev_error_drop:
            prev_error_drop = error_drop
            return_mode = mode

            return_scale = ratio
            return_offset = offset

    if verbose:
        print(">>>> return_mode: ", return_mode)
        print(">>>> return_scale: ", return_scale)
        print(">>>> return_offset: ", return_offset)
    return return_scale, return_offset, return_mode


def label_2D_kpt_on_img(
    kpts_2d: torch.Tensor,
    img_cv2: np.array,
    color: tuple = (255, 0, 0),
) -> np.array:
    # Draw each point
    points = kpts_2d.cpu().numpy().astype(np.int32)
    for point in points:
        x, y = point
        cv2.circle(img_cv2, (x, y), radius=5, color=color, thickness=-1)

    # BGR for cv2
    cv2.circle(img_cv2, (100, 100), radius=25, color=(0, 255, 0), thickness=-1)
    cv2.circle(img_cv2, (100, 200), radius=25, color=(255, 0, 0), thickness=-1)
    cv2.circle(img_cv2, (100, 500), radius=25, color=(0, 0, 255), thickness=-1)

    return img_cv2


def draw_2D_on_img(
    kpts_2d: torch.Tensor,
    img_cv2: np.array,
) -> np.array:
    # Add additional alpha channel to the 2D kpts
    kpts_2d = torch.cat(
        [kpts_2d.cpu(), torch.ones(kpts_2d.shape[0], 1).cpu()], dim=-1
    ).numpy()
    kpts_proj_2d_imgs = []
    assert kpts_2d.shape[1] == 3
    cam_view = render_openpose(img_cv2, kpts_2d)
    kpts_proj_2d_imgs.append(cam_view / 255.0)
    kpts_proj_2d_img = np.concatenate(kpts_proj_2d_imgs, axis=1)

    # Convert the kpts_proj_2d_img to normal 2D iamge without alpha channel
    kpts_proj_2d_img = (kpts_proj_2d_img[:, :, :3] * 255.0).astype(np.uint8)

    return kpts_proj_2d_img


def project_hamer_3D_to_2D(
    kpts_3d: torch.Tensor,
    IMG_W: int,
    IMG_H: int,
    camera_translation: Optional[torch.Tensor] = None,
    scaled_focal_length: Optional[torch.Tensor] = None,
) -> np.array:
    # Here the image_size is wrong!
    batch_size = 1

    rotation = torch.eye(3).unsqueeze(0)
    assert camera_translation is not None
    # camera_translation = camera_translation.cpu()
    camera_translation = camera_translation.clone().cuda()
    kpts_3d = kpts_3d.clone().cuda()
    rotation = rotation.cuda()

    # focal_length_x, focal_length_y = focal_length
    scaled_focal_length = torch.tensor(
        [scaled_focal_length, scaled_focal_length]
    ).reshape(1, 2)

    camera_center = torch.tensor([IMG_W, IMG_H], dtype=torch.float).reshape(1, 2) / 2.0
    kpts_2d = perspective_projection(
        kpts_3d.reshape(batch_size, -1, 3),
        rotation=rotation.repeat(batch_size, 1, 1),
        translation=camera_translation.reshape(batch_size, -1),
        focal_length=scaled_focal_length.repeat(batch_size, 1),
        camera_center=camera_center.repeat(batch_size, 1),
    ).reshape(batch_size, -1, 2)
    kpts_2d = kpts_2d[0]

    return kpts_2d
