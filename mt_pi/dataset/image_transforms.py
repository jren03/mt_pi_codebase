from typing import Any, Callable, Dict, List

import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables import KeypointsOnImage
from termcolor import cprint

from dataset.common import normalize_data

np.bool = np.bool_  # needed for imgaug


def image_preprocess_transform(target_size: int) -> iaa.Sequential:
    """
    Using imgaug change colorspace and resize to handle keypoints.

    Note imgaug uses (H, W, C) format for images, whereas PyTorch uses (C, H, W).

    Since the cameras already output RGB images, we don't need to change colorspace.
    - iaa.color.ChangeColorspace(from_colorspace="BGR", to_colorspace="RGB"),
    """
    return iaa.Sequential(
        [
            iaa.Resize({"height": target_size, "width": target_size}),
        ]
    )


def get_augmentation_transform_by_name(
    name: str,
    obs_stats: Dict[str, np.ndarray],
    normalize_image: bool = True,
) -> iaa.SomeOf:
    """
    Returns an imgaug augmentation transform based on the specified name.

    Parameters:
    - name (str): The name of the augmentation transform.
    - obs_stats (Dict[str, np.ndarray]): The statistics used for normalization.
    - normalize_image (bool): Whether to normalize the image after augmentation.

    Returns:
    - iaa.SomeOf: The imgaug augmentation transform.

    Raises:
    - ValueError: If the specified name is not recognized.
    """
    name = name.lower()
    if name == "none":
        imgaug_list = [iaa.Noop()]
    elif name == "easy":
        imgaug_list = [iaa.CropAndPad(percent=(-0.15, 0.15))]
    elif name == "medium-affine":
        imgaug_list = [
            iaa.CropAndPad(percent=(-0.15, 0.15)),
            iaa.Affine(
                scale={
                    "x": (0.9, 1.1),
                    "y": (0.9, 1.1),
                },
                translate_percent={
                    "x": (-0.08, 0.08),
                    "y": (-0.08, 0.08),
                },
                order=[0, 1],
                cval=(0, 20),
                mode=["constant", "edge"],
            ),
        ]
    elif name == "medium-geometric":
        imgaug_list = [
            iaa.Add((-10, 10), per_channel=True),
            iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255)),
            iaa.Dropout(p=0.15),
            iaa.MotionBlur(k=5),
            iaa.GammaContrast((0.95, 1.05)),
            # iaa.LinearContrast((0.95, 1.05), per_channel=0.15),
            # iaa.GammaContrast((0.95, 1.05)),
            # iaa.MultiplySaturation((0.95, 1.05)),
            # iaa.MultiplyHue((0.9, 1.1)),
            # iaa.CoarseDropout(0.1, size_percent=0.05),
        ]
    elif name == "medium-flip":
        imgaug_list = [
            iaa.Add((-10, 10), per_channel=True),
            iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255)),
            iaa.Dropout(p=0.15),
            iaa.MotionBlur(k=5),
            iaa.GammaContrast((0.95, 1.05)),
            iaa.Fliplr(0.5),
            # iaa.LinearContrast((0.95, 1.05), per_channel=0.15),
            # iaa.GammaContrast((0.95, 1.05)),
            # iaa.MultiplySaturation((0.95, 1.05)),
            # iaa.MultiplyHue((0.9, 1.1)),
            # iaa.CoarseDropout(0.1, size_percent=0.05),
        ]
    elif name == "hard":
        imgaug_list = [
            iaa.CropAndPad(percent=(-0.15, 0.15)),
            iaa.Affine(
                scale={
                    "x": (0.9, 1.1),
                    "y": (0.9, 1.1),
                },
                translate_percent={
                    "x": (-0.08, 0.08),
                    "y": (-0.08, 0.08),
                },
                rotate=(-5, 5),
                shear=(-5, 5),
                order=[0, 1],
                cval=(0, 20),
                mode=["constant", "edge"],
            ),
            iaa.LinearContrast((0.85, 1.15), per_channel=0.25),
            iaa.Add((-40, 40), per_channel=False),
            iaa.GammaContrast((0.95, 1.05)),
            iaa.MultiplySaturation((0.95, 1.05)),
            iaa.MultiplyHue((0.5, 1.5)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.0125 * 255)),
        ]
    else:
        raise ValueError(f"Unknown augmentation transform: {name}")

    def _to_tensor_img_func_normalize(images, random_state, parents, hooks):
        for i in range(len(images)):
            images[i] = normalize_data(images[i], stats=obs_stats)
            images[i] = images[i].transpose(2, 0, 1)
        return images

    def _to_tensor_img_func(images, random_state, parents, hooks):
        for i in range(len(images)):
            images[i] = images[i].transpose(2, 0, 1)
        return images

    def _noop_kpt_func(kpt, random_state, parents, hooks):
        return kpt

    if normalize_image:
        cprint(
            f"Using augmentation: {name} and normalizing image", "cyan", attrs=["bold"]
        )
        return iaa.Sequential(
            [
                iaa.SomeOf((0, None), imgaug_list, random_order=True),
                iaa.Lambda(_to_tensor_img_func_normalize, _noop_kpt_func),
            ]
        )
    else:
        cprint(
            f"Using augmentation: {name} but not normalizing image",
            "cyan",
            attrs=["bold"],
        )
        return iaa.Sequential(
            [
                iaa.SomeOf((0, None), imgaug_list, random_order=True),
                iaa.Lambda(_to_tensor_img_func, _noop_kpt_func),
            ]
        )


def transform_image_and_action_kpts(
    images: np.ndarray,
    actions: np.ndarray,
    tf: Callable[[np.ndarray, np.ndarray], Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses a single image and its corresponding action keypoints using the specified transformation function.
    If passed a batch of images and actions, this function will apply the same augmentation over the entire batch.

    Parameters:
    - images (np.ndarray): The input image or batch of images with shape (height, width, channels) or (batch_size, height, width, channels).
    - actions (np.ndarray): The action keypoints or batch of action keypoints with shape (num_keypoints*2,) or (batch_size, num_keypoints*2).
    - tf: The transformation function to be applied. It must accept `image` and `keypoints` as arguments and return the transformed image and keypoints.

    Returns:
    - tuple[np.ndarray, np.ndarray]: A tuple containing the transformed image and action keypoints. If the input did not have a batch dimension, the output will also not have a batch dimension.
    """
    has_batch_dim = len(images.shape) == 4
    augment_actions = actions is not None
    if not augment_actions:
        # create dummy actions
        actions = np.zeros((images.shape[0], 2)) if has_batch_dim else np.zeros(2)
    if not has_batch_dim:
        images = images[np.newaxis]
        actions = actions[np.newaxis]
    action_bsize = actions.shape[0]
    # (B, num_points*2) -> (B*num_points, 2)
    actions_on_images = KeypointsOnImage.from_xy_array(
        actions.reshape(-1, 2), shape=images.shape[1:]
    )
    # Apply same transform over obs horizon
    tf_det = tf.to_deterministic()
    aug_imgs = []
    for image in images:
        aug_img, aug_kpts = tf_det(image=image, keypoints=actions_on_images)
        aug_imgs.append(aug_img)
        # Take just the keypoints from the first image in the batch
        aug_kpts = aug_kpts.to_xy_array().reshape(action_bsize, -1)
    if not has_batch_dim:
        aug_imgs = aug_imgs[0]
        aug_kpts = aug_kpts[0]
    aug_imgs = np.array(aug_imgs)
    return aug_imgs, aug_kpts


def resize_and_normalize_image(
    image: np.ndarray,
    action: np.ndarray,
    target_size: int,
    image_stats: Dict[str, np.array],
    action_stats: Dict[str, np.array],
    to_tensor: bool = True,
    add_grasp_info_to_tracks: bool = False,
    normalize_image: bool = True,
) -> List[np.ndarray]:
    """
    Takes a raw image as input and applies the same sequence of resizing and
    normalization as train time. If to_tensor is True, the image will be converted
    from (H)

    NOTE: This function only processes a single image at a time,
    and does not work with batched images.

    Parameters:
    - image (np.ndarray): The input image with shape (height, width, channels).
    - target_size (int): The target size for the image.

    Returns:
    - np.ndarray: The resized and normalized image with shape (target_size, target_size, channels).
    - np.ndarray: The normalized action keypoints with shape (num_keypoints*2,).
    """
    if action is None:
        # Dummy actions, remove grasp and terminal dimensions
        action = np.zeros((action_stats["min"].shape[0] - 2,))
    action = action.reshape(-1, 2)
    actions_on_images = KeypointsOnImage.from_xy_array(action, shape=image.shape)
    image, action = image_preprocess_transform(target_size)(
        image=image, keypoints=actions_on_images
    )
    action = action.to_xy_array().reshape(-1)
    # Add terminal flag + possible grasp info
    extra_act_dims = 1 + int(add_grasp_info_to_tracks)
    action = np.concatenate((action, np.zeros(extra_act_dims)), axis=0)
    action = normalize_data(action, stats=action_stats)
    action = action[:-extra_act_dims]
    if normalize_image:
        # Don't need to normalize if using Voltron
        image = normalize_data(image, stats=image_stats)
    if to_tensor:
        # (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1)
    return image, action


def get_resize_to_original_image_fn(
    orig_image: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    This method returns a function that can be called on an image and
    action keypoints to resize them to the original size.

    """

    orig_height, orig_width = orig_image.shape[:2]
    resize_aug = iaa.Resize({"height": orig_height, "width": orig_width})

    def resize_to_original_image(
        image: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns actions in the form of (act_horizon, num_points*2)"""
        has_act_hor = False
        if len(actions.shape) == 3:
            has_act_hor = True
            action_horizon = actions.shape[0]
        actions_on_images = KeypointsOnImage.from_xy_array(
            actions.reshape(-1, 2), shape=image.shape[:2]
        )
        image, actions_on_images = resize_aug(image=image, keypoints=actions_on_images)
        if has_act_hor:
            actions = actions_on_images.to_xy_array().reshape(action_horizon, -1)
        else:
            actions = actions_on_images.to_xy_array().reshape(-1)
        return image, actions

    return resize_to_original_image
