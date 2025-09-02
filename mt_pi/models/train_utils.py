import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


def process_namedtuple_batch(named_tuple: Any, device: str) -> Any:
    """
    Processes a namedtuple batch, moving any tensor fields to the specified device and converting them to float.

    Args:
        named_tuple (Any): The namedtuple instance containing the batch data.
        device (str): The device to which tensor fields should be moved. Typically 'cuda' or 'cpu'.

    Returns:
        Any: A new namedtuple instance with tensor fields moved to the specified device and converted to float.
    """
    cuda_namedtuple = named_tuple.__class__
    cuda_fields = {
        field: (
            getattr(named_tuple, field).to(device).float()
            if isinstance(getattr(named_tuple, field), torch.Tensor)
            else getattr(named_tuple, field)
        )
        for field in named_tuple._fields
    }
    return cuda_namedtuple(**cuda_fields)


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility.

    Parameters:
    - seed (int): The seed to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def flatten_dict(d, parent_key="", sep="_", no_parent_key=False):
    """
    Flattens a nested dictionary, combining keys with an underscore. Optionally, it can exclude the parent key
    in the combined keys based on the `no_parent_key` flag.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): The base key to use for the current level of depth. Defaults to ''.
        sep (str, optional): The separator to use between keys. Defaults to '_'.
        no_parent_key (bool, optional): If True, do not include the parent key in the flattened keys. Defaults to False.

    Returns:
        dict: A new dictionary with flattened keys.
    """
    items = []
    for k, v in d.items():
        new_key = k if no_parent_key or parent_key == "" else f"{parent_key}{sep}{k}"
        if isinstance(v, dict):
            items.extend(
                flatten_dict(v, new_key, sep=sep, no_parent_key=no_parent_key).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)


def save_sweep_to_csv(final_val_loss: int, log_dir: str, cfg: Any) -> None:
    """
    Saves sweep parameters to a CSV file. If the file doesn't exist, it creates it and adds a header.

    Parameters:
    - final_val_loss (int): The final validation loss.
    - log_dir (str): The directory where the log file is saved.
    - args (Any): An object (usually from argparse) containing the sweep parameters.
    """
    csv_file_path = Path("sweep_logs", f"{cfg.experiment_name}.csv")
    csv_file_path.parent.mkdir(exist_ok=True, parents=True)
    file_exists = csv_file_path.is_file()

    # Construct header and row data
    cfg_dict = vars(cfg)
    string_args = {k: v for k, v in cfg_dict.items() if isinstance(v, str)}
    bool_args = {k: v for k, v in cfg_dict.items() if isinstance(v, bool)}
    int_args = {k: v for k, v in cfg_dict.items() if isinstance(v, int)}
    list_args = {k: v for k, v in cfg_dict.items() if isinstance(v, list)}
    sorted_args = {**string_args, **bool_args, **int_args, **list_args}
    header = ["log_dir", "final_val_loss"] + list(sorted_args.keys())
    row = [log_dir, final_val_loss] + [str(v) for v in sorted_args.values()]

    # Write to CSV
    with open(csv_file_path, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(header)  # Write header only if the file was just created
        csvwriter.writerow(row)
    print(f"Saved sweep parameters to {csv_file_path}")


def plot_losses(train_losses: List[float], val_losses: List[float], save_dir: str):
    """Plot the training and validation losses."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # create a figure with two subplots
    axs[0].plot(np.arange(len(train_losses)), train_losses, label="train_loss")
    axs[0].set_title("Train Loss")
    axs[0].text(
        len(train_losses) - 1,
        train_losses[-1],
        f"Final Value: {train_losses[-1]:.4f}",
        ha="right",
    )
    axs[0].legend()

    # Plot validation loss on the second subplot
    if len(val_losses) == 0:
        val_losses = [0]
    axs[1].plot(np.arange(len(val_losses)), val_losses, label="val_loss")
    axs[1].set_title("Validation Loss")
    axs[1].text(
        len(val_losses) - 1,
        val_losses[-1],
        f"Final Value: {val_losses[-1]:.4f}",
        ha="right",
    )
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/losses.png")
    plt.close()


def absolute_to_delta_action(
    actions: torch.Tensor, add_grasp_info_to_tracks: bool
) -> torch.Tensor:
    """
    Given a list of aboslute actions, return the delta of each action
    with respect to the first action. In other words, the first residual should just be 0.
    This function also separates the grasp and termination information from the actions,
    which are not taken a delta over.

    Args:
        abs_actions (torch.Tensor): The absolute actions to convert.
        first_dim_is_abs (bool): Whether the first action is absolute or not.
        add_grasp_info_to_tracks (bool): Whether additional grasp information is added to the points.

    Returns:
        torch.Tensor: The delta actions + grasp and termination information.
    """
    grasp_and_term_dims = 1 + int(add_grasp_info_to_tracks)
    abs_actions = actions[:, :, :-grasp_and_term_dims]
    grasp_and_term = actions[:, :, -grasp_and_term_dims:]

    # prepend the first action to the start of the list
    # so that the first residual is 0 and dimensions match
    delta_actions = torch.diff(abs_actions, prepend=abs_actions[:, :1], dim=1)
    final_actions = torch.cat([delta_actions, grasp_and_term], dim=-1)
    return final_actions


def delta_to_absolute_action(
    actions: torch.Tensor,
    first_action_abs: torch.Tensor,
    add_grasp_info_to_tracks: bool,
) -> torch.Tensor:
    """
    Given a list of delta actions, return the absolute actions by accumulating the deltas.
    The first action is assumed to be absolute.

    Args:
        delta_actions (torch.Tensor): The delta actions to convert.
        first_action_abs (torch.Tensor): First action in absolute coordinates.
            NOTE: this function assumes first_action_abs does not have grip and term dims.
        add_grasp_info_to_tracks (bool): Whether additional grasp information is added to the points.

    Returns:
        torch.Tensor: The absolute actions + grasp and termination information.
    """
    grasp_and_term_dims = 1 + int(add_grasp_info_to_tracks)
    delta_actions = actions[:, :, :-grasp_and_term_dims]
    grasp_and_term = actions[:, :, -grasp_and_term_dims:]
    if first_action_abs.shape[-1] == delta_actions.shape[-1] + grasp_and_term_dims:
        first_grasp_and_term = first_action_abs[:, :, -grasp_and_term_dims:]
        first_action_abs = first_action_abs[:, :, :-grasp_and_term_dims]
    abs_actions = first_action_abs + torch.cumsum(delta_actions, dim=1)
    final_actions = torch.cat([abs_actions, grasp_and_term], dim=-1)
    return final_actions


def create_optim(
    policy: str, nets: Dict[str, torch.nn.Module], num_training_steps: int
) -> Tuple[Optimizer, _LRScheduler]:
    """
    Creates an optimizer and learning rate scheduler based on the specified policy and network architecture.

    Args:
        policy (str): The policy to use for creating the optimizer. Currently supports 'transformer'.
        nets (Dict[str, torch.nn.Module]): A dictionary containing the network modules. Expected keys are
                                           'noise_pred_net' and 'vision_encoder' for the 'transformer' policy.
        num_training_steps (int): The total number of training steps for which the scheduler will be used.

    Returns:
        Tuple[Optimizer, _LRScheduler]: A tuple containing the created optimizer and learning rate scheduler.
    """
    if policy == "transformer":
        transformer_weight_decay = 1e-3
        obs_encoder_weight_decay = 1e-6
        lr = 1e-4
        betas = [0.9, 0.95]
        num_warmup_steps = 500
        optim_groups = nets["noise_pred_net"].get_optim_groups(
            weight_decay=transformer_weight_decay
        )
        optim_groups.append(
            {
                "params": nets["vision_encoder"].parameters(),
                "weight_decay": obs_encoder_weight_decay,
            }
        )
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
    else:
        num_warmup_steps = 500
        optimizer = torch.optim.AdamW(
            params=nets.parameters(), lr=1e-4, weight_decay=1e-5
        )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, lr_scheduler
