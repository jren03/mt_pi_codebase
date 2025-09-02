import torch
import torch.nn as nn
from typing import List, Tuple
from models.diffusion_policy import DiffusionPolicy


class MultiViewWrapper(nn.Module):
    def __init__(self, policies: List[Tuple[str, DiffusionPolicy]]):
        super().__init__()
        self.policies = {vp: policy for vp, policy in policies}

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        state_cond: torch.Tensor,
        vp: str,
        *,
        first_action_abs: torch.Tensor = None,
        return_full_action: bool = False,
        cpu: bool = True,
    ) -> torch.Tensor:
        """
        Generate action from noise for the specified viewpoint.

        Args:
            obs (torch.Tensor): The observation tensor.
            state_cond (torch.Tensor): The state condition tensor.
            vp (str): The viewpoint to use for action generation.
            first_action_abs (torch.Tensor, optional): The first absolute action.
            return_full_action (bool, optional): Whether to return the full action sequence.
            cpu (bool, optional): Whether to return the result on CPU.

        Returns:
            torch.Tensor: The generated action.
        """
        if vp not in self.policies:
            raise ValueError(f"Viewpoint '{vp}' not found in available policies.")

        policy = self.policies[vp]
        return policy.act(
            obs,
            state_cond,
            first_action_abs=first_action_abs,
            return_full_action=return_full_action,
            cpu=cpu,
        )

    def __getattr__(self, name):
        """
        Delegate any other method calls to the first policy in the list.
        This allows access to other methods and attributes of the wrapped DiffusionPolicy.
        """
        return getattr(next(iter(self.policies.values())), name)
