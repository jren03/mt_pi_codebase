import heapq
from pathlib import Path
from typing import Any, Optional

import torch
from termcolor import cprint


class Checkpointer:
    def __init__(self, save_dir: Path, num_best: int = 1, num_recent: int = 2):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoints = []
        self.recent_checkpoints = []
        self.latest_checkpoint = None
        self.num_best = num_best
        self.num_recent = num_recent

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
    ):
        checkpoint = {
            "model_state_dict": model.state_dict()
            if not hasattr(model, "_orig_mod")
            else model._orig_mod.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }

        checkpoint_path = (
            self.save_dir / f"checkpoint_epoch_{epoch}_val_loss_{val_loss:.5f}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # Manage best checkpoints (keep only 2)
        if len(self.best_checkpoints) < self.num_best:
            heapq.heappush(self.best_checkpoints, (-val_loss, checkpoint_path))
        else:
            heapq.heappushpop(self.best_checkpoints, (-val_loss, checkpoint_path))

        # Manage recent checkpoints (keep only 5)
        self.recent_checkpoints.append((epoch, checkpoint_path))
        if len(self.recent_checkpoints) > self.num_recent:
            _, old_path = self.recent_checkpoints.pop(0)
            if old_path not in {path for _, path in self.best_checkpoints}:
                old_path.unlink()

        # Update latest checkpoint
        self.latest_checkpoint = checkpoint_path

        # Remove checkpoints that are neither in the best 2 nor in the recent 5
        keep_checkpoints = {path for _, path in self.best_checkpoints}.union(
            {path for _, path in self.recent_checkpoints}
        )
        for checkpoint_path in self.save_dir.glob("checkpoint_*.pth"):
            if checkpoint_path not in keep_checkpoints:
                checkpoint_path.unlink()

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        best_k: int = 0,
    ):
        if not self.best_checkpoints:
            return None
        best_k = min(best_k, len(self.best_checkpoints) - 1)
        best_checkpoint_path = sorted(
            self.best_checkpoints, key=lambda x: x[0], reverse=True
        )[best_k][-1]
        return self._load_checkpoint(best_checkpoint_path, model, optimizer)

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        latest_k: int = 0,
    ):
        if not self.latest_checkpoint:
            return None
        latest_k = min(latest_k, len(self.recent_checkpoints) - 1)
        latest_checkpoint_path = sorted(
            self.recent_checkpoints, key=lambda x: x[1], reverse=True
        )[latest_k][-1]
        return self._load_checkpoint(latest_checkpoint_path, model, optimizer)

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        checkpoint = torch.load(checkpoint_path)
        if hasattr(model, "_orig_mod"):
            model._orig_mod.load_state_dict(checkpoint["model_state_dict"])
        else:
            # cprint("[WARNING] Loading with strict=False", "red", attrs=["bold"])
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def update_checkpoint_from_dir(self, experiment_dir: Path) -> None:
        if not experiment_dir.exists():
            print(f"No checkpoints directory found in {experiment_dir}")
            return

        checkpoint_files = []
        for ckpt_path in experiment_dir.glob("checkpoint_*.pth"):
            if "last" not in ckpt_path.name:
                epoch = int(ckpt_path.name.split("_")[2])
                val_loss = float(ckpt_path.name.split("_")[-1].replace(".pth", ""))
                checkpoint_files.append((-val_loss, epoch, ckpt_path))

        # Update the best_checkpoints list
        self.best_checkpoints = heapq.nsmallest(
            self.num_best, checkpoint_files, key=lambda x: x[0]
        )

        # Update the recent_checkpoints list
        self.recent_checkpoints = sorted(
            checkpoint_files, key=lambda x: x[1], reverse=True
        )[: self.num_recent]

        if self.recent_checkpoints:
            self.latest_checkpoint = self.recent_checkpoints[0][2]

        self._print_updated_ckpts()

    def _print_updated_ckpts(self) -> None:
        print("Best checkpoints:")
        for i, (val_loss, _, ckpt_path) in enumerate(sorted(self.best_checkpoints)):
            print(f"  {i}: {ckpt_path} with val loss {-val_loss:.5f}")

        print("\nRecent checkpoints:")
        for i, (_, epoch, ckpt_path) in enumerate(self.recent_checkpoints):
            print(f"  {i}: {ckpt_path} from epoch {epoch}")

        print(f"\nLatest checkpoint: {self.latest_checkpoint}")

    @staticmethod
    def load_checkpoint(
        policy: Any,
        checkpointer: Any,
        checkpoint_type: str,
        optim: Optional[Any] = None,
    ) -> Any:
        if checkpoint_type.startswith("best"):
            best_k = int(checkpoint_type.split("_")[1]) if "_" in checkpoint_type else 0
            loaded_checkpoint = checkpointer.load_best(policy, optim, best_k)
        elif checkpoint_type.startswith("latest"):
            latest_k = (
                int(checkpoint_type.split("_")[1]) if "_" in checkpoint_type else 0
            )
            loaded_checkpoint = checkpointer.load_latest(policy, optim, latest_k)
        else:
            raise ValueError(
                "Invalid checkpoint_type. Use 'best', 'best_k', or 'latest' and 'latest_k'."
            )

        if loaded_checkpoint is not None:
            cprint(
                f"Loaded {checkpoint_type} checkpoint from epoch {loaded_checkpoint['epoch']} "
                f"with validation loss {loaded_checkpoint['val_loss']:.5f}",
                "green",
                attrs=["bold"],
            )
        else:
            cprint(
                "No checkpoint found. Using the current model state.",
                "red",
                attrs=["bold"],
            )

        # Recompile the model if necessary
        return torch.compile(policy._orig_mod) if hasattr(
            policy, "_orig_mod"
        ) else policy, loaded_checkpoint
