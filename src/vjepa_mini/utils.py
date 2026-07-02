"""Small utilities: reproducibility, checkpoints, metering."""

import random
from pathlib import Path
from typing import Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


class AverageMeter:
    """Running average of a scalar (loss, std, ...)."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def save_checkpoint(path: Union[str, Path], **objects) -> None:
    """save_checkpoint('ckpt.pt', model=model.state_dict(), step=100)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(objects, path)


def load_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
