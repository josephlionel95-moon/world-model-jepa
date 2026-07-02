"""On-the-fly Moving MNIST video generator.

Why generate instead of download?
  1. Infinite data — the model never sees the same clip twice, which mirrors
     V-JEPA's 2M-video setting better than a fixed 10k-clip file.
  2. We keep the *generative factors* (digit class, position, velocity) as
     labels. That lets us probe the learned representation for both
     appearance (which digit?) and motion (which direction? how fast?) —
     a miniature version of the paper's Kinetics-vs-SSv2 evaluation.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class MovingMNIST(Dataset):
    """Videos of MNIST digits bouncing inside the frame.

    Each item returns:
        clip:  float tensor [C=1, T, H, W] in [0, 1]
        labels: dict with
            "digit":      int64 tensor [num_digits] — digit classes
            "velocity":   float tensor [num_digits, 2] — (vy, vx) px/frame
            "speed":      float tensor [num_digits] — |v|
            "direction":  int64 tensor [num_digits] — one of 8 compass bins
    """

    def __init__(
        self,
        root: str = "./mnist_data",
        train: bool = True,
        num_frames: int = 8,
        img_size: int = 64,
        num_digits: int = 1,
        max_speed: float = 4.0,
        min_speed: float = 1.0,
        length: int = 60_000,
        seed: Optional[int] = None,
        download: bool = True,
    ) -> None:
        super().__init__()
        self.mnist = datasets.MNIST(root=root, train=train, download=download)
        self.num_frames = num_frames
        self.img_size = img_size
        self.num_digits = num_digits
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.length = length
        # A fixed seed makes the dataset deterministic (needed for probe
        # train/val splits); seed=None gives fresh clips every epoch.
        self.seed = seed
        self.digit_size = 28

    def __len__(self) -> int:
        return self.length

    def _rng(self, index: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed * 1_000_003 + index)

    def __getitem__(self, index: int):
        rng = self._rng(index)
        canvas = np.zeros(
            (self.num_frames, self.img_size, self.img_size), dtype=np.float32
        )
        limit = self.img_size - self.digit_size

        digits, velocities = [], []
        for _ in range(self.num_digits):
            idx = int(rng.integers(0, len(self.mnist)))
            img, label = self.mnist[idx]
            img = np.asarray(img, dtype=np.float32) / 255.0

            # Random start position and velocity.
            y, x = rng.uniform(0, limit, size=2)
            speed = rng.uniform(self.min_speed, self.max_speed)
            angle = rng.uniform(0, 2 * np.pi)
            vy, vx = speed * np.sin(angle), speed * np.cos(angle)

            for t in range(self.num_frames):
                iy, ix = int(round(y)), int(round(x))
                canvas[t, iy : iy + self.digit_size, ix : ix + self.digit_size] = (
                    np.maximum(
                        canvas[t, iy : iy + self.digit_size, ix : ix + self.digit_size],
                        img,
                    )
                )
                # Advance and bounce off walls (billiard dynamics).
                y, x = y + vy, x + vx
                if y < 0 or y > limit:
                    vy = -vy
                    y = float(np.clip(y, 0, limit))
                if x < 0 or x > limit:
                    vx = -vx
                    x = float(np.clip(x, 0, limit))

            digits.append(label)
            velocities.append((vy, vx))

        clip = torch.from_numpy(canvas).unsqueeze(0)  # [1, T, H, W]
        velocity = torch.tensor(velocities, dtype=torch.float32)
        speed = velocity.norm(dim=-1)
        # Bin the *final* direction into 8 compass sectors for a
        # classification probe (N, NE, E, SE, S, SW, W, NW).
        angle = torch.atan2(velocity[:, 0], velocity[:, 1])  # (-pi, pi]
        direction = ((angle + np.pi) / (2 * np.pi / 8)).long() % 8

        labels = {
            "digit": torch.tensor(digits, dtype=torch.long),
            "velocity": velocity,
            "speed": speed,
            "direction": direction,
        }
        return clip, labels


def collate_clips(batch) -> Tuple[torch.Tensor, dict]:
    """Stack clips; stack each label field (all clips have num_digits digits)."""
    clips = torch.stack([b[0] for b in batch])
    labels = {
        k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]
    }
    return clips, labels
