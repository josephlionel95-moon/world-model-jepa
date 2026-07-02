"""3D multi-block masking — the heart of V-JEPA's prediction task.

Paper (Sec 3.2): sample several spatially-contiguous blocks, repeat each
block across ALL frames, and take their union as the target region y.
The context x is the complement. Two mask "flavours":

    short-range: union of 8 blocks, each covering 15% of a frame
    long-range:  union of 2 blocks, each covering 70% of a frame

Both average ~90% of tokens masked. Blocks span the full temporal axis
because video is temporally redundant: if frame t were visible at some
location, predicting frame t+1 there would be trivial copying, and the
encoder would learn nothing about objects or motion.

Batching subtlety: samples with different numbers of visible tokens can't
be stacked into one tensor. Like the official implementation, we sample
ONE mask per batch and share it across all clips in that batch.
"""

from typing import Tuple

import math
import torch


class MultiBlockMaskGenerator:
    """Generates (context_indices, target_indices) over the token grid.

    The token grid has shape [grid_t, grid_h, grid_w]; indices are into the
    flattened grid of N = grid_t * grid_h * grid_w tokens.
    """

    def __init__(
        self,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        num_blocks: int = 8,
        spatial_scale: float = 0.15,
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        generator: torch.Generator | None = None,
    ) -> None:
        self.grid_t, self.grid_h, self.grid_w = grid_t, grid_h, grid_w
        self.num_blocks = num_blocks
        self.spatial_scale = spatial_scale
        self.aspect_ratio = aspect_ratio
        self.g = generator

    def _rand(self, lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(lo, hi, generator=self.g))

    def _sample_block(self) -> torch.Tensor:
        """One spatial block as a [grid_h, grid_w] boolean mask."""
        # Target block area = scale * full frame; shape jittered by aspect ratio.
        area = self.spatial_scale * self.grid_h * self.grid_w
        ar = self._rand(*self.aspect_ratio)
        h = max(1, min(self.grid_h, round(math.sqrt(area * ar))))
        w = max(1, min(self.grid_w, round(math.sqrt(area / ar))))
        top = int(self._rand(0, self.grid_h - h + 1))
        left = int(self._rand(0, self.grid_w - w + 1))
        mask = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        mask[top : top + h, left : left + w] = True
        return mask

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (context_idx [L], target_idx [M]) — disjoint, covering N."""
        spatial = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for _ in range(self.num_blocks):
            spatial |= self._sample_block()  # union of (possibly overlapping) blocks

        # Never mask everything: the context needs at least one token.
        if spatial.all():
            spatial[0, 0] = False
        if not spatial.any():
            spatial[0, 0] = True

        # Repeat across time: same spatial mask for every temporal slice.
        mask3d = spatial.unsqueeze(0).expand(self.grid_t, -1, -1)  # [T', H', W']
        flat = mask3d.reshape(-1)  # True = target (to predict), False = context
        target_idx = flat.nonzero(as_tuple=True)[0]
        context_idx = (~flat).nonzero(as_tuple=True)[0]
        return context_idx, target_idx


class VJEPAMasks:
    """Convenience wrapper: the paper's short-range + long-range pair.

    Each call randomly picks one flavour (the paper computes losses for both
    per clip; alternating between them is a simpler, nearly equivalent
    mini-scale choice — and a good thing to ablate yourself).
    """

    def __init__(self, cfg) -> None:
        self.short = MultiBlockMaskGenerator(
            cfg.grid_t, cfg.grid_h, cfg.grid_w,
            num_blocks=cfg.short_range_num_blocks,
            spatial_scale=cfg.short_range_scale,
            aspect_ratio=cfg.aspect_ratio,
        )
        self.long = MultiBlockMaskGenerator(
            cfg.grid_t, cfg.grid_h, cfg.grid_w,
            num_blocks=cfg.long_range_num_blocks,
            spatial_scale=cfg.long_range_scale,
            aspect_ratio=cfg.aspect_ratio,
        )

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        gen = self.short if torch.rand(1).item() < 0.5 else self.long
        return gen()
