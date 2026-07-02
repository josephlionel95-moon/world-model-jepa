"""Video tokenization: 3D "tubelet" embedding + 3D sin-cos position codes.

Paper (App. B): a 3D convolution with kernel/stride (2, 16, 16) maps a
16x224x224 clip to an 8x14x14 grid of tokens, then absolute 3D sin-cos
positional embeddings are added. We do exactly the same at 1/8 scale.
"""

import math

import torch
import torch.nn as nn


class TubeletEmbed(nn.Module):
    """[B, C, T, H, W] -> [B, N, D] where N = (T/tt) * (H/p) * (W/p).

    A single Conv3d does three jobs at once: cut the video into
    non-overlapping tubelets (tt x p x p), flatten each tubelet's pixels,
    and linearly project to D dims. It is exactly a linear layer applied
    to each tubelet — a conv is just the efficient way to write it.
    """

    def __init__(
        self, in_channels: int, dim: int, tubelet_t: int, patch_size: int
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            dim,
            kernel_size=(tubelet_t, patch_size, patch_size),
            stride=(tubelet_t, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, D, T', H', W']
        return x.flatten(2).transpose(1, 2)  # [B, N, D]  (row-major: t, h, w)


def sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    """Standard transformer sin-cos code for a 1D coordinate. [len] -> [len, dim]."""
    assert dim % 2 == 0
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
    omega = 1.0 / (10000.0 ** omega)  # frequencies from 1 to 1/10000
    out = positions.float().unsqueeze(1) * omega.unsqueeze(0)  # [len, dim/2]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [len, dim]


def sincos_3d(dim: int, grid_t: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """3D positional code: concat separate codes for (t, h, w). -> [N, dim].

    Why not learned embeddings? Sin-cos codes are parameter-free, cannot
    overfit, and make token positions comparable across runs — useful when
    the predictor must reason about *where* a masked token is.
    """
    # Split channels between axes; time gets the remainder.
    dim_h = dim_w = dim // 4 * 1  # quarter each for h and w ...
    dim_h = dim_w = (dim // 4) & ~1  # ... rounded to even
    dim_t = dim - dim_h - dim_w
    if dim_t % 2:  # keep every part even
        dim_t -= 1
        dim_h += 1
        dim_w = dim - dim_t - dim_h

    t = sincos_1d(dim_t, torch.arange(grid_t))  # [T', dim_t]
    h = sincos_1d(dim_h, torch.arange(grid_h))  # [H', dim_h]
    w = sincos_1d(dim_w, torch.arange(grid_w))  # [W', dim_w]

    pe = torch.zeros(grid_t, grid_h, grid_w, dim)
    pe[..., :dim_t] = t[:, None, None, :]
    pe[..., dim_t : dim_t + dim_h] = h[None, :, None, :]
    pe[..., dim_t + dim_h :] = w[None, None, :, :]
    return pe.reshape(-1, dim)  # row-major (t, h, w) — must match TubeletEmbed
