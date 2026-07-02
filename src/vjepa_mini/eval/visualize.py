"""Visual diagnostics: look at what the model sees and learns."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def show_clip(
    clip: torch.Tensor, title: str = "", mask: Optional[torch.Tensor] = None,
    grid_hw: Optional[tuple] = None, patch: int = 8, tubelet_t: int = 2,
) -> None:
    """Display a [C, T, H, W] clip as a filmstrip.

    If `mask` (flat token indices of TARGETS) is given, masked regions are
    dimmed — this is literally what the x-encoder does NOT see.
    """
    clip = clip.detach().cpu()
    T = clip.shape[1]
    fig, axes = plt.subplots(1, T, figsize=(1.6 * T, 1.9))
    keep = None
    if mask is not None and grid_hw is not None:
        gt, gh, gw = grid_hw
        vis = torch.ones(gt * gh * gw)
        vis[mask] = 0.25  # dim the target region
        keep = vis.reshape(gt, gh, gw)
    for t in range(T):
        frame = clip[0, t].numpy()
        if keep is not None:
            scale = keep[t // tubelet_t].repeat_interleave(patch, 0)
            scale = scale.repeat_interleave(patch, 1).numpy()
            frame = frame * scale + 0.08 * (scale < 1)  # faint haze on hidden area
        axes[t].imshow(frame, cmap="gray", vmin=0, vmax=1)
        axes[t].set_axis_off()
        axes[t].set_title(f"t={t}", fontsize=8)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def pca_feature_map(
    encoder: torch.nn.Module, clip: torch.Tensor, device: torch.device,
    grid: tuple, k: int = 3,
) -> np.ndarray:
    """Project each token embedding to its top-k PCA components and render
    them as RGB — a quick qualitative check that features track content.

    Returns array [T', H', W', 3] in [0, 1].
    """
    encoder.eval()
    tokens = encoder(clip.unsqueeze(0).to(device))[0].cpu()  # [N, D]
    x = tokens - tokens.mean(0)
    # PCA via SVD: principal directions = right singular vectors.
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    proj = x @ v[:k].T                                        # [N, k]
    proj = (proj - proj.min(0).values) / (
        proj.max(0).values - proj.min(0).values + 1e-8
    )
    gt, gh, gw = grid
    return proj.reshape(gt, gh, gw, k).numpy()


def plot_pca_map(pca: np.ndarray, title: str = "PCA of token features") -> None:
    T = pca.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(2.0 * T, 2.2))
    if T == 1:
        axes = [axes]
    for t in range(T):
        axes[t].imshow(pca[t])
        axes[t].set_axis_off()
        axes[t].set_title(f"t'={t}", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_history(history: List[Dict[str, float]]) -> None:
    """Loss + collapse diagnostics from Trainer.history."""
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(steps, [h["loss"] for h in history])
    axes[0].set_title("L1 prediction loss")
    axes[1].plot(steps, [h["target_std"] for h in history], label="target std")
    axes[1].plot(steps, [h["pred_std"] for h in history], label="pred std")
    axes[1].axhline(0.05, color="r", ls="--", lw=0.8, label="collapse zone")
    axes[1].legend()
    axes[1].set_title("Feature std (collapse watch)")
    axes[2].plot(steps, [h["lr"] for h in history], label="lr")
    ax2 = axes[2].twinx()
    ax2.plot(steps, [h["momentum"] for h in history], color="g", label="EMA m")
    axes[2].set_title("lr (left) / EMA momentum (right)")
    for ax in axes:
        ax.set_xlabel("step")
    plt.tight_layout()
    plt.show()
