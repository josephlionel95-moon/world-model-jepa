"""Attentive probing — V-JEPA's frozen evaluation protocol (App. D.1).

The encoder outputs one embedding per token; a classifier needs one vector
per clip. Average pooling treats all tokens equally, but a bouncing digit
occupies few tokens — its signal would drown in background. The attentive
probe learns WHERE to look: a single cross-attention block with one
learnable query pools the tokens, then a linear layer classifies.

Crucially the encoder stays FROZEN. We measure what the representation
already contains, not what fine-tuning could squeeze into it.
"""

from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class AttentiveProbe(nn.Module):
    def __init__(
        self, dim: int, num_classes: int, num_heads: int = 6
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv = nn.Linear(dim, dim * 2)
        self.q_proj = nn.Linear(dim, dim)
        # Small MLP after pooling (paper: 2-layer MLP with GeLU + LayerNorm).
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, D] frozen encoder outputs -> logits [B, classes]."""
        B, N, D = tokens.shape
        q = self.q_proj(self.query).expand(B, -1, -1)  # [B, 1, D]
        k, v = self.kv(tokens).chunk(2, dim=-1)        # each [B, N, D]

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        pooled = F.scaled_dot_product_attention(split(q), split(k), split(v))
        pooled = pooled.transpose(1, 2).reshape(B, 1, D)
        pooled = pooled + q                  # residual onto the query
        pooled = pooled + self.mlp(pooled)   # residual MLP
        return self.head(self.norm(pooled)).squeeze(1)


@torch.no_grad()
def extract_features(
    encoder: nn.Module, loader: DataLoader, device: torch.device,
    label_fn: Callable[[dict], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen encoder over a loader once; cache tokens + labels.

    Caching features makes probe training ~100x faster than re-encoding
    every epoch — the standard trick for frozen evaluation.
    """
    encoder.eval()
    feats, labels = [], []
    for video, meta in loader:
        tokens = encoder(video.to(device))
        feats.append(tokens.cpu())
        labels.append(label_fn(meta))
    return torch.cat(feats), torch.cat(labels)


def train_probe(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    label_fn: Callable[[dict], torch.Tensor],
    num_classes: int,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = True,
) -> Tuple[AttentiveProbe, Dict[str, List[float]]]:
    """Train an attentive probe on frozen features; return probe + metrics."""
    x_tr, y_tr = extract_features(encoder, train_loader, device, label_fn)
    x_va, y_va = extract_features(encoder, val_loader, device, label_fn)

    dim = x_tr.shape[-1]
    probe = AttentiveProbe(dim, num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history: Dict[str, List[float]] = {"train_acc": [], "val_acc": []}

    n = x_tr.shape[0]
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n)
        correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_tr[idx].to(device), y_tr[idx].to(device)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += (logits.argmax(-1) == yb).sum().item()
        sched.step()
        train_acc = correct / n

        probe.eval()
        with torch.no_grad():
            correct = 0
            for i in range(0, x_va.shape[0], batch_size):
                xb = x_va[i : i + batch_size].to(device)
                yb = y_va[i : i + batch_size].to(device)
                correct += (probe(xb).argmax(-1) == yb).sum().item()
            val_acc = correct / x_va.shape[0]

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if verbose:
            print(
                f"probe epoch {epoch + 1:>2d}/{epochs} | "
                f"train {train_acc:.3f} | val {val_acc:.3f}"
            )
    return probe, history
