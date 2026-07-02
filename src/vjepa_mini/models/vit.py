"""A clean Vision Transformer for video, written from scratch.

No timm, no einops — every operation is visible. The encoder must accept a
*subset* of tokens (the unmasked context), which is why positional
embeddings are added BEFORE tokens are dropped: each surviving token keeps
the code of its true spatio-temporal location.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_mini.models.patch_embed import TubeletEmbed, sincos_3d


class Attention(nn.Module):
    """Multi-head self-attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_head)) V
    computed in parallel for `num_heads` independent subspaces.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must divide evenly across heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, heads, N, head_dim]
        # Fused, memory-efficient attention kernel (FlashAttention on GPU).
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class MLP(nn.Module):
    """Position-wise feed-forward: expand -> GELU -> contract."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x)).

    Pre-norm (LayerNorm inside the residual branch) keeps the residual
    stream an identity path, which is what lets deep ViTs train without
    fragile warmup tricks.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VideoViT(nn.Module):
    """The V-JEPA encoder E_theta.

    forward(video, keep_idx) returns one embedding per *kept* token.
    With keep_idx=None it embeds the full video (used for the target
    encoder and for downstream evaluation).
    """

    def __init__(
        self,
        img_size: int = 64,
        num_frames: int = 8,
        in_channels: int = 1,
        tubelet_t: int = 2,
        patch_size: int = 8,
        dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.grid_t = num_frames // tubelet_t
        self.grid_h = self.grid_w = img_size // patch_size

        self.patch_embed = TubeletEmbed(in_channels, dim, tubelet_t, patch_size)
        pe = sincos_3d(dim, self.grid_t, self.grid_h, self.grid_w)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList(
            [Block(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, video: torch.Tensor, keep_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """video: [B, C, T, H, W]; keep_idx: [L] indices into the token grid."""
        x = self.patch_embed(video)          # [B, N, D]
        x = x + self.pos_embed               # position BEFORE dropping tokens
        if keep_idx is not None:
            x = x[:, keep_idx, :]            # [B, L, D] — masked tokens removed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
