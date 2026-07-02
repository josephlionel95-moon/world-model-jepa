"""The V-JEPA predictor P_phi — a narrow transformer.

Paper (Sec 3.3): the predictor receives (a) the context embeddings from the
x-encoder and (b) one learnable "mask token" per target position, each
carrying the positional embedding of the location it must predict. After
self-attention over the combined sequence, ONLY the mask-token outputs are
kept and projected back to encoder width.

Why narrow (96 vs 192 dim here; 384 vs 1024+ in the paper)? The predictor
is a scratchpad, not the product. If it were as powerful as the encoder it
could solve the task internally and relieve the encoder of learning good
features. Keeping it weak pushes the burden of understanding into E_theta.
"""

import torch
import torch.nn as nn

from vjepa_mini.models.patch_embed import sincos_3d
from vjepa_mini.models.vit import Block


class Predictor(nn.Module):
    def __init__(
        self,
        enc_dim: int = 192,
        pred_dim: int = 96,
        depth: int = 4,
        num_heads: int = 6,
        grid_t: int = 4,
        grid_h: int = 8,
        grid_w: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        # Project encoder outputs into the predictor's (narrower) width.
        self.embed = nn.Linear(enc_dim, pred_dim)
        # One shared learnable vector for "unknown content here".
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        pe = sincos_3d(pred_dim, grid_t, grid_h, grid_w)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList(
            [Block(pred_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        # Project predictions back to encoder width for the L1 loss.
        self.proj = nn.Linear(pred_dim, enc_dim)

    def forward(
        self,
        context_feats: torch.Tensor,  # [B, L, enc_dim] from the x-encoder
        context_idx: torch.Tensor,    # [L] grid positions of context tokens
        target_idx: torch.Tensor,     # [M] grid positions to predict
    ) -> torch.Tensor:
        B, L, _ = context_feats.shape
        M = target_idx.numel()

        # Context tokens: narrow projection + their true positional codes.
        ctx = self.embed(context_feats) + self.pos_embed[:, context_idx, :]

        # Mask tokens: identical content vector, distinguished ONLY by the
        # positional code of the location each must predict.
        tgt = self.mask_token.expand(B, M, -1) + self.pos_embed[:, target_idx, :]

        x = torch.cat([ctx, tgt], dim=1)  # [B, L + M, pred_dim]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        preds = x[:, L:, :]               # keep only mask-token outputs
        return self.proj(preds)           # [B, M, enc_dim]
