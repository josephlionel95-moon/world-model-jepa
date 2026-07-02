"""The full V-JEPA system: x-encoder + EMA y-encoder + predictor + L1 loss.

Objective (paper Eq. 1):

    minimize || P_phi( E_theta(x), positions(y) ) - sg( E_ema(y) ) ||_1

Three collapse-prevention ingredients work together:
  1. stop-gradient  — no gradient flows into the target branch;
  2. EMA target     — targets move slower than the online encoder, so the
                      predictor stays near-optimal for the current targets;
  3. the predictor  — an asymmetric module between the two branches.

Remove any one of them and the constant-output solution ("collapse")
becomes reachable. Notebook 04 lets you verify this empirically.
"""

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_mini.config import VJEPAConfig
from vjepa_mini.models.predictor import Predictor
from vjepa_mini.models.vit import VideoViT


class VJEPA(nn.Module):
    def __init__(self, cfg: VJEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = VideoViT(
            img_size=cfg.img_size,
            num_frames=cfg.num_frames,
            in_channels=cfg.in_channels,
            tubelet_t=cfg.tubelet_t,
            patch_size=cfg.patch_size,
            dim=cfg.enc_dim,
            depth=cfg.enc_depth,
            num_heads=cfg.enc_heads,
            mlp_ratio=cfg.mlp_ratio,
        )
        self.predictor = Predictor(
            enc_dim=cfg.enc_dim,
            pred_dim=cfg.pred_dim,
            depth=cfg.pred_depth,
            num_heads=cfg.pred_heads,
            grid_t=cfg.grid_t,
            grid_h=cfg.grid_h,
            grid_w=cfg.grid_w,
            mlp_ratio=cfg.mlp_ratio,
        )

        # Target encoder: an exact copy, updated only by EMA — never by SGD.
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """theta_ema <- m * theta_ema + (1 - m) * theta  (Polyak averaging)."""
        for p_ema, p in zip(
            self.target_encoder.parameters(), self.encoder.parameters()
        ):
            p_ema.mul_(momentum).add_(p, alpha=1.0 - momentum)

    def forward(
        self,
        video: torch.Tensor,       # [B, C, T, H, W]
        context_idx: torch.Tensor, # [L]
        target_idx: torch.Tensor,  # [M]
    ) -> Dict[str, torch.Tensor]:
        # ---- target branch (no gradients) --------------------------------
        with torch.no_grad():
            # The y-encoder sees the FULL video; targets are its outputs at
            # the masked positions. Full-video encoding = "contextualized
            # targets": each target token summarizes its whole neighbourhood,
            # so predicting it requires understanding, not texture copying.
            full = self.target_encoder(video)          # [B, N, D]
            targets = full[:, target_idx, :]           # [B, M, D]
            if self.cfg.norm_targets:
                # Parameter-free LayerNorm keeps target scale stable so the
                # L1 loss is comparable across training (as in I-JEPA).
                targets = F.layer_norm(targets, (targets.shape[-1],))

        # ---- online branch ------------------------------------------------
        ctx_feats = self.encoder(video, keep_idx=context_idx)   # [B, L, D]
        preds = self.predictor(ctx_feats, context_idx, target_idx)  # [B, M, D]

        loss = F.l1_loss(preds, targets)

        # Diagnostics: the standard deviation of target features across the
        # batch. If this heads to ~0, the representation is collapsing.
        with torch.no_grad():
            target_std = targets.float().std(dim=0).mean()
            pred_std = preds.float().std(dim=0).mean()

        return {"loss": loss, "target_std": target_std, "pred_std": pred_std}
