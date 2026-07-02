"""Smoke tests: shapes, masking invariants, one full training step.

Run with:  python tests/test_smoke.py   (or pytest tests/)
CPU-only and fast (~30s) — run these before any long Colab session.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from vjepa_mini.config import VJEPAConfig
from vjepa_mini.data.masking import MultiBlockMaskGenerator, VJEPAMasks
from vjepa_mini.models.vjepa import VJEPA


def make_cfg() -> VJEPAConfig:
    return VJEPAConfig(batch_size=2, use_amp=False)


def test_masking() -> None:
    cfg = make_cfg()
    gen = MultiBlockMaskGenerator(
        cfg.grid_t, cfg.grid_h, cfg.grid_w, num_blocks=8, spatial_scale=0.15
    )
    for _ in range(20):
        ctx, tgt = gen()
        n = cfg.num_tokens
        assert len(ctx) + len(tgt) == n, "context + target must cover the grid"
        assert len(set(ctx.tolist()) & set(tgt.tolist())) == 0, "must be disjoint"
        assert len(ctx) >= 1 and len(tgt) >= 1
        # temporal consistency: token i masked => same (h, w) masked at all t
        per_frame = cfg.grid_h * cfg.grid_w
        spatial = set(t.item() % per_frame for t in tgt)
        assert len(tgt) == len(spatial) * cfg.grid_t, "mask must span all frames"
    print("ok: masking invariants")


def test_forward_shapes() -> None:
    cfg = make_cfg()
    model = VJEPA(cfg)
    video = torch.rand(2, cfg.in_channels, cfg.num_frames, cfg.img_size, cfg.img_size)
    ctx, tgt = VJEPAMasks(cfg)()
    out = model(video, ctx, tgt)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    full = model.encoder(video)
    assert full.shape == (2, cfg.num_tokens, cfg.enc_dim)
    print("ok: forward shapes, loss =", round(out["loss"].item(), 4))


def test_training_step_and_ema() -> None:
    cfg = make_cfg()
    model = VJEPA(cfg)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    video = torch.rand(2, cfg.in_channels, cfg.num_frames, cfg.img_size, cfg.img_size)
    ctx, tgt = VJEPAMasks(cfg)()

    before_online = model.encoder.blocks[0].mlp.fc1.weight.clone()
    before_ema = model.target_encoder.blocks[0].mlp.fc1.weight.clone()

    loss = model(video, ctx, tgt)["loss"]
    loss.backward()
    # Target branch must receive NO gradient (stop-grad + requires_grad=False).
    assert all(p.grad is None for p in model.target_encoder.parameters())
    opt.step()
    model.update_target_encoder(momentum=0.99)

    assert not torch.allclose(before_online, model.encoder.blocks[0].mlp.fc1.weight)
    after_ema = model.target_encoder.blocks[0].mlp.fc1.weight
    assert not torch.allclose(before_ema, after_ema), "EMA must move"
    delta_ema = (after_ema - before_ema).abs().mean()
    delta_online = (model.encoder.blocks[0].mlp.fc1.weight - before_online).abs().mean()
    assert delta_ema < delta_online, "EMA must move slower than online weights"
    print("ok: training step + EMA update")


def test_param_counts() -> None:
    cfg = make_cfg()
    model = VJEPA(cfg)
    enc = sum(p.numel() for p in model.encoder.parameters())
    pred = sum(p.numel() for p in model.predictor.parameters())
    print(f"ok: encoder {enc/1e6:.2f}M params, predictor {pred/1e6:.2f}M params")
    assert enc < 10e6, "mini encoder should stay small"


if __name__ == "__main__":
    test_masking()
    test_forward_shapes()
    test_training_step_and_ema()
    test_param_counts()
    print("\nall smoke tests passed")
