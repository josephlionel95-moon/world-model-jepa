"""Single source of truth for all hyper-parameters.

Every value here has a counterpart in the V-JEPA paper (Appendix B/C).
Comments give the paper's value so you can see exactly how we scaled down.
"""

from dataclasses import dataclass, field


@dataclass
class VJEPAConfig:
    # ------------------------------------------------------------------ data
    # Paper: 16 frames of 224x224 RGB, frameskip 4 (~2s of video).
    # Mini:  8 frames of 64x64 grayscale Moving MNIST.
    img_size: int = 64
    num_frames: int = 8
    in_channels: int = 1

    # ----------------------------------------------------------- tokenization
    # Paper: 3D conv "tubelets" of 2x16x16 -> 8x14x14 = 1568 tokens.
    # Mini:  tubelets of 2x8x8            -> 4x8x8   = 256 tokens.
    tubelet_t: int = 2
    patch_size: int = 8

    # ---------------------------------------------------------------- encoder
    # Paper: ViT-L/16 (1024 dim, 24 layers, 16 heads) up to ViT-H/16.
    # Mini:  ~2.8M param ViT. Big enough to learn, small enough for a T4.
    enc_dim: int = 192
    enc_depth: int = 6
    enc_heads: int = 6
    mlp_ratio: float = 4.0

    # -------------------------------------------------------------- predictor
    # Paper: narrow ViT, 384 dim, 12 layers (encoder is 1024+ dim).
    # Mini:  narrow ViT, 96 dim, 4 layers. The predictor is deliberately
    #        *weaker* than the encoder: it should succeed only if the
    #        encoder's features carry the information.
    pred_dim: int = 96
    pred_depth: int = 4
    pred_heads: int = 6

    # ----------------------------------------------------------------- masking
    # Paper: short-range = union of 8 blocks @ 15% spatial scale,
    #        long-range  = union of 2 blocks @ 70% spatial scale,
    #        aspect ratio in (0.75, 1.5), masks span ALL frames.
    # Mini:  identical strategy on the 8x8 token grid.
    short_range_num_blocks: int = 8
    short_range_scale: float = 0.15
    long_range_num_blocks: int = 2
    long_range_scale: float = 0.7
    aspect_ratio: tuple = (0.75, 1.5)

    # ---------------------------------------------------------------- training
    # Paper: batch 3072, 90k iters, AdamW, lr 6.25e-4 (ViT-L), wd 0.04->0.4,
    #        EMA momentum 0.998 -> 1.0, schedules stretched by 1.25x.
    batch_size: int = 64
    total_steps: int = 6000
    warmup_steps: int = 600
    lr: float = 1.5e-3
    final_lr: float = 1e-6
    wd_start: float = 0.04
    wd_end: float = 0.4
    ema_start: float = 0.996
    ema_end: float = 1.0
    schedule_scale: float = 1.25  # the paper's "scale schedules 25% beyond" trick
    grad_clip: float = 10.0
    use_amp: bool = True

    # Normalize targets with a parameter-free LayerNorm (as in I-JEPA).
    # Stabilizes the L1 loss scale; try switching it off as an experiment.
    norm_targets: bool = True

    seed: int = 0

    # ------------------------------------------------------------ derived sizes
    @property
    def grid_t(self) -> int:
        return self.num_frames // self.tubelet_t

    @property
    def grid_h(self) -> int:
        return self.img_size // self.patch_size

    @property
    def grid_w(self) -> int:
        return self.img_size // self.patch_size

    @property
    def num_tokens(self) -> int:
        return self.grid_t * self.grid_h * self.grid_w
