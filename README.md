# vjepa-from-scratch

A complete, small-scale reimplementation of **V-JEPA** (Bardes et al., 2024, *"Revisiting Feature Prediction for Learning Visual Representations from Video"*) built for learning and research on a **free Colab T4 GPU** — plus a graduate-level course in the `docs/` folder that takes you from "what is attention?" to "here is my research proposal."

Every component of the real method is here at 1/100th scale: 3D tubelet tokenization, multi-block video masking, ViT encoder on visible tokens only, narrow predictor with mask tokens, EMA target encoder with stop-gradient, L1 feature-space loss, hyper-parameter schedules (including the paper's 1.25× stretch trick), and frozen evaluation with attentive probes.

## Quick start (Colab)

1. Push this repo to your GitHub (or upload the zip to Colab).
2. Open `notebooks/01_vit_patches_and_attention.ipynb` in Colab.
3. Runtime → Change runtime type → **T4 GPU**.
4. Run the setup cell and follow along.

Local install:

```bash
pip install -e .
python tests/test_smoke.py   # ~30s on CPU; run before any long training
```

## The course

Read `docs/00_START_HERE.md` first. Path: world-models motivation → ViTs → MAE → I-JEPA → the V-JEPA paper line-by-line → collapse & training → evaluation → research gaps. Docs are theory; notebooks are practice; do both, in order.

| Notebook | You build | New idea |
|---|---|---|
| 01 | patches, attention, ViT block | tokenization & attention |
| 02 | mini-MAE on MNIST | masked prediction (pixel targets) |
| 03 | mini I-JEPA on MNIST | feature targets, EMA, collapse |
| 04 | **mini V-JEPA on Moving MNIST** | video, 3D masking, full method |
| 05 | attentive probes & analysis | frozen evaluation, the paper's claims |

Notebook 04 trains in ~30–60 min on a T4. Notebook 05 reproduces the paper's central "appearance AND motion" claim using digit-class and motion-direction probes with ground-truth labels from the data generator.

## Repository map

```
docs/            the course (start at 00_START_HERE.md)
notebooks/       5 Colab lessons, in order
src/vjepa_mini/
  config.py      every hyper-parameter, annotated against the paper's values
  data/          Moving MNIST generator (labels = generative factors) + 3D multi-block masking
  models/        tubelet embed, ViT, predictor, full VJEPA module
  train/         schedules, trainer (AMP, EMA, collapse diagnostics)
  eval/          attentive probe, PCA feature maps, run plots
tests/           smoke tests: shapes, masking invariants, stop-grad, EMA
```

## Fidelity to the paper

| Component | Paper | Here |
|---|---|---|
| Objective | L1 feature prediction, stop-grad, EMA target | identical |
| Masking | multi-block (8×15% + 2×70%), spatial blocks repeated over time | identical strategy, smaller grid |
| Encoder | ViT-L/H, tubelets 2×16×16 | mini ViT 2.8M, tubelets 2×8×8 |
| Predictor | narrow ViT, 384-d × 12 | narrow ViT, 96-d × 4 |
| Targets | contextualized (full-clip EMA encoding) + LayerNorm | identical |
| Schedules | warmup+cosine lr, wd 0.04→0.4, EMA 0.998→1, 1.25× stretch | identical shapes |
| Eval | frozen backbone + attentive probe | identical |
| Data | 2M internet videos | Moving MNIST (infinite, with ground-truth factors) |

## References

- Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA), 2024. arXiv:2404.08471
- Assran et al., *I-JEPA*, 2023. arXiv:2301.08243
- Assran et al., *V-JEPA 2*, 2025. arXiv:2506.09985
- He et al., *Masked Autoencoders*, 2021. arXiv:2111.06377
- Grill et al., *BYOL*, 2020. arXiv:2006.07733
- LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022.

MIT License. Built as a learning vehicle — expect to break things; that is the point.
