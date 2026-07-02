# Start Here: Your Path Through This Repository

You are going to build V-JEPA from scratch, understand every design decision in the paper, and finish with the tools to find and attack a research gap. This page is the map.

## What you will be able to do at the end

1. Explain, from first principles, why predicting *features* beats predicting *pixels* for representation learning.
2. Implement every component of V-JEPA — tokenizer, encoder, predictor, masking, EMA target, loss — without reference code.
3. Diagnose representation collapse and explain the three mechanisms that prevent it.
4. Evaluate a self-supervised model the way the paper does (frozen backbone + attentive probe).
5. Read the V-JEPA paper line by line and know why each choice was made.
6. Name the open problems in the JEPA line of work and design an experiment around one of them.

## The learning path

Work in this order. Each step has a doc (theory) and most have a notebook (practice). Read the doc first, then run the notebook and do its exercises before moving on.

| Step | Doc | Notebook | Time |
|---|---|---|---|
| 1 | `01_world_models_and_jepa_vision.md` | — | 1–2 h |
| 2 | `02_vision_transformers.md` | `01_vit_patches_and_attention.ipynb` | 3–4 h |
| 3 | `03_from_mae_to_ijepa.md` | `02_masked_autoencoder_mini.ipynb`, `03_ijepa_mini_images.ipynb` | 4–6 h |
| 4 | `04_vjepa_paper_walkthrough.md` (with the paper open) | — | 3–4 h |
| 5 | `05_collapse_ema_and_training.md` | `04_vjepa_mini_moving_mnist.ipynb` | 4–6 h |
| 6 | `06_evaluation_and_probing.md` | `05_probing_and_analysis.ipynb` | 3–4 h |
| 7 | `07_research_gaps_and_your_path.md` | your own experiments | open-ended |

Total: roughly two to three focused weekends.

## Why this build-up order

V-JEPA is a composition of four ideas, each of which is confusing if you meet it for the first time inside V-JEPA:

```
ViT (how to tokenize and process images/video with attention)
 └─> MAE (learn by masking and reconstructing pixels)
      └─> I-JEPA (same masking idea, but predict FEATURES, not pixels)
           └─> V-JEPA (extend to video: 3D masking, tubelets, motion)
```

Each notebook introduces exactly one new idea on top of the previous one. By the time you train V-JEPA in notebook 04, nothing in it will be new except the video dimension.

## Hardware reality check

The real V-JEPA trained a ViT-H (632M params) on 2 million videos with a batch size of 3072 — a multi-week job on a large A100 cluster. A free Colab T4 has 16 GB and modest compute. So we scale everything down ~100x while keeping the *method* identical:

| | Paper | This repo |
|---|---|---|
| Data | 2M real videos, 16×224×224 RGB | infinite Moving MNIST, 8×64×64 gray |
| Encoder | ViT-L/H (300M–632M) | mini ViT (~2.8M) |
| Predictor | 384-dim, 12 layers | 96-dim, 4 layers |
| Tokens per clip | 8×14×14 = 1568 | 4×8×8 = 256 |
| Batch × steps | 3072 × 90k | 64 × 6k |
| Training time | weeks on a cluster | ~30–60 min on a T4 |

Nothing conceptual is lost: masking strategy, EMA, L1-in-feature-space, attentive probing — all identical. What IS lost is semantic richness (MNIST digits vs the visual world), so absolute numbers here say nothing about ImageNet. That trade is exactly right for learning.

## How to run on Colab

Each notebook starts with a setup cell. Two options:

1. **GitHub (recommended).** Push this repo to your GitHub, then the setup cell does `git clone` + `pip install -e .`
2. **Zip upload.** Zip the repo, upload to Colab (or Google Drive), unzip, `pip install -e .`

Always select Runtime → Change runtime type → T4 GPU first.

## Rules of engagement (how to actually learn this)

- Type code yourself before reading mine. Each notebook poses the problem first.
- When a hyper-parameter looks arbitrary, change it and watch what breaks. The config file (`src/vjepa_mini/config.py`) documents the paper's value next to ours for every single knob.
- Keep a lab notebook (a markdown file) of every run: what you changed, what you predicted would happen, what happened. This habit is 50% of becoming a researcher.
- Do the exercises. Reading is not learning; the exercises are calibrated to expose the gaps reading leaves.

## Repository layout

```
vjepa-from-scratch/
├── docs/            <- theory, math, paper walkthrough (you are here)
├── notebooks/       <- 5 Colab lessons, in order
├── src/vjepa_mini/  <- the library the notebooks import
│   ├── config.py    <- every hyper-parameter, annotated vs the paper
│   ├── data/        <- Moving MNIST generator + 3D multi-block masking
│   ├── models/      <- tokenizer, ViT, predictor, full V-JEPA
│   ├── train/       <- schedules, EMA, training loop
│   └── eval/        <- attentive probe, visualization
└── tests/           <- smoke tests (run before long training sessions)
```
