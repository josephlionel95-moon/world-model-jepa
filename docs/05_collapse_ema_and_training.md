# 5. Collapse, EMA, and How to Actually Train This Thing

**Learning objectives.** Understand representation collapse deeply enough to cause it on purpose, explain each anti-collapse ingredient, and know every training-stability lever in the codebase.

Companion notebook: `04_vjepa_mini_moving_mnist.ipynb` (includes the sabotage experiments).

## 5.1 Collapse: the failure mode that shapes everything

The naive joint-embedding objective is

```
min_θ,φ || P_φ(E_θ(x)) − E_θ(y) ||
```

Set E_θ(anything) = constant c. Then P only needs to output c, loss = 0, global optimum reached, and the "representation" contains nothing. Gradient descent finds this attractor happily — it is the *easiest* direction of improvement, because both branches share θ and can conspire.

Partial collapse is sneakier and more common than total collapse: the representation keeps a few informative dimensions and zeroes the rest, or keeps information at low variance. The loss looks fine; probe accuracy is mediocre. This is why we log `target_std` (the across-batch standard deviation of target features) throughout training — it is the vital sign. Healthy runs hold it well above zero; a monotone slide toward zero is collapse in progress.

## 5.2 The three-ingredient vaccine

V-JEPA (following BYOL/I-JEPA) prevents collapse with architecture, not with negatives or explicit variance penalties:

**Ingredient 1 — stop-gradient.** No gradient flows through the target branch. The encoder can no longer *directly* move the targets toward its predictions; it can only move the predictions toward the targets. Collapse requires the two branches to conspire; stop-grad cuts one side of the deal.

**Ingredient 2 — EMA target encoder.** θ̄ ← m·θ̄ + (1−m)·θ with m ≈ 0.996→1.0. The target network is a heavily smoothed, delayed copy: targets move slowly and consistently. Two intuitions for why this matters:

- *Teacher-student:* the student chases a teacher who is an average of the student's own past selves — stable enough to be a fixed point to learn toward, current enough to improve as the student improves.
- *The optimal-predictor argument* (paper Sec 3.1, derived in chapter 4): if targets move slowly relative to the predictor's learning speed, the predictor stays near-optimal; and *for an optimal predictor* the encoder's effective objective becomes "minimize the conditional spread (MAD) of targets given context" — i.e. maximize predictive information — rather than "make everything constant." Collapse is only reachable when the predictor is stale.

**Ingredient 3 — the predictor itself.** The asymmetry between branches (one has P_φ, the other doesn't) means the online branch never needs to *equal* the target — it needs to be *mappable onto* it. BYOL's surprising discovery was that this asymmetry, with EMA, suffices even with no negatives; removing the predictor collapses BYOL immediately.

None of the three is sufficient alone. Notebook 04's sabotage suite has you remove each one and watch `target_std`:

| Sabotage | Expected outcome |
|---|---|
| momentum m = 0 (target = current encoder) | rapid partial/total collapse |
| remove predictor (score context features directly at target positions) | collapse or trivial features |
| loss on visible tokens too | copying, degenerate features |
| m = 1 forever (frozen random target) | no collapse, but ceiling-limited features (you're distilling a random net) |

That last row is instructive: a *frozen random* target already avoids collapse and produces non-trivial features — the EMA schedule's job is to let target quality *improve* over training without ever moving fast enough to enable conspiracy.

## 5.3 Why L1 (recap and practical view)

Chapter 4 derived: optimal-L1-predictor = conditional median; encoder gradient = ∇MAD of targets. Practically: with a drifting target network, target vectors occasionally jump; L2 gradients scale with error magnitude and let those jumps yank the encoder, while L1 gradients are bounded (±1 per dimension). The paper simply reports L1 "more stable." We also LayerNorm targets (parameter-free) so the loss scale is comparable across training — an I-JEPA detail that matters more at small scale where feature norms drift a lot.

## 5.4 The schedules (what `train/schedules.py` implements)

- **LR: warmup → cosine.** Warmup exists because Adam's second-moment estimates are garbage for the first few hundred steps; a full-size LR then can push a pre-norm ViT into a bad basin.
- **Weight decay: 0.04 → 0.4, increasing.** Unusual (most recipes fix wd). Rationale: light regularization early while features are forming; strong late, acting like a growing pull toward simpler solutions as learning saturates.
- **EMA momentum: 0.996 → 1.0, linear.** Early: teacher tracks student quickly (student is improving fast, teacher must not be stale). Late: teacher freezes (stable targets for final convergence).
- **The 1.25× stretch-and-truncate trick** (App. C): compute all schedules as if training were 25% longer, then stop early. Avoids the aggressive tail of cosine/linear schedules. Cheap to implement (one multiplier — `schedule_scale` in our config), reportedly worth real accuracy.

## 5.5 T4-specific engineering notes

- **AMP (mixed precision)** is on by default: T4 tensor cores double throughput and memory headroom. GradScaler guards against fp16 underflow; we `unscale_` before clipping so the clip threshold is in true gradient units.
- **One mask per batch.** Different masks give different visible-token counts, which can't be stacked into one tensor. Official V-JEPA shares masks within a batch; so do we. Slight variance reduction loss, big simplicity win.
- **Data pipeline**: Moving MNIST is generated on the CPU per item. If GPU utilization is low, raise `DataLoader(num_workers=2)` (Colab gives 2 CPU cores).
- **Batch size 64** fits comfortably; you can go to 128 with this model size. Remember lr should scale roughly linearly with batch size if you change it.

## 5.6 Diagnostics: reading a run like a doctor

Healthy mini-V-JEPA run on Moving MNIST:

- L1 loss: drops fast for ~500 steps, then declines slowly. It will NOT go near zero — targets keep improving under it. A loss that plummets to ~0 is a red flag (collapse), not a victory.
- `target_std`: dips slightly in early training, then stabilizes (with target LayerNorm, at ~1 by construction per-vector; the across-batch std we log stays clearly positive). Slide toward 0 = collapse.
- `pred_std` tracking `target_std` from below is normal (median-seeking predictions are conservative).
- Final verdict comes ONLY from the probe (chapter 6). Tattoo this on the inside of your eyelids: *pretraining loss is not a quality metric in joint-embedding land.*

## 5.7 Common mistakes (all made by me or the field at some point)

1. Optimizer given target-encoder params (collapse). Filter by `requires_grad`.
2. EMA update forgotten, or applied before `optimizer.step()` (subtle staleness).
3. Buffers vs parameters: EMA-copying `parameters()` misses buffers; fine here (our buffers are constant sin-cos tables), but with BatchNorm it's a classic bug.
4. Evaluating with the *online* encoder and wondering why results differ — the paper evaluates the EMA (target) encoder; it's the smoothed, better one. Try both in notebook 05.
5. Changing model width without changing predictor width proportionally — a too-strong predictor weakens the encoder's training signal.
6. Judging experiments off single seeds at this scale. Mini-scale runs are noisy; ±1–2% probe accuracy between seeds is normal. Run 2–3 seeds before believing a difference.

## 5.8 Exercises

1. Run the full sabotage suite (notebook 04) and write one paragraph per row of the table in 5.2 explaining the mechanism of what you observed.
2. Swap L1 → L2 (one-line change). Compare loss curves and probe accuracy over 3 seeds each.
3. Turn off target LayerNorm. What happens to the loss scale over training and why?
4. Implement a variance regularizer (VICReg-style: hinge loss pushing per-dim std above a threshold) as an *alternative* to EMA: set m=0, add the regularizer. Can you rescue the run? This is a genuinely instructive mini-research exercise.

## 5.9 Check your understanding

1. Explain, using the optimal-predictor argument, why a *fast-moving* target enables collapse while a slow-moving one doesn't.
2. Why does a frozen random target avoid collapse yet cap representation quality?
3. Why is `target_std` measured across the batch rather than across feature dimensions? What would each variant of the diagnostic tell you?

## 5.10 Reading

- Grill et al., *BYOL* (2020) — the origin of the EMA+predictor recipe; read Sec 3 + the collapse discussion.
- Tian et al., *Understanding Self-Supervised Learning Dynamics without Contrastive Pairs* (2021) — the theory paper the V-JEPA analysis builds on.
- Bardes et al., *VICReg* (2022) — collapse prevention by explicit variance/covariance regularization; the "other road" to exercise 4 (and same first author as V-JEPA).
