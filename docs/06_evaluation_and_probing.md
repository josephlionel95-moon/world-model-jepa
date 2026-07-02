# 6. Evaluation: How to Know if a Representation Is Good

**Learning objectives.** Understand frozen evaluation and why the field moved to attentive probes; design a probe suite for our Moving MNIST model that mirrors the paper's appearance-vs-motion evaluation.

Companion notebook: `05_probing_and_analysis.ipynb`.

## 6.1 The evaluation problem

A self-supervised model outputs feature vectors. Feature vectors have no accuracy. To measure quality you must attach a task, and *how* you attach it changes what you measure:

- **Fine-tuning** (unfreeze everything, train on the task): measures how good an *initialization* the pretraining provides. Powerful but expensive per task, and it can mask representation deficiencies — with enough labeled data, fine-tuning can repair a mediocre representation. VideoMAE looks great under fine-tuning.
- **Frozen probing** (encoder fixed, train a small head): measures what the representation *already contains*, cheaply, per task. One backbone amortizes across all tasks. This is where V-JEPA plants its flag, and it's the honest metric if what you care about is the representation itself.

The paper's argument (Sec. 5, App. E): the *linear* probe is too weak an instrument for spatially-structured features, so they use the **attentive probe** — and they re-evaluate all baselines under it too, for fairness.

## 6.2 The attentive probe, derived

The encoder emits N token embeddings; a classifier needs one vector. Options, in increasing power:

1. **Mean pool + linear.** Democratic averaging. A digit occupying 6% of tokens contributes 6% of the pooled vector; background dominates. Motion direction, which lives in *which tokens changed where*, is smeared away entirely.
2. **Attentive pool + linear** (V-JEPA's choice): a single cross-attention with one learnable query q:

```
pooled = Σ_i softmax_i(qᵀ W_k s_i) · W_v s_i
```

The probe *learns where to look* — softmax weights concentrate on informative tokens. Then residual + small MLP + linear head (`eval/attentive_probe.py` matches App. D.1, including 12-head attention). It's still "frozen evaluation": ~100k probe params reading a fixed representation, versus millions in fine-tuning.

Question to sit with: at what probe capacity does "reading the representation" become "computing the answer yourself"? An attentive probe can compare tokens to each other (via softmax competition) — is that reading or computing? There's no bright line; the field's convention is that one cross-attention block is acceptable. Be suspicious of papers that win only with ever-bigger probes.

## 6.3 Our Moving MNIST probe suite (mini-K400 and mini-SSv2)

The paper's central claim is *versatility*: one frozen backbone good at appearance (K400) AND motion (SSv2). Our generator keeps every clip's generative factors, so we can build the exact analogue:

| Probe task | Type | Analogue of | Chance |
|---|---|---|---|
| digit class (0–9) | appearance | Kinetics-400 | 10% |
| motion direction (8 compass bins) | motion | SSv2 | 12.5% |
| speed regression | motion | — | — |

The punchline experiment of the whole course (notebook 05): train probes on (a) your V-JEPA encoder, (b) a *randomly initialized* encoder (the crucial baseline — random ViT features are shockingly non-trivial), (c) your MAE encoder from notebook 02 adapted to video if you did the challenge, (d) optionally supervised features. If V-JEPA beats random-init clearly on BOTH digit and direction probes, you have reproduced the paper's core claim at 1/100th scale on your free GPU.

Also compare mean-pool vs attentive-pool probes on the *direction* task specifically — you should see the paper's App. E effect: motion suffers far more from mean pooling than appearance does.

## 6.4 Qualitative diagnostics (cheap and revealing)

- **PCA feature maps** (`eval/visualize.py`): project token features to top-3 principal components as RGB. Digits should segment from background as coherent colored regions that *track* motion across time steps. (DINOv2's famous PCA figures are this.)
- **Feature cosine-similarity across time**: pick a token on the digit at t=0, plot its similarity to all tokens at later times — a bright spot should follow the digit like a tracker.
- The paper's fancy version — a diffusion decoder over predicted features — is out of T4 scope, but a lightweight cousin is in notebook 05's challenge: train a small deconv decoder from *frozen* features to pixels and look at what survives. Remember the epistemics: the decoder is trained separately, so blurriness tells you about missing information, not about the objective.

## 6.5 Evaluation sins (learn them so you can spot them in papers)

1. **Tuning on the test metric.** If you pick pretraining hyper-parameters by probe accuracy and report the same probe as your result, you've overfit the protocol. Keep a held-out seed/config for the final claim.
2. **Comparing probes at unequal budgets** (probe epochs, lr, augmentation). The paper re-runs *all baselines* under its own protocol for exactly this reason.
3. **Single-task evaluation.** A representation optimized for one probe silently overfits to it; versatility across task types is the point.
4. **Ignoring the random baseline.** At small scale a random ViT + attentive probe on MNIST digits scores well above chance. Improvement *over that* is your signal, not raw accuracy.
5. **Single seeds** (mini-scale runs are noisy; see 5.7.6).

## 6.6 Exercises

1. Compute the mean-pool vs attentive-pool gap on digit and direction probes; explain the asymmetry you find.
2. Probe *each layer* of your trained encoder (not just the last). Where does motion information peak? Appearance? (In big models these peak at different depths — the paper probes intermediate layers for detection too, App. D.)
3. Probe the online encoder vs the EMA encoder. Which wins, by how much, and is it consistent across 3 seeds?
4. Challenge: implement the deconv decoder of 6.4 and produce your own version of the paper's Figure 7 (qualitative prediction rendering).

## 6.7 Check your understanding

1. Why would results under fine-tuning *understate* the difference between pixel-target and feature-target pretraining?
2. Your attentive probe on random features already hits 60% digit accuracy. What are two distinct reasons this can happen, and why isn't it a bug?
3. Why does the paper use multiple clips per video at evaluation (8 for K400, 2 for SSv2)? What property of the tasks drives the different counts?
