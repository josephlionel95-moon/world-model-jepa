# 4. The V-JEPA Paper, Line by Line

*Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran†, Ballas† — "Revisiting Feature Prediction for Learning Visual Representations from Video" (2024, arXiv:2404.08471).*

Read this chapter with the PDF open. We go section by section: what it says, why the authors did it that way, and what to question.

## 4.0 The one-sentence claim

Feature prediction, *alone* — no pixels, no text, no pretrained image encoder, no negatives, no augmentation-invariance — is a sufficient objective to learn strong, versatile video representations, and it does so with better compute efficiency than pixel-based masked modeling.

The word "revisiting" is doing work: predictive features are an old idea (Rao & Ballard 1999; slow feature analysis, 2002). The paper's contribution is showing the old idea *wins* once you equip it with 2024 tools — ViTs, JEPA-style asymmetry, multi-block masking, 2M videos.

## 4.1 Section 1–2: Introduction and related work

The intro frames the question ("How effective is feature prediction as a stand-alone objective...?") and previews the answers:

- Versatility under *frozen* evaluation: one backbone, no fine-tuning, good at both appearance tasks (Kinetics-400: what objects/scenes are present) and motion tasks (Something-Something-v2: which *direction* did the hand move — appearance alone cannot solve it). Pixel-prediction models and image models each tend to win only one of these; Figure 1 plots exactly this trade-off plane.
- Efficiency: shorter training schedules than VideoMAE-style pixel prediction.
- Their ViT-H/16, video-only, gets 77.9% frozen on *ImageNet* — a model that never saw a single still image is competitive on image classification. Videos contain images; the reverse direction had been the dominant assumption.

The related-work section organizes history as: slow features / local invariance → predictive features with frozen encoders or contrastive anti-collapse → JEPA (predict features with encoder+predictor trained jointly, collapse handled by architecture rather than negatives). Notice what the taxonomy implies: the *anti-collapse mechanism* is the axis along which the field evolved.

## 4.2 Section 3: Methodology — every design decision

### 3.1 The objective

```
minimize_θ,φ  || P_φ(E_θ(x), Δy) − sg(E_θ̄(y)) ||₁
```

Chapter 3 gave you every piece except the norm. **Why L1, not L2?** The paper says "more stable," and offers a theoretical nugget worth deriving (they adapt BYOL's argument):

For an optimal predictor, the prediction that minimizes E|ŷ − Y| is the *conditional median* of Y (whereas L2 gives the conditional mean). Substituting the optimal predictor into the loss, the gradient received by the encoder becomes

```
∇θ E | P*(E_θ(x)) − Y |₁ = ∇θ MAD( Y | E_θ(x) )
```

where MAD is the median absolute deviation — a robust measure of the *spread* of targets given the context. So the encoder is being trained to make targets *predictable* (low conditional spread), i.e. to capture as much predictive information about the video as possible. The same derivation under L2 yields conditional variance; variance is dominated by outliers, and with a drifting target network outliers are common — hence L1's stability advantage. The hypothesis for why EMA prevents collapse also lives here: the slow target ensures the predictor "evolves faster than the encoder," staying near-optimal, and a near-optimal predictor turns the encoder's objective into spread-minimization rather than a race to a constant.

### 3.2 The prediction task: 3D multi-block masking

To sample the target region y: sample several spatially contiguous blocks with random aspect ratio in (0.75, 1.5), take their union, and **repeat the mask across every frame**. Context x = complement. Two flavours per clip:

- **short-range**: union of 8 blocks each covering 15% of the frame;
- **long-range**: union of 2 blocks each covering 70%.

Both land near 90% masking. The temporal repetition is the crucial video-specific decision: if a region were visible at any time step, temporal redundancy would let the model copy rather than understand ("limits information leakage"). Section 4's ablation (Table 4) backs this empirically — their multi-block beats random-tube and causal variants.

Ask yourself before reading on: what does spatially-repeated masking make *impossible to learn*? (Answer: nothing is visible at masked locations *ever*, so predictions there must be inferred entirely from surrounding context — but note it also means the model never practices short-horizon "continue this visible object's motion" forecasting. Keep that thought for chapter 7.)

### 3.3 Network parameterization

- Encoder: standard ViT on tubelet tokens (2×16×16 conv3d), 3D sin-cos positions.
- Predictor: **narrow** transformer — 12 blocks, 384-dim regardless of encoder size (ViT-L is 1024-dim). Mask tokens with positional codes mark what to predict; only their outputs are scored.
- Targets are the EMA encoder's outputs *at masked positions after encoding the full clip* — contextualized targets (from data2vec).

Everything here maps 1:1 onto `src/vjepa_mini/models/`. Our config file lists the paper's number beside each of ours.

### 3.4 Pretraining data

**VideoMix2M**: ~2M videos from HowTo100M + Kinetics-400/600/700 + Something-Something-v2 (labels discarded). Clips: 16 frames, frameskip 4 (~3 s). Batch 3072, 90k iterations, AdamW, wd 0.04→0.4, EMA 0.998→1.0, all schedules stretched 1.25× and truncated (App. C — an empirical trick: the last quarter of a cosine schedule changes hyper-parameters too aggressively).

## 4.3 Section 4: "What matters" — the ablations (the most instructive section)

Run on ViT-L/16, frozen attentive-probe evaluation. Three findings:

1. **Data distribution matters and scale helps** (Table 2): mixing datasets beats any single one; more unique videos beat more epochs of fewer videos.
2. **Masking design matters** (Table 4 & Fig. 6): their multi-block beats random-tube masking and causal variants ("predict the future frames from the first p frames" — interestingly, *worse* than spatial multi-block for representation quality). Also: needing BOTH short- and long-range masks together beats either alone.
3. **Feature vs pixel targets, controlled** (Table 5–6): same architecture, same masking, swap only the target space (V-JEPA vs a VideoMAE-style pixel objective). Feature prediction wins clearly under frozen evaluation everywhere, and under fine-tuning wins on SSv2 while roughly matching on K400 — with ~2× fewer pretraining samples seen. This is the paper's cleanest scientific result: the *only* variable is what space the loss lives in.

Also in Section 5: attentive probing (chapter 6 here) and the observation that *average*-pooled probes are much worse — spatial structure carries the motion information, and pooling destroys it.

## 4.4 Section 6–7: Headline comparisons

- vs pixel-prediction SOTA (VideoMAE, VideoMAEv2, OmniMAE, Hiera): V-JEPA wins frozen eval broadly, especially on SSv2 (motion), and needs far fewer samples seen.
- vs image models (DINOv2, OpenCLIP, I-JEPA): image models remain slightly ahead on pure-appearance tasks (they trained on curated images), but they *fail on motion* (SSv2) — a static-image model has no way to represent temporal direction. V-JEPA is the first video model to beat them there while staying close on appearance. ImageNet: 77.9% frozen (an attentive probe on a video-only model).
- Low-shot: with 5–10% of labels, V-JEPA's frozen features degrade more gracefully than competitors — a signature of representations that already contain the task-relevant structure.

## 4.5 The qualitative check (Section "Evaluating the Predictor")

They freeze encoder+predictor, train a conditional diffusion decoder to render *predicted features* into pixels, and look. The samples are spatio-temporally coherent — objects with plausible motion appear in masked regions, and varying the random seed shows the predictor carries *positional uncertainty* correctly. Note the epistemics: the decoder is trained separately, so this visualizes what information the features contain without that information ever having been optimized for rendering.

## 4.6 Reading the appendices like a researcher

- App. B–C: exact architectures and every hyper-parameter, including the schedule-stretching trick. This is where reproduction lives.
- App. D: evaluation protocol details — attentive probe structure, multi-clip evaluation (8 clips on K400, 2 on SSv2 — temporal coverage matters).
- The theory box (Sec. 3.1) is short; redo the median/MAD derivation on paper. Exercise: show that argmin_c E|Y − c| is the median of Y. (Hint: differentiate E|Y−c| in c and set the derivative — P(Y<c) − P(Y>c) — to zero.)

## 4.7 What to question (your training as a reviewer)

1. The masking ablation searches a small design space. Is spatial-block-repeated-in-time optimal, or just the best of five tried? What would a *learned* masking policy do?
2. Frozen-probe evaluation favours methods that pack linearly-accessible info into features. Is that the right notion of "good representation" for downstream *control/planning*?
3. The predictor is discarded after pretraining. For a "world model" narrative, the predictor IS the dynamics model — nothing in the evaluation tests multi-step rollout or planning. (V-JEPA 2, released 2025, addresses exactly this: action-conditioned prediction and robot planning. Chapter 7.)
4. All targets come from ~3-second clips. What temporal abstraction can you learn when nothing exceeds 3 seconds?

These four questions are not rhetorical — each one is an active research direction, and chapter 7 turns them into concrete project ideas.

## 4.8 Check your understanding

1. State precisely what differs between V-JEPA's loss and I-JEPA's loss (two things).
2. Why must the mask repeat across all frames? What failure occurs with per-frame independent masks?
3. In Table 5's controlled comparison, why is "same masking, same architecture, swap the target space" the right experimental design? What confound does it remove?
4. Why does average pooling destroy motion information while attentive pooling preserves it?
