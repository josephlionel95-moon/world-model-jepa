# 3. From Masked Autoencoders to I-JEPA

**Learning objectives.** Understand masked prediction as a pretext task, implement MAE and see its limits with your own eyes, then make the single conceptual move — pixels → features — that creates I-JEPA. After this chapter, V-JEPA is "just" I-JEPA plus time.

Companion notebooks: `02_masked_autoencoder_mini.ipynb`, `03_ijepa_mini_images.ipynb`.

## 3.1 Masked prediction: the pretext task that won

BERT (2018) showed that hiding 15% of words and predicting them teaches deep language understanding. Vision resisted for three years — until MAE (He et al., 2021) found the two adjustments images need:

1. **Mask much more (75%+).** Language tokens are information-dense; pixels are hugely redundant. Hide 15% of an image and interpolation solves it — no understanding required. Hide 75–90% and you must know what objects are to fill the gaps.
2. **Asymmetric encoder/decoder.** The encoder sees *only visible patches* (cheap — 25% of tokens); a small decoder gets encoder outputs plus mask tokens and reconstructs pixels. After pretraining, throw the decoder away.

Loss: plain L2 on masked-patch pixels (usually per-patch-normalized). Study this architecture carefully in notebook 02, because V-JEPA reuses its skeleton exactly: encoder-on-visible-tokens + lightweight-predictor-with-mask-tokens. What changes later is only *what the predictor's output is compared against*.

## 3.2 What's wrong with pixel targets (see it, don't take my word)

Notebook 02 has you train a mini-MAE on MNIST and inspect reconstructions. You will observe:

- Reconstructions are *blurry* where the input is ambiguous. With 90% masking, a digit could be a 4 or a 9 — L2 forces the output to be the average of both, a ghost. (Chapter 1, question 1: the minimizer of E||y−ŷ||² is the conditional mean.)
- The loss curve keeps improving long after representation quality (probe accuracy) plateaus — the extra capacity goes into sharpening pixel details that don't help recognition.
- MAE representations famously need *fine-tuning* to shine; their frozen linear-probe accuracy is mediocre. The encoder keeps low-level information because the pixel loss demands it, and that clutters the feature space.

That last point is the practical motivation for JEPA: the V-JEPA paper's Table 5–6 story is precisely "we win under *frozen* evaluation and match under fine-tuning, at less compute."

## 3.3 The I-JEPA move: change the target space

I-JEPA (Assran et al., 2023) keeps MAE's skeleton and changes one thing: the predictor's output is compared against *features* of the hidden region, produced by a second encoder — not against pixels.

```
MAE:     Encoder(visible) → Decoder → pixels(hidden)         vs pixels: fixed target
I-JEPA:  Encoder(visible) → Predictor → features(hidden)     vs EMA-Encoder(hidden): learned target
```

Three consequences cascade from this one change:

1. **The target is learned and adaptive** — it can discard unpredictable detail (the whole point, per chapter 1).
2. **The target can collapse** — so I-JEPA imports the BYOL toolkit: the target encoder is an exponential moving average (EMA) of the online encoder, gradients are stopped through the target branch, and the predictor provides asymmetry. (Full theory in chapter 5.)
3. **Masking must get smarter.** With adaptive targets, trivially-solvable masks yield trivial features. I-JEPA masks large contiguous *blocks* (not random scattered patches): scattered masks can be solved by local texture interpolation, while a missing 40%-of-the-image block can only be predicted from *semantic* understanding. It also uses a multi-block setup: one context block, several target blocks per image.

Also note a subtle detail you'll meet in the code: I-JEPA's targets are *contextualized* — the target encoder processes the FULL image and targets are its outputs at masked positions. Each target vector thus encodes its region *in context*, which makes targets more semantic than encoding the crop alone would.

## 3.4 The I-JEPA objective, precisely

Context block x (visible tokens), target blocks y (masked positions), positional information Δy of targets. With online encoder E_θ, EMA encoder E_θ̄, predictor P_φ:

```
L = || P_φ( E_θ(x), Δy )  −  sg( LayerNorm( E_θ̄(y_full)[targets] ) ) ||²
θ̄ ← m·θ̄ + (1−m)·θ         (each step; m ≈ 0.996 → 1.0)
```

sg = stop-gradient. I-JEPA uses L2; V-JEPA switches to L1 (why — chapter 5). The conditioning on Δy is what makes this a *predictive* architecture rather than an invariance method: the predictor is asked "what would the encoder say about location Δy?", a different question for every location.

## 3.5 Common mistakes when implementing this family

- **Forgetting stop-grad / letting the optimizer see target params.** If target-encoder parameters end up in the optimizer, collapse follows. Our smoke test asserts target params have no grads.
- **EMA update before the optimizer step**, or using the wrong momentum schedule direction (it *increases* toward 1.0).
- **Loss on all tokens instead of masked ones only.** Scoring visible tokens rewards copying.
- **Random scattered masking.** Works for MAE (pixel targets), fails to produce semantic features with JEPA. Block structure matters — this is one of I-JEPA's central ablations, and V-JEPA's Table 4 repeats it for video.
- **Judging by pretraining loss.** A lower feature-prediction loss does not imply better features (a slightly-collapsed encoder has a lower loss!). Only probing tells the truth. Always evaluate with a probe.

## 3.6 Exercises

1. (Notebook 02) Train mini-MAE at masking ratios 25/50/75/90% and plot probe accuracy vs ratio. Explain the shape of the curve.
2. (Notebook 03) Train mini I-JEPA on static MNIST. Compare frozen-probe accuracy against your MAE at equal compute.
3. (Notebook 03) Sabotage experiment: set EMA momentum to 0 (target = online encoder, still stop-gradded). Watch `target_std`. What happens and how fast?
4. Challenge: replace block masking with uniform random masking at the same ratio in I-JEPA and measure the probe gap.

## 3.7 Check your understanding

1. Why does MAE need a decoder but I-JEPA only a (smaller) predictor? What information does a decoder have to produce that a predictor doesn't?
2. Why are contextualized targets (encode full image, select masked positions) better than encoding the target crop alone?
3. Your I-JEPA loss drops to 0.001 within 200 steps. Are you happy? What do you check first?

## 3.8 Reading

- He et al., *Masked Autoencoders Are Scalable Vision Learners* (2021).
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA, 2023) — the direct parent of V-JEPA; read fully.
- Baevski et al., *data2vec* (2022) — the same idea discovered from the speech/NLP side; V-JEPA cites it for contextualized targets.
