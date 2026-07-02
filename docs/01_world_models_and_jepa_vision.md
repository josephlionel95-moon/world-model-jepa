# 1. World Models and the JEPA Vision

**Learning objectives.** After this chapter you can (a) define a world model and explain why prediction is the engine of representation learning, (b) contrast generative and joint-embedding approaches from an energy-based-model viewpoint, and (c) state precisely what problem JEPA was invented to solve.

## 1.1 What is a world model?

An agent — animal, robot, or program — benefits enormously from an internal model that answers: *"if the world is in state s and event a happens, what happens next?"* That internal simulator is a world model. With one, you can plan (imagine action sequences and pick the best), predict (anticipate hazards before they arrive), and fill in gaps (infer what is behind an occluder).

The catch: the world delivers raw sensory data — millions of pixels per second — and almost all of it is irrelevant detail. A useful world model cannot operate on pixels. It needs a *representation*: a compact description that keeps what matters (objects, positions, motion) and discards what doesn't (leaf textures, sensor noise). So the world-model problem splits in two:

1. **Representation learning** — map raw observations x to useful state s = E(x).
2. **Dynamics learning** — predict future states: s' = P(s, a).

V-JEPA lives in part 1, but with a twist that involves part 2: it *uses prediction itself as the training signal for the representation*. This is the "predictive feature principle" (Rao & Ballard, 1999, cited in the paper's first paragraph): representations of temporally or spatially adjacent stimuli should be predictive of each other.

## 1.2 Why prediction is such a good training signal

Consider what it takes to predict the hidden part of a video from the visible part. You must implicitly know: what objects are present, where they are, how they move, how they interact, what is rigid, what continues behind an occluder. None of this needs labels — the future (or the hidden region) is its own ground truth. Prediction turns the raw data stream into an unlimited supply of exam questions about how the world works.

This is why self-supervised learning (SSL) chose prediction-flavoured objectives: there are ~10^9 videos on the internet and ~10^7 labeled ones. Whoever can learn from the unlabeled pile wins.

## 1.3 The central dilemma: predict in what space?

Here is the fork in the road that defines the whole field. You want to train E (the encoder) by making some predictor P predict hidden content. Two options:

**Option A — generative / reconstruction.** Predict the hidden content *in pixel space*: decode back to pixels and score with a pixel loss (MAE, VideoMAE, diffusion world models like GameNGen or DIAMOND work here).

**Option B — joint embedding.** Predict the hidden content *in representation space*: score the prediction against E(hidden part), never touching pixels (BYOL, DINO, I-JEPA, V-JEPA).

Option A has a fundamental inefficiency. The world is stochastic and detail-rich: given the visible frames, the exact pixels of the hidden region are *not determined*. Precise position of every MNIST stroke pixel, the flicker of leaves, camera grain — unpredictable. A pixel loss forces the model to spend capacity modeling exactly this unpredictable detail, and when the target is genuinely multimodal, an L2 pixel loss makes the model predict the *mean of all futures* — a blurry ghost. Capacity spent sharpening ghosts is capacity not spent understanding scenes.

Option B lets the encoder *choose what to keep*. If leaf-flicker is unpredictable, the encoder can simply not represent it, and then the predictor is never punished for failing to predict it. The loss lives in a learned space where irrelevant detail has been discarded. That is the "joint embedding" bet: **abstraction and prediction should happen in the same learned space.**

The paper's Figure 5 result makes this concrete: V-JEPA (feature prediction) reaches better frozen-evaluation performance than VideoMAE (pixel prediction) with *shorter* training schedules — understanding-per-FLOP is higher when you don't pay the pixel tax.

But Option B has a lethal failure mode that Option A cannot have: **collapse**. If the encoder outputs the same constant vector for every input, prediction becomes trivially perfect and the loss goes to zero. A pixel target can't collapse (pixels are fixed); a *learned* target can. The entire design of BYOL/DINO/JEPA methods is shaped by the need to dodge collapse. Chapter 5 is devoted to this.

## 1.4 The energy-based view (LeCun's framing)

LeCun's position paper ("A Path Towards Autonomous Machine Intelligence", 2022) frames both options as energy-based models. An EBM assigns a scalar energy F(x, y) to a pair (context x, outcome y): low energy = compatible, high = incompatible. Learning shapes the energy landscape so real pairs sit in valleys.

- A **generative** model computes energy through pixel-space reconstruction error: F(x,y) = ||Decoder(E(x)) − y||².
- A **JEPA** computes energy in representation space: F(x,y) = ||P(E(x)) − E(y)|| — for us, with the L1 norm.

The JEPA is LeCun's proposed core module for machine intelligence, with three properties he insists on: (1) it predicts in representation space, so it can ignore the unpredictable; (2) a latent/conditioning variable z tells the predictor *which* transformation relates x to y (in V-JEPA, z = the positions of the masked region); (3) stacked hierarchically, JEPAs could predict at multiple levels of abstraction and time scale — short-term detailed predictions at the bottom, long-horizon abstract plans at the top. V-JEPA implements (1) and (2); (3) remains open research — remember that for chapter 7.

## 1.5 The SSL family tree (where JEPA sits)

Four families, distinguished by how they get a training signal without labels and how they avoid collapse:

| Family | Signal | Anti-collapse | Examples |
|---|---|---|---|
| Contrastive | pull augmented views together, push different images apart | negatives repel | SimCLR, MoCo, CPC |
| Distillation / invariance | make two views' embeddings equal | EMA teacher + asymmetry | BYOL, DINO, DINOv2 |
| Masked generative | reconstruct masked pixels/tokens | targets are fixed data | BERT, MAE, VideoMAE |
| Joint-embedding predictive | predict *features* of masked/other region | EMA teacher + stop-grad + predictor | data2vec, I-JEPA, **V-JEPA** |

Two contrasts worth internalizing:

*JEPA vs contrastive:* contrastive methods need negatives (or huge batches) and learn *invariance* to hand-designed augmentations — crops, color jitter. The augmentations smuggle in human prior knowledge about what shouldn't matter. JEPA needs no negatives and no augmentations; masking is its only corruption. (V-JEPA's ablations confirm it trains well without color jitter or flips.)

*JEPA vs distillation (BYOL/DINO):* BYOL makes two views of the same image map to the *same* embedding — pure invariance, spatial information is pooled away. JEPA instead makes one region's embedding *predict a different region's* embedding, keeping the predictor conditioned on location. Prediction subsumes invariance and preserves spatial/equivariant structure — which V-JEPA needs, because motion tasks (Something-Something-v2) die without it.

## 1.6 Common misconceptions

- *"JEPA is an autoencoder without a decoder."* No — an autoencoder's target is its own input in pixel space. JEPA's target is a *learned representation of a different (hidden) region*, produced by a slowly-moving copy of the encoder. Both the "what is predicted" and "who produces the target" differ.
- *"Feature prediction must be easier than pixel prediction, that's why it works better."* It's not about easy. Predicting features of a moving target network is in some ways harder (the target drifts). It works better because the *target space is adaptive*: it can drop unpredictable information.
- *"Collapse is a rare edge case."* Collapse is the default behaviour of the naive objective. The surprise, made precise in chapter 5, is that such indirect mechanisms (EMA + stop-grad + predictor) reliably prevent it.

## 1.7 Check your understanding

1. Why does an L2 pixel loss produce blurry predictions when the future is multimodal? (Hint: what minimizes E[||y − ŷ||²] over ŷ?)
2. Give a concrete piece of information in a Moving MNIST clip that a pixel-space model must represent but a feature-space model can safely discard.
3. In the EBM framing, why can't a JEPA trained with the naive objective assign meaningfully low energy to *correct* pairs only? What does the flat-zero-energy landscape correspond to in encoder terms?
4. Contrastive learning imports human priors through augmentation choices. What is the analogous "human prior" input in V-JEPA? (There is one — think about what masking strategy encodes.)

Answers are discussed as they arise in later chapters; write yours down first.

## 1.8 Reading

- LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — read Sections 1–4 for the JEPA vision.
- Bardes et al., *V-JEPA* (2024) — read only Sections 1–2 for now.
- Optional: Rao & Ballard (1999), predictive coding — the neuroscience ancestor of all of this.
