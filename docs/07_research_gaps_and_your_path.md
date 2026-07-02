# 7. Research Gaps and Your Path to a Contribution

**Learning objectives.** Know what has happened since the V-JEPA paper, identify the live open problems, and pick a T4-scale research question with a credible path to a real finding.

## 7.1 What happened after the paper (state of play, mid-2026)

**V-JEPA 2** (Meta, June 2025, arXiv:2506.09985) is essential context — it answers chapter 4's question 3 (the discarded predictor) directly:

- Scaled to 1.2B params on ~1M hours of internet video; state-of-the-art motion understanding and strong video QA when aligned with an LLM.
- **V-JEPA 2-AC**: freeze the pretrained encoder, post-train an *action-conditioned* predictor on <62 hours of unlabeled robot video (DROID). The predictor now IS a usable world model: given current state + candidate action sequence, it predicts future latent states.
- **Zero-shot planning**: model-predictive control on real Franka arms — sample action sequences, roll them out in latent space, pick the sequence whose predicted final state is closest to an image goal's embedding. Pick-and-place in labs the model never saw, no rewards, no task training.
- Meta also released physical-reasoning benchmarks (IntPhys 2, MVPBench, CausalVQA) on which even frontier models lag humans — a signpost for where the gaps are.

So the JEPA line went: features (V-JEPA) → dynamics + planning (V-JEPA 2). The 2018 "World Models" promise — learn a model, plan inside it — now runs on real robots with self-supervised video pretraining.

## 7.2 The live gaps

Each of these is acknowledged in the literature (V-JEPA 2's own limitations section, LeCun's position paper, and 2025–26 follow-ups) and none is closed as of this writing:

**Gap 1 — Long-horizon rollout instability.** Latent rollouts drift: small prediction errors compound, and MPC beyond a few seconds degrades. Follow-up work (e.g. hierarchical latent planners in the FF-JEPA direction) decomposes long tasks into subgoals, but "how to plan far ahead in latent space" is open.

**Gap 2 — No hierarchy.** LeCun's blueprint calls for stacked JEPAs predicting at multiple time scales and abstraction levels (H-JEPA). Nobody has convincingly trained one. What should a higher level predict, at what rate, and how do levels talk?

**Gap 3 — Uncertainty and multimodal futures.** The predictor outputs a point estimate (the conditional median under L1!). The future is multimodal. How should a JEPA represent *distributions* over futures — latent variables z sampled at inference, variational objectives, ensembles? Probabilistic-JEPA formulations are appearing but nothing is settled.

**Gap 4 — Collapse theory is incomplete.** We prevent collapse with a recipe (EMA + stop-grad + predictor) that works empirically but whose theory (chapter 5) is partial. When exactly is the recipe safe? Can we design objectives with *guarantees*? (VICReg-style regularizers are one road; nobody has a clean story for JEPAs at scale.)

**Gap 5 — Masking as a hand-designed prior.** Multi-block masking is a human-designed curriculum (chapter 1, question 4). Learned/adaptive masking policies, curriculum masking, or object-aware masking are underexplored — and the paper's own ablation shows masking design moves results a lot.

**Gap 6 — Goal specification.** V-JEPA 2-AC plans toward *image goals* — you must photograph the desired end state. Language goals, reward-free subgoal discovery, and goal abstraction are open.

**Gap 7 — Memory.** JEPAs see a ~3s (V-JEPA) to ~few-minute (V-JEPA 2 with tricks) window. Persistent memory of "what's behind me / where things were" — the object-permanence-at-scale problem — is unsolved for this family.

## 7.3 Gap → T4-sized research question

You cannot compete on scale. You CAN compete on *controlled science at small scale* — exactly what this repo is built for. Moving MNIST (and its extensions you can write in an afternoon: occluders, gravity, object interactions, color) gives you ground-truth generative factors, so you can measure things frontier labs can't measure cleanly on YouTube data.

Ranked by (feasibility × relevance), here are concrete projects. Each is a genuine open question projected down to mini-scale:

**P1 (Gap 3) — Multimodal futures in mini-JEPA.** Make Moving MNIST futures explicitly multimodal (e.g. digits randomly bounce OR pass through walls, 50/50). Compare: L1 point predictor vs predictor with sampled latent z vs small ensemble. Measure: does the point predictor's output land "between" modes in feature space? Does a latent-variable predictor separate them? You have ground truth for both modes — measurable cleanly.

**P2 (Gap 5) — Learned masking curriculum.** Baseline: paper's multi-block. Variants: adversarial masking (mask what the model currently predicts *best*), easy-to-hard curriculum on block size, object-tracking masks (you know digit positions!). Metric: probe accuracy at equal compute. The paper's Table 4 is your template for presenting it.

**P3 (Gap 1) — Rollout drift at mini-scale.** Add an action-conditioned predictor to your mini V-JEPA (actions = digit velocity changes you control in the generator — a mini V-JEPA 2-AC). Measure feature-space error vs rollout depth; test whether multi-step training objectives (predict t+k directly vs compose k one-step predictions) change the drift curve.

**P4 (Gap 7) — Occlusion memory.** Add a static occluder bar to Moving MNIST. Probe: can frozen features report a digit's position *while it is hidden*? Compare context windows and predictor depths. This is object permanence, quantified, for ~$0 of compute.

**P5 (Gap 4) — Collapse phase diagram.** Sweep (EMA momentum × predictor depth × lr) on 2-hour runs; map where collapse happens (`target_std` as the order parameter). A careful empirical phase diagram of the BYOL/JEPA mechanism at controlled scale would be a useful workshop paper — measurement, not scale, is the contribution.

Start with P4 or P5 (least engineering risk), do P1 or P2 for a real shot at novelty.

## 7.4 How to run a research project (the process)

1. **One-sentence hypothesis**, falsifiable: "Latent-variable predictors reduce between-mode feature error by X% on bimodal futures."
2. **Baseline first.** Reproduce the vanilla number 3 seeds deep before touching the variant. Most "improvements" in student projects are baseline bugs.
3. **One change at a time**, everything else pinned (the paper's Table 5 discipline).
4. **Decide metrics before running.** Post-hoc metric shopping is how you fool yourself.
5. **Lab notebook** per run: config hash, prediction, result, surprise.
6. **Negative results are results** at this stage — a clean "adversarial masking does NOT help at this scale, here's why" teaches you more than a noisy win.
7. Write up as you go; the write-up exposes the holes. Target format: a 4-page workshop paper (NeurIPS/ICLR workshop style). Workshops on SSL and world models exist every cycle and welcome small-scale careful studies.

## 7.5 Scaling the ladder later

When a mini-scale finding holds: Colab Pro / a rented A100 gets you to real video (e.g. Something-Something-v2 at reduced resolution, or the released V-JEPA 2 checkpoints as frozen encoders with *your* predictor/masking innovation post-trained on top — the V-JEPA 2-AC recipe proves 62 hours of data can be enough when the encoder is frozen). That's the credible route from this repo to a publishable result without a cluster: **innovate on the cheap moving part (predictor, masking, objective), reuse the expensive part (pretrained encoder).**

## 7.6 Reading

- Assran et al., *V-JEPA 2* (2025, arXiv:2506.09985) — read in full with chapter 4's method; the AC section is your P3 template.
- LeCun (2022) Sections 5–8 — the H-JEPA blueprint (Gap 2).
- Meta's V-JEPA 2 blog + benchmarks (IntPhys 2, MVPBench, CausalVQA) — where models still fail.
- Ha & Schmidhuber, *World Models* (2018) — the ancestor; compare its VAE+RNN decomposition with JEPA's encoder+predictor.
- Recent arXiv: search "JEPA" monthly. The family grows fast (hierarchical planners, probabilistic variants, multimodal JEPAs); part of research training is maintaining your own map.
