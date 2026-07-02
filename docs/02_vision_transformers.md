# 2. Vision Transformers, from Pixels to Video Tokens

**Learning objectives.** Derive attention from scratch, explain every line of a ViT block, and understand the two video-specific extensions V-JEPA needs: tubelet embedding and 3D positional codes.

Companion notebook: `01_vit_patches_and_attention.ipynb`.

## 2.1 Why transformers took over vision

Convolutions hard-code an assumption: useful relationships are *local* (a pixel relates to its neighbours). That prior is efficient but rigid — relating a digit at the top-left of frame 0 to the same digit at the bottom-right of frame 7 requires stacking many layers to grow the receptive field.

A transformer makes the opposite bet: *let every element attend to every other element, and learn which relationships matter.* For masked video prediction this is exactly right — the model must relate visible patches to arbitrarily distant masked positions in space *and* time, in a single layer if needed. The price is O(N²) cost in the number of tokens N, which is why token count (and hence masking, which removes tokens) dominates the compute budget.

## 2.2 From image to tokens

A transformer eats a *sequence of vectors*. An image is not one, so ViT (Dosovitskiy et al., 2020) makes it one: cut the image into non-overlapping p×p patches, flatten each patch's pixels, and apply one shared linear layer.

For video, V-JEPA (following VideoMAE) uses **tubelets**: little space-time boxes of shape (t_t × p × p) = 2×16×16 in the paper, 2×8×8 here. Each tubelet spans *two frames*, so a token carries a snapshot of local motion, not just appearance. A 16×224×224 clip becomes an 8×14×14 grid of tokens (1568 tokens); our 8×64×64 clips become 4×8×8 = 256.

Implementation trick (see `models/patch_embed.py`): a `Conv3d` with kernel = stride = tubelet size is *identical* to "flatten each tubelet and apply a shared linear layer" — a conv is just the vectorized way to write it. Verify this equivalence yourself in notebook 01; it is exercise 1.

## 2.3 Deriving attention

Goal: each token should update itself using information from the tokens most relevant to it. We need a mechanism that (a) decides relevance, (b) aggregates.

Give each token three learned projections of its embedding x_i:

- query q_i = W_q x_i — "what am I looking for?"
- key k_i = W_k x_i — "what do I contain?"
- value v_i = W_v x_i — "what do I offer if you attend to me?"

Relevance of token j to token i: the dot product q_i·k_j. Turn scores into weights with a softmax, then take the weighted average of values:

```
attention(i) = Σ_j softmax_j( q_i·k_j / √d ) v_j
```

Why the √d? If q and k have d independent components with unit variance, q·k has variance d, so its standard deviation grows as √d. Softmax of large-magnitude inputs saturates to a one-hot vector, and the gradient through a saturated softmax vanishes. Dividing by √d keeps the scores at unit scale regardless of width. Derive this variance claim yourself: Var(Σ q_i k_i) = Σ Var(q_i k_i) = d for independent zero-mean unit-variance terms.

**Multi-head:** run h attention operations in parallel on d/h-dim slices, concatenate. Different heads can specialize (one tracks position, another shape). Cost is the same as one full-width head; expressiveness is higher because each head has its own softmax pattern.

**Self- vs cross-attention:** self-attention has Q, K, V all from one sequence. Cross-attention takes Q from one place, K and V from another. Remember cross-attention — the attentive probe in chapter 6 is exactly one cross-attention with a single learned query.

## 2.4 The transformer block

```
x  ── LayerNorm ── Attention ──(+)──  LayerNorm ── MLP ──(+)── out
 \___________________________/    \______________________/
        residual                        residual
```

Two sublayers, each wrapped in residual + pre-norm. The parts and their reasons:

- **Residual connections** make each block a *refinement* of the identity function. Gradients flow unimpeded through the additive path, which is what makes 12–32-layer stacks trainable.
- **Pre-norm** (LayerNorm before the sublayer, not after) keeps the residual stream itself unnormalized, avoiding the fragile warmup requirements of the original post-norm transformer.
- **MLP** (expand 4×, GELU, contract): attention *moves* information between tokens but is (nearly) linear in the values; the MLP is where per-token nonlinear computation happens. A transformer alternates "communicate" (attention) and "compute" (MLP).

Our implementation is ~60 lines in `models/vit.py`. Read it after this section — every one of these ideas maps to a specific line.

## 2.5 Positional embeddings, and why V-JEPA's are 3D sin-cos

Attention is permutation-equivariant: shuffle the tokens and the output shuffles identically. Position must be injected explicitly, by adding a position-dependent vector to each token before the first block.

The sin-cos code for one coordinate places each position on a set of sinusoids at geometrically spaced frequencies (`patch_embed.py: sincos_1d`). Key property: the code of position p+k is a *linear function* of the code of position p — so relative offsets are easy for the network to compute. For video, V-JEPA concatenates independent codes for t, h, w into one vector (`sincos_3d`).

Why fixed codes instead of learned embeddings? Two V-JEPA-specific reasons. First, the *predictor's only clue* about which location it must predict is the positional code attached to each mask token — a clean, structured code makes "reason about where" easier. Second, mask tokens for arbitrary positions must be constructible even for positions rarely seen unmasked during training; fixed codes generalize by construction.

**The ordering trap** (the classic silent bug): the token sequence from the tokenizer and the positional code table must use the same flattening order. Ours are both row-major (t, h, w). If you ever permute one and not the other, the model still trains — attention doesn't care — but position information becomes garbage and masked prediction quietly degrades. The smoke test in `tests/test_smoke.py` can't catch this; notebook 01's visualization exercise can.

## 2.6 Masked encoding: the detail everything hinges on

V-JEPA's encoder must process *only the visible tokens* (~10% of the clip — this is where the 10x compute saving comes from, an idea inherited from MAE). Look at `VideoViT.forward`:

```python
x = self.patch_embed(video)   # [B, N, D]  all tokens
x = x + self.pos_embed        # add position FIRST
if keep_idx is not None:
    x = x[:, keep_idx, :]     # THEN drop masked tokens
```

Position is added *before* dropping, so each surviving token keeps the code of its true location in the full grid. Add position after dropping and every token thinks it lives in a dense little grid — the encoder can no longer tell the predictor where the visible evidence actually was.

## 2.7 Scaling behaviour you should know

- Token count N: attention is O(N²·d), MLP O(N·d²). At our scale (N=256, d=192) the MLP dominates; at the paper's scale (N=1568 visible + predicted, d=1024+) attention matters a lot, and masking 90% of tokens cuts encoder attention cost by ~100x.
- ViT names: ViT-S/B/L/H ≈ 22M/86M/300M/632M params. "/16" = patch size 16. V-JEPA's headline model is ViT-H/16.

## 2.8 Exercises (do these in notebook 01)

1. Prove-by-code that Conv3d with kernel=stride equals per-tubelet linear projection (max abs difference < 1e-5).
2. Implement single-head attention with explicit softmax; check it matches `F.scaled_dot_product_attention`.
3. Remove the √d scaling at d=192 and plot the softmax entropy of attention maps at init. What do you see, and why does it hurt gradients?
4. Visualize the 3D sin-cos code: show the [N×N] cosine-similarity matrix of `sincos_3d` and explain the block structure you see.
5. Challenge: swap sin-cos for learned positional embeddings in the mini V-JEPA later (notebook 04) and compare probe accuracy. Predict the outcome first.

## 2.9 Check your understanding

1. Why must positional embeddings be added before token dropping in a masked encoder?
2. Attention moves information; the MLP transforms it. What would a transformer with no MLPs be unable to do?
3. Why does masking give a quadratic (not linear) reduction in attention FLOPs?

## 2.10 Reading

- Dosovitskiy et al., *An Image is Worth 16x16 Words* (2020) — sections 1–3.
- Illustrated Transformer (Jay Alammar) if attention is still hazy.
- V-JEPA paper Appendix B (architecture details) — now fully readable.
