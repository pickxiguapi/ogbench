# LeWM JAX: verified ViT reproduction and explicit CNN variant

The public default (`from lewm_jax import LeWM`) is the verified reference
ViT implementation. The model factory supports two explicit architectures:

- `reference_vit_66d47b6`: ViT-Tiny/14 corresponding to the original LeWM
  architecture and proven on Server 23 at a true batch size of 128.
- `lewm_impala_variant`: the OGBench IMPALA-small CNN used by visual
  GCIQL/HIQL, paired with the same LeWM predictor and objective.

Use `--encoder=vit_tiny14` for paper reproduction and
`--encoder=impala_small` for the CNN experiment. The architecture identifier is
stored in every checkpoint and evaluation reconstructs the matching model;
there is no implicit fallback between them.

The formal `vit_tiny14` reproduction uses `vit.py`, restored from the
Server-23-proven commit `66d47b6`: TwoRoom completed 10 epochs and the other
three tasks completed five epochs at true batch 128 before interruption. The
JEPA, predictor, rollout, and SIGReg organization also follows
[`dhidary/le-wm-jax`](https://github.com/dhidary/le-wm-jax) at commit
`e52c1a0`. That project provides strong inference/checkpoint parity evidence,
but no training loop. This package therefore uses trainable Flax primitives
and the local OGBench training pipeline rather than copying its inference-only
Equinox `Linear`, `LayerNorm`, and `BatchNorm1dEval` wrappers.

## Encoder preprocessing

- The source HDF5 `pixels` column is `uint8` RGB in HWC layout. The Lance
  conversion stores each frame as JPEG bytes (quality 95 by default), and the
  lazy loader decodes it back to `uint8` RGB HWC.
- The formal ViT backend receives ImageNet-normalized float NHWC images,
  matching the original LeWM preprocessing and its proven execution graph.
- `impala_small` performs `/255` internally through the shared OGBench encoder.

The ViT keeps the attention execution path proven by the first working JAX
port: explicit linear layers produce Q/K/V, tensors are explicitly laid
out as `B,H,Q,D`, QK logits are bf16, softmax is evaluated in float32 and cast
back to bf16, and probability-times-V is bf16. This avoids the much larger XLA
backward temporary produced by the generic `B,Q,H,D` attention path at
batch 128. The checkpoint parameter tree and attention formula remain the
HuggingFace ViT layout; only the implementation is made explicit.

The selected encoder is stored in every checkpoint. Evaluation restores both
the encoder and its preprocessing from that field.

## Default training configuration

The LeWM reproduction protocol defaults to 10 epochs. Apart from the selected
visual encoder, both variants share the LeWM configuration: seed 3072, 90/10
clip split, batch 128, image size 224, latent
dimension 192, history 3, one prediction, frameskip/action chunk 5, predictor
depth 6 with 16 heads and 2048-wide MLP, AdamW at 5e-5 with weight decay 1e-3,
global gradient clipping at 1.0, SIGReg weight 0.09 with 17 knots and 1024
projections, bf16 compute, and a 1%-warmup cosine schedule advanced once per
optimizer update. All architecture and optimization values are written into
the checkpoint config and restored explicitly by evaluation.

The prediction MSE and SIGReg follow PyTorch bf16 autocast boundaries rather
than converting all embeddings to fp32. SIGReg uses bf16 projected embeddings,
fp32 trigonometric quadrature, then returns to bf16 for its final contraction.

The reference dataset-goal evaluation uses the published configuration's
default `history_len=1`, even though training clips contain three frames. This
is kept deliberately for paper-protocol reproduction; a real-history-3 planner
must be recorded as a separate ablation.

The train/validation split and epoch shuffles share one seeded NumPy random
generator. This preserves the reference split ratio and deterministic behavior
without adding PyTorch to the JAX environment; exact sample permutations are
backend-specific. Non-pixel action columns are cached from the sibling source
HDF5 when available, preserving Reacher's float64 action statistics before
normalized actions are converted to float32 for the model.

## Compatibility

These new Flax parameter trees are not compatible with checkpoints produced by
the earlier monolithic `impls/lewm_jax.py`. Keep the code revision that created
an old checkpoint when evaluating that checkpoint.
