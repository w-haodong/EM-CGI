# Core API

## Construction

```python
from emcgi import EMCGI, model_config
cfg = model_config(points=68, image_hw=(1024, 512))
model = EMCGI(cfg, prior)
```

`prior` contains `count` (K), `edges` (E 脳 2 landmark indices), `means` (E 脳 2 relative mean positions), and `precisions` (E 脳 2 脳 2 positive definite matrices). For edge (i, j), means represent p_i - p_j. All positions use normalized (x, y) coordinates; convert pixels with (x / W, y / H), and convert back with (x * W, y * H). Precision matrices use this same normalized coordinate system. Use at least one valid edge and match the landmark count and order. The example's artificial chain is not an anatomical prior.

## Inputs and outputs

`image`: float tensor B 脳 3 脳 H 脳 W, values in [0, 1]. The model internally maps it to [-1, 1]. Input images must be single-channel grayscale after application-specific preprocessing. The release does not implement the original dataset preprocessing. Use dimensions divisible by 32 for the backbone.

`model(image)` returns `points` (B 脳 K 脳 2), `logits` (B 脳 K 脳 h 脳 w under independent heatmaps), unary estimates, a detached `reference`, and auxiliary diagnostics. Solver coordinates are double precision and are not globally clamped to [0, 1].

`model(image, history)` performs a cumulative correction. `history` contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| current | B 脳 K 脳 2 | Most recent output coordinates |
| old | B 脳 K 脳 2 | Coordinates immediately before each landmark was confirmed |
| confirmed | B 脳 K 脳 2 | Confirmed positions; initialize unconfirmed entries to zero |
| clicked | B 脳 K boolean | Cumulative confirmed-landmark mask |
| reference_q, reference_rhs | B 脳 2K 脳 2K; B 脳 2K | Fixed automatic quadratic reference |
| automatic_points | B 脳 K 脳 2 | Automatic positions p0 |
| automatic_unary_mean | B 脳 K 脳 2 | Initial unary means |
| automatic_precision | B 脳 K 脳 2 脳 2 | Initial unary precisions |

Copy all reference fields directly from the initial `automatic['reference']`. Initialize `current` and `old` from automatic points. Before adding a new click, copy that landmark's current position into `old`, set its confirmed coordinate, and set its clicked flag. Preserve all previous confirmations. After each forward call, update `current` from the returned points. Keep the original reference throughout the sequence. All tensors and the model must share a device. Keep `current`, `old` and `confirmed` in float32 to match visual features; cast solver outputs to float32 when updating `current`. Preserve the reference tensors in their returned precision.

The optional `prior_transform` is B 脳 2 脳 2 and transforms the structural prior under a consistent affine coordinate transformation. It must be consistent with the image and coordinate transformation; it does not independently resize image coordinates.

## Parameters and checkpoints

`model_config` retains the architecture schema required by this implementation. Model parameters start randomly initialized. This release deliberately omits the internal predecessor-checkpoint initialization helper. A compatible state dictionary can be loaded using `model.load_state_dict(state_dict)`; no checkpoint is supplied and no result reproduction is claimed.

The public class is `EMCGI`. Internal configuration identifiers (`SILC` and `v132_silc_joint_residual_v2`) retain the original implementation schema for state validation; they refer to this implementation, not a separate released method.

## Inference-only scope

Call `model.eval()` and use `torch.no_grad()` for inference, as shown in the example. Training-specific BatchNorm policy, losses, optimization, data loaders and experiment configuration are omitted. The released architecture and settings are sufficient for the synthetic interface example, not for reconstructing the unpublished training workflow.
