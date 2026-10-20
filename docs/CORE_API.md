# Core API

## Model

```python
from emcgi import EMCGI, model_config

cfg = model_config(points=68, image_hw=(1024, 512))
model = EMCGI(cfg, prior).eval()
```

`prior` defines the landmark graph:

| Key | Shape | Meaning |
| --- | --- | --- |
| count | Integer K | Number of landmarks |
| edges | E x 2 | Landmark index pairs (i, j); at least one edge |
| means | E x 2 | Mean relative coordinates p_i - p_j |
| precisions | E x 2 x 2 | Positive definite edge precision matrices |

Coordinates use normalized (x, y): `(x_px / W, y_px / H)`. Prior means and precisions use the same coordinate system and landmark order.

## Inference

Use `torch.no_grad()` with the model in evaluation mode.

- **Input:** float32 grayscale image, shape `B x 1 x H x W`, values in `[0, 1]`. H and W should be divisible by 32.
- **Automatic localization:** `automatic = model(image)`.
- **Interactive correction:** `corrected = model(image, history)`.
- **Outputs:** `points` (`B x K x 2`), `logits` (`B x K x h x w` with independent heatmaps), unary estimates, `reference` and diagnostics. Points use float64 and may fall outside `[0, 1]`.

## Interaction history

| Key | Shape | Meaning |
| --- | --- | --- |
| current | B x K x 2 | Latest output coordinates |
| old | B x K x 2 | Position immediately before each landmark was confirmed |
| confirmed | B x K x 2 | Confirmed coordinates; zero for unconfirmed entries |
| clicked | B x K, boolean | Cumulative confirmation mask |
| reference_q | B x 2K x 2K | Fixed automatic quadratic matrix |
| reference_rhs | B x 2K | Fixed automatic right-hand side |
| automatic_points | B x K x 2 | Initial automatic positions |
| automatic_unary_mean | B x K x 2 | Initial unary means |
| automatic_precision | B x K x 2 x 2 | Initial unary precisions |

1. Copy reference fields from `automatic['reference']`; initialize `current` and `old` from automatic points.
2. For a new confirmation, copy that landmark's current position into `old`, set `confirmed` and mark `clicked`.
3. Run `model(image, history)`, then update `current` from the returned points.

Keep prior confirmations and the initial reference throughout the sequence. Use float32 for `current`, `old` and `confirmed`; preserve reference dtypes. Keep the model and tensors on the same device.

The optional `prior_transform` (`B x 2 x 2`) applies a linear transformation to the structural prior, consistent with the image-coordinate transformation.

## Model weights

Load compatible parameters with `model.load_state_dict(state_dict)`.

See [the synthetic example](../examples/synthetic_demo.py) for a complete interaction sequence.
