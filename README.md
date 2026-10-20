# EM-CGI: Evidence-Modulated Constrained Graph Inference

EM-CGI performs interactive vertebral landmark correction through two modules:

- **Evidence-conditioned Feature Modulation (ECFM):** uses landmark confirmations to sample and match local and structural evidence, then update localization features.
- **Relation-guided Joint Correction (RJC):** combines visual displacements and learned landmark relations to jointly update coordinates while preserving confirmed positions.

## Framework

![EM-CGI framework](assets/emcgi-framework.png)

[Full-resolution PNG](assets/emcgi-framework.png) · [PDF](assets/emcgi-framework.pdf)

## Dataset

[AASCE (SpineWeb Dataset 16)](http://spineweb.digitalimaginggroup.ca/)

## Quick start

Python 3.10+ and PyTorch 2.0+.

```sh
python -m pip install -r requirements.txt
python -m examples.synthetic_demo
```

The CPU example uses synthetic inputs, random weights and an artificial prior to demonstrate automatic localization and cumulative landmark corrections. Default model settings use 68 landmarks and a 1024 × 512 image (height × width).

## Code structure

```text
emcgi/
  config.py              Architecture and inference settings
  compatibility.py       Model state validation
  models/
    model.py             Automatic and interactive forward paths
    backbone.py          Multi-resolution backbone
    visual.py            Visual encoder and heatmap head
    evidence.py          Evidence sampling, matching and modulation
    interaction.py       Correction-conditioned feature updates
    relations.py         Relation weights and relative corrections
    geometry.py          Heatmap decoding and constrained localization
    residual.py          Joint displacement solver
examples/
  synthetic_demo.py      Synthetic inference example
assets/
  emcgi-framework.png    Framework figure
  emcgi-framework.pdf    Framework figure in PDF
```

[Core API documentation](docs/CORE_API.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)
