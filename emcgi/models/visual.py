"""MVE: multi-scale visual encoder with the shared localization head."""
import torch
from torch import nn
from .backbone import MultiResolutionBackbone


class MultiScaleVisualEncoder(nn.Module):
    def __init__(self, points, independent=False):
        super().__init__()
        self.network = MultiResolutionBackbone(out_chs=points if independent else points+1)
        self.network.init_weights()

    def forward(self, image, condition=None):
        return self.network(image*2.-1., condition)
