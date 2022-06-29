# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn

from model.stg_lib.utils import get_batcnnorm, get_dropout, get_activation

__all__ = [
    'LinearLayer', 'MLPLayer', 'FeatureSelector',
]

class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu