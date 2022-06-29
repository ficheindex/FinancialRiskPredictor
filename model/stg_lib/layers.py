# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn

from model.stg_lib.utils import get_batcnnorm, get_dropout, get_activation

__all__ = [
    'LinearLayer', 'MLPLayer', 'FeatureSelector',
]

class FeatureSelector(nn.Module):
    def __