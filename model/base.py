#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

# import torch
import torch.nn as nn
# import torch.nn.functional as F
from sklearn.base import BaseEstimator


class ModelSML(BaseEstimator):

    def __init__(self, args):
        self.a