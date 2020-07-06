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
        self.args = args
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        r