
# -*- coding:utf-8 -*-

# import math
import numpy as np
import torch.nn as nn
import torch
# from torch.autograd import Variable


from model.stg_lib.layers import MLPLayer, FeatureSelector, GatingLayer
from model.stg_lib.losses import PartialLogLikelihood

__all__ = ["MLPModel", "MLPRegressionModel", "MLPClassificationModel",
           "LinearRegressionModel", "LinearClassificationModel"]


class ModelIOKeysMixin(object):
    def _get_input(self, feed_dict):
        return feed_dict["input"]

    def _get_label(self, feed_dict):
        return feed_dict["label"]

    def _get_covariate(self, feed_dict):
        """For cox"""
        return feed_dict["X"]

    def _get_fail_indicator(self, feed_dict):
        """For cox"""
        return feed_dict["E"].reshape(-1, 1)

    def _get_failure_time(self, feed_dict):
        """For cox"""
        return feed_dict["T"]

    def _compose_output(self, value):
        return dict(pred=value)


class MLPModel(MLPLayer):
    def freeze_weights(self):
        for name, p in self.named_parameters():
            if name != "mu":
                p.requires_grad = False

    def get_gates(self, mode):
        if mode == "raw":
            return self.mu.detach().cpu().numpy()
        elif mode == "prob":
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5)) 
        else:
            raise NotImplementedError()


class L1RegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation="relu",
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = nn.MSELoss()
        self.lam = lam

    def forward(self, feed_dict):
        pred = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            reg = torch.mean(torch.abs(self.mlp[0][0].weight)) 
            total_loss = loss + self.lam * reg
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)
