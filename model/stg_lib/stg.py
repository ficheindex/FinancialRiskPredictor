# -*- coding:utf-8 -*-

import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.stg_lib.models import STGClassificationModel, STGRegressionModel, \
    MLPClassificationModel, MLPRegressionModel, STGCoxModel, MLPCoxModel, L1RegressionModel, \
    SoftThreshRegressionModel, L1GateRegressionModel
from model.stg_lib.utils import get_optimizer, as_tensor, as_float, as_numpy, as_cpu, SimpleDataset, \
    FastTensorDataLoader, probe_infnan
from model.stg_lib.io import load_state_dict, state_dict
from model.stg_lib.meter import GroupMeters
from model.stg_lib.losses import calc_concordance_index, PartialLogLikelihood

import logging

# import logging.config

logger = logging.getLogger("my-logger")

__all__ = ['STG']


def _standard_truncnorm_sample(lower_bound, upper_bound, sample_shape=torch.Size()):
    r"""
    Implements accept-reject algorithm for doubly truncated standard normal distribution.
    (Section 2.2. Two-sided truncated normal distribution in [1])
    [1] Robert, Christian P. "Simulation of truncated normal variables." Statistics and computing 5.2 (1995): 121-125.
    Available online: https://arxiv.org/abs/0907.4010
    Args:
        lower_bound (Tensor): lower bound for standard normal distribution. Best to keep it greater than -4.0 for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than 4.0 for
        stable results
    """
    x = torch.randn(sample_shape)
    done = torch.zeros(sample_shape).byte()
    while not done.all():
        proposed_x = lower_bound + torch.rand(sample_shape) * (upper_bound - lower_bound)
        if (upper_bound * lower_bound).lt(0.0):  # of opposite sign
            log_prob_accept = -0.5 * proposed_x ** 2
        elif upper_bound < 0.0:  # both negative
            log_prob_accept = 0.5 * (upper_bound ** 2 - proposed_x ** 2)
        else:  # both positive
            assert (lower_bound.gt(0.0))
            log_prob_accept = 0.5 * (lower_bound ** 2 - proposed_x ** 2)
        prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        accept = torch.bernoulli(prob_accept).byte() & ~done
        if accept.any():
            accept = accept.bool()
            x[accept] = proposed_x[accept]
            accept = accept.byte()
            done |= accept
    return x


class STG(object):
    def __init__(self, device, input_dim=784, output_dim=10, hidden_dims=[400, 200],
                 activation='relu', sigma=0.5, lam=0.1,
                 optimizer='Adam', learning_rate=1e-5, batch_size=100, freeze_onward=None,
                 feature_selection=True, weight_decay=1e-3,
                 task_type='classification', report_maps=False, random_state=1, extra_args=None):
        self.batch_size = batch_size
        self.activation = activation
        self.device = device  # self.get_device(device)
        self.report_maps = report_maps
        self.random_state = random_state
        self.task_type = task_type
        self.extra_args = extra_args
        self.freeze_onward = freeze_onward
        self._model = self.build_model(input_dim, output_dim, hidden_dims, activation, sigma, lam,
                                       task_type, feature_selection)
        self._model.apply(self.init_weights)
        self._model = self._model.to(device)
        self._optimizer = get_optimizer(optimizer, self._model, lr=learning_rate, weight_decay=weight_decay)

    def get_device(self, device):
        if device == "cpu":
            device = torch.device("cpu")
        elif device is None:
            args_cuda = torch.cuda.is_available()
            device = device = torch.device("cuda" if args_cuda else "cpu")
        else:
            raise NotImplementedError("Only 'cpu' or 'cuda' is a valid option.")
        return device

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            stddev = torch.tensor(0.1)
            shape = m.weight.shape
            m.weight = nn.Parameter(_standard_truncnorm_sample(lower_bound=-2 * stddev, upper_bound=2 * stddev,
                                                               sample_shape=shape))
            torch.nn.init.zeros_(m.bias)

    def build_model(self, input_dim, output_dim, hidden_dims, activation, sigma, lam, task_ty