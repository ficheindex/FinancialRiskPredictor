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