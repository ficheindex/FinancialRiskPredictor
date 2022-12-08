# -*- coding:utf-8 -*-

import os
import six
import copy
import collections
from collections import defaultdict

import h5py
import numpy as np
# from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

SKIP_TYPES = six.string_types


class SimpleDataset(Dataset):
    """
    Assuming X and y are numpy arrays and
     with X.shape = (n_samples, n_features)
          y.shape = (n_samples,)
    """
    def __init__(self, X, y=None):
        self.X = X
   