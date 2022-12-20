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
        self.y = y
    
    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i]
        data = np.array(data).astype(np.float32)
        if self.y is not None:
            return dict(input=data, label=self.y[i])
        else:
            return dict(input=data)


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, tensor_names, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param tensor_names: name of tensors (for feed_dict)
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle