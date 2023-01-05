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
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.tensor_names = tensor_names

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = {}
        for k in range(len(self.tensor_names)):
            batch.update({self.tensor_names[k]: self.tensors[k][self.i:self.i+self.batch_size]})
        self.i += self.batch_size
        return batch
        

    def __len__(self):
        return self.n_batches


'''standardize_dataset function is from utils_jared.py'''
def standardize_dataset(dataset, offset, scale):
    norm_ds = copy.deepcopy(dataset)
    norm_ds['x'] = (norm_ds['x'] - offset) / scale
    return norm_ds


'''load_datasets function is from utils_jared.py'''
def load_datasets(dataset_file):
    datasets = defaultdict(dict)
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets

def load_cox_gaussian_data():
    dataset_file = os.path.join(os.path.dirname(__file__), 
        'datasets/gaussian_survival_data.h5')
    datasets = defaultdict(dict)
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets

def prepare_data(x, label):
    if isinstance(label, dict):
       e, t = label['e'], label['t']

    # Sort training data for accurate partial likelihood calculation.
    sort_idx = np.argsort(t)[::-1]
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    #return x, {'e': e, 't': t} this is for parse_data(x, label); see the third line in the parse_data function. 
    #return {'x': x, 'e': e, 't': t}
    return x, e, t

def probe_infnan(v, name, extras={}):
    nps = torch.isnan(v)
    s = nps.sum().item()
    if s > 