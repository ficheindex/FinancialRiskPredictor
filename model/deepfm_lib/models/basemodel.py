# -*- coding:utf-8 -*-

from __future__ import print_function

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from model.deepfm_lib.inputs import build_input_features, SparseFeat, DenseFeat, \
    VarLenSparseFeat, get_varlen_pooling_list, create_embedding_matrix,