# -*- coding: utf-8 -*-

from abc import ABC
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model.basemodel_torch import BaseModelTorch
from utils.io_utils import get_output_path

"""
VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain

Paper: https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf
Code: https://github.com/jsyoon0823/VIME
"""


class VIME(BaseModelTorch, ABC):

    def __init__(self, params=None, args=None):
        super().__init__(params, args)

        self.device = args.device
        self.model_self = VIMESelf(args.num_features).to(self.device)
        self.model_semi = VIMESemi(args, args.num_features, args.num_classes).to(self.device)

        if self.args.data_parallel:
            self.model_self = nn.DataParallel(self.model_self, device_ids=self.args.gpu_ids)
            self.model_