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
            self.model_semi = nn.DataParallel(self.model_semi, device_ids=self.args.gpu_ids)

        self.params = dict({
            "p_m": 0.113,
            "K": 15,
            "alpha": 9.83,
            "beta": 0.313,
        })

        print("On Device:", self.device)

        self.encoder_layer = None

    def fit(self, X, y, X_val=None, y_val=None, optimizer=None, criterion=None):
        X = np.array(X, dtype=np.float)
        X_val = np.array(X_val, dtype=np.float)

        X_unlab = np.concatenate([X, X_val], axis=0)

        self.fit_self(X_unlab, p_m=self.params["p_m"], alpha=self.params["alpha"])

        if self.args.data_parallel:
            self.encoder_layer = self.model_self.module.input_layer
        else:
            self.encoder_layer = self.model_self.input_layer

        loss_history, val_loss_history = self.fit_semi(
            X, y, X, X_val, y_val, p_m=self.params["p_m"], K=self.params["K"], beta=self.params["beta"])

        self.load_model(filename_extension="best", directory="tmp")
        return loss_history, val_loss_