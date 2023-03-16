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
        return loss_history, val_loss_history

    def predict_helper(self, X):
        self.model_self.eval()
        self.model_semi.eval()

        X = np.array(X, dtype=np.float)
        X = torch.tensor(X).float()

        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                 num_workers=2)

        predictions = []

        with torch.no_grad():
            for batch_X in test_loader:
                X_encoded = self.encoder_layer(batch_X[0].to(self.device))
                preds = self.model_semi(X_encoded)

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "p_m": trial.suggest_float("p_m", 0.1, 0.9),
            "alpha": trial.suggest_float("alpha", 0.1, 10),
            "K": trial.suggest_categorical("K", [2, 3, 5, 10, 15, 20]),
            "beta": trial.suggest_float("beta", 0.1, 10),
        }
        return params

    def fit_self(self, X, p_m=0.3, alpha=2):
        optimizer = optim.RMSprop(self.model_self.p