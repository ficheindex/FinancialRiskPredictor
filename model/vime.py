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
        optimizer = optim.RMSprop(self.model_self.parameters(), lr=0.001)
        loss_func_mask = nn.BCELoss()
        loss_func_feat = nn.MSELoss()

        m_unlab = mask_generator(p_m, X)
        m_label, x_tilde = pretext_generator(m_unlab, X)

        x_tilde = torch.tensor(x_tilde).float()
        m_label = torch.tensor(m_label).float()
        X = torch.tensor(X).float()
        train_dataset = TensorDataset(x_tilde, m_label, X)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.bsz, shuffle=True, num_workers=2)

        for epoch in range(self.args.epoch):
            for batch_X, batch_mask, batch_feat in train_loader:
                out_mask, out_feat = self.model_self(batch_X.to(self.device))

                loss_mask = loss_func_mask(out_mask, batch_mask.to(self.device))
                loss_feat = loss_func_feat(out_feat, batch_feat.to(self.device))

                loss = loss_mask + loss_feat * alpha

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("Fitted encoder")

    def fit_semi(self, X, y, x_unlab, X_val=None, y_val=None, p_m=0.3, K=3, beta=1):
        X = torch.tensor(X).float().to(self.device)
        y = torch.tensor(y).to(self.device)
        x_unlab = torch.tensor(x_unlab).float().to(self.device)

        X_val = torch.tensor(X_val).float().to(self.device)
        y_val = torch.tensor(y_val).to(self.device)

        # if self.args.objective == "regression":
        #     loss_func_supervised = nn.MSELoss()
        #     y = y.float()
        #     y_val = y_val.float()
        # elif self.args.objective == "classification":
        #     loss_func_supervised = nn.CrossEntropyLoss()
        # else:
        #     loss_func_supervised = nn.BCEWithLogitsLoss()
        #     y = y.float()
        #     y_val = y_val.float()

        if self.args.objective == "regression":
            loss_func_supervised = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        else:
            loss_func_supervised = nn.CrossEntropyLoss()
            y = y.long()
            y_val = y_val.long()

        optimizer = optim.AdamW(self.model_semi.parameters())

        train_dataset = TensorDataset(X, y, x_unlab)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.bsz, shuffle=True, num_workers=2,
                                  drop_last=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.bsz, shuffle=False)

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epoch):
            for i, (batch_X, batch_y, batch_unlab) in enumerate(train_loader):

                batch_X_encoded = self.encoder_layer(batch_X.to(self.device))
                y_hat = self.model_semi(batch_X_encoded)

                yv_hats = torch.empty(K, self.args.bsz, self.args.num_classes)
                for rep in range(K):
                    m_batch = mask_generator(p_m, batch_unlab)
                    _, batch_unlab_tmp = pretext_generator(m_batch, batch_unlab)

                    batch_unlab_encoded = self.encoder_layer(batch_unlab_tmp.float().to(self.device))
                    yv_hat = self.model_semi(batch_unlab_encoded)
                    yv_hats[rep] = yv_hat

                if self.args.objective == "regression" or self.args.objective == "binary":
                    y_hat = y_hat.squeeze()

                y_loss = loss_func_supervised(y_hat, batch_y.to(self.device))
                yu_loss = torch.mean(torch.var(yv_hats, dim=0))
                loss = y_loss + beta * yu_loss
                lo