# -*- coding: utf-8 -*-

from abc import ABC
import torch

from utils.io_utils import get_output_path
from model.basemodel_torch import BaseModelTorch

from model.stg_lib import STG as STGModel

"""
Feature Selection using Stochastic Gates

Paper: https://arxiv.org/pdf/1810.04247.pdf
Code: https://github.com/runopti/stg
"""


class STG(BaseModelTorch, ABC):

    def __init__(self, params=None, args=None):
        super().__init__(params, args)

        self.params = dict({
            "learning_rate": 4.64e-3,
            "lam": 6.2e-3,
            "hidden_dims": [500, 50, 10]
        })

        task = "classification" if self.args.objective == "binary" else self.args.objective
        out_dim = 2 if self.args.objective == "binary" else self.args.num_classes

        self.device = args.device
        self.model = STGModel(device=self.device, task_type=task, input_dim=self.args.num_features,
                              output_dim=out_dim, activation="tanh", sigma=0.5,
                              optimizer="SGD", feature_selection=True, random_state=1,
                              batch_size=self.args.bsz, **self.params)
        # self.model = STGModel(device=self.device, task_type=task, input_dim=self.args.num_features,
        #                       output_dim=out_dim, activation="tanh", sigma=0.5,
        #                       optimizer="SGD", feature_selection=True, random_state=1,
        #                       batch_size=self.args.bsz)  # hidden_dims=[500, 50, 10],

    def fit(self, X, y, X_val=None, y_val=None, optimizer=None, criterion=None):
        X, X_val = X.astype("float"), X_val.astype("float")

        if self.args.objective == "regression":
            y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)

        # self.args.logging_period # early_stop=True
        loss, val_loss = self.model.fit(
            X, y, nr_epochs=self.args.epoch, valid_X=X_val, valid_y=y_val,
            print_interval=100,  # original: 1
        )

        return loss, val_loss

    def predict_helper(self, X):
        return self.model.predict(X)

    def save_model(s