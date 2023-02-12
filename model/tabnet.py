# -*- coding: utf-8 -*-

from abc import ABC
import numpy as np
import torch

from model.basemodel_torch import BaseModelTorch
from utils.io_utils import save_model_to_file, load_model_from_file

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

"""
TabNet: Attentive Interpretable Tabular Learning

Paper: https://arxiv.org/pdf/1908.07442.pdf
Paper: https://ojs.aaai.org/index.php/AAAI/article/download/16826/16633
Code: https://github.com/dreamquark-ai/tabnet
"""


class TabNet(BaseModelTorch, ABC):

    def __init__(self, params=None, args=None):
        super().__init__(params, args)

        self.params = dict({
            "n_d": 14,
            "n_steps": 3,
            "gamma": 1.76,
            "cat_emb_dim": 2,
            "n_independent": 3,
            "n_shared": 2,
            "momentum": 0.319,
            "mask_type": "entmax"
        })

        # Paper recommends to be n_d and n_a the same
        self.params["n_a"] = self.params["n_d"]

        self.params["cat_idxs"] = args.cat_idx if args.cat_idx else []
        self.params["cat_dims"] = args.cat_dim

        self.params["device_name"] = self.device

        if args.objective == "regression":
            self.model = TabNetRegressor(**self.params)
            self.metric = ["rmse"]
        elif args.objective == "classification" or args.objective == "binary":
            self.model = TabNetClassifier(**self.params)
            self.metric = ["logloss"]

    def fit(self, X, y, X_val=None, y_val=None, optimizer=None, criterion=None):
        if self.args.objective == "regression":
            y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)

        X = X.astype(np.float32)

        self.model.fit(X, y, eval_set=[(X_val, y_val)], eval_name=["eval"], eval_metric=self.metri