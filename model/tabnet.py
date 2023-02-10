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
        self.params["cat_dims"] = arg