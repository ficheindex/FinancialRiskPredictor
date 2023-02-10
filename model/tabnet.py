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
         