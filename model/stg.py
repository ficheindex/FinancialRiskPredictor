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
        s