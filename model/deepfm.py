# -*- coding: utf-8 -*-

from abc import ABC
import numpy as np
import torch

from model.basemodel_torch import BaseModelTorch
from model.deepfm_lib.models.deepfm import DeepFM as DeepFMModel
from model.deepfm_lib.inputs import SparseFeat, DenseFeat

"""
DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

Paper: https://www.ijcai.org/proceedings/2017/0239.pdf
Code: https://github.com/shenweichen/DeepCTR-Torch
"""


class DeepFM(BaseModelTorch, ABC):

    def __init__(self, para