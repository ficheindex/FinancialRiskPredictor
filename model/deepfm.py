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

    def __init__(self, params=None, args=None):
        super().__init__(params, args)

        self.params = dict({
            "dnn_dropout": 0.4
        })

        # if args.objective == "classification":
        #     print("DeepFM not yet implemented for classification")
        #     import sys
        #     sys.exit()

        if args.cat_idx:
            dense_features = list(set(range(args.num_features)) - set(args.cat_idx))
            fixlen_feature_columns = [SparseFeat(str(feat), args.cat_dim[idx])
                                      for idx, feat in enumerate(args.cat_idx)] + \
                                     [DenseFeat(str(feat), 1, ) for feat in dense_features]

        else:
            # Add dummy sparse feature, otherwise it will crash...
            fixlen_feature_columns = [SparseFeat("dummy", 1)] + \
                                     [De