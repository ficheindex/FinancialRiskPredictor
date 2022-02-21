
# -*- coding: utf-8 -*-

import numpy as np
from model.base import ModelSML


class ModelLogisticRegression(ModelSML):

    def __init__(self, args,
                 random_state=None, C=None, solver=None, max_iter=None):
        super().__init__(args)

        from sklearn.linear_model import LogisticRegression

        self.random_state = random_state
        self.C = C
        self.solver = solver
        self.max_iter = max_iter

        self.param_grid = {
            "random_state": [0, 1, 42, 1234],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            "max_iter": [100, 1000],
        }