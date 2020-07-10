
# -*- coding: utf-8 -*-

from abc import ABC
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model.basemodel import BaseModel
from utils.io_utils import get_output_path


class BaseModelTorch(BaseModel, ABC):

    def __init__(self, params=None, args=None):
        super().__init__(params, args)
        self.device = self.get_device()
        self.gpus = args.cuda if args.cuda != "cpu" and torch.cuda.is_available() and args.data_parallel else None

    def to_device(self):
        if self.args.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        print("On Device:", self.device)
        self.model.to(self.device)

    def get_device(self):
        if hasattr(self.args, "device"):
            return torch.device(self.args.device)

        if self.args.cuda != "cpu" and torch.cuda.is_available():
            if self.args.data_parallel:
                device = "cuda"  # + "".join(str(i) + "," for i in self.args.gpu_ids)[:-1]
            else:
                device = "cuda"
        else:
            device = "cpu"

        return torch.device(device)

    def fit(self, X, y, X_val=None, y_val=None, optimizer=None, criterion=None):
        if optimizer is None:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)

        X = torch.tensor(X).float().to(self.device)
        X_val = torch.tensor(X_val).float().to(self.device)

        y = torch.tensor(y).to(self.device)
        y_val = torch.tensor(y_val).to(self.device)

        # if self.args.objective == "regression":
        #     loss_func = nn.MSELoss()
        #     y = y.float()
        #     y_val = y_val.float()
        # elif self.args.objective == "classification":