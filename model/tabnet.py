# -*- coding: utf-8 -*-

from abc import ABC
import numpy as np
import torch

from model.basemodel_torch import BaseModelTorch
from utils.io_utils import save_model_to_file, load_model_from_file

from pytorch_tabnet.tab_model import TabNetCla