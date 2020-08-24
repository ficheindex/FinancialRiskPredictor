# -*- coding:utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.deepfm_lib.layers.activation import activation_layer


class LocalActivationUnit(nn.Module):
    """The LocalActivationUnit used in DIN with which the representation of
        user interests varies a