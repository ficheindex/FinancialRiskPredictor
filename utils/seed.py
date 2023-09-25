#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import random
import numpy as np

import torch
from torch.backends import cudnn


def set_seed(seed: int):
    r"""
    Sets the seed for generating random numbers in Py