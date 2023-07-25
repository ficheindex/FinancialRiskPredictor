#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import logging
import argparse
import gc

import numpy as np

import torch
from datasets import load_dataset

from model import *
from criterion import *
from utils.seed import set_seed


def run_baselines_nn(cur_ds_name, cur_model_name):
    logger.info(f"\n\n\n *** *** cur_ds_name: {cur_ds_name}; cur_model_name: {cur_model_name}")
    data = load_dataset("yuweiyin/FinBench", cur_ds_name, cache_dir=cache_ds)
    train_set = data["train"] if "train" in data else []
    val_set = data["validation"] if "train" in data else []
    test_set = data["test"] if "train" in data else []

    args.dev