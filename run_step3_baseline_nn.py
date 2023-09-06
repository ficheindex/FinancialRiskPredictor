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

    args.device = device
    args.num_classes = train_set[0]["num_classes"]  # int (The total number of classes)
    args.num_features = train_set[0]["num_features"]  # int (The total number of features)
    args.num_idx = train_set[0]["num_idx"]  # List[int] (The indices of the numerical datatype columns)
    args.cat_idx = train_set[0]["cat_idx"]  # List[int] (The indices of the categorical datatype columns)
    args.cat_dim = train_set[0]["cat_dim"]  # List[int] (The dimension of each categorical column)
    args.cat_str = train_set[0]["cat_str"]  # List[List[str]] (The category names of categorical columns)
    args.col_name = train_set[0]["col_name"]  # List[str] (The name of each column)

    x_key = "X_ml"
    # x_key = "X_ml_unscale"
    train_X_ml, train_y = np.asarray(train_set[x_key], dtype=np.float32), np.asarray(train_set["y"], dtype=np.int64)
    val_X_ml, val_y = np.asarray(val_set[x_key], dtype=np.float32), np.asarray(val_set["y"], dtype=np.int64)
    test_X_ml, test_y = np.asarray(test_set[x_key], dtype=np.float32), np.asarray(test_set["y"], dtype=np.int64)

    assert isinstance(criterion_name, str) and criterion_name in CRITERION_DICT
    criterion = CRITERION_DICT[criterion_name]()
    logger.info(criterion)

    model = MODEL_DICT[cur_model_name](args=args)
    # model = model.to(device=device)

    eval_results_train[cur_ds_name] = []
    eval_results_val[cur_ds_name] = []
    eval_results_test[cur_ds_name] = []

    model.fit(X=train_X_ml, y=train_y, X_val=val_X_ml, y_val=val_y, optimizer=None, criterion=criterion)

    train_eval_res = model.evaluate(X=train_X_ml, y=train_y)
    val_eval_res = model.evaluate(X=val_X_ml, y=val_y)
    test_eval_res = model.evaluate(X=test_X_ml, y=test_y)

    logger.info(f">>> Dataset = {cur_ds_name}; Model = {cur_model_name}")
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Training set): {train_eval_res}")
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Validation set): {val_eval_res}")
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Test set): {test_eval_res}")

    del data
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Baseline-NN args")

    parser.add_argument("--cuda", type=str, default="cpu", help="Specify which device to use")
    parser.add_argument("--seed", type=int, default=0, help="Seed of random modules")
    parser.add_argument("--task", type=str, default="binary", help="binary/classification OR regression")
    parser.add_argument("--ds_name", type=str, default="cd1", help="Specify which dataset to use.",
                        choices=["cd1