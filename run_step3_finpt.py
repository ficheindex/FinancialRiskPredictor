
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
import wandb

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login

from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer, LlamaTokenizer

from model.finpt_bert import FinptBertForSequenceClassification
from model.finpt_llama import FinptLlamaForSequenceClassification
from model.finpt_t5 import FinptT5ForSequenceClassification
from model.finpt_gpt import FinptGPT2ForSequenceClassification
from utils.seed import set_seed


def get_finpt_data(cur_ds_name: str, fix_seq_len: int = None):
    global bsz

    data = load_dataset("yuweiyin/FinBench", cur_ds_name, cache_dir=cache_ds)
    fin_text_train = data["train"]["X_profile"]
    fin_text_validation = data["validation"]["X_profile"]
    fin_text_test =  data["test"]["X_profile"]

    label_train = data["train"]["y"]
    label_validation = data["validation"]["y"]
    label_test = data["test"]["y"]
    logger.info(f">>> len(fin_text_train) = {len(fin_text_train)}; len(label_train) = {len(label_train)};\n"
                f">>> len(fin_text_val) = {len(fin_text_validation)}; len(label_val) = {len(label_validation)};\n"