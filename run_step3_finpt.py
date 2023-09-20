
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
                f">>> len(fin_text_test) = {len(fin_text_test)}; len(label_test) = {len(label_test)}")

    fin_text_train_len = [len(tokenizer.encode(text)) for text in fin_text_train]
    fin_text_val_len = [len(tokenizer.encode(text)) for text in fin_text_validation]
    fin_text_test_len = [len(tokenizer.encode(text)) for text in fin_text_test]
    assert len(fin_text_train_len) > 0 and len(fin_text_val_len) > 0 and len(fin_text_test_len) > 0
    fin_text_train_len_avg = sum(fin_text_train_len) / len(fin_text_train_len)
    fin_text_val_len_avg = sum(fin_text_val_len) / len(fin_text_val_len)
    fin_text_test_len_avg = sum(fin_text_test_len) / len(fin_text_test_len)
    logger.info(f">>> fin_text_train_len_avg = {fin_text_train_len_avg}")
    logger.info(f">>> fin_text_val_len_avg = {fin_text_val_len_avg}")
    logger.info(f">>> fin_text_test_len_avg = {fin_text_test_len_avg}")

    if isinstance(fix_seq_len, int) and fix_seq_len > 0:
        seq_len = fix_seq_len
    else:
        if fin_text_train_len_avg <= 128.0:
            seq_len = 128
        elif fin_text_train_len_avg <= 256.0:
            seq_len = 256
            bsz = int(bsz / 2)
        else:
            seq_len = 512
            bsz = int(bsz / 4)
        if bsz <= 1:
            bsz = 1
    logger.info(f">>> seq_len = {seq_len}; bsz = {bsz}")

    label_train_text = ["Yes" if label == 1 else "No" for label in label_train]
    label_validation_text = ["Yes" if label == 1 else "No" for label in label_validation]
    label_test_text = ["Yes" if label == 1 else "No" for label in label_test]

    fin_text_train = [f"{t_in} {tokenizer.eos_token}" for t_in in fin_text_train]
    fin_text_validation = [f"{t_in} {tokenizer.eos_token}" for t_in in fin_text_validation]
    fin_text_test = [f"{t_in} {tokenizer.eos_token}" for t_in in fin_text_test]
    ds_train = {"sentence": fin_text_train, "labels": label_train, "labels_text": label_train_text, }
    ds_val = {"sentence": fin_text_validation, "labels": label_validation, "labels_text": label_validation_text, }
    ds_test = {"sentence": fin_text_test, "labels": label_test, "labels_text": label_test_text, }

    ds_train = Dataset.from_dict(ds_train)
    ds_val = Dataset.from_dict(ds_val)
    ds_test = Dataset.from_dict(ds_test)

    ds_train = ds_train.map(
        lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length", max_length=seq_len),
        remove_columns=["sentence"], batched=True, num_proc=4)
    ds_val = ds_val.map(
        lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length", max_length=seq_len),
        remove_columns=["sentence"], batched=True, num_proc=4)
    ds_test = ds_test.map(
        lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length", max_length=seq_len),
        batched=True, num_proc=4)

    ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    ds_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    ds_test.set_format(type="torch", columns=["input_ids", "attention_mask", "sentence", "labels", "labels_text"])

    return ds_train, ds_val, ds_test


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="FinPT args")
    parser.add_argument("--cuda", type=str, default="cpu", help="Specify which device to use")
    parser.add_argument("--seed", type=int, default=0, help="Seed of random modules")
    parser.add_argument("--ds_name", type=str, default="cd1", help="Specify which dataset to use.",
                        choices=["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"])
    parser.add_argument("--model_name", type=str, default="gpt2", help="Specify which model to use.",
                        choices=["bert", "finbert", "gpt2", "t5-base", "flan-t5-base",
                                 "t5-xxl", "flan-t5-xxl", "llama-7b", "llama-13b"])
    parser.add_argument("--bsz", type=int, default=128, help="TrainingArguments: per_device_train/eval_batch_size")
    parser.add_argument("--epoch", type=int, default=100, help="TrainingArguments: num_train_epochs")
    parser.add_argument("--fp8", action="store_true", help="Using 8bit precision")
    parser.add_argument("--fp16", action="store_true", help="Using fp16 precision")
    parser.add_argument("--bf16", action="store_true", help="Using bf16/mixture precision")
    parser.add_argument("--freeze", action="store_true", help="Freeze all params except the last nn.Linear head")
    parser.add_argument("--use_pos_weight", action="store_true", help="Use pos_weight for unbalanced binary-cls data")
    parser.add_argument("--save_ckpt", action="store_true", help="Save the final model checkpoint")
    args = parser.parse_args()

    logger.info(args)

    cuda = str(args.cuda)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    has_cuda = torch.cuda.is_available()
    cnt_cuda = torch.cuda.device_count()
    device = torch.device("cpu" if not has_cuda else f"cuda")
    logger.info(f"has_cuda: {has_cuda}; cnt_cuda: {cnt_cuda}; device: {device}")

    seed = int(args.seed)
    ds_name = str(args.ds_name)
    model_name = str(args.model_name)
    bsz = int(args.bsz)
    epoch = int(args.epoch)
    fp8 = bool(args.fp8)
    fp16 = bool(args.fp16)
    bf16 = bool(args.bf16)
    freeze = bool(args.freeze)
    use_pos_weight = bool(args.use_pos_weight)
    save_ckpt = bool(args.save_ckpt)

    set_seed(seed)

    # hf_token = "YOUR_ACCESS_TOKENS"  # TODO: https://huggingface.co/settings/tokens
    # login(token=hf_token)

    cache_dir = "~/.cache/huggingface/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cache_model = os.path.join(cache_dir, "models")
    cache_ds = os.path.join(cache_dir, "datasets")

    if model_name == "bert":
        model_class = "bert"
        hf_model_id = "bert-base-cased"
        tokenizer = BertTokenizer.from_pretrained(hf_model_id, cache_dir=cache_model)
        model = FinptBertForSequenceClassification.from_pretrained(
            hf_model_id, num_labels=2, cache_dir=cache_model)
        fp8 = False
        freeze = False
    elif model_name == "finbert":
        model_class = "bert"
        hf_model_id = "yiyanghkust/finbert-pretrain"
        tokenizer = BertTokenizer.from_pretrained(hf_model_id, cache_dir=cache_model)
        model = FinptBertForSequenceClassification.from_pretrained(
            hf_model_id, num_labels=2, cache_dir=cache_model)
        fp8 = False
        freeze = False
    elif model_name == "gpt2":
        model_class = "gpt"
        hf_model_id = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(hf_model_id, cache_dir=cache_model)
        model = FinptGPT2ForSequenceClassification.from_pretrained(
            hf_model_id, cache_dir=cache_model, load_in_8bit=fp8)
        freeze = False
    elif model_name == "t5-base":
        model_class = "t5"
        hf_model_id = "t5-base"
        tokenizer = T5Tokenizer.from_pretrained(hf_model_id, cache_dir=cache_model)
        model = FinptT5ForSequenceClassification.from_pretrained(
            hf_model_id, cache_dir=cache_model, load_in_8bit=fp8)
        freeze = False
    elif model_name == "flan-t5-base":
        model_class = "t5"
        hf_model_id = "google/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(hf_model_id, cache_dir=cache_model)
        model = FinptT5ForSequenceClassification.from_pretrained(
            hf_model_id, cache_dir=cache_model, load_in_8bit=fp8)
        freeze = False
    elif model_name == "t5-xxl":
        model_class = "t5"
        hf_model_id = "t5-11b"
        tokenizer = T5Tokenizer.from_pretrained(hf_model_id, cache_dir=cache_model)
        model = FinptT5ForSequenceClassification.from_pretrained(
            hf_model_id, cache_dir=cache_model, load_in_8bit=fp8)
        freeze = True
    elif model_name == "flan-t5-xxl":
        model_class = "t5"
        hf_model_id = "google/flan-t5-xxl"
        tokenizer = T5Tokenizer.from_pretrained(hf_model_id, cache_dir=cache_model)
        model = FinptT5ForSequenceClassification.from_pretrained(
            hf_model_id, cache_dir=cache_model, load_in_8bit=fp8)
        freeze = True
    elif model_name == "llama-7b":
        model_class = "llama"
        hf_model_id = "openlm-research/open_llama_7b"
        tokenizer = LlamaTokenizer.from_pretrained(hf_model_id, cache_dir=cache_model)
        model = FinptLlamaForSequenceClassification.from_pretrained(
            hf_model_id, cache_dir=cache_model, load_in_8bit=fp8)
        freeze = True
    elif model_name == "llama-13b":
        model_class = "llama"
        hf_model_id = "openlm-research/open_llama_13b"
        tokenizer = LlamaTokenizer.from_pretrained(hf_model_id, cache_dir=cache_model)
        model = FinptLlamaForSequenceClassification.from_pretrained(
            hf_model_id, cache_dir=cache_model, load_in_8bit=fp8)
        freeze = True
    else:
        raise ValueError(f">>> ValueError: model_name = {model_name}")

    if not fp8:
        if fp16:
            model = model.to(device=device, dtype=torch.float16)
        else:
            model = model.to(device=device)

    logger.info(f">>> tokenizer.all_special_tokens (before): {tokenizer.all_special_tokens}")
    if model_class == "bert" and tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
    if tokenizer.cls_token is None:
        tokenizer.cls_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.mask_token = tokenizer.eos_token
    logger.info(f">>> tokenizer.all_special_tokens (after): {tokenizer.all_special_tokens}")

    dataset_train, dataset_val, dataset_test = get_finpt_data(cur_ds_name=ds_name)

    # get pos ratio of the training set for loss computing
    dataset_train_y = dataset_train["labels"]
    total_y = float(len(dataset_train_y))
    pos_y = float(sum(dataset_train_y))
    assert total_y >= pos_y > 0.0
    neg_to_pos = float((total_y - pos_y) / pos_y)
    pos_ratio = float(pos_y / total_y)
    logger.info(f">>> pos_ratio = {pos_ratio}; neg_to_pos = {neg_to_pos}")
