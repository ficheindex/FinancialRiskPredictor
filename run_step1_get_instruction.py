#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import json
import logging
import argparse

from datasets import load_dataset


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Step1 Get_Instruction Args")
    args = parser.parse_args()

    logger.info(args)

    cache_dir = "~/.cache/huggingface/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cache_model = os.path.join(cache_dir, "models")
    cache_ds = os.path.join(cache_dir, "datasets")

    profile_root_dir = os.path.join("./data/profile")
    ds_name_list = ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
    for ds_name in ds_name_list:
        logger.info(f"\n\n>>> ds_name: {ds_name}")
        profile_dir = os.path.join(profile_root_dir, ds_name)
        os.makedirs(profile_dir, exi