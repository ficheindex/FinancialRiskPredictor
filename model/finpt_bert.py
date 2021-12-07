
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, \
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class FinptBertForSequenceClassification(BertPreTrainedModel):
    _CHECKPOINT_FOR_DOC = "bert-base-uncased"
    _CONFIG_FOR_DOC = "BertConfig"

    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
    _SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
    _SEQ_CLASS_EXPECTED_LOSS = 0.01

    _keys_to_ignore_on_load_missing = [r"classifier.weight"]

    def __init__(self, config):
        super().__init__(config)