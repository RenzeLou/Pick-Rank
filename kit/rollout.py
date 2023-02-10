import logging
import math
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import wandb
from tqdm import tqdm
import copy
import random
import torch
import string

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets.utils import set_progress_bar_enabled
from datasets import load_dataset, load_metric

import transformers
# from transformers.utils import logging as hug_logging
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
# from ni_collator_augment_v2 import DataCollatorForNI
from ni_collator_adv_train import DataCollatorForNI
# from ni_trainer_augment_v2 import NITrainer, DenserEvalCallback
from ni_trainer_adv_train import NITrainer, DenserEvalCallback
from compute_metrics import compute_grouped_metrics_add_f, compute_grouped_metrics_add_f_v2, compute_metrics, compute_grouped_metrics, compute_metrics_add_f
from modeling_t5_adv_train import T5ForConditionalGeneration_neg


def rollout_mc_search(model:AutoModelForSeq2SeqLM,tokenizer:AutoTokenizer,batch_inputs:dict,
                      current_action:torch.tensor,max_seq_len:int,multinomial:bool=True,exp:bool=True):
    ''' retuen a tensor of searched sequence [batch_size, max_seq_len] 
    batch_inputs: {input_ids,attention_mask,decoder_input_ids,labels}
    current_action: [batch_size, time_step]
    '''
    softmax = torch.nn.Softmax(dim=2)
    while current_action.shape[1] < max_seq_len:
        forward_kwargs = {
            "input_ids" : batch_inputs["input_ids"],
            "attention_mask" : batch_inputs["attention_mask"],
        }
        # if any((ids not in range(len(tokenizer)) for ids in current_action[:,-1].detach().cpu().tolist())):
        #     wait = True
        # there could be some ids out of vocab. e.g., -100
        zeros = torch.zeros_like(current_action)
        min_ids, max_ids = 0, len(tokenizer)-1
        # make sure the ids are in the range of vocab
        current_action = torch.where(torch.lt(current_action,min_ids) | torch.gt(current_action,max_ids),zeros,current_action)
        outputs = model(**forward_kwargs,decoder_input_ids=current_action)
        # outputs = model(**forward_kwargs,decoder_input_ids=current_action)
        prob = softmax(outputs.logits)  ## [batch_size, time_step, vocab_size]
        next_token_prob = prob[:, -1:, :].squeeze(1)  ## [batch_size, vocab_size]
        next_token = torch.multinomial(next_token_prob, 1) if not exp else torch.multinomial(torch.exp(next_token_prob), 1) ## [batch_size, 1]
        # next_token_lis = next_token.squeeze().detach().cpu().tolist()
        # if any((ids not in range(len(tokenizer)) for ids in next_token_lis)):
        #     wait = True
        current_action = torch.cat([current_action,next_token],dim=1)  ## [batch_size, time_step + 1]
    return current_action # [batch_size, max_seq_len]