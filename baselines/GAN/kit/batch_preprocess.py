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

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )

def pad_tensors_to_max_len(tensor, max_length,tokenizer:AutoTokenizer=None):
    if tokenizer is not None and hasattr(tokenizer, "pad_token_id"):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    else:
        raise NotImplementedError
        # if self.model.config.pad_token_id is not None:
        #     pad_token_id = self.model.config.pad_token_id
        # else:
        #     raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

    padded_tensor = pad_token_id * torch.ones(
        (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor

def construct_nli_instance(definition,ins_list:list,label:int)->list:
    ''' return a list containing [def, (x,y) pair, label] '''
    results = []
    for ins in ins_list:
        sen1 = f'The task definition:\n{definition}'
        sen2 = f'The input:\n{ins["input"]}\n\nThe output:\n{ins["output"][0]}'
        results.append((sen1,sen2,label))
    
    return results

def construct_ni_input(definition:str,input:str):
    def construct_mul_def(definition:str):
        definition = "Definition: " + definition.strip()
        if not definition[-1] in string.punctuation:
            definition += "."
        definition += "\n\n"
        
        return definition
    definition = construct_mul_def(definition)
    task_input = ""
    # add the input first.
    task_input += "Now complete the following example -\n"
    task_input += f"Input: {input.strip()}"
    if not task_input[-1] in string.punctuation:
        task_input += "."
    task_input += "\n"
    task_input += "Output: "
    
    source = definition + task_input
    return source


def feature2input_G(batch_features:list,tokenizer:AutoTokenizer,device,prepare_decoder_input_ids_from_labels,
                    max_source_length=1024,max_target_length=128,padding="longest",pad_to_multiple_of=None,
                    label_pad_token_id=-100)->dict:
    ## batch_features: [{def,x,y,label}]
    ## try to make generator predict the y_, namely (def,x) -> (y_)
    ## use this func to convert ori data into tensor
    ## return {input_ids,attention_mask,decoder_input_ids,labels}
    ## note that this function can only be used for training, not for eval, since we randomly chose one y from the list
    if not isinstance(batch_features[0],dict):
        batch_features = batch_features[0]
    sources,labels = [],[]
    for item_ in batch_features:
        item = copy.deepcopy(item_)
        source = construct_ni_input(item["def"],item["x"])
        tokenized_source = tokenizer(source)["input_ids"]
        if isinstance(item["y"],str):
            labels.append(item["y"])
        elif isinstance(item["y"],list):
            labels.append(random.choice(item["y"]))
        else:
            raise RuntimeError("y should be str or list")
        # ensure the input length is not too along after encoding
        # each element in the sources is a string 
        if len(tokenized_source) <= max_source_length:
            sources.append(source)
        else:
            sources.append(tokenizer.decode(tokenized_source[:max_source_length], skip_special_tokens=True))

    model_inputs = tokenizer(
                sources, 
                max_length=max_source_length, 
                padding=padding,
                return_tensors="pt", 
                truncation=True,
                pad_to_multiple_of=pad_to_multiple_of)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            labels,
            max_length=max_target_length,
            padding=padding,
            return_tensors="pt",
            truncation=True,
            pad_to_multiple_of=pad_to_multiple_of
        )
    label_mask = labels["attention_mask"].bool()
    model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, label_pad_token_id)
    
    decoder_input_ids = prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
    model_inputs["decoder_input_ids"] = decoder_input_ids
    
    # move to device
    model_inputs = dict([(k,v.to(device)) for k,v in model_inputs.items()])

    return model_inputs

def feature2input_D(batch_features:list,tokenizer:AutoTokenizer,device,add_def:bool=True)->dict:
    ## batch_features: [{def,x,y,label}]
    ## try to make classifier predict the label, namely (def,x,y) -> 0/1
    ## use this func to convert ori data into tensor
    ## return {input_ids,attention_mask,token_type_ids,labels}
    if not isinstance(batch_features[0],dict):
        batch_features = batch_features[0]
    batch_inputs = dict()
    def_lis, xy_lis, label_lis = [],[],[]
    for item_ in batch_features:
        item = copy.deepcopy(item_)
        if add_def:
            sen1 = f'The task definition:\n{item["def"]}'
            sen2 = f'The input:\n{item["x"]}\n\nThe output:\n{item["y"]}'
        else:
            sen1 = f'The input:\n{item["x"]}'
            sen2 = f'The output:\n{item["y"]}'
        def_lis.append(sen1)
        xy_lis.append(sen2)
        label_lis.append(item["label"])
    # inputs = tokenizer(*(def_lis,xy_lis),truncation=True,padding=True,return_tensors="pt")
    inputs = tokenizer(*(def_lis,xy_lis),truncation=True,padding='longest',return_tensors="pt")
    batch_inputs["input_ids"] = inputs.input_ids.to(device)
    batch_inputs["attention_mask"] = inputs.attention_mask.to(device)
    try:
        # for roberta, there is no token_type_ids
        batch_inputs["token_type_ids"] = inputs.token_type_ids.to(device)
    except AttributeError:
        # do not do anything
        roberta = True
        
    batch_inputs["labels"] = torch.LongTensor(label_lis).to(device)

    return batch_inputs