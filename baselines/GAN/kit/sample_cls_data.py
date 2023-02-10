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
from transformers.trainer import nested_numpify
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
# from ni_collator_augment_v2 import DataCollatorForNI
from ni_collator_adv_train import DataCollatorForNI
# from ni_trainer_augment_v2 import NITrainer, DenserEvalCallback
from ni_trainer_adv_train import NITrainer, DenserEvalCallback
from compute_metrics import compute_grouped_metrics_add_f, compute_grouped_metrics_add_f_v2, compute_metrics, compute_grouped_metrics, compute_metrics_add_f
from modeling_t5_adv_train import T5ForConditionalGeneration_neg
from batch_preprocess import pad_tensors_to_max_len, feature2input_G
from compute_metrics import metric_max_over_ground_truths, exact_match_score,rougeL_score

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
)
## data instances: [{def,x,y,label}]


def FindAllSuffix(task_path,sufix="json"):
    all_path = os.listdir(task_path)
    result = []
    for p in all_path:
        if not os.path.isdir(p) and sufix in p:
            result.append(os.path.join(task_path,p))
            
    return result

def sample_pos(task_path:str="data/tasks/",split_path:str="data/splits/default",sample_num_per_task:int=25,
               diff:bool=False)->tuple:
    del_ent = lambda x:x[:-1]
    ori_sample_num = sample_num_per_task
    sample_num_per_task = sample_num_per_task * 2 if diff else sample_num_per_task  ## use to sample different set for pos and neg
    with open(split_path+"/train_tasks.txt","r") as sp:
        all_tr_tasks = sp.readlines()
        all_tr_tasks = list(map(del_ent,all_tr_tasks))
        all_tr_tasks_key = dict([(t,1) for t in all_tr_tasks])
        
    all_tasks_pt = FindAllSuffix(task_path,"json")
    PREFIX = len(task_path) if task_path.endswith("/") else len(task_path) + 1
    train_num = 0
    
    # sample from the training set
    pos_exps, pos_exps_diff = [],[]
    for _,tk_pt in enumerate(tqdm(all_tasks_pt)):
        tk_name = tk_pt[PREFIX:len(tk_pt)-5]
        if all_tr_tasks_key.get(tk_name,0) == 1:
            train_num += 1
            with open(tk_pt,"r",encoding="utf-8") as tk:
                tk_info = json.load(tk) 
                definition = copy.deepcopy(tk_info["Definition"][0])
                s_n = min(len(tk_info["Instances"]),sample_num_per_task)
                examples = random.sample(tk_info["Instances"],s_n)
                for cnt, exp in enumerate(examples):
                    item = dict()
                    item["x"], item["y"] = exp["input"], exp["output"][0]
                    item["def"] = definition
                    item["label"] = 1
                    if cnt < ori_sample_num:
                        pos_exps.append(item)
                    else:
                        pos_exps_diff.append(item)
                    
    assert train_num == len(all_tr_tasks)
    
    return pos_exps, pos_exps_diff

def sample_neg(pos_exps:list,generator:AutoModelForSeq2SeqLM,tokenizer:AutoTokenizer,batch_size:int=4,
               num_beams:int=1,max_length:int=128,multinomial:bool=False)->list:
    ''' use generator to predict the silver y, namely (def, x) -> (y_) '''
    neg_exps = []
    
    generator.eval()
    with torch.no_grad():
        # for each batch
        for _,index in enumerate(tqdm(range(0,len(pos_exps),batch_size))):
            batch_features = pos_exps[index:index+batch_size]
            if len(batch_features) == 0:
                continue
            batch_inputs = feature2input_G(batch_features,tokenizer,next(generator.parameters()).device,
                                           generator.prepare_decoder_input_ids_from_labels)
            gen_kwargs = {
                "max_length": max_length,
                "synced_gpus": False,
                "attention_mask" : batch_inputs.get("attention_mask", None)
            }
            if multinomial:
                # multinomial sampling
                gen_kwargs["do_sample"] = True
            else:
                # beam search 
                gen_kwargs["num_beams"] = num_beams
            generated_tokens = generator.generate(
                batch_inputs['input_ids'],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"],tokenizer)
            
            preds = nested_numpify(generated_tokens)
            # silver y
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)    
            
            # generate neg samples
            batch_features_neg = copy.deepcopy(batch_features)
            for ins,y_ in zip(batch_features_neg,decoded_preds):
                # ins.pop("y")
                # ins["y_"], ins["label"] = y_, 0
                ins["y"], ins["label"] = y_, 0
                neg_exps.append(ins)
    
    return neg_exps

def sample_pos_neg(generator:AutoModelForSeq2SeqLM,tokenizer:AutoTokenizer,threshold:float=1.0,
                   task_path:str="data/tasks/",split_path:str="data/splits/default",batch_size:int=4,
                   num_beams:int=1,max_length:int=128,multinomial:bool=False,sample_num_per_task:int=50)->tuple:
    del_ent = lambda x:x[:-1]
    ori_sample_num = sample_num_per_task
    sample_num_per_task = sample_num_per_task * 2  ## use to sample different set for pos and neg
    with open(split_path+"/train_tasks.txt","r") as sp:
        all_tr_tasks = sp.readlines()
        all_tr_tasks = list(map(del_ent,all_tr_tasks))
        all_tr_tasks_key = dict([(t,1) for t in all_tr_tasks])
        
    all_tasks_pt = FindAllSuffix(task_path,"json")
    PREFIX = len(task_path) if task_path.endswith("/") else len(task_path) + 1
    train_num = 0
    
    # sample POS from the training set
    candidate_exps = []
    for _,tk_pt in enumerate(tqdm(all_tasks_pt)):
        tk_name = tk_pt[PREFIX:len(tk_pt)-5]
        if all_tr_tasks_key.get(tk_name,0) == 1:
            train_num += 1
            with open(tk_pt,"r",encoding="utf-8") as tk:
                tk_info = json.load(tk) 
                definition = copy.deepcopy(tk_info["Definition"][0])
                s_n = min(len(tk_info["Instances"]),sample_num_per_task)
                examples = random.sample(tk_info["Instances"],s_n)
                for cnt, exp in enumerate(examples):
                    item = dict()
                    item["x"], item["y"] = exp["input"], exp["output"]
                    item["def"] = definition
                    item["label"] = 1
                    candidate_exps.append(item)
                    
    assert train_num == len(all_tr_tasks)
    
    # divide the candidate_exps
    t = int(len(candidate_exps)/2)
    random.shuffle(candidate_exps)
    candidate_exps_t = candidate_exps[:t]
    candidate_exps_diff = candidate_exps[t:]
    candidate_exps = candidate_exps_t
    # sample NEG from the training set
    pos_exps, neg_exps = [],[]
    # use generator to predict the silver y, namely (def, x) -> (y_)
    generator.eval()
    all_preds = []
    with torch.no_grad():
        # for each batch
        for _,index in enumerate(tqdm(range(0,len(candidate_exps),batch_size))):
            batch_features = candidate_exps[index:index+batch_size]
            if len(batch_features) == 0:
                continue
            batch_inputs = feature2input_G(batch_features,tokenizer,next(generator.parameters()).device,
                                           generator.prepare_decoder_input_ids_from_labels)
            gen_kwargs = {
                "max_length": max_length,
                "synced_gpus": False,
                "attention_mask" : batch_inputs.get("attention_mask", None)
            }
            if multinomial:
                # multinomial sampling
                gen_kwargs["do_sample"] = True
            else:
                # beam search 
                gen_kwargs["num_beams"] = num_beams
            generated_tokens = generator.generate(
                batch_inputs['input_ids'],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"],tokenizer)
            
            preds = nested_numpify(generated_tokens)
            # silver y
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)    
            all_preds.extend(decoded_preds)
            
    assert len(candidate_exps) == len(all_preds)        
    
    # select POS and NEG examples 
    for ins,y_ in zip(candidate_exps,all_preds):
        # if y equals to y_, then it is a POS example
        # otherwise, it is a NEG example
        # here, we compute the EM and Rouge score, if the score is high, then it is a POS example
        exact_match = float(metric_max_over_ground_truths(exact_match_score, prediction=y_, ground_truths=ins["y"]))
        rougeL = metric_max_over_ground_truths(rougeL_score, prediction=y_, ground_truths=ins["y"])
        new_ins = copy.deepcopy(ins)
        if exact_match >= threshold or rougeL >= threshold:
            new_ins["y"], new_ins["label"] = random.choice(ins["y"]),1
            pos_exps.append(new_ins)
        else:
            new_ins["y"], new_ins["label"] = y_, 0
            neg_exps.append(new_ins)
        
    assert len(neg_exps) + len(pos_exps) == len(candidate_exps)
    
    # make POS and NEG balanced
    candidate_exps_diff_new = []
    for ins in candidate_exps_diff:
        ins["y"] = random.choice(ins["y"])
        candidate_exps_diff_new.append(ins)
    # at most of time, NEG is much more than POS, so use the candidate_exps_diff to supplement the POS
    if len(pos_exps) > len(neg_exps):
        logger.info("cutting POS: {} -> {}".format(len(pos_exps),len(neg_exps)))
        pos_exps = random.sample(pos_exps,len(neg_exps))
    else:
        logger.info("filling POS: {} -> {}".format(len(pos_exps),len(neg_exps)))
        num_s = min(len(neg_exps)-len(pos_exps),len(candidate_exps_diff_new))
        pos_exps.extend(random.sample(candidate_exps_diff_new,num_s))
        
    # print("\n==== POS: {}, NEG: {} ==== ".format(len(pos_exps),len(neg_exps)))
    
    return pos_exps, neg_exps