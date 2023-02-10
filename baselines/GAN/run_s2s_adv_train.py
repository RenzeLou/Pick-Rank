#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import copy
import logging
import math
import os
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import wandb
from sklearn.metrics import accuracy_score, f1_score
import jsonlines

sys.path.append(os.path.abspath("./kit/"))
sys.path.append(os.path.abspath("./"))

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets.utils import set_progress_bar_enabled
from datasets import load_dataset, load_metric

import transformers
import torch
# from transformers.utils import logging as hug_logging
from filelock import FileLock
from transformers.trainer import nested_numpify
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
from transformers.optimization import AdamW
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
# from ni_collator_augment_v2 import DataCollatorForNI
from ni_collator_adv_train import DataCollatorForNI
# from ni_trainer_augment_v2 import NITrainer, DenserEvalCallback
from ni_trainer_adv_train import NITrainer, DenserEvalCallback
from compute_metrics import compute_grouped_metrics_add_f, compute_grouped_metrics_add_f_v2, compute_metrics, compute_grouped_metrics, compute_metrics_add_f
from modeling_t5_adv_train import T5ForConditionalGeneration_neg
from sample_cls_data import sample_pos, sample_neg, sample_pos_neg
from tool_box import SimpleDataloader, SimpleDataset
from batch_preprocess import feature2input_G, feature2input_D
from rollout import rollout_mc_search
from tqdm import tqdm
# from tool_box import seed_torch

# import warnings

# warnings.filterwarnings('ignore')

set_progress_bar_enabled(False)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

my_env = os.environ.copy()
my_env["PATH"] = "/home/tuq59834/code/project/Tk-ins/Tk-Instruct/download/bin:" +\
                "/home/tuq59834/code/project/Tk-ins/Tk-Instruct/download/lib64:" +\
                "/home/tuq59834/anaconda3/envs/Tk-ins/bin:" +\
                my_env["PATH"]
my_env["LD_LIBRARY_PATH"] = "/home/tuq59834/code/project/Tk-ins/Tk-Instruct/download/lib/:" +\
                my_env["LD_LIBRARY_PATH"]
os.environ.update(my_env)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    pretrain_G_path: Optional[str] = field(
        default="/home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/POS_augmentation-num_0",
        metadata={"help": "Path to pretrained generator."},    #######
    )
    pretrain_D_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained discriminator."},   #####
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=500, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    tk_instruct: Optional[bool] = field(
        default=False,
        metadata={"help": "tk_instruct will train a model combining all valid instruction encodings. This will overwrite the other settings about instruction encoding."} 
    )
    exp_name:Optional[str] = field(
        default=None,
        metadata={"help":"exp dir"}
    )
    data_dir_argu: str = field(
        default=None, metadata={"help": "The directory for train/dev/test splits (include the negative instructions)."}
    )
    task_dir_argu: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files (include the negative instructions)."}
    )
    sample_num_neg: int = field(
        default=None, metadata={"help": "How many negative definitions sampled when training."}
    )
    sample_num_pos: int = field(
        default=None, metadata={"help": "How many positive definitions sampled when training."}
    )
    main_loss_warm: int = field(
        default=0, metadata={"help": "epoch num used for warm up, before add additional loss."}
    )
    task_dir_adv: str = field(
        default=None  ######
    )
    
    def __post_init__(self):
        pass


@dataclass
class NITrainingArguments(Seq2SeqTrainingArguments):
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    loss_mix_ratio_null: float = field(
        default=0.5, metadata={"help": "The ratio between pos and null losses."}
    )
    loss_mix_ratio_neg: float = field(
        default=0.5, metadata={"help": "The ratio between pos and neg losses."}
    )
    loss_mix_ratio_out: float = field(
        default=0.5, metadata={"help": "The ratio of out constrain loss."}
    )
    margin_pos: float = field(
        default=0.0003, metadata={"help": "The margin used in contrastive loss (pos)."}
    )
    margin_null: float = field(
        default=0.001, metadata={"help": "The margin used in contrastive loss (null)."}
    )
    margin_neg: float = field(
        default=0.001, metadata={"help": "The margin used in contrastive loss (neg)."}
    )
    margin_out: float = field(
        default=0.001, metadata={"help": "The margin used in contrastive loss (out constrain)."}
    )
    pos_neg_ratio: float = field(
        default=0.5,metadata={"help": "The trade-off between pos and neg."}
    )
    # neg_loss_only: Optional[bool] = field(
    #     default=False, metadata={"help": "Only use constractive loss, to test whether the ranking method makes sense."}
    # )
    # neg_loss_type: str = field(
    #     default="contrastive_loss_all", metadata={"help": "The loss function."}
    # )
    neg_loss_type: str = field(
        default=None, metadata={"help": "The loss function for negative definition."}
    )
    null_loss_type: str = field(
        default=None, metadata={"help": "The loss function for null definition."}
    )
    out_loss_type: str = field(
        default=None, metadata={"help": "The loss function for neg_output."}
    )
    add_task_definition_train: bool = field(
        default=True, metadata={"help": "The loss function."}
    )
    add_task_definition_test: bool = field(
        default=True, metadata={"help": "The loss function."}
    )
    training_ins_num: int = field(
        default=75317, metadata={"help": "The pre-defined training instance num."}
    )
    neg_out_sample_num: int = field(
        default=None, metadata={"help": "The sample num of neg otuputs that used for ranking loss."}
    )
    pre_train_batch_size: int = field(
        default=4  ####
    )
    force_pretrain: bool = field(
        default=False  ####
    )
    multinomial: bool = field(
        default=False,   ####
        metadata={"help": "use multinomial sampling insteaf of beam search to sample predictions from generator."}
    )
    pretrain_save_path: str = field(
        default="/home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/pretrain_classifier",   ####
    )
    d_step: int = field(
        default=1,   ####
    )
    d_epoch: int = field(
        default=2,   ####
    )
    adv_g_step: int = field(
        default=10,   ####
    )
    adv_d_step: int = field(
        default=10,   ####
    )
    adv_d_epoch: int = field(
        default=1,   ####
    )
    lr_D: float = field(
        default=2e-5,   ####
    )
    lr_G: float = field(
        default=5e-5,   ####
    )
    ADV_train_epoch: int = field(
        default=100,   ####
    )
    add_def: Optional[bool] = field(
        default=False,   ####
    )
    sample_batch_size: int = field(
        default=64,   ####
    )
    mc_search_num: int = field(
        default=5,   ####
    )
    clip_norm: Optional[int] = field(
        default=None,   ####
    )
    generate_examples: Optional[bool] = field(
        default=False,   ####
    )
    log_likelihood: Optional[bool ] = field(
        default=False,   ####
    )

def train_generator(model:AutoModelForSeq2SeqLM,tokenizer:AutoTokenizer,model_D:AutoModelForSequenceClassification,tokenizer_D:AutoTokenizer,
                    optimizer_G:AdamW,training_args,data_args,adv_epoch:int,save_generator:bool = True, sample_num_per_task:int = 2):
    adv_g_step = training_args.adv_g_step
    batch_size = training_args.per_device_train_batch_size
    sample_batch_size = training_args.sample_batch_size
    add_def = training_args.add_def
    mc_search_num = training_args.mc_search_num
    multinomial = training_args.multinomial
    clip_norm = training_args.clip_norm
    log_likelihood = training_args.log_likelihood
    # sample the negative samples (predicted by generator), which is used for adv training
    _,neg_exps = sample_pos_neg(model,tokenizer,task_path=data_args.task_dir_adv,split_path=data_args.data_dir,
                                batch_size=sample_batch_size,multinomial=multinomial,
                                sample_num_per_task=sample_num_per_task)
    data_iter = SimpleDataloader(SimpleDataset(neg_exps),batch_size=batch_size,pin_memory=True,shuffle=True)
    all_tr_loss = []
    for i, batch_features in enumerate(tqdm(data_iter)):
        if i >= adv_g_step:
            break  # for each step, train the generator for 1 batch
        batch_inputs = feature2input_G(batch_features,tokenizer,next(model.parameters()).device,model.prepare_decoder_input_ids_from_labels)
        # use generator.forward to get the output logits (i.e., vocab prob [batch_size, seq_len, vocab_size])
        # get the prob corresponding to the silver y
        model.train()
        logits,labels_prob,prob_mask = model(**batch_inputs,get_prob=True)
        prob_mask = prob_mask.squeeze(-1)  # [batch_size, seq_len]
        labels_prob = labels_prob.squeeze(-1)  # [batch_size, seq_len]
        # for each time step, use silver y as the current action, predict the future tokens with generator & sampling
        # then use classifier to predict the score for this sequence, use this score as reward
        # do this sampling for several times, and use the average score as the final reward
        with torch.no_grad():
            # when getting reward, we do not need to compute the gradient
            model.eval()
            model_D.eval()
            decoder_inputs,labels = batch_inputs["decoder_input_ids"],batch_inputs["labels"]
            assert decoder_inputs.shape[1] == labels.shape[1]
            bs = labels.shape[0]
            start_letter = torch.ones(bs,1,dtype=torch.long).to(labels.device) * tokenizer.pad_token_id  # t5 use the pad token as the starting letter
            sequence = torch.cat([start_letter,labels],dim=1)  # [batch_size, seq_len+1]
            rewards = []
            for time_step in range(2,sequence.shape[1]+1):
                current_action = sequence[:,:time_step]  # [batch_size, time_step]
                if time_step == sequence.shape[1]:
                    # the last time step, we dont do the MC search
                    sequence_searched = current_action[:,1:] ## [batch_size, seq_len]
                    sequence_searched = sequence_searched.mul(prob_mask.long())  # mask the padding tokens
                    # use classifier to predict the score for this sequence
                    pd = nested_numpify(sequence_searched)
                    pd_str = tokenizer.batch_decode(pd,skip_special_tokens=True)  # a lit of string
                    # get the reward of current time step
                    reward = get_reward(model_D,tokenizer_D,batch_features,pd_str,add_def=add_def)  # [batch_size]
                    reward.unsqueeze_(1)  # [batch_size, 1] 
                    rewards.append(reward)
                else:
                    # MC search
                    reward = torch.zeros(bs).to(labels.device)  # [batch_size]
                    assert mc_search_num > 0
                    for _ in range(mc_search_num):
                        sequence_searched = rollout_mc_search(model,tokenizer,batch_inputs,current_action,sequence.shape[1],multinomial) ## [batch_size, seq_len + 1]
                        sequence_searched = sequence_searched[:,1:] ## [batch_size, seq_len]
                        sequence_searched = sequence_searched.mul(prob_mask.long())  # mask the padding tokens
                        # use classifier to predict the score for this sequence
                        pd = nested_numpify(sequence_searched)
                        pd_str = tokenizer.batch_decode(pd,skip_special_tokens=True)  # a lit of string
                        # get the reward of current time step
                        reward_ = get_reward(model_D,tokenizer_D,batch_features,pd_str,add_def=add_def)  # [batch_size]
                        reward = reward + reward_
                    reward = reward / mc_search_num
                    reward.unsqueeze_(1)  # [batch_size, 1] 
                    rewards.append(reward)
            rewards = torch.cat(rewards,dim=1)  # [batch_size, seq_len]
        # reward ([batch_size, seq_len]) dot prob ([batch_size, seq_len])
        # note the mask for the padding tokens
        model.train()
        rewards = rewards.mul(prob_mask)
        final_reward = labels_prob.mul(rewards)  # [batch_size, seq_len]
        sum_mask = torch.sum(prob_mask)
        if sum_mask.item() != 0.:
            if log_likelihood:
                # use log likelihood as the loss, and do backpropagation
                final_loss = model.neg_log(final_reward,prob_mask)
                adv_loss = torch.sum(final_loss) / sum_mask
            else:
                # take avg token-level reward as the reward for current batch
                # use -reward as the loss, and do backpropagation (w.r.t. minimize the loss <==> maximize the reward)
                r = torch.sum(final_reward) / sum_mask
                adv_loss = -r
        else:
            adv_loss = torch.tensor(0.).type_as(final_reward.detach())
            adv_loss.requires_grad_()
        
        # optimize
        optimizer_G.zero_grad()
        adv_loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer_G.step()
        all_tr_loss.append(adv_loss.item())
        
    # for observation    
    logger.info(f"[ADV-GEN] | epoch {adv_epoch+1} | loss: {np.mean(all_tr_loss)}")    
    with open(os.path.join(training_args.output_dir,"adv_gen_loss.txt"),"a") as f:
        f.write(f"[ADV-GEN] | epoch {adv_epoch+1} | loss: {np.mean(all_tr_loss)}\n")
    if save_generator:
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"save the generator at {training_args.output_dir}")

def train_classifier(model_D:AutoModelForSequenceClassification,tokenizer_D:AutoTokenizer,
                     model_G:AutoModelForSeq2SeqLM,tokenizer_G:AutoTokenizer,
                    training_args:NITrainingArguments,data_args:DataTrainingArguments,
                    optimizer_D:AdamW,sample_num_per_task=50,phase="Pretrain",clip_norm=None,adv_epoch:int=None):
    assert phase in ["ADV","Pretrain"]
    save_path = training_args.pretrain_save_path if phase=="Pretrain" else os.path.join(training_args.output_dir,"classifier")
    d_step = training_args.d_step if phase=="Pretrain" else training_args.adv_d_step
    d_epoch = training_args.d_epoch if phase=="Pretrain" else training_args.adv_d_epoch 
    batch_size = training_args.pre_train_batch_size if phase=="Pretrain" else training_args.per_device_train_batch_size
    add_def = training_args.add_def
    sample_num_per_task = 2 if phase=="ADV" else sample_num_per_task
    ## randomly sample positive and negative samples
    logger.info(f" '{phase}': sample POS and NEG examples")
    pos_exps, neg_exps = sample_pos_neg(generator=model_G,tokenizer=tokenizer_G,task_path=data_args.task_dir_adv,
                                        split_path=data_args.data_dir,batch_size=training_args.sample_batch_size,
                                        multinomial=training_args.multinomial,sample_num_per_task=sample_num_per_task)
    logger.info(f"training num ==> POS: {len(pos_exps)}; NEG: {len(neg_exps)}")
    ## shuffle and split train/eval (9:1)
    train_num_pos, train_num_neg = int(len(pos_exps) * 9 / 10), int(len(neg_exps) * 9 / 10)
    train_data_D = pos_exps[:train_num_pos] + neg_exps[:train_num_neg]
    eval_data_D = pos_exps[train_num_pos:] + neg_exps[train_num_neg:]
    eval_loader_D = SimpleDataloader(SimpleDataset(eval_data_D),batch_size=batch_size,pin_memory=True,shuffle=True)
    train_loader_D = SimpleDataloader(SimpleDataset(train_data_D),batch_size=batch_size,pin_memory=True,shuffle=True)
    softmax = torch.nn.Softmax(dim=1)
    test_results = f"[ADV-CLS] | epoch {adv_epoch+1}|\ntraining num ==> POS: {train_num_pos}; NEG: {train_num_neg}\n"
    test_results += f"eval num ==> POS: {len(pos_exps)-train_num_pos}; NEG: {len(neg_exps)-train_num_neg}\n"
    for step in range(d_step):
        logger.info(f" Train Classifier Step {step+1}")
        ## train the classifier
        for epoch in range(d_epoch):
            logger.info(f" === Epoch {epoch+1} === ")
            all_tr_loss = []
            model_D.train()
            for i, batch_features in enumerate(tqdm(train_loader_D)):
                batch_inputs = feature2input_D(batch_features,tokenizer_D,next(model_D.parameters()).device,add_def=add_def)
                output = model_D(**batch_inputs)
                loss = output.loss
                optimizer_D.zero_grad()
                loss.backward()
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model_D.parameters(), clip_norm)
                optimizer_D.step()
                
                all_tr_loss.append(loss.item())
                ## when doing ADV train, just training 10 batches each epoch
                if phase == "ADV" and i >= 10:
                    logger.info("==> avg tr loss = %.4f" % (np.mean(all_tr_loss))) 
                    break
            logger.info("==> avg tr loss = %.4f" % (np.mean(all_tr_loss))) 
        # each step, shuffle the trainng data
        train_loader_D = SimpleDataloader(SimpleDataset(train_data_D),batch_size=batch_size,pin_memory=True,shuffle=True)
        ## test (for observation)
        all_te_loss = []
        preds,labels,ins_lis = [],[],[]
        model_D.eval()
        with torch.no_grad():
            for i, batch_features in enumerate(tqdm(eval_loader_D)):
                batch_inputs = feature2input_D(batch_features,tokenizer_D,next(model_D.parameters()).device,add_def=add_def)
                outputs = model_D(**batch_inputs)
                prob = softmax(outputs.logits)  # [1,2]
                prob = prob.detach().cpu().numpy()
                pred = np.argmax(prob,axis=1)
                preds.extend(pred.tolist())
                labels.extend(batch_inputs["labels"].detach().cpu().numpy().tolist())
                ins_lis.extend(batch_features) if isinstance(batch_features[0],dict) else ins_lis.extend(batch_features[0]) 
                all_te_loss.append(outputs.loss.item())
        acc = accuracy_score(labels,preds)
        ma_f1 = f1_score(labels,preds,average="macro")
        mi_f1 = f1_score(labels,preds,average="micro")
        logger.info("==> avg te loss = %.4f;  te_acc = %.2f; te_macro_f1 = %.2f; te_micro_f1 = %.2f" % (np.mean(all_te_loss),acc*100,ma_f1*100,mi_f1*100)) 
        test_results += "==> avg te loss = %.4f;  te_acc = %.2f; te_macro_f1 = %.2f; te_micro_f1 = %.2f\n" % (np.mean(all_te_loss),acc*100,ma_f1*100,mi_f1*100)
    
    if save_path is not None:
        os.makedirs(save_path,exist_ok=True)
        model_D.save_pretrained(save_path)
        tokenizer_D.save_pretrained(save_path)
        # save eval results
        with open(os.path.join(save_path,"eval_results.txt"),"w") as f:
            f.write(test_results)
        # save eval predictions
        assert len(preds) == len(labels) == len(ins_lis)
        with jsonlines.open(os.path.join(save_path,"eval_predictions.jsonl"),"w") as f:
            for p,l,ins in zip(preds,labels,ins_lis):
                ins.pop("label")
                a = {"pred":p,"label":l}
                a.update(ins)
                jsonlines.Writer.write(f,a)
        logger.info(f"Save the classifier at {save_path}. Save the eval results at {save_path}/eval_results.txt. Save the eval predictions at {save_path}/eval_predictions.jsonl.")

def get_reward(model_D:AutoModelForSequenceClassification,tokenizer_D:AutoTokenizer,batch_features:list,
               samples:list,add_def:bool=False):
    ''' return a tensor of rewards: [batch_size,] '''
    # use sampled y to replace the original y in the batch_features
    if not isinstance(batch_features[0],dict):
        batch_features = batch_features[0]
    assert len(batch_features) == len(samples)
    new_batch_features = []
    for ins,pred in zip(batch_features,samples):
        new_ins = copy.deepcopy(ins)
        new_ins["y"] = pred
        new_batch_features.append(new_ins)
    # preprocess the input
    softmax = torch.nn.Softmax(dim=1)
    batch_inputs = feature2input_D(new_batch_features,tokenizer_D,next(model_D.parameters()).device,add_def=add_def)
    outputs = model_D(**batch_inputs)
    prob = softmax(outputs.logits)  # [batch_size,2]  
    rewards = prob[:,1]  # [batch_size,]
    return rewards  

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NITrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    
    model_args.model_name_or_path = model_args.pretrain_G_path if model_args.pretrain_G_path is not None else model_args.model_name_or_path

    # Setup logging
    # hug_logging.set_verbosity_info()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
        
    ## forbidden wandb for large-scale experiments
    wandb.init(mode="disabled")
    
    ## using the data argument (i.e., negative training)
    data_args.data_dir = data_args.data_dir_argu if data_args.data_dir_argu is not None else data_args.data_dir
    data_args.task_dir = data_args.task_dir_argu if data_args.task_dir_argu is not None else data_args.task_dir
    if data_args.data_dir_argu is not None and data_args.task_dir_argu is not None:
        logger.info("===> Note that you are useing data argument!")

    ## reset the output dir according to the exp_name to avoid results overwriting
    training_args.output_dir = os.path.join(training_args.output_dir,data_args.exp_name) if data_args.exp_name is not None \
                                else training_args.output_dir
    os.makedirs(training_args.output_dir,exist_ok=True)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    ''' 1. load the pretrained generator '''
    logger.info(f" ** 1. load the generator: {model_args.model_name_or_path} ** ")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    model = T5ForConditionalGeneration_neg.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    ## optimizer
    optimizer_G = AdamW(model.parameters(),lr=training_args.lr_G)

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id


    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )     
        
        
    ''' 2. pretrain the classifier '''
    logger.info(" ** 2. pretrain the classifier ** ")
    path_D = model_args.pretrain_D_path if model_args.pretrain_D_path is not None else "bert-base-cased"
    config_D = AutoConfig.from_pretrained(
        path_D,
        num_labels=2,
        finetuning_task="adv_train_cls",
        cache_dir=model_args.cache_dir
    )
    tokenizer_D = AutoTokenizer.from_pretrained(
        path_D,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )
    model_D = AutoModelForSequenceClassification.from_pretrained(
        path_D,
        config=config_D,
        cache_dir=model_args.cache_dir
    )
    model_D.cuda()
    ## optimizer
    optimizer_D = AdamW(model_D.parameters(),lr=training_args.lr_D)
    if model_args.pretrain_D_path is None or training_args.force_pretrain:
        train_classifier(model_D,tokenizer_D,model,tokenizer,training_args,data_args,optimizer_D=optimizer_D)
    else:
        logger.info(f"==> load the pre-trained classifier at {model_args.pretrain_D_path}")
    
    ''' 3. ADV training '''
    logger.info(" ** 3. ADV training begin ** ")
    for adv_epoch in range(training_args.ADV_train_epoch):
        logger.info('\n-----\nADV EPOCH %d\n-----' % (adv_epoch + 1))
        ''' 3.1 train generator '''
        train_generator(model,tokenizer,model_D,tokenizer_D,optimizer_G,training_args,data_args,adv_epoch)
        ''' 3.2 train classifier '''
        train_classifier(model_D,tokenizer_D,model,tokenizer,training_args,data_args,phase="ADV",optimizer_D=optimizer_D,adv_epoch=adv_epoch)
        
    ''' 4. save the generator and use it to predict the examples'''
    logger.info(" ** 4. save and predict ** ")
    # must save the generator
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"save the generator at {training_args.output_dir}")
    # load the examples in the test data
    # use the generator to predict the silver y
    # save those silver (x,y) pairs for further prediction
    if training_args.generate_examples:
        # use pos examples as the testing instances
        raw_datasets = load_dataset(  
            "src/ni_dataset_adv_train_use_examples_as_instance.py", 
            data_dir="/home/tuq59834/code/project/Tk-ins/Tk-Instruct/data/splits/pred_only",   ## use this path, to ensusre there are training and dev split
            task_dir=data_args.task_dir,   ## all data along with instructions
            cache_dir=model_args.cache_dir,
            max_num_instances_per_task=data_args.max_num_instances_per_task,
            max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
            sample_num_pos=data_args.sample_num_pos,
            sample_num_neg=data_args.sample_num_neg,
            training_ins_num=training_args.training_ins_num,
            add_task_definition_train=training_args.add_task_definition_train,
            add_task_definition_test=training_args.add_task_definition_test,
            neg_loss_type=training_args.neg_loss_type,
            null_loss_type=training_args.null_loss_type,
            out_loss_type=training_args.out_loss_type
        )

       
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        # Data collator
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForNI(
            tokenizer,
            model=model,
            padding="max_length" if data_args.pad_to_max_length else "longest",
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            add_task_name=data_args.add_task_name,
            add_task_definition=data_args.add_task_definition,
            num_pos_examples=data_args.num_pos_examples,
            num_neg_examples=data_args.num_neg_examples,
            neg_out_sample_num=training_args.neg_out_sample_num,
            add_explanation=data_args.add_explanation,
            tk_instruct=data_args.tk_instruct
        )
        
        training_args.remove_unused_columns = False 


        def compute_ni_metrics(dataset, preds, save_prefix=None):
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            references = [e["Instance"]["output"] for e in dataset]
            result = compute_metrics_add_f(predictions=decoded_preds, references=references)
            # result_per_task = compute_grouped_metrics_add_f(predictions=decoded_preds, references=references, groups=dataset["Task"])
            # result.update(result_per_task)
            categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
            # result_per_category = compute_grouped_metrics_add_f(predictions=decoded_preds, references=references, groups=categories)
            result_per_category = compute_grouped_metrics_add_f_v2(predictions=decoded_preds, references=references, groups=categories)
            result.update(result_per_category)
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            if save_prefix is not None:
                with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                    for example, pred in zip(dataset, decoded_preds):
                        fout.write(json.dumps({
                            "Task": example["Task"],
                            "Definition": example["Definition"],
                            "Instance": example["Instance"],
                            "Prediction": pred
                        }) + "\n")
            return result

        # Initialize our Trainer
        trainer = NITrainer(
            model=model,
            args=training_args,
            train_dataset= None,
            eval_dataset= None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
            callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
        )

        all_metrics = {"run_name": training_args.run_name}

        # Evaluation
        results = {}
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.max_target_length
        )
        num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
        
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        all_metrics.update(metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                # output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                # with open(output_prediction_file, "w") as writer:
                #     writer.write("\n".join(predictions))
                output_prediction_file = os.path.join(training_args.output_dir, "predicted_examples_for_inference.jsonl")
                with open(output_prediction_file, "w") as fout:
                    for example, prediction in zip(predict_dataset, predictions):
                        example["prediction"] = prediction
                        fout.write(json.dumps(example) + "\n")

    
    if training_args.do_predict:
        # 1. test the generator to see if the ADV training can directly improve the performance
        # 2. and also save the prediction results (i.e., predicted_examples.jsonl)
        # run the generate_silver_output_for_examples.py add the predicted y (silver y) into dataset
        # then give these silver y to gold_y model to do the test
        raw_datasets = load_dataset(  
            "src/ni_dataset_adv_train.py", 
            data_dir=data_args.data_dir,   ## train/test/dev split
            task_dir=data_args.task_dir,   ## all data along with instructions
            cache_dir=model_args.cache_dir,
            max_num_instances_per_task=data_args.max_num_instances_per_task,
            max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
            sample_num_pos=data_args.sample_num_pos,
            sample_num_neg=data_args.sample_num_neg,
            training_ins_num=training_args.training_ins_num,
            add_task_definition_train=training_args.add_task_definition_train,
            add_task_definition_test=training_args.add_task_definition_test,
            neg_loss_type=training_args.neg_loss_type,
            null_loss_type=training_args.null_loss_type,
            out_loss_type=training_args.out_loss_type
        )

        if training_args.do_predict:
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        # Data collator
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForNI(
            tokenizer,
            model=model,
            padding="max_length" if data_args.pad_to_max_length else "longest",
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            add_task_name=data_args.add_task_name,
            add_task_definition=data_args.add_task_definition,
            num_pos_examples=data_args.num_pos_examples,
            num_neg_examples=data_args.num_neg_examples,
            neg_out_sample_num=training_args.neg_out_sample_num,
            add_explanation=data_args.add_explanation,
            tk_instruct=data_args.tk_instruct
        )
        # if training_args.do_train:
        #     if "train" not in raw_datasets:
        #         raise ValueError("--do_train requires a train dataset")
        #     train_dataset = raw_datasets["train"]
        #     if data_args.max_train_samples is not None:
        #         train_dataset = train_dataset.select(range(data_args.max_train_samples))

        # if training_args.do_eval:
        #     if "validation" not in raw_datasets:
        #         raise ValueError("--do_eval requires a validation dataset")
        #     eval_dataset = raw_datasets["validation"]
        #     if data_args.max_eval_samples is not None:
        #         eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        
        
        # we don't want to remove unused columns because we will prepare each batch during training, 
        # and some of the information will aslo be used in evaluation.
        training_args.remove_unused_columns = False 


        def compute_ni_metrics(dataset, preds, save_prefix=None):
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            references = [e["Instance"]["output"] for e in dataset]
            result = compute_metrics_add_f(predictions=decoded_preds, references=references)
            # result_per_task = compute_grouped_metrics_add_f(predictions=decoded_preds, references=references, groups=dataset["Task"])
            # result.update(result_per_task)
            categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
            # result_per_category = compute_grouped_metrics_add_f(predictions=decoded_preds, references=references, groups=categories)
            result_per_category = compute_grouped_metrics_add_f_v2(predictions=decoded_preds, references=references, groups=categories)
            result.update(result_per_category)
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            if save_prefix is not None:
                with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                    for example, pred in zip(dataset, decoded_preds):
                        fout.write(json.dumps({
                            "Task": example["Task"],
                            "Definition": example["Definition"],
                            "Instance": example["Instance"],
                            "Prediction": pred
                        }) + "\n")
            return result

        # Initialize our Trainer
        trainer = NITrainer(
            model=model,
            args=training_args,
            train_dataset= None,
            eval_dataset= None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
            callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
        )
        
        all_metrics = {"run_name": training_args.run_name}

        # Evaluation
        results = {}
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.max_target_length
        )
        num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
        
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        all_metrics.update(metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                # output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                # with open(output_prediction_file, "w") as writer:
                #     writer.write("\n".join(predictions))
                output_prediction_file = os.path.join(training_args.output_dir, "predicted_examples.jsonl")
                with open(output_prediction_file, "w") as fout:
                    for example, prediction in zip(predict_dataset, predictions):
                        example["prediction"] = prediction
                        fout.write(json.dumps(example) + "\n")

    if (training_args.do_train or training_args.do_eval or training_args.do_predict) and trainer.is_world_process_zero():
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()