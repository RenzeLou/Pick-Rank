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

import logging
import math
import os
import sys
import json
import torch
import random
import wandb
import datasets
import nltk 
import numpy as np
import copy
import transformers

from dataclasses import dataclass, field
from typing import Optional
from datasets.utils import set_progress_bar_enabled
from datasets import load_dataset, load_metric
from torch import nn
from transformers.utils import logging as hug_logging
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
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

# Following imports are the key scripts for this project
from data_collator import DataCollatorForNI
from trainer_pointer import NITrainer, DenserEvalCallback
from modeling_pointer import T5ForConditionalGeneration_neg, T5Stack
from compute_metrics import compute_grouped_metrics_add_f, compute_grouped_metrics_add_f_v2, compute_metrics, compute_grouped_metrics, compute_metrics_add_f

# set env variable 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # enable the deterministic method

set_progress_bar_enabled(False)
logger = logging.getLogger(__name__)


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
    sample_num_neg: int = field(
        default=None  ### deprecated
    )
    sample_num_pos: int = field(
        default=None, 
        metadata={"help": "Number of sentences. We random select several sentences from the task definition as the candidates of the Pointer Network."}
    )
    main_loss_warm: int = field(
        default=0, metadata={"help": "epoch num used for warm up, before add additional loss."}  ### deprecated, set to 0
    )
    
    def __post_init__(self):
        pass


@dataclass
class NITrainingArguments(Seq2SeqTrainingArguments):
    num_train_epochs: float = field(
        default=2,
        metadata={"help": "Total number of training epochs to perform."}
    )
    seed: int = field(
        default=42,
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    loss_mix_ratio_null: float = field(
        default=1, metadata={"help": "The ratio for the pos loss (rep vs. ori)."}
    )
    loss_mix_ratio_neg: float = field(
        default=1, metadata={"help": "The ratio for the neg loss (rep vs. del)."}
    )
    loss_mix_ratio_pos: float = field(
        default=1, metadata={"help": "The ratio for the null loss (rep vs. null)."}
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
    neg_loss_type: str = field(
        default=None, metadata={"help": "The loss function for negative definition. e.g., ``contrastive_loss_max_v6''"}
    )
    null_loss_type: str = field(
        default="contrastive_loss_max_v6", metadata={"help": "The loss function for null definition. e.g., contrastive_loss_max_v6."}
    )
    pos_loss_type: str = field(
        default=None, metadata={"help": "The loss function for repeated definition. e.g., ``contrastive_loss_repeat_v2.''"}
    )
    add_task_definition_train: bool = field(
        default=True, metadata={"help": "Whether to add task definition when training."}
    )
    add_task_definition_test: bool = field(
        default=True, metadata={"help": "Whether to add task definition when testing."}
    )
    training_ins_num: int = field(
        default=75317, metadata={"help": "The pre-defined training instance num."}  ### deprecated
    )
    neg_out_sample_num: int = field(
        default=None, metadata={"help": "The sample num of neg otuputs that used for ranking loss."} ### deprecated
    )
    pooling: Optional[str] = field(
        default="mean", metadata={"help": "The token-level pooling method"}
    )
    lr_proj: Optional[float] = field(
        default=None, metadata={"help": "The learning rate for the Linear Projector of the Pointer Network."}
    )
    pointer_hidden_dim: Optional[int] = field(
        default=None 
    )
    act_func: Optional[str] = field(
        default="relu" 
    )
    main_loss_on_rep: Optional[bool] = field(
        default=False, metadata={"help": "Set ``True'' to calculate the cross entropy loss on the repeated definition."}
    )
    predict_on_rep: Optional[bool] = field(
        default=False 
    )
    save_pointer_choice: Optional[bool] = field(
        default=False, metadata={"help": "Save the choices of the Pointer Network for observation."}
    )
    pointer_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "The path to the Linear Projector of the Pointer Network. e.g., pointer.pth.tar"}
    )
    add_pointer_encoder: Optional[bool] = field(
        default=False, metadata={"help": "Whether to add a T5-encoder to the Pointer Network."}
    )
    pointer_encoder_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "The path to the T5-encoder checkpoint of the Pointer Network."}
    )
    lr_encoder: Optional[float] = field(
        default=None, metadata={"help": "The learning rate of the T5-encoder."}
    )
    sample_times: Optional[int] = field(
        default=1, metadata={"help": "The sampling times for Gumbel-Softmax."}
    )
    pointer_train_epoch: Optional[float] = field(
        default=0, 
        metadata={"help": "The training epoch for the Pointer Network (no ranking loss added). The total epoch is ``num_train_epochs''."}
    )
    ranking_forbiden_on_pointer: Optional[bool] = field(
        default=False, metadata={"help": "Whether to fix the parameter of Pointer Network when doing ranking loss."}
    )
    prob_save_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The path to save the token-level probability of the Pointer Network. Used for analysis."}
    )
    prob_save_on_rep: Optional[bool] = field(
        default=False
    )
    wandb_enable: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable wandb."}
    )
    last_sen_num: Optional[int] = field(
        default=None  ### deprecated, pls set ``None'' or 0.
    )
    random_words_neg_def: Optional[bool] = field(
        default=False, ### deprecated
    )
    words_num: int = field(
        default=1, ### deprecated
    )
    reverse: Optional[bool] = field(
        default=False, ### deprecated
    )
    
def seed_torch(seed=42,deter=False):
    '''
    `deter` means use deterministic algorithms for GPU training reproducibility, 
    if set `deter=True`, please set the environment variable `CUBLAS_WORKSPACE_CONFIG` in advance
    '''
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(deter) # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    

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

    # check the epoch
    assert training_args.num_train_epochs >= training_args.pointer_train_epoch, "``pointer_train_epoch'' should be smaller than ``num_train_epochs''"

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
    if not training_args.wandb_enable:
        wandb.init(mode="disabled")
        
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
    seed_torch(training_args.seed,deter=False)  ## you can set deter=True to avoid nondeterministic algorithms, this is helpful for reproducibility on the same machine

    # Get the NaturalInstructions dataset
    raw_datasets = load_dataset( 
        "src/dataset_pointer.py", 
        data_dir=data_args.data_dir,   ## train/test/dev split
        task_dir=data_args.task_dir,   ## all data along with instructions
        cache_dir=model_args.cache_dir,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        sample_num_pos=data_args.sample_num_pos,
        sample_num_neg=data_args.sample_num_neg,
        training_ins_num=training_args.training_ins_num,
        neg_loss_type=training_args.neg_loss_type,
        null_loss_type=training_args.null_loss_type,
        pos_loss_type=training_args.pos_loss_type,
        last_sen_num=training_args.last_sen_num,
    )

    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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
    # add special tokens
    # tokenizer.add_special_tokens({"additional_special_tokens": ["[REPEAT]","[DEL]", "[NULL]"]})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[REPEAT]"]})
       
    model = T5ForConditionalGeneration_neg.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # add pointer to the model
    model.add_pointer(config.d_model,training_args.pointer_hidden_dim,training_args.act_func,1)
    
    # load pre-trained pointer
    if training_args.pointer_checkpoint is not None:
        pointer_save_file = os.path.join(training_args.output_dir, training_args.pointer_checkpoint)
        if os.path.isfile(pointer_save_file):
            logger.info("=> loading pointer checkpoint '{}'".format(pointer_save_file))
            pointer_checkpoint = torch.load(pointer_save_file)
            model.pointer_projector.load_state_dict(pointer_checkpoint['state_dict'])
        else:
            raise ValueError("=> no such pointer checkpoint: '{}', please check your chechpoint name!".format(pointer_save_file))
        
    # add additional encoder for pointer 
    # by default, we use the same encoder as the T5 model for the Pointer Network, but these two encoders are trained separately
    if training_args.add_pointer_encoder:
        if training_args.pointer_encoder_checkpoint is not None:
            pointer_encoder_save_dir= os.path.join(training_args.output_dir, training_args.pointer_encoder_checkpoint)
            # if the checkpoint is a directory, load the model from the directory
            if os.path.isdir(pointer_encoder_save_dir):
                # load the saved config
                encoder_config = AutoConfig.from_pretrained(pointer_encoder_save_dir)
                encoder_config.is_decoder = False
                encoder_config.use_cache = False
                encoder_config.is_encoder_decoder = False
                # load the saved embedding for the pointer_encoder
                embed_tokens = nn.Embedding(config.vocab_size, config.d_model) # the shape should be (config.vocab_size, config.d_model)
                embedding_file = os.path.join(pointer_encoder_save_dir, "pointer_encoder_embed_tokens.pth.tar")
                embedding_checkpoint = torch.load(embedding_file)
                embed_tokens.load_state_dict(embedding_checkpoint['state_dict'])
                # load the saved T5Stack
                model.pointer_encoder = T5Stack.from_pretrained(
                    pointer_encoder_save_dir,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=encoder_config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    embed_tokens=embed_tokens
                )
            else:
                raise ValueError("=> no such pointer encoder checkpoint: '{}', please check your chechpoint name!".format(pointer_encoder_save_dir))
        else:
            logger.info("**"*10 + "=> no pointer encoder checkpoint, copy the encoder of the T5 model" + "**"*10)
            pointer_encoder = copy.deepcopy(model.encoder)
            model.pointer_encoder = pointer_encoder
    else:
        model.pointer_encoder = None
    
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

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

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
    # we don't want to remove unused columns because we will prepare each batch during training, 
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False 


    # Metric
    def compute_ni_metrics(dataset, preds, save_prefix=None):
        # we also add F1 score in this function, just ignore it if you don't need it.
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [e["Instance"]["output"] for e in dataset]
        result = compute_metrics_add_f(predictions=decoded_preds, references=references)
        categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
        try:
            result_per_category = compute_grouped_metrics_add_f_v2(predictions=decoded_preds, references=references, groups=categories)
        except KeyError:
            # eval on dev set
            result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
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
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
    )
    
    # add some hyperparameters to the trainer
    save_choice_file= os.path.join(training_args.output_dir, "pointer_choice_test.txt") 
    trainer.init_hyper(margin_pos=training_args.margin_pos,
                       margin_null=training_args.margin_null,
                       margin_neg=training_args.margin_neg,
                       neg_loss_type=training_args.neg_loss_type,
                       null_loss_type=training_args.null_loss_type,
                       pos_loss_type=training_args.pos_loss_type,
                       loss_mix_ratio_neg=training_args.loss_mix_ratio_neg,
                       loss_mix_ratio_null=training_args.loss_mix_ratio_null,
                       loss_mix_ratio_pos=training_args.loss_mix_ratio_pos,
                       sample_num_neg=data_args.sample_num_neg,
                       sample_num_pos=data_args.sample_num_pos,
                       main_loss_warm=data_args.main_loss_warm,
                       pooling=training_args.pooling,
                       reverse=training_args.reverse,
                       lr_proj=training_args.lr_proj,
                       lr_encoder=training_args.lr_encoder,
                       save_pointer_choice=training_args.save_pointer_choice,
                       predict_on_rep=training_args.predict_on_rep,
                       main_loss_on_rep=training_args.main_loss_on_rep,
                       save_choice_file=save_choice_file,
                       sample_times=training_args.sample_times,
                       pointer_train_epoch=training_args.pointer_train_epoch,
                       ranking_forbiden_on_pointer=training_args.ranking_forbiden_on_pointer,
                       prob_save_file=training_args.prob_save_file,
                       prob_save_on_rep=training_args.prob_save_on_rep,
                       prob_save_path=training_args.output_dir,
                       )
    
    if trainer.args.num_train_epochs <= trainer.main_loss_warm:
        logger.warning("num_train_epochs: {} <= main_loss_warm: {}, that is, no constractive will be applied!".format(
            trainer.args.num_train_epochs,
            trainer.main_loss_warm))

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["all_optimize_steps"] = model.all_step
        metrics["obvious_neg_steps"] = model.obvious
        trainer.log_metrics("train", metrics)
        
        if model.all_step != 0:
            metrics["obvious_percentage"] = str(round((model.obvious / model.all_step) * 100, 2)) + "%"
        else:
            metrics["obvious_percentage"] = "NAN"
        metrics["loss_record"] = model.loss_record
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        all_metrics.update(metrics)
        
        trainer.save_model()  # Saves the tokenizer too for easy upload
        if hasattr(model,"pointer_projector"):
            # save the pointer by default 
            pointer_save_file = os.path.join(training_args.output_dir, "pointer.pth.tar")
            torch.save({'state_dict': model.pointer_projector.state_dict()}, pointer_save_file)  
        if hasattr(model,"pointer_encoder"):
            # save T5 Stack
            pointer_encoder_save_dir = os.path.join(training_args.output_dir, "pointer_encoder")
            model.pointer_encoder.save_pretrained(pointer_encoder_save_dir)
            # save the token_embedding
            if hasattr(model.pointer_encoder, "embed_tokens"):
                pointer_encoder_save_file = os.path.join(pointer_encoder_save_dir, "pointer_encoder_embed_tokens.pth.tar")
                torch.save({'state_dict': model.pointer_encoder.embed_tokens.state_dict()}, pointer_encoder_save_file)
                logger.info("save pointer_encoder embed_tokens to {}, shape: {}".format(pointer_encoder_save_file, model.pointer_encoder.embed_tokens.weight.shape))

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        all_metrics.update(metrics)

    if training_args.do_predict:
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
                output_prediction_file = os.path.join(training_args.output_dir, "predicted_examples.jsonl")
                with open(output_prediction_file, "w") as fout:
                    for example, prediction in zip(predict_dataset, predictions):
                        example["prediction"] = prediction
                        fout.write(json.dumps(example) + "\n")

    if (training_args.do_train or training_args.do_eval or training_args.do_predict) and trainer.is_world_process_zero():
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

    if training_args.do_demo:
        logger.info("Serving the model as a demo...")
        user_input = ''
        trainer._max_length = max_length
        trainer._num_beams = num_beams
        while True:
            user_input = input("Please enter your input to the model, or enter 'quit' to exit: ")
            if user_input.lower() == "quit":
                break
            inputs = tokenizer([user_input], return_tensors="pt")
            _, preds, _ = trainer.prediction_step(model, inputs=inputs, prediction_loss_only=False)
            print(f"Model generates: {tokenizer.decode(preds[0], skip_special_tokens=True)}\n\n")
            
    if training_args.save_pointer_choice:
        save_choice_file= os.path.join(training_args.output_dir, "pointer_choice_test.txt")  
        # because use "a" to save the choices, need to have a indicator of different exps
        with open(save_choice_file, "a") as fout:
            fout.write("="*20 + "one exp ended!" + "="*20 + "\n")
            
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()