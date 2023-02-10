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
from dataclasses import dataclass, field
from typing import Optional
import wandb

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets.utils import set_progress_bar_enabled
from datasets import load_dataset, load_metric

import transformers
from transformers.utils import logging as hug_logging
from filelock import FileLock
from transformers import (
    AutoConfig,
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
# from ni_collator_augment_v2 import DataCollatorForNI
from ni_collator_first_last_repeat import DataCollatorForNI
# from ni_trainer_augment_v2 import NITrainer, DenserEvalCallback
from ni_trainer_first_last_repeat import NITrainer, DenserEvalCallback
from compute_metrics import compute_grouped_metrics_add_f, compute_grouped_metrics_add_f_v2, compute_metrics, compute_grouped_metrics, compute_metrics_add_f
from modeling_t5_first_last_repeat import T5ForConditionalGeneration_neg
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
    random_words_neg_def: Optional[bool] = field(
        default=False
    )
    words_num: int = field(
        default=1
    )
    q_k_projector_dim: Optional[int] = field(
        default = 768,  
    )
    memory_projector_dim: Optional[int] = field(
        default = 768,  
    )
    add_attention_projector: Optional[bool] = field(
        default=False, metadata={"help": "whether to add attention projector."} 
    )
    add_memory_projector: Optional[bool] = field(
        default=False 
    )
    # pooling : Optional[str] = field(
    #     default="mean", metadata={"help": "pooling method for the attention projector."}
    # )
    reverse: Optional[bool] = field(
        default=False 
    )
    pooling_memory : Optional[str] = field(
        default="mean" 
    )
    pooling_att: Optional[str] = field(
        default="max",  
    )
    lr_proj: Optional[float] = field(
        default=None,  
    )
    use_first: Optional[bool] = field(
        default=False,  
    )
    use_last: Optional[bool] = field(
        default=False,  
    )
    use_first_last_random: Optional[bool] = field(
        default=False, 
    )
    add_tk_sen_projector: Optional[bool] = field(
        default=False  ###
    )
    tk_sen_projector_dim: Optional[int] = field(
        default = 768,  ###
    )

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

    # Get the NaturalInstructions dataset
    # "src/ni_dataset_argument.py"
    # if not training_args.random_words_neg_def:
    raw_datasets = load_dataset( 
        "src/ni_dataset_first_last_repeat.py", 
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
        out_loss_type=training_args.out_loss_type,
        use_first=training_args.use_first,
        use_last=training_args.use_last,
        use_first_last_random=training_args.use_first_last_random,
    )
    # else:
    #     raw_datasets = load_dataset(
    #         "src/ni_dataset_argument_output_constrain_random_words_neg.py", 
    #         data_dir=data_args.data_dir,   ## train/test/dev split
    #         task_dir=data_args.task_dir,   ## all data along with instructions
    #         cache_dir=model_args.cache_dir,
    #         max_num_instances_per_task=data_args.max_num_instances_per_task,
    #         max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
    #         sample_num_pos=data_args.sample_num_pos,
    #         sample_num_neg=data_args.sample_num_neg,
    #         training_ins_num=training_args.training_ins_num,
    #         add_task_definition_train=training_args.add_task_definition_train,
    #         add_task_definition_test=training_args.add_task_definition_test,
    #         neg_loss_type=training_args.neg_loss_type,
    #         null_loss_type=training_args.null_loss_type,
    #         out_loss_type=training_args.out_loss_type,
    #         words_num=training_args.words_num
    #     )

    # Load pretrained model and tokenizer
    #
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
    ## add special tokens
    tokenizer.add_special_tokens({"additional_special_tokens": ["[REPEAT]"]})
    
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
    
    # if training_args.add_attention_projector:
    #     model.add_q_k_projector(config.d_model,training_args.q_k_projector_dim)
    # else:
    #     model.add_q_k_projector(None,None)
    
    # if training_args.add_memory_projector:
    #     model.add_memory_projector(3*config.d_model,training_args.memory_projector_dim)
    #     model.add_hidden_projector(training_args.memory_projector_dim+config.d_model,config.d_model)
    # else:
    #     model.add_memory_projector(None,None)
    #     model.add_hidden_projector(4*config.d_model,config.d_model)
    # if training_args.diff:
    #     if training_args.add_tk_sen_projector:
    #         model.add_tk_sen_projector(config.d_model,training_args.tk_sen_projector_dim)
    #         model.add_hidden_projector(2*training_args.tk_sen_projector_dim,config.d_model)
    #     else:
    #         model.add_tk_sen_projector(None,None)
    #         model.add_hidden_projector(2*config.d_model,config.d_model)
    # else:
    #     model.add_hidden_projector(2*config.d_model,config.d_model)

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
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
    )
    
    # add mix ratio
    trainer.init_hyper(pos_neg_ratio=training_args.pos_neg_ratio,margin_pos=training_args.margin_pos,
                       margin_null=training_args.margin_null,margin_out=training_args.margin_out,
                       margin_neg=training_args.margin_neg,neg_loss_type=training_args.neg_loss_type,
                       null_loss_type=training_args.null_loss_type,out_loss_type=training_args.out_loss_type,
                       loss_mix_ratio_neg=training_args.loss_mix_ratio_neg,
                       loss_mix_ratio_null=training_args.loss_mix_ratio_null,
                       loss_mix_ratio_out=training_args.loss_mix_ratio_out,
                       sample_num_neg=data_args.sample_num_neg,
                       sample_num_pos=data_args.sample_num_pos,main_loss_warm=data_args.main_loss_warm,
                       pooling_memory=training_args.pooling_memory,pooling_att=training_args.pooling_att,
                       reverse=training_args.reverse,
                       lr_proj=training_args.lr_proj)
    if trainer.args.num_train_epochs <= trainer.main_loss_warm:
        logger.warning("num_train_epochs: {} <= main_loss_warm: {}, that is, no constractive will be applied!".format(
            trainer.args.num_train_epochs,
            trainer.main_loss_warm))
    # trainer.init_hyper(loss_mix_ratio=training_args.loss_mix_ratio,margin=training_args.margin,neg_loss_only=training_args.neg_loss_only,neg_loss_type=training_args.neg_loss_type)

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

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

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()