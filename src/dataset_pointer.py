# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""Natural Instruction V2 Dataset."""

'''
This script is used to process and cache the Natural Instruction V2 dataset.
'''


import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
"""

_DESCRIPTION = """
Natural-Instructions v2 is a benchmark of 1,600+ diverse language tasks and their expert-written instructions. 
It covers 70+ distinct task types, such as tagging, in-filling, and rewriting. 
These tasks are collected with contributions of NLP practitioners in the community and 
through an iterative peer review process to ensure their quality. 
"""

_URL = "https://instructions.apps.allenai.org/"

class NIConfig(datasets.BuilderConfig):
    def __init__(self, *args, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None,
                 sample_num_neg=None, sample_num_pos=None,training_ins_num=None,
                 neg_loss_type=None,null_loss_type=None,pos_loss_type=None,last_sen_num=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.sample_num_neg: int = sample_num_neg
        self.sample_num_pos: int = sample_num_pos
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task
        self.training_ins_num = training_ins_num if training_ins_num is not None else 75317  ## a pre-defined training instance num, equal to the setting of the baseline
        self.current_training_ins_num = 0
        self.neg_loss_type=neg_loss_type
        self.null_loss_type=null_loss_type
        self.pos_loss_type=pos_loss_type
        self.last_sen_num=last_sen_num

class NaturalInstructions(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = NIConfig
    BUILDER_CONFIGS = [
        NIConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            ## this is the features for each example
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Contributors": datasets.Value("string"),
                    "Source": [datasets.Value("string")],
                    "URL": [datasets.Value("string")],
                    "Categories": [datasets.Value("string")],
                    "Reasoning": [datasets.Value("string")],
                    "Definition": [datasets.Value("string")],
                    "Positive Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Negative Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Input_language": [datasets.Value("string")],
                    "Output_language": [datasets.Value("string")],
                    "Instruction_language": [datasets.Value("string")],
                    "Domains": [datasets.Value("string")],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        "output": [datasets.Value("string")],
                        "neg_output": [datasets.Value("string")]
                    },
                    "Instance License": [datasets.Value("string")]
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/allenai/natural-instructions",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            dl_path = dl_manager.download_and_extract(_URL)
            self.config.data_dir = self.config.data_dir or os.path.join(dl_path, "splits")
            self.config.task_dir = self.config.task_dir or os.path.join(dl_path, "tasks")

        split_dir = self.config.data_dir
        task_dir = self.config.task_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(split_dir, "train_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(split_dir, "dev_tasks.txt"), 
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(split_dir, "test_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test"
                }),
        ]

    def _generate_examples(self, path=None, task_dir=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
        with open(path, encoding="utf-8") as split_f:
            for line in split_f:
                task_name = line.strip()
                task_path = os.path.join(task_dir, task_name + ".json")
                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)  ## a dict format of each json file
                    task_data["Task"] = task_name
                    if "Instruction Source" in task_data:
                        task_data.pop("Instruction Source")
                    all_instances = task_data.pop("Instances")
                    # get all the individual sentences in the definitions, i.e., ''Definition_POS''
                    # note that ``Definition_NEG"'' is derecated
                    neg_def = task_data.pop("Definition_NEG") if task_data.get("Definition_NEG",None) is not None else []
                    pos_def = task_data.pop("Definition_POS") if task_data.get("Definition_POS",None) is not None else []
                    # randomly sample neg definitions
                    sample_num_neg = self.config.sample_num_neg
                    sample_num_neg = len(neg_def) if sample_num_neg is None else sample_num_neg
                    sample_num_pos = self.config.sample_num_pos
                    sample_num_pos = len(pos_def) if sample_num_pos is None else sample_num_pos
                    
                    pos_def_sampled = []
                    sample_num_pos = min(len(pos_def),sample_num_pos)
                    if self.config.last_sen_num is not None:
                        # during our experiments, we found that the last few sentences of the definition are usually more important
                        # speficially, if we only use the last sentence, the performance is much better than only using the first sentence:
                        '''
                        definition      Rouge-L
                        last:           27.9514
                        first:          14.2735
                        first + last:   28.1063
                        '''
                        # that's why we have these lines of code sampling the last few sentences
                        # however, this assumption didn't provide much improvement on our method, so just ignore it
                        sample_num_pos = sample_num_pos - self.config.last_sen_num
                        if sample_num_pos > 0:
                            pos_def_sampled = random.sample(pos_def[:len(pos_def)-self.config.last_sen_num],sample_num_pos)
                            pos_def_sampled = pos_def_sampled + pos_def[len(pos_def)-self.config.last_sen_num:]
                        elif sample_num_pos == 0:
                            pos_def_sampled = pos_def[len(pos_def)-self.config.last_sen_num:]
                        else:
                            if self.config.last_sen_num >= len(pos_def):
                                pos_def_sampled = pos_def
                            else:
                                pos_def_sampled = pos_def[len(pos_def)-self.config.last_sen_num:]
                    else:
                        # note that the validation and test set do not have neg definition
                        pos_def_sampled = random.sample(pos_def,sample_num_pos)
                    
                    # there could be multiple definitions
                    ori_def = [" ".join(task_data["Definition"])]
                    assert len(neg_def) == 0
                    assert len(pos_def) > 0, "we need all seperated sentences in the original definition."
                    # the first def is the original def
                    # the following defs are the sampled individual sentences from the original def
                    all_def = ori_def + pos_def_sampled
                    if subset == "test" or subset == "dev":
                        # for testing tasks, 100 instances are selected for efficient evaluation and they are label-balanced.
                        # we put them in the first for reproducibility.
                        # so, we use them here
                        instances = all_instances[:100]
                    else:
                        instances = all_instances
                        if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                            # select those instances having ``neg_output'' as training examples
                            instances_has_neg, instances_no_neg = [],[]
                            for instance in instances:
                                if len(instance["neg_output"]) > 0:
                                    instances_has_neg.append(instance)
                                else:
                                    instances_no_neg.append(instance)
                            # pad the instances_has_neg to meet the ins num requirment 
                            if len(instances_has_neg) >= max_num_instances_per_task:
                                random.shuffle(instances_has_neg)
                                instances = instances_has_neg[:max_num_instances_per_task]
                            else:
                                random.shuffle(instances_no_neg)
                                pad_num = max_num_instances_per_task - len(instances_has_neg)
                                instances = instances_has_neg + instances_no_neg[:pad_num]
                                random.shuffle(instances)
                            # instances = instances[:max_num_instances_per_task]
                    for idx, instance in enumerate(instances):
                        example = task_data.copy()
                        example["id"] = instance["id"]
                        if subset == "test" or subset == "dev" or self.config.pos_loss_type is None:
                            ## note that if pos_loss_type is none, means we don not do the output constrain,
                            ## there is no need to add "neg_output" in the instance
                            instance["neg_output"] = []
                        example["Instance"] = instance  ## instance is a dict containing input, output, id, etc.
                        example["Definition"] = all_def
                        if subset == "train":
                            self.config.current_training_ins_num += 1
                        yield f"{task_name}_{idx}", example