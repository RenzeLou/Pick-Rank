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
    def __init__(self, *args, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, sample_num=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.sample_num: int = sample_num
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task


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
                    # "Definition_NEG": [datasets.Value("string")],
                    "indicator": [datasets.Value("string")],  ## indicate whether the definition is pos (1) or neg (0)
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
                    # "Instances": [{
                    #     "input": datasets.Value("string"),
                    #     "output": [datasets.Value("string")]
                    # }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        "output": [datasets.Value("string")]
                    },
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
                    # get negative definitions
                    neg_def = task_data.pop("Definition_NEG")
                    # randomly sample neg definitions
                    sample_num = self.config.sample_num
                    sample_num = len(neg_def) if sample_num is None else sample_num
                    # note that the validation and test set do not have neg definition
                    sample_num = min(len(neg_def),sample_num)
                    neg_def_sampled = random.sample(neg_def,sample_num)
                    # create indicators
                    neg_indicator = [0] * len(neg_def_sampled)
                    # there could be multiple definitions
                    pos_def = [" ".join(task_data["Definition"])]
                    all_def = pos_def + neg_def_sampled
                    ## all_indicator = [1] * len(task_data["Definition"]) + neg_indicator
                    all_indicator = [1] + neg_indicator
                    # make sure the length
                    assert len(all_def) == len(all_indicator)
                    if subset == "test":
                        # for testing tasks, 100 instances are selected for efficient evaluation and they are label-balanced.
                        # we put them in the first for reproducibility.
                        # so, we use them here
                        assert len(all_def) == 1
                        instances = all_instances[:100]
                    else:
                        instances = all_instances
                    if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                        random.shuffle(instances)
                        instances = instances[:max_num_instances_per_task]
                    # concat each instance with the meta information (i.e., task definition, pos/neg examples)
                    idx = -1
                    for _, instance in enumerate(instances):
                        for _idx,(df,indicator) in enumerate(zip(all_def,all_indicator)):
                            idx += 1
                            example = task_data.copy()
                            example["id"] = instance["id"] + "-" + str(_idx)
                            example["Instance"] = instance  ## instance is a dict containing input, output, id, etc.
                            example["Definition"] = [df]
                            assert len(example["Definition"]) == 1
                            example["indicator"] = [str(indicator)]
                            yield f"{task_name}_{idx}", example

