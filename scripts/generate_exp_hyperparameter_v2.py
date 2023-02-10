import os
from random import sample
import numpy as np

"""
generate several scripts, each script correspondents to a specific loss_mix_ratio (fix sample_num == 1),
in every script, there are a series of experiments to find a correct margin for the corresponding ratio,
that is, each script contains the experiments with one fixed ratio and several different margins.
"""

model_name = "t5-base"
# sample_num_list = [1,2,3,4,5]  ## there are 5 kinds of negative instructions
sample_num = 1
round_all = lambda x:round(x,1)
# loss_mix_ratio_list = list(map(round_all,list(np.arange(0.1,1,0.1))))
loss_mix_ratio_list = [0.00001,0.00003,0.00006,0.0001,0.0003,0.0006,0.001,0.003,0.006,0.01,0.03,0.06,0.1,0.3,0.6,1]  ## TODO: 0.1,0.3,0.6,1  remaining exps
loss_mix_ratio_list_already = [0.0001,0.0003,0.001,0.003,0.01,0.03]
# margin_list = [0.1,0.3,1,3,10,30]  # for log
margin_list = [0.0001,0.0003,0.0006,0.001,0.003,0.006,0.01,0.03,0.06,0.1,0.3]  # for no log
margin_list_already = [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3]


for loss_mix_ratio in loss_mix_ratio_list:
    shell_name = "{}-search_ratio_{}".format(model_name,loss_mix_ratio)

    # ======== shell head (fixed parameters) =======

    template = '''#!/bin/bash
set -x

gpu=$1
batch=$2

export model="{}"
export sample_num="{}"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

export loss_mix_ratio="{}"
'''

    template = template.format(model_name,sample_num,loss_mix_ratio)

    # ======== large scale experiments (tuning parameters) =========
    t = 'export margin="{}"'
    same = '''
export EXP="${model}-sample-${sample_num}-ratio-${loss_mix_ratio}-margin-${margin}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_test.py \\
    --do_train \\
    --do_predict \\
    --predict_with_generate \\
    --model_name_or_path $model \\
    --max_source_length 1024 \\
    --max_target_length 128 \\
    --generation_max_length 128 \\
    --max_num_instances_per_task 100 \\
    --max_num_instances_per_eval_task 100 \\
    --add_task_name False \\
    --add_task_definition True \\
    --num_pos_examples 2 \\
    --num_neg_examples 0 \\
    --add_explanation False \\
    --tk_instruct False \\
    --data_dir data/splits/default \\
    --task_dir data/tasks/neg_mixed \\
    --output_dir output/ \\
    --cache_dir ./cache/ \\
    --overwrite_cache \\
    --per_device_train_batch_size $batch \\
    --per_device_eval_batch_size $batch \\
    --gradient_accumulation_steps 2 \\
    --learning_rate 5e-05 \\
    --num_train_epochs 2 \\
    --lr_scheduler_type constant \\
    --warmup_steps 0 \\
    --logging_strategy steps \\
    --logging_steps 500 \\
    --evaluation_strategy no \\
    --save_strategy no \\
    --save_steps 2500 \\
    --deepspeed ds_configs/stage2.config \\
    --bf16 \\
    --run_name t5-experiment \\
    --exp_name $EXP \\
    --seed 42 \\
    --sample_num $sample_num \\
    --loss_mix_ratio $loss_mix_ratio \\
    --margin $margin
    '''
    already_exp = False
    for margin in margin_list:
        if margin in margin_list_already and loss_mix_ratio in loss_mix_ratio_list_already:
            already_exp = True
            continue
        template += t.format(margin) + same + "\n"

    # already_exp = any([True for margin in margin_list if margin in margin_list_already])

    if not shell_name.endswith(".sh"):
        shell_name += ".sh"
        
    if already_exp:
        save_dir = "./already"
        os.makedirs(save_dir,exist_ok=True)
        with open(os.path.join(save_dir,shell_name),"w",encoding="utf-8") as f:
            f.write(template)
    else:
        with open(shell_name,"w",encoding="utf-8") as f:
            f.write(template)
    
    print("create {} successfully!".format(shell_name))