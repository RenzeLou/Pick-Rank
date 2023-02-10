import os
from random import sample

"""
fix the mix ratio to 0.5, search a best sample num
"""

model_name = "t5-base"
loss_mix_ratio = 0.5
sample_num_list = [0,1,2,3,4,5]  ## there are 5 kinds of negative instructions

shell_name = "{}-search_sample_num.sh".format(model_name)

# ======== shell head (fixed parameters) =======

template = '''#!/bin/bash
set -x

gpu=$1
batch=$2

export model="{}"
export loss_mix_ratio="{}"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

'''

template = template.format(model_name,loss_mix_ratio)

# ======== large scale experiments (tuning parameters) =========
t = 'export sample_num="{}"'
same = '''
export EXP="tuning-${model}-sample-${sample_num}"
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
    --overwrite_output_dir \\
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
    --save_strategy epoch \\
    --save_steps 2500 \\
    --deepspeed ds_configs/stage2.config \\
    --bf16 \\
    --run_name t5-experiment \\
    --exp_name $EXP \\
    --seed 42 \\
    --sample_num $sample_num \\
    --loss_mix_ratio $loss_mix_ratio
'''

for sample_num in sample_num_list:
    template += t.format(sample_num) + same + "\n"

if not shell_name.endswith(".sh"):
    shell_name += ".sh"
    
with open(shell_name,"w",encoding="utf-8") as f:
    f.write(template)
    
print("create {} successfully!".format(shell_name))