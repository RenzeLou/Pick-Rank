from operator import ne
import os
from random import sample
import numpy as np

"""
only consider neg example, fix sample_num to 3, loss_mix_ratio to 1
try to tune a good margin
"""

model_name = "t5-base"
round_all = lambda x:round(x,1)
# loss_mix_ratio_list = list(map(round_all,list(np.arange(0.1,1,0.1))))
loss_mix_ratio_list = [1]  # 0.1,0.3,0.6
# loss_mix_ratio_list_already = []
margin_pos_list = [0.0001,0.001,0.01,0.1]
margin_neg_list = [0.0001]  # 0.000001,0.00001,0.001
sample_num_pos = [0]
sample_num_neg = [3]
pos_neg_ratio_list = [1] 
lr_list = [1e-05,3e-05,1e-04] 

## When doing neg_loss_only, increase the value of loss_mix_ratio
# if neg_loss_only:
#     print("only neg loss, so increase the ratio")
#     loss_mix_ratio_list = [1]

# for margin in margin_neg_list:
#     shell_name = "only-neg-margin_{}".format(margin)
# for ratio in loss_mix_ratio_list:
#     shell_name = "only-neg-ratio_{}".format(ratio)
# for num in sample_num_neg:
#     shell_name = "only-neg-num_{}".format(num)
for lr in lr_list:
    shell_name = "only-neg-lr_{}".format(lr)
    # ======== shell head (fixed parameters) =======

    template = '''#!/bin/bash
set -x

gpu=$1
batch=$2

export model="{}"
export margin_pos="{}"
export margin_neg="{}"
export sample_num_pos="{}"
export sample_num_neg="{}"
export pos_neg_ratio="{}"
export loss_mix_ratio="{}"
export learning_rate="{}"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

'''

    template = template.format(model_name,margin_pos_list[0],margin_neg_list[0],sample_num_pos[0],sample_num_neg[0],pos_neg_ratio_list[0],loss_mix_ratio_list[0],lr)

    # ======== large scale experiments (tuning parameters) =========
    same = '''
export EXP="Only-NEG-margin_neg_${margin_neg}-mix_ratio_${loss_mix_ratio}-num_${sample_num_neg}-lr_${learning_rate}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_test_pos.py \\
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
    --task_dir data/tasks/pos_neg_mixed_v3 \\
    --output_dir output/ \\
    --cache_dir ./cache/ \\
    --overwrite_cache \\
    --overwrite_output_dir \\
    --per_device_train_batch_size $batch \\
    --per_device_eval_batch_size $batch \\
    --gradient_accumulation_steps 2 \\
    --learning_rate ${learning_rate} \\
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
    --neg_loss_type contrastive_loss_max_v4 \\
    --seed 42 \\
    --sample_num_pos ${sample_num_pos} \\
    --sample_num_neg ${sample_num_neg} \\
    --loss_mix_ratio ${loss_mix_ratio} \\
    --margin_pos ${margin_pos} \\
    --margin_neg ${margin_neg} \\
    --pos_neg_ratio ${pos_neg_ratio}
    '''
    
    # if neg_loss_only:
    #     same += ''' \\
    # --neg_loss_only
    #     '''
    
    already_exp = False
    # for margin in margin_list:
    #     if margin in margin_list_already and loss_mix_ratio in loss_mix_ratio_list_already:
    #         already_exp = True
    #         continue
    #     template += t.format(margin) + same + "\n"
    # for ratio in loss_mix_ratio_list:
    #     for loss_func in neg_loss_type_list:
    #         for l in lr:
    #             for e in epoch:
    #                 template += t1.format(loss_func) + t2.format(ratio) + t3.format(l) + t4.format(e) + same + "\n"
    
    # already_exp = any([True for margin in margin_list if margin in margin_list_already])
    template += same + "\n"


    if not shell_name.endswith(".sh"):
        shell_name += ".sh"
        
    # if already_exp:
    #     save_dir = "./already"
    #     os.makedirs(save_dir,exist_ok=True)
    #     with open(os.path.join(save_dir,shell_name),"w",encoding="utf-8") as f:
    #         f.write(template)
    # else:
    with open(shell_name,"w",encoding="utf-8") as f:
        f.write(template)
    
    print("create {} successfully!".format(shell_name))