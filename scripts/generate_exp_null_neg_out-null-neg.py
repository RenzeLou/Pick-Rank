from operator import ne
import os
from random import sample
import numpy as np

"""
only consider pos example, fix sample_num to 2, loss_mix_ratio to 1
try to tune a good margin
"""

model_name = "t5-base"
round_all = lambda x:round(x,1)
# loss_mix_ratio_list = list(map(round_all,list(np.arange(0.1,1,0.1))))
loss_mix_ratio_null_list = [0.01]  # 0.1,0.3,0.6
# loss_mix_ratio_list_already = []
margin_null_list = [0.001]

loss_mix_ratio_neg_list = [0.0001,0.001,0.003,0.01,0.03,0.1,0.3]  # 0.1,0.3,0.6 TODO: i am not sure about the range
margin_neg_list = [0.0001,0.001,0.01,0.006,0.003,0.03,0.06]

sample_num_pos = [1]
sample_num_neg = [1]
# pos_neg_ratio_list = [1]
lr_list = [5e-05] 
out_dir = "output_null_neg"

## When doing neg_loss_only, increase the value of loss_mix_ratio
# if neg_loss_only:
#     print("only neg loss, so increase the ratio")
#     loss_mix_ratio_list = [1]

# for margin in margin_pos_list:
#     shell_name = "only-pos-margin_{}".format(margin)
# for ratio in loss_mix_ratio_null_list:
#     shell_name = "null-ratio_{}".format(ratio)
for ratio in loss_mix_ratio_neg_list:
    shell_name = "neg-ratio_{}".format(ratio)
# for lr in lr_list:
#     shell_name = "only-pos-lr_{}".format(lr)
    # ======== shell head (fixed parameters) =======

    template = '''#!/bin/bash
set -x

gpu=$1
batch=$2

export model="{}"
export sample_num_pos="{}"
export sample_num_neg="{}"
export null_loss_type="contrastive_loss_max_v4"
export neg_loss_type="contrastive_loss_max_v4"
export loss_mix_ratio_null="{}"
export loss_mix_ratio_neg="{}"
export margin_null="{}"
export learning_rate="{}"
export out_dir="{}"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

'''

    template = template.format(model_name,sample_num_pos[0],sample_num_neg[0],loss_mix_ratio_null_list[0],ratio,margin_null_list[0],lr_list[0],out_dir)

    # ======== large scale experiments (tuning parameters) =========
    t = 'export margin_neg="{}"'
    same = '''
export EXP="ratio_${loss_mix_ratio_neg}-margin_${margin_neg}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_output_constrain.py \\
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
    --num_pos_examples 0 \\
    --num_neg_examples 0 \\
    --add_explanation False \\
    --tk_instruct False \\
    --data_dir data/splits/default \\
    --task_dir data/tasks/def_pos2_neg_3-add_neg_out \\
    --output_dir ${out_dir}/ \\
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
    --null_loss_type ${null_loss_type} \\
    --neg_loss_type ${neg_loss_type} \\
    --seed 42 \\
    --sample_num_pos ${sample_num_pos} \\
    --sample_num_neg ${sample_num_neg} \\
    --loss_mix_ratio_null ${loss_mix_ratio_null} \\
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \\
    --margin_null ${margin_null} \\
    --margin_neg ${margin_neg}    
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
    for m in margin_neg_list:
        template += t.format(m) + same + "\n"


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