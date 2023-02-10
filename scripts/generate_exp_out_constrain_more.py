import os
from random import sample
import numpy as np

"""
similar to generate_exp_out_constrain.py
tune the neg_out_sample_num and loss_ratio
Specifically, for each script, fix neg_out_num, change the loss ratio
"""

model_name = "t5-base"
round_all = lambda x:round(x,1)
# loss_mix_ratio_list = list(map(round_all,list(np.arange(0.1,1,0.1))))
neg_out_sample_num = [1,2,3,4,5,6,7,8,9,10]  ## TODO: my assumption is that, the neg out is iid, so the total num of neg out will impact the performance
loss_mix_ratio_list = [[0,0.05],  ## for 1  ## ,0.01,0.05,0.1,0.2,0.4,0.6
                       [0.2],  ## for 2  ## ,0.001,0.005,0.01,0.05,0.1,0.2,0.4
                       [0.1,],  ## 3  ## 0.001,0.005,0.01,0.05,0.1,0.2,0.4
                       [0.005,],  ## 4  ## 0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2
                       [0.0005],  ## 5  ## ,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2
                       [0.0005],  ## 6  ## ,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2
                       [0.1],  ## 7  ## 0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2
                       [0.001,],  ## 8  ## 0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2
                       [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2],  ## 9
                       [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2]]  ## 10
margin_neg_list = [0.01]  # ,0.001,0.003,0.01,0.03,0.1,0.3
## fixed 
sample_num_pos = [0]
sample_num_neg = [0]
lr_list = [5e-05] # 1e-05,3e-05,1e-04main_loss_warm
main_loss_warm_list = [0]
epochs = [2]
## dataset 
NEG_OUT_NUM = 5
data_dir = ["add_neg_output_pos_10_more"]  # add_neg_output, add_neg_output_pos_1_neg_0
output_dir = ["output_add_neg_out-more_{}".format(NEG_OUT_NUM)] # output_add_neg_out
## When doing neg_loss_only, increase the value of loss_mix_ratio
# if neg_loss_only:
#     print("only neg loss, so increase the ratio")
#     loss_mix_ratio_list = [1]

# for margin in margin_pos_list:
#     shell_name = "only-pos-margin_{}".format(margin)
# for ratio in loss_mix_ratio_list:
#     shell_name = "only-pos-ratio_{}".format(ratio)
for num, loss_mix_ratio in zip(neg_out_sample_num,loss_mix_ratio_list):
    shell_name = "out_constrain_num_of_{}_{}".format(num,NEG_OUT_NUM)
# for margin in margin_neg_list:
#     shell_name = "out_constrain_{}_pos_1".format(margin)
    # ======== shell head (fixed parameters) =======

    template = '''#!/bin/bash
set -x

gpu=$1
batch=$2

export model="{}"
export neg_out_sample_num="{}"
export margin_neg="{}"
export sample_num_pos="0"
export sample_num_neg="0"
export learning_rate="{}"
export main_loss_warm="{}"
export num_train_epochs="{}"
export data_dir="{}"
export output_dir="{}"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

'''

    template = template.format(model_name,num,margin_neg_list[0],lr_list[0],main_loss_warm_list[0],epochs[0],data_dir[0],output_dir[0])

    # ======== large scale experiments (tuning parameters) =========
    t = "export loss_mix_ratio='{}'"
    same = '''
export EXP="t5-base-out_constrain-num_${neg_out_sample_num}-ratio_${loss_mix_ratio}-margin_${margin_neg}-warm_${main_loss_warm}-epoch_${num_train_epochs}"
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
    --task_dir data/tasks/${data_dir}/ \\
    --output_dir ${output_dir}/ \\
    --cache_dir ./cache/ \\
    --overwrite_cache \\
    --overwrite_output_dir \\
    --per_device_train_batch_size $batch \\
    --per_device_eval_batch_size $batch \\
    --gradient_accumulation_steps 2 \\
    --learning_rate ${learning_rate} \\
    --num_train_epochs ${num_train_epochs} \\
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
    --neg_loss_type contrastive_loss_out_constrain_all \\
    --seed 42 \\
    --sample_num_pos ${sample_num_pos} \\
    --sample_num_neg ${sample_num_neg} \\
    --loss_mix_ratio ${loss_mix_ratio} \\
    --margin_neg ${margin_neg} \\
    --main_loss_warm ${main_loss_warm} \\
    --neg_out_sample_num ${neg_out_sample_num}
    '''
    
    # if neg_loss_only:
    #     same += ''' \\
    # --neg_loss_only
    #     '''
    for ratio in loss_mix_ratio:
        template += t.format(ratio) + same + "\n"


    if not shell_name.endswith(".sh"):
        shell_name += ".sh"
        
    with open(shell_name,"w",encoding="utf-8") as f:
        f.write(template)
    
    print("create {} successfully!".format(shell_name))