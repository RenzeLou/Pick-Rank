from operator import ne
import os
from random import sample
import numpy as np

"""
generate several scripts, each script correspondents to a specific margin,
fix the sample_num
change the loss_function, ratio, lr, epoch
to find a proporate combination of those hyper-parameter (margin,ratio,loss, lr and epoch) for sample_num == 9 
"""

model_name = "t5-base"
sample_num_list = [9]  ## 1,2,3,4,5,6 ## there are 5 kinds of negative instructions
# sample_num = 1
round_all = lambda x:round(x,1)
# loss_mix_ratio_list = list(map(round_all,list(np.arange(0.1,1,0.1))))
loss_mix_ratio_list = [0.001,0.005,0.01,0.05,0.1,0.5,1]  ## seems like the 0.001 become the best? 
loss_mix_ratio_list_already = []
# margin_list = [0.1,0.3,1,3,10,30]  # for log
margin_list = [0.0001,0.0003,0.001,0.003,0.01,0.03,]  # for no log
margin_list_already = []
epoch = [2]
lr = [5e-05,] # 1e-05,3e-05,
neg_loss_type_list = ["contrastive_loss_all"]  # "contrastive_loss_max_v2","contrastive_loss_all","contrastive_loss_softmax"
# neg_loss_only = False  

## When doing neg_loss_only, increase the value of loss_mix_ratio
# if neg_loss_only:
#     print("only neg loss, so increase the ratio")
#     loss_mix_ratio_list = [1]

for margin in margin_list:
    shell_name = "all-search_margin_{}".format(margin)
    # ======== shell head (fixed parameters) =======

    template = '''#!/bin/bash
set -x

gpu=$1
batch=$2

export model="{}"
export margin="{}"
export sample_num="{}"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

'''

    template = template.format(model_name,margin,sample_num_list[0])

    # ======== large scale experiments (tuning parameters) =========
    t1 = 'export neg_loss_type="{}"\n'
    t2 = 'export loss_mix_ratio="{}"\n'
    t3 = 'export learning_rate="{}"\n'
    t4 = 'export num_train_epochs="{}"\n'
    
    same = '''
export EXP="all-margin_${margin}-num_${sample_num}-ratio_${loss_mix_ratio}-func_${neg_loss_type}-lr_${learning_rate}-epoch_${num_train_epochs}"
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
    --task_dir data/tasks/neg_mixed_v2 \\
    --output_dir output/ \\
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
    --seed 42 \\
    --sample_num $sample_num \\
    --loss_mix_ratio ${loss_mix_ratio} \\
    --margin $margin \\
    --neg_loss_type ${neg_loss_type}
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
    for ratio in loss_mix_ratio_list:
        for loss_func in neg_loss_type_list:
            for l in lr:
                for e in epoch:
                    template += t1.format(loss_func) + t2.format(ratio) + t3.format(l) + t4.format(e) + same + "\n"
    
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