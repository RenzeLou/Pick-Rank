import os

shell_name = "T5-base-pos"
model = ["t5-base"]
seed = [42]

train_template = "python src/run_s2s_test.py --do_train --do_predict --predict_with_generate \
                --max_source_length 1024 --max_target_length 128 --generation_max_length 128 \
                --max_num_instances_per_task 100 --max_num_instances_per_eval_task 100 \
                --add_task_name False --add_task_definition True \
                --num_pos_examples 2 --num_neg_examples 0 \
                --add_explanation False --tk_instruct False \
                --data_dir data/splits/default --task_dir data/tasks --output_dir output/ --overwrite_output_dir --cache_dir ./cache/ --overwrite_cache \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --gradient_accumulation_steps 2 --learning_rate 5e-05 --num_train_epochs 2 \
                --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps \
                --logging_steps 500 --evaluation_strategy no --save_strategy steps \
                --save_steps 2500 --deepspeed ds_configs/stage2.config \
                --bf16 \
                --run_name t5-experiment \
                --exp_name {} \
                --model_name_or_path {} --seed {} \n"

eval_template = "python src/run_s2s_test.py --do_predict --predict_with_generate \
                --evaluation_strategy no --max_source_length 1024 --max_target_length 128 \
                --generation_max_length 128 --max_num_instances_per_task 100 --max_num_instances_per_eval_task 100 \
                --add_task_name False --add_task_definition True --num_pos_examples 2 --num_neg_examples 0 \
                --add_explanation False --tk_instruct False \
                --data_dir data/splits/default --task_dir data/tasks \
                --output_dir output/ --overwrite_output_dir --cache_dir ./cache/ \
                --overwrite_cache --per_device_eval_batch_size 32 \
                --exp_name {} \
                --model_name_or_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/output/{}/checkpoint-7500 \
                --seed {} \n"
                
template = """
gpu=$1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface

echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
"""
template += "\n"

for m in model:
    for s in seed:
        template += train_template.format(shell_name,m,s)
        template += eval_template.format(shell_name,shell_name,s)

with open("{}.sh".format(shell_name),"w") as f:
    f.write(template)

print("generate shell: {}".format(shell_name+".sh"))
        
'''
--max_source_length 1024 --max_target_length 128 --generation_max_length 128 \
                --max_num_instances_per_task 100 --max_num_instances_per_eval_task 100 \
                --add_task_name False --add_task_definition True \
                --num_pos_examples 2 --num_neg_examples 0 \
                --add_explanation False --tk_instruct False \
                --data_dir data/splits/default --task_dir data/tasks --output_dir output/ --overwrite_output_dir --cache_dir ./cache/ --overwrite_cache \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --gradient_accumulation_steps 2 --learning_rate 5e-05 --num_train_epochs 2 \
                --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps \
                --logging_steps 500 --evaluation_strategy no --save_strategy steps \
                --save_steps 2500 --deepspeed ds_configs/stage2.config \
                --bf16 \
                --run_name t5-experiment \
                    
--evaluation_strategy no --max_source_length 1024 --max_target_length 128 \
                --generation_max_length 128 --max_num_instances_per_task 100 --max_num_instances_per_eval_task 100 \
                --add_task_name False --add_task_definition True --num_pos_examples 2 --num_neg_examples 0 \
                --add_explanation False --tk_instruct False \
                --data_dir data/splits/default --task_dir data/tasks \
                --output_dir output/ --overwrite_output_dir --cache_dir ./cache/ \
                --overwrite_cache --per_device_eval_batch_size 32 \
'''