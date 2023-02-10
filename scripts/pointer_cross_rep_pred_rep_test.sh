#!/bin/bash
# only cross_entropy loss
# use repeated def to train and test
set -x

gpu=$1
batch=$2
batch2=$3
sample_num_pos=$4

export model="t5-base"
export sample_num_pos=$sample_num_pos
export learning_rate="5e-05"
export out_dir="output_pointer-cross_rep-pred_rep"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

export pooling="mean"
export lr_proj=3e-4

export pointer_hidden_dim=128
export act_func="tanh"

export EXP="pointer_hidden_dim_${pointer_hidden_dim}-act_func_${act_func}-sample_num_${sample_num_pos}"
echo "experiment name: $EXP"

# python src/run_s2s_pointer.py \
#     --do_train \
#     --model_name_or_path $model \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --generation_max_length 128 \
#     --max_num_instances_per_task 100 \
#     --max_num_instances_per_eval_task 100 \
#     --add_task_name False \
#     --add_task_definition True \
#     --num_pos_examples 0 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --tk_instruct False \
#     --data_dir data/splits/default \
#     --task_dir data/tasks/pos_neg_def_segmentation \
#     --output_dir ${out_dir}/ \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size $batch \
#     --per_device_eval_batch_size $batch \
#     --gradient_accumulation_steps 2 \
#     --learning_rate ${learning_rate} \
#     --num_train_epochs 2 \
#     --lr_scheduler_type constant \
#     --warmup_steps 0 \
#     --logging_strategy steps \
#     --logging_steps 500 \
#     --evaluation_strategy no \
#     --save_strategy no \
#     --save_steps 2500 \
#     --bf16 \
#     --run_name t5-experiment \
#     --exp_name $EXP \
#     --seed 42 \
#     --sample_num_pos ${sample_num_pos} \
#     --lr_proj ${lr_proj} \
#     --pooling ${pooling} \
#     --pointer_hidden_dim ${pointer_hidden_dim} \
#     --act_func ${act_func} \
#     --save_pointer_choice \
#     --predict_on_rep \
#     --main_loss_on_rep

python src/run_s2s_pointer.py \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $out_dir/$EXP \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/default \
    --task_dir data/tasks/pos_neg_def_segmentation \
    --output_dir ${out_dir}/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size $batch2 \
    --per_device_eval_batch_size $batch2 \
    --gradient_accumulation_steps 2 \
    --learning_rate ${learning_rate} \
    --num_train_epochs 2 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --pointer_hidden_dim ${pointer_hidden_dim} \
    --act_func ${act_func} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --pointer_checkpoint pointer.pth.tar

export EXP="pointer_hidden_dim_None-sample_num_${sample_num_pos}"
echo "experiment name: $EXP"

# python src/run_s2s_pointer.py \
#     --do_train \
#     --model_name_or_path $model \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --generation_max_length 128 \
#     --max_num_instances_per_task 100 \
#     --max_num_instances_per_eval_task 100 \
#     --add_task_name False \
#     --add_task_definition True \
#     --num_pos_examples 0 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --tk_instruct False \
#     --data_dir data/splits/default \
#     --task_dir data/tasks/pos_neg_def_segmentation \
#     --output_dir ${out_dir}/ \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size $batch \
#     --per_device_eval_batch_size $batch \
#     --gradient_accumulation_steps 2 \
#     --learning_rate ${learning_rate} \
#     --num_train_epochs 2 \
#     --lr_scheduler_type constant \
#     --warmup_steps 0 \
#     --logging_strategy steps \
#     --logging_steps 500 \
#     --evaluation_strategy no \
#     --save_strategy no \
#     --save_steps 2500 \
#     --bf16 \
#     --run_name t5-experiment \
#     --exp_name $EXP \
#     --seed 42 \
#     --sample_num_pos ${sample_num_pos} \
#     --lr_proj ${lr_proj} \
#     --pooling ${pooling} \
#     --act_func ${act_func} \
#     --save_pointer_choice \
#     --predict_on_rep \
#     --main_loss_on_rep

python src/run_s2s_pointer.py \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $out_dir/$EXP \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/default \
    --task_dir data/tasks/pos_neg_def_segmentation \
    --output_dir ${out_dir}/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size $batch2 \
    --per_device_eval_batch_size $batch2 \
    --gradient_accumulation_steps 2 \
    --learning_rate ${learning_rate} \
    --num_train_epochs 2 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --act_func ${act_func} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --pointer_checkpoint pointer.pth.tar

export act_func="relu"

export EXP="pointer_hidden_dim_${pointer_hidden_dim}-act_func_${act_func}-sample_num_${sample_num_pos}"
echo "experiment name: $EXP"

# python src/run_s2s_pointer.py \
#     --do_train \
#     --model_name_or_path $model \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --generation_max_length 128 \
#     --max_num_instances_per_task 100 \
#     --max_num_instances_per_eval_task 100 \
#     --add_task_name False \
#     --add_task_definition True \
#     --num_pos_examples 0 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --tk_instruct False \
#     --data_dir data/splits/default \
#     --task_dir data/tasks/pos_neg_def_segmentation \
#     --output_dir ${out_dir}/ \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size $batch \
#     --per_device_eval_batch_size $batch \
#     --gradient_accumulation_steps 2 \
#     --learning_rate ${learning_rate} \
#     --num_train_epochs 2 \
#     --lr_scheduler_type constant \
#     --warmup_steps 0 \
#     --logging_strategy steps \
#     --logging_steps 500 \
#     --evaluation_strategy no \
#     --save_strategy no \
#     --save_steps 2500 \
#     --bf16 \
#     --run_name t5-experiment \
#     --exp_name $EXP \
#     --seed 42 \
#     --sample_num_pos ${sample_num_pos} \
#     --pooling ${pooling} \
#     --lr_proj ${lr_proj} \
#     --pointer_hidden_dim ${pointer_hidden_dim} \
#     --act_func ${act_func} \
#     --save_pointer_choice \
#     --predict_on_rep \
#     --main_loss_on_rep

python src/run_s2s_pointer.py \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $out_dir/$EXP \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/default \
    --task_dir data/tasks/pos_neg_def_segmentation \
    --output_dir ${out_dir}/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size $batch2 \
    --per_device_eval_batch_size $batch2 \
    --gradient_accumulation_steps 2 \
    --learning_rate ${learning_rate} \
    --num_train_epochs 2 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --pooling ${pooling} \
    --lr_proj ${lr_proj} \
    --pointer_hidden_dim ${pointer_hidden_dim} \
    --act_func ${act_func} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --pointer_checkpoint pointer.pth.tar