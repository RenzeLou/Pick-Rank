#!/bin/bash
set -x

gpu=$1
batch=$2
sample_num_pos=$3

export model="t5-base"
export sample_num_pos=$sample_num_pos
export learning_rate="5e-05"
export out_dir="output_memory_def_seg"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"


export k_projector_dim=200
export v_projector_dim=768
export lr_proj=3e-4

export pooling_memory="mean"
export pooling_att="mean"

export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}"
echo "experiment name: $EXP"

python src/run_s2s_def_segmentation.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
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
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
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
    --add_k_projector \
    --add_v_projector \
    --k_projector_dim ${k_projector_dim} \
    --v_projector_dim ${v_projector_dim} \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --lr_proj ${lr_proj}


export pooling_memory="max"
export pooling_att="mean"

export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}"
echo "experiment name: $EXP"

python src/run_s2s_def_segmentation.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
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
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
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
    --add_k_projector \
    --add_v_projector \
    --k_projector_dim ${k_projector_dim} \
    --v_projector_dim ${v_projector_dim} \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --lr_proj ${lr_proj}

export pooling_memory="mean"
export pooling_att="max"

export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}"
echo "experiment name: $EXP"

python src/run_s2s_def_segmentation.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
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
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
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
    --add_k_projector \
    --add_v_projector \
    --k_projector_dim ${k_projector_dim} \
    --v_projector_dim ${v_projector_dim} \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --lr_proj ${lr_proj}

export pooling_memory="max"
export pooling_att="max"

export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}"
echo "experiment name: $EXP"

python src/run_s2s_def_segmentation.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
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
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
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
    --add_k_projector \
    --add_v_projector \
    --k_projector_dim ${k_projector_dim} \
    --v_projector_dim ${v_projector_dim} \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --lr_proj ${lr_proj}


export pooling_memory="mean"
export pooling_att="mean"

export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}-reverse"
echo "experiment name: $EXP"

python src/run_s2s_def_segmentation.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
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
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
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
    --add_k_projector \
    --add_v_projector \
    --k_projector_dim ${k_projector_dim} \
    --v_projector_dim ${v_projector_dim} \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --lr_proj ${lr_proj} \
    --reverse


export pooling_memory="max"
export pooling_att="mean"

export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}-reverse"
echo "experiment name: $EXP"

python src/run_s2s_def_segmentation.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
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
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
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
    --add_k_projector \
    --add_v_projector \
    --k_projector_dim ${k_projector_dim} \
    --v_projector_dim ${v_projector_dim} \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --lr_proj ${lr_proj} \
    --reverse

export pooling_memory="mean"
export pooling_att="max"

export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}-reverse"
echo "experiment name: $EXP"

python src/run_s2s_def_segmentation.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
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
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
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
    --add_k_projector \
    --add_v_projector \
    --k_projector_dim ${k_projector_dim} \
    --v_projector_dim ${v_projector_dim} \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --lr_proj ${lr_proj} \
    --reverse

export pooling_memory="max"
export pooling_att="max"

export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}-reverse"
echo "experiment name: $EXP"

python src/run_s2s_def_segmentation.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
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
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
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
    --add_k_projector \
    --add_v_projector \
    --k_projector_dim ${k_projector_dim} \
    --v_projector_dim ${v_projector_dim} \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --lr_proj ${lr_proj} \
    --reverse