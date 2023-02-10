#!/bin/bash
set -x

gpu=$1
batch=$2
sample_num_pos=$3
sample_num_neg=0

export model="t5-base"
export sample_num_pos=$sample_num_pos
export sample_num_neg=$sample_num_neg
export null_loss_type="contrastive_loss_attention"
export neg_loss_type="contrastive_loss_attention"
export loss_mix_ratio_neg="1"
export learning_rate="5e-05"
export out_dir="output_memory_def_del"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"
export pooling_memory="mean"
export pooling_att="max"

port=$(shuf -i25000-30000 -n1)

export margin_neg="0.01"

export projector_dim=200
# export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}"
# echo "experiment name: $EXP"

# deepspeed --master_port $port  src/run_s2s_memory_def_del.py \
#     --do_train \
#     --do_predict \
#     --predict_with_generate \
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
#     --task_dir data/tasks/pos_neg_def_delete \
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
#     --deepspeed ds_configs/stage2.config \
#     --bf16 \
#     --run_name t5-experiment \
#     --exp_name $EXP \
#     --neg_loss_type ${neg_loss_type} \
#     --seed 42 \
#     --sample_num_pos ${sample_num_pos} \
#     --sample_num_neg ${sample_num_neg} \
#     --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
#     --margin_neg ${margin_neg} \
#     --add_attention_projector \
#     --q_k_projector_dim ${projector_dim} \
#     --add_memory_projector \
#     --pooling_memory ${pooling_memory} \
#     --pooling_att ${pooling_att}

export lr_proj="1e-3"
export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}-reverse-lr_proj_${lr_proj}"
echo "experiment name: $EXP"

python src/run_s2s_memory_def_del.py \
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
    --task_dir data/tasks/pos_neg_def_delete \
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
    --neg_loss_type ${neg_loss_type} \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --margin_neg ${margin_neg} \
    --add_attention_projector \
    --q_k_projector_dim ${projector_dim} \
    --add_memory_projector \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --reverse \
    --lr_proj ${lr_proj}

export lr_proj="5e-3"
export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}-reverse-lr_proj_${lr_proj}"
echo "experiment name: $EXP"

python src/run_s2s_memory_def_del.py \
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
    --task_dir data/tasks/pos_neg_def_delete \
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
    --neg_loss_type ${neg_loss_type} \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --margin_neg ${margin_neg} \
    --add_attention_projector \
    --q_k_projector_dim ${projector_dim} \
    --add_memory_projector \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --reverse \
    --lr_proj ${lr_proj}

export lr_proj="5e-4"
export EXP="sample_num_${sample_num_pos}-pooling_memory_${pooling_memory}-pooling_att_${pooling_att}-reverse-lr_proj_${lr_proj}"
echo "experiment name: $EXP"

python src/run_s2s_memory_def_del.py \
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
    --task_dir data/tasks/pos_neg_def_delete \
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
    --neg_loss_type ${neg_loss_type} \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --margin_neg ${margin_neg} \
    --add_attention_projector \
    --q_k_projector_dim ${projector_dim} \
    --add_memory_projector \
    --pooling_memory ${pooling_memory} \
    --pooling_att ${pooling_att} \
    --reverse \
    --lr_proj ${lr_proj}
