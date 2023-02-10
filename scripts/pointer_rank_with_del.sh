#!/bin/bash
# only ranking with del
set -x

gpu=$1
batch=$2
sample_num_pos=$3
seq_len=$4
last_sen_num=$5
sample_times=$6
margin_neg=$7
loss_mix_ratio_neg=$8

export model="t5-base"
export sample_num_pos=$sample_num_pos
export learning_rate="5e-05"
export out_dir="output_pointer-cross_rep-pred_rep"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"
# add ranking loss between ori and rep
export null_loss_type="contrastive_loss_max_v5"
export neg_loss_type="contrastive_loss_max_v5"
export margin_null=0.001
export margin_neg=$margin_neg
export loss_mix_ratio_null=1
export loss_mix_ratio_neg=$loss_mix_ratio_neg  ## fix the ratio to 1, just tune the margin

export pooling="mean"
export lr_encoder=5e-6
export lr_proj=3e-4

export EXP="del_sample_num_${sample_num_pos}-seq_len_${seq_len}-last_sen_num_${last_sen_num}-sample_times_${sample_times}-margin_neg_${margin_neg}-loss_mix_ratio_neg_${loss_mix_ratio_neg}"
echo "experiment name: $EXP"

python src/run_s2s_pointer_add_encoder_multi_sample.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
    --max_source_length $seq_len \
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
    --neg_loss_type ${neg_loss_type} \
    --margin_null ${margin_null} \
    --margin_neg ${margin_neg} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --add_pointer_encoder \
    --lr_encoder ${lr_encoder} \
    --last_sen_num ${last_sen_num} \
    --sample_times ${sample_times}