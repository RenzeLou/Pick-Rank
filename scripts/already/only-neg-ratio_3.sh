#!/bin/bash
set -x

gpu=$1
batch=$2

export model="t5-base"
export margin_pos="0.0001"
export margin_neg="0.0001"
export sample_num_pos="0"
export sample_num_neg="3"
export pos_neg_ratio="1"
export loss_mix_ratio="3"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)


export EXP="Only-NEG-margin_neg_${margin_neg}-mix_ratio_${loss_mix_ratio}-num_${sample_num_neg}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_test_pos.py \
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
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/default \
    --task_dir data/tasks/pos_neg_mixed_v3 \
    --output_dir output/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-05 \
    --num_train_epochs 2 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --neg_loss_type contrastive_loss_max_v4 \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio ${loss_mix_ratio} \
    --margin_pos ${margin_pos} \
    --margin_neg ${margin_neg} \
    --pos_neg_ratio ${pos_neg_ratio}
    
