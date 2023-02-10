#!/bin/bash
set -x

gpu=$1
batch=$2
model=$3
sample_num=$4
step=$5
echo "export CUDA_VISIBLE_DEVICES=$gpu"
export CUDA_VISIBLE_DEVICES=${gpu}
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
export EXP=${model}-neg-${sample_num}

deepspeed --master_port $port  src/run_s2s.py \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "no" \
    --model_name_or_path "/home/tuq59834/code/project/Tk-ins/Tk-Instruct/output/${EXP}/checkpoint-${step}" \
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
    --task_dir data/tasks/neg_mixed \
    --output_dir output/ \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_eval_batch_size $batch \
    --exp_name $EXP \
    --seed 42
