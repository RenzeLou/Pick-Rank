#!/bin/bash
set -x

gpu=$1
batch=$2
margin=$3
ratio=$4


export margin_neg="0.1"
export sample_num_pos="0"
export sample_num_neg="0"
export learning_rate="5e-05"
export main_loss_warm="0"
export num_train_epochs="2"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

export EXP="t5-base-out_constrain-margin_${margin}-ratio_${ratio}-warm_${main_loss_warm}-epoch_${num_train_epochs}"
export model="output_add_neg_out/${EXP}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_output_constrain.py \
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
    --task_dir data/tasks/add_neg_output/ \
    --output_dir output_add_neg_out/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps 2 \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
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
    --neg_loss_type contrastive_loss_out_constrain_all \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio ${ratio} \
    --margin_neg ${margin} \
    --main_loss_warm ${main_loss_warm}
    