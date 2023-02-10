
#!/bin/bash
set -x

gpu=$1
batch=$2
model=$3
method=$4
# port=$4
echo "export CUDA_VISIBLE_DEVICES=$gpu"
export CUDA_VISIBLE_DEVICES=${gpu}
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
# export MASTER_PORT=$port

port=$(shuf -i25000-30000 -n1)
export EXP=${model}-neg-${method}
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_test.py \
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
    --task_dir data/tasks \
    --output_dir output/ \
    --overwrite_output_dir     --cache_dir ./cache/     --overwrite_cache     --per_device_train_batch_size $batch     --per_device_eval_batch_size $batch     --gradient_accumulation_steps 2     --learning_rate 5e-05     --num_train_epochs 2     --lr_scheduler_type constant     --warmup_steps 0     --logging_strategy steps     --logging_steps 500     --evaluation_strategy no     --save_strategy steps     --save_steps 2500     --deepspeed ds_configs/stage2.config     --bf16     --run_name t5-experiment     --exp_name $EXP     --seed 42     --data_dir_argu data/splits/argument     --task_dir_argu data/tasks/${method}/next
