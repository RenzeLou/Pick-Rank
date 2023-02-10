#!/bin/bash
set -x

gpu=$1
batch=$2

export model="/home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/dense_retriever-gold_y_pretrained_model"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

export method="rerank"
export pos_num=1
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="rerank"
export pos_num=2
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="rerank"
export pos_num=3
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="rerank"
export pos_num=4
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="rerank"
export pos_num=5
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="rerank"
export pos_num=6
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="rerank"
export pos_num=7
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="rerank"
export pos_num=8
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="rerank"
export pos_num=9
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="rerank"
export pos_num=10
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=1
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=2
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=3
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=4
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=5
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=6
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=7
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=8
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=9
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42

export method="random"
export pos_num=10
export EXP="pred_${method}_${pos_num}"
echo "experiment name: $EXP"

python src/pred_examples_from_training.py \
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
    --num_pos_examples ${pos_num} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/pred_only \
    --task_dir data/tasks/silver_examples_from_training/main_loss-bert_base/${method} \
    --output_dir pred_examples_from_training/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_eval_batch_size $batch \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42