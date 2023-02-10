#!/bin/bash
set -x

gpu=$1
batch=$2

export model="t5-base"
export sample_num_pos="1"
export sample_num_neg="1"
export null_loss_type="contrastive_loss_max_v4"
export neg_loss_type="contrastive_loss_max_v4"
export out_loss_type="contrastive_loss_out_constrain_all"
export loss_mix_ratio_null="0.01"
export loss_mix_ratio_neg="0.0001"
export loss_mix_ratio_out="0.4"
export margin_null="0.001"
export margin_neg="0.0001"
export learning_rate="5e-05"
export out_dir="output_null_neg_out"
export neg_out_sample_num="2"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

export margin_out="0.001"
export EXP="ratio_${loss_mix_ratio_out}-margin_${margin_out}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_output_constrain.py \
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
    --task_dir data/tasks/def_pos2_neg_3-add_neg_out \
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
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --null_loss_type ${null_loss_type} \
    --neg_loss_type ${neg_loss_type} \
    --out_loss_type ${out_loss_type} \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --loss_mix_ratio_out ${loss_mix_ratio_out} \
    --margin_null ${margin_null} \
    --margin_neg ${margin_neg} \
    --margin_out ${margin_out} \
    --neg_out_sample_num ${neg_out_sample_num}
    
export margin_out="0.003"
export EXP="ratio_${loss_mix_ratio_out}-margin_${margin_out}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_output_constrain.py \
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
    --task_dir data/tasks/def_pos2_neg_3-add_neg_out \
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
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --null_loss_type ${null_loss_type} \
    --neg_loss_type ${neg_loss_type} \
    --out_loss_type ${out_loss_type} \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --loss_mix_ratio_out ${loss_mix_ratio_out} \
    --margin_null ${margin_null} \
    --margin_neg ${margin_neg} \
    --margin_out ${margin_out} \
    --neg_out_sample_num ${neg_out_sample_num}
    
export margin_out="0.01"
export EXP="ratio_${loss_mix_ratio_out}-margin_${margin_out}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_output_constrain.py \
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
    --task_dir data/tasks/def_pos2_neg_3-add_neg_out \
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
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --null_loss_type ${null_loss_type} \
    --neg_loss_type ${neg_loss_type} \
    --out_loss_type ${out_loss_type} \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --loss_mix_ratio_out ${loss_mix_ratio_out} \
    --margin_null ${margin_null} \
    --margin_neg ${margin_neg} \
    --margin_out ${margin_out} \
    --neg_out_sample_num ${neg_out_sample_num}
    
export margin_out="0.03"
export EXP="ratio_${loss_mix_ratio_out}-margin_${margin_out}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_output_constrain.py \
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
    --task_dir data/tasks/def_pos2_neg_3-add_neg_out \
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
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --null_loss_type ${null_loss_type} \
    --neg_loss_type ${neg_loss_type} \
    --out_loss_type ${out_loss_type} \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --loss_mix_ratio_out ${loss_mix_ratio_out} \
    --margin_null ${margin_null} \
    --margin_neg ${margin_neg} \
    --margin_out ${margin_out} \
    --neg_out_sample_num ${neg_out_sample_num}
    
export margin_out="0.1"
export EXP="ratio_${loss_mix_ratio_out}-margin_${margin_out}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_output_constrain.py \
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
    --task_dir data/tasks/def_pos2_neg_3-add_neg_out \
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
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --null_loss_type ${null_loss_type} \
    --neg_loss_type ${neg_loss_type} \
    --out_loss_type ${out_loss_type} \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --loss_mix_ratio_out ${loss_mix_ratio_out} \
    --margin_null ${margin_null} \
    --margin_neg ${margin_neg} \
    --margin_out ${margin_out} \
    --neg_out_sample_num ${neg_out_sample_num}
    
export margin_out="0.3"
export EXP="ratio_${loss_mix_ratio_out}-margin_${margin_out}"
echo "experiment name: $EXP"

deepspeed --master_port $port  src/run_s2s_output_constrain.py \
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
    --task_dir data/tasks/def_pos2_neg_3-add_neg_out \
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
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --null_loss_type ${null_loss_type} \
    --neg_loss_type ${neg_loss_type} \
    --out_loss_type ${out_loss_type} \
    --seed 42 \
    --sample_num_pos ${sample_num_pos} \
    --sample_num_neg ${sample_num_neg} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_neg ${loss_mix_ratio_neg} \
    --loss_mix_ratio_out ${loss_mix_ratio_out} \
    --margin_null ${margin_null} \
    --margin_neg ${margin_neg} \
    --margin_out ${margin_out} \
    --neg_out_sample_num ${neg_out_sample_num}
    
