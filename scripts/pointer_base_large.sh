#!/bin/bash
set -x

gpu=$1
batch=$2
# beam=$3
sample_num_pos=3
margin_pos=0.01

export model="t5-large"
export sample_num_pos=$sample_num_pos
export learning_rate="5e-05"
export out_dir="output_pointer_first_train-v2_add_pos"
# export neg_loss_type="contrastive_loss_max_v5"
export null_loss_type="contrastive_loss_max_v5"
export pos_loss_type="contrastive_loss_repeat"
export margin_null=0.001
export margin_pos=$margin_pos
export loss_mix_ratio_null=1
export loss_mix_ratio_pos=1  ## fix the ratio to 1, just tune the margin
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

export pooling="mean"
export seq_len=1536 # 2048

export lr_proj=3e-4

export beam=1

export seed=42
export EXP="baseline-beam_${beam}_seed_${seed}"
echo "experiment name: $EXP"

python src/run_s2s_pointer.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
    --max_source_length ${seq_len} \
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
    --seed $seed \
    --sample_num_pos ${sample_num_pos} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --base \
    --lr_proj ${lr_proj} \
    --num_beams ${beam} 

export seed=3407
export EXP="baseline-beam_${beam}_seed_${seed}"
echo "experiment name: $EXP"

python src/run_s2s_pointer.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
    --max_source_length ${seq_len} \
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
    --seed $seed \
    --sample_num_pos ${sample_num_pos} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --base \
    --lr_proj ${lr_proj} \
    --num_beams ${beam} 

export seed=114514
export EXP="baseline-beam_${beam}_seed_${seed}"
echo "experiment name: $EXP"

python src/run_s2s_pointer.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
    --max_source_length ${seq_len} \
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
    --seed $seed \
    --sample_num_pos ${sample_num_pos} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --base \
    --lr_proj ${lr_proj} \
    --num_beams ${beam} 

export beam=2
export EXP="baseline-beam_${beam}"
echo "experiment name: $EXP"

python src/run_s2s_pointer.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
    --max_source_length ${seq_len} \
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
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --base \
    --lr_proj ${lr_proj} \
    --num_beams ${beam} 

export beam=3
export EXP="baseline-beam_${beam}"
echo "experiment name: $EXP"

python src/run_s2s_pointer.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
    --max_source_length ${seq_len} \
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
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --base \
    --lr_proj ${lr_proj} \
    --num_beams ${beam} 

export beam=4
export EXP="baseline-beam_${beam}"
echo "experiment name: $EXP"

python src/run_s2s_pointer.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
    --max_source_length ${seq_len} \
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
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --base \
    --lr_proj ${lr_proj} \
    --num_beams ${beam} 


export beam=5
export EXP="baseline-beam_${beam}"
echo "experiment name: $EXP"

python src/run_s2s_pointer.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
    --max_source_length ${seq_len} \
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
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --base \
    --lr_proj ${lr_proj} \
    --num_beams ${beam} 