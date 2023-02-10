#!/bin/bash
# this scripts only has pos_ranking
# run it again with neg_ranking
set -x

gpu=$1
batch=$2
sample_num_pos=$3
margin_pos=$4

export model="t5-base"
export sample_num_pos=$sample_num_pos
export learning_rate="5e-05"
export out_dir="output_pointer-cross_ori-pred_rep-rank_rep"
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
export EXP="pointer-lr_${lr_proj}-sample_num_${sample_num_pos}-margin_pos_${margin_pos}-loss_mix_ratio_pos_${loss_mix_ratio_pos}-seq_${seq_len}"
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
    --lr_proj ${lr_proj} \
    --null_loss_type ${null_loss_type} \
    --pos_loss_type ${pos_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --lr_proj ${lr_proj} \
    --save_pointer_choice \
    --predict_on_rep

export lr_proj=5e-4
export EXP="pointer-lr_${lr_proj}-sample_num_${sample_num_pos}-margin_pos_${margin_pos}-loss_mix_ratio_pos_${loss_mix_ratio_pos}-seq_${seq_len}"
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
    --lr_proj ${lr_proj} \
    --null_loss_type ${null_loss_type} \
    --pos_loss_type ${pos_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --lr_proj ${lr_proj} \
    --save_pointer_choice \
    --predict_on_rep

export lr_proj=1e-4
export EXP="pointer-lr_${lr_proj}-sample_num_${sample_num_pos}-margin_pos_${margin_pos}-loss_mix_ratio_pos_${loss_mix_ratio_pos}-seq_${seq_len}"
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
    --lr_proj ${lr_proj} \
    --null_loss_type ${null_loss_type} \
    --pos_loss_type ${pos_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --lr_proj ${lr_proj} \
    --save_pointer_choice \
    --predict_on_rep

# python src/run_s2s_pointer.py \
#     --do_predict \
#     --predict_with_generate \
#     --model_name_or_path $out_dir/$EXP \
#     --max_source_length ${seq_len} \
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
#     --task_dir data/tasks/pos_neg_def_segmentation \
#     --output_dir ${out_dir}/ \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size $batch2 \
#     --per_device_eval_batch_size $batch2 \
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
#     --bf16 \
#     --run_name t5-experiment \
#     --exp_name $EXP \
#     --seed 42 \
#     --sample_num_pos ${sample_num_pos} \
#     --lr_proj ${lr_proj} \
#     --null_loss_type ${null_loss_type} \
#     --pos_loss_type ${pos_loss_type} \
#     --margin_null ${margin_null} \
#     --margin_pos ${margin_pos} \
#     --loss_mix_ratio_null ${loss_mix_ratio_null} \
#     --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
#     --pooling ${pooling} \
#     --lr_proj ${lr_proj} \
#     --save_pointer_choice \
#     --predict_on_rep \
#     --pointer_checkpoint pointer.pth.tar


export lr_proj=1e-3
export EXP="pointer-lr_${lr_proj}-sample_num_${sample_num_pos}-margin_pos_${margin_pos}-loss_mix_ratio_pos_${loss_mix_ratio_pos}-seq_${seq_len}"
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
    --lr_proj ${lr_proj} \
    --null_loss_type ${null_loss_type} \
    --pos_loss_type ${pos_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --pooling ${pooling} \
    --lr_proj ${lr_proj} \
    --save_pointer_choice \
    --predict_on_rep

# python src/run_s2s_pointer.py \
#     --do_predict \
#     --predict_with_generate \
#     --model_name_or_path $out_dir/$EXP \
#     --max_source_length ${seq_len} \
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
#     --task_dir data/tasks/pos_neg_def_segmentation \
#     --output_dir ${out_dir}/ \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size $batch2 \
#     --per_device_eval_batch_size $batch2 \
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
#     --bf16 \
#     --run_name t5-experiment \
#     --exp_name $EXP \
#     --seed 42 \
#     --sample_num_pos ${sample_num_pos} \
#     --lr_proj ${lr_proj} \
#     --null_loss_type ${null_loss_type} \
#     --pos_loss_type ${pos_loss_type} \
#     --margin_null ${margin_null} \
#     --margin_pos ${margin_pos} \
#     --loss_mix_ratio_null ${loss_mix_ratio_null} \
#     --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
#     --pooling ${pooling} \
#     --lr_proj ${lr_proj} \
#     --save_pointer_choice \
#     --predict_on_rep \
#     --pointer_checkpoint pointer.pth.tar
