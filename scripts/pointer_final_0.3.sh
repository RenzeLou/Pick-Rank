#!/bin/bash
set -x

gpu=$1
batch=$2
train_epochs=$3
pointer_train_epoch=$4
loss_mix_ratio_null=0.3
# margin_null=$5

export sample_num_pos=5
export seq_len=1536
export last_sen_num=0
export sample_times=2


export model="t5-base"
export sample_num_pos=$sample_num_pos
export learning_rate="5e-05"
export out_dir="output_pointer_final"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"
# add ranking loss between ori and rep
export null_loss_type="contrastive_loss_max_v6"
export pos_loss_type="contrastive_loss_repeat_v2"
export train_epochs=$train_epochs
export pointer_train_epoch=$pointer_train_epoch
export loss_mix_ratio_null=$loss_mix_ratio_null  
export loss_mix_ratio_pos=1
export margin_pos=0.1

export pooling="mean"
export lr_encoder=5e-6
export lr_proj=3e-4


export margin_null=0.1
export EXP="epochs_${train_epochs}-pointer_epochs_${pointer_train_epoch}--loss_mix_ratio_null_${loss_mix_ratio_null}-margin_null_${margin_null}"
echo "experiment name: $EXP"

python src/run_s2s_pointer_train_first.py \
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
    --num_train_epochs ${train_epochs} \
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
    --null_loss_type ${null_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --add_pointer_encoder \
    --lr_encoder ${lr_encoder} \
    --last_sen_num ${last_sen_num} \
    --sample_times ${sample_times} \
    --del_detach \
    --pointer_train_epoch ${pointer_train_epoch}


export margin_null=0.03
export EXP="epochs_${train_epochs}-pointer_epochs_${pointer_train_epoch}--loss_mix_ratio_null_${loss_mix_ratio_null}-margin_null_${margin_null}"
echo "experiment name: $EXP"

python src/run_s2s_pointer_train_first.py \
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
    --num_train_epochs ${train_epochs} \
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
    --null_loss_type ${null_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --add_pointer_encoder \
    --lr_encoder ${lr_encoder} \
    --last_sen_num ${last_sen_num} \
    --sample_times ${sample_times} \
    --del_detach \
    --pointer_train_epoch ${pointer_train_epoch}

export margin_null=0.01
export EXP="epochs_${train_epochs}-pointer_epochs_${pointer_train_epoch}--loss_mix_ratio_null_${loss_mix_ratio_null}-margin_null_${margin_null}"
echo "experiment name: $EXP"

python src/run_s2s_pointer_train_first.py \
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
    --num_train_epochs ${train_epochs} \
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
    --null_loss_type ${null_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --add_pointer_encoder \
    --lr_encoder ${lr_encoder} \
    --last_sen_num ${last_sen_num} \
    --sample_times ${sample_times} \
    --del_detach \
    --pointer_train_epoch ${pointer_train_epoch}

export margin_null=0.003
export EXP="epochs_${train_epochs}-pointer_epochs_${pointer_train_epoch}--loss_mix_ratio_null_${loss_mix_ratio_null}-margin_null_${margin_null}"
echo "experiment name: $EXP"

python src/run_s2s_pointer_train_first.py \
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
    --num_train_epochs ${train_epochs} \
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
    --null_loss_type ${null_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --add_pointer_encoder \
    --lr_encoder ${lr_encoder} \
    --last_sen_num ${last_sen_num} \
    --sample_times ${sample_times} \
    --del_detach \
    --pointer_train_epoch ${pointer_train_epoch}

export margin_null=0.001
export EXP="epochs_${train_epochs}-pointer_epochs_${pointer_train_epoch}--loss_mix_ratio_null_${loss_mix_ratio_null}-margin_null_${margin_null}"
echo "experiment name: $EXP"

python src/run_s2s_pointer_train_first.py \
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
    --num_train_epochs ${train_epochs} \
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
    --null_loss_type ${null_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --add_pointer_encoder \
    --lr_encoder ${lr_encoder} \
    --last_sen_num ${last_sen_num} \
    --sample_times ${sample_times} \
    --del_detach \
    --pointer_train_epoch ${pointer_train_epoch}

export margin_null=0.0003
export EXP="epochs_${train_epochs}-pointer_epochs_${pointer_train_epoch}--loss_mix_ratio_null_${loss_mix_ratio_null}-margin_null_${margin_null}"
echo "experiment name: $EXP"

python src/run_s2s_pointer_train_first.py \
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
    --num_train_epochs ${train_epochs} \
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
    --null_loss_type ${null_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --add_pointer_encoder \
    --lr_encoder ${lr_encoder} \
    --last_sen_num ${last_sen_num} \
    --sample_times ${sample_times} \
    --del_detach \
    --pointer_train_epoch ${pointer_train_epoch}

export margin_null=0.0001
export EXP="epochs_${train_epochs}-pointer_epochs_${pointer_train_epoch}--loss_mix_ratio_null_${loss_mix_ratio_null}-margin_null_${margin_null}"
echo "experiment name: $EXP"

python src/run_s2s_pointer_train_first.py \
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
    --num_train_epochs ${train_epochs} \
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
    --null_loss_type ${null_loss_type} \
    --margin_null ${margin_null} \
    --margin_pos ${margin_pos} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --loss_mix_ratio_pos ${loss_mix_ratio_pos} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --add_pointer_encoder \
    --lr_encoder ${lr_encoder} \
    --last_sen_num ${last_sen_num} \
    --sample_times ${sample_times} \
    --del_detach \
    --pointer_train_epoch ${pointer_train_epoch}