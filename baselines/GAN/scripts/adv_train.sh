#!/bin/bash
set -x

gpu=$1
batch=$2
ADV_train_epoch=$3
adv_g_step=$4
adv_d_epoch=$5
adv_d_step=$6
batch_pt=24
sample_batch_size=64
mc_search_num=5

export out_dir="output_adv"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

export EXP="test"
echo "experiment name: $EXP"

python src/run_s2s_adv_train.py \
    --do_predict \
    --generate_examples \
    --predict_with_generate \
    --model_name_or_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/POS_augmentation-num_0 \
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
    --task_dir_adv data/tasks \
    --task_dir data/tasks/silver_output_for_examples \
    --output_dir ${out_dir}/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --pre_train_batch_size $batch_pt \
    --sample_batch_size $sample_batch_size \
    --gradient_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42 \
    --ADV_train_epoch $ADV_train_epoch \
    --lr_G 5e-5 \
    --lr_D 2e-5 \
    --adv_g_step $adv_g_step \
    --adv_d_epoch $adv_d_epoch \
    --adv_d_step $adv_d_step \
    --d_epoch 1 \
    --d_step 6 \
    --pretrain_save_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/pretrain_classifier \
    --multinomial \
    --pretrain_D_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/pretrain_classifier_bert-base-cased_add-def \
    --pretrain_G_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/POS_augmentation-num_0 \
    --add_def \
    --mc_search_num $mc_search_num \
    --log_likelihood

export EXP="test_no_log"
echo "experiment name: $EXP"

python src/run_s2s_adv_train.py \
    --do_predict \
    --generate_examples \
    --predict_with_generate \
    --model_name_or_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/POS_augmentation-num_0 \
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
    --task_dir_adv data/tasks \
    --task_dir data/tasks/silver_output_for_examples \
    --output_dir ${out_dir}/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --pre_train_batch_size $batch_pt \
    --sample_batch_size $sample_batch_size \
    --gradient_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2500 \
    --bf16 \
    --run_name t5-experiment \
    --exp_name $EXP \
    --seed 42 \
    --ADV_train_epoch $ADV_train_epoch \
    --lr_G 5e-5 \
    --lr_D 2e-5 \
    --adv_g_step $adv_g_step \
    --adv_d_epoch $adv_d_epoch \
    --adv_d_step $adv_d_step \
    --d_epoch 1 \
    --d_step 6 \
    --pretrain_save_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/pretrain_classifier \
    --multinomial \
    --pretrain_D_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/pretrain_classifier_bert-base-cased_add-def \
    --pretrain_G_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/POS_augmentation-num_0 \
    --add_def \
    --mc_search_num $mc_search_num