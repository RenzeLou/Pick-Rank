#!/bin/bash
# used for training and evaluation
set -x

export gpu=0
export batch=2
export out_dir="output"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

# We use T5-base by default, 
# If you use the other models (e.g., t5-large, t5-3b), please tune the hyperparameters again on the dev set
export model="t5-base"  
# ======================================== Hyperparameters ========================================
export learning_rate="5e-05"
export sample_num_pos=5  # maximum candidate sentences
export seq_len=1536  # maximum input length
export last_sen_num=0 # deprecated, set to 0
export sample_times=2 # the Gumbel-Softmax sampling times (i.e., the $k$ in our paper)
export loss_mix_ratio_null=1
export margin_null=0.1
export num_train_epochs=2
export pointer_train_epoch=1  # epoch number for pre-tuning the pointer network
export pooling="mean" # the pooling method for the sentence-level representations in the pointer network
export lr_encoder=5e-6 # learning rate for the encoder
export lr_proj=3e-4 # learning rate for the projection layer


# ======================================== Training ========================================
export EXP="${model}"
echo "experiment name: $EXP"

python src/run.py \
    --do_train \
    --do_eval \
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
    --data_dir data/splits/add_dev \
    --task_dir data/tasks/def_segmentation \
    --output_dir ${out_dir}/ \
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
    --seed 3407 \
    --sample_num_pos ${sample_num_pos} \
    --margin_null ${margin_null} \
    --loss_mix_ratio_null ${loss_mix_ratio_null} \
    --lr_proj ${lr_proj} \
    --pooling ${pooling} \
    --save_pointer_choice \
    --predict_on_rep \
    --main_loss_on_rep \
    --add_pointer_encoder \
    --lr_encoder ${lr_encoder} \
    --last_sen_num ${last_sen_num} \
    --sample_times ${sample_times} \
    --pointer_train_epoch ${pointer_train_epoch} \
    --prob_save_file gold_token_prob_on_repeated_def \
    --prob_save_on_rep \
    --ranking_forbiden_on_pointer
