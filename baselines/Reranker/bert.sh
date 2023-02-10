# bert-base & bert-large
gpu=$1
batch1=$2
batch2=$3


export model="bert-base-cased"
export out_dir="out"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"


python run_nli.py \
    --model_name_or_path ${model} \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ./data/train.csv \
    --validation_file ./data/dev.csv \
    --test_file ./data/test.csv \
    --max_seq_length 1024 \
    --per_device_train_batch_size ${batch1} \
    --per_device_eval_batch_size ${batch1} \
    --cache_dir ./cache/ \
    --output_dir ./${out_dir}/${model}/ \
    --overwrite_output_dir \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --save_strategy no \
    --evaluation_strategy epoch \
    --seed 42

export model="bert-large-cased"
export out_dir="out"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"


python run_nli.py \
    --model_name_or_path ${model} \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ./data/train.csv \
    --validation_file ./data/dev.csv \
    --test_file ./data/test.csv \
    --max_seq_length 1024 \
    --per_device_train_batch_size ${batch2} \
    --per_device_eval_batch_size ${batch2} \
    --cache_dir ./cache/ \
    --output_dir ./${out_dir}/${model}/ \
    --overwrite_output_dir \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --save_strategy no \
    --evaluation_strategy epoch \
    --seed 42