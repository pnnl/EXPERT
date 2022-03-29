#!/usr/bin/bash
trap "kill 0" EXIT

### TO USE THIS SCRIPT: e.g., bash <name>.sh 4 "0,1,2,3" <lr>

# high level
single_device_cuda="0"
num_gpus=$1 # 4, 1, etc
multi_device_cuda=$2 # "0,1,2,3", "0", etc
hf_cache="./cache"
core_lm_name="allenai/scibert_scivocab_cased" # "allenai/scibert_scivocab_cased", "bert-large-cased"
main_log_dir="/./logging" # change later if requiring large storage
global_seed=2022

task_train_data_cap=100
task_eval_data_cap=300
num_train_epochs=20
gradient_accumulation_steps=8
align_weight_over_uniform=0.5
learning_rate=$3
repr_mode="embedding"

# setup accelerate config
accelerate_config="./$1gpu.yaml"
# CUDA_VISIBLE_DEVICES=${multi_device_cuda} HF_HOME=${hf_cache} accelerate config --config_file ${accelerate_config}

# directories
processed_data_dir="${main_log_dir}/processed_data"
lm_properties_dir="${main_log_dir}/lm_properties"
task_model_dir="${main_log_dir}/task_model"

################

rm -rf ${task_model_dir} # prevent the model from loading task head from previously trained task model

# remove --tuned_model_dir for vanilla model
available_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
CUDA_VISIBLE_DEVICES=${multi_device_cuda} HF_HOME=${hf_cache} accelerate launch --config_file ${accelerate_config} --main_process_port ${available_port} my_train_task.py \
    --model_name_or_path ${core_lm_name} \
    --train_file "./SciREX/scirex_dataset/release_data/train.jsonl" \
    --validation_file "./SciREX/scirex_dataset/release_data/dev.jsonl" \
    --test_file "./SciREX/scirex_dataset/release_data/test.jsonl" \
    --train_data_cap ${task_train_data_cap} \
    --eval_data_cap ${task_eval_data_cap} \
    --seed ${global_seed} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --use_slow_tokenizer \
    --max_seq_length 512 \
    --output_dir ${task_model_dir} \
    --tokenized_data_file_path ${processed_data_dir}/processed_scirex_datasets.pkl \
    --context_sentence_k 2 \
    --if_create_tokenized_data_file "no" \
    --align_weight_over_uniform ${align_weight_over_uniform} \
    --tuned_model_dir ${lm_properties_dir}_${repr_mode}
    # --freeze_bert
