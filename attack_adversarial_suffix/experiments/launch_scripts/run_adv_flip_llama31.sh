#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=1
export model=llama31 # qwen25 or llama31
seed=2025

# Create results folder if it doesn't exist
if [ ! -d "../results_${model}_individual" ]; then
    mkdir "../results_${model}_individual"
    echo "Folder '../results_${model}_individual' created."
else
    echo "Folder '../results_${model}_individual' already exists."
fi

for offset in {0..15}
do
    python -u ../main.py \
        --config="../configs/transfer_${model}.py" \
        --config.attack=gcg \
        --config.train_data="../../data/jailflipbench.csv" \
        --config.result_prefix="../results_${model}_individual/jailflip_result_${model}_n${n}_seed${seed}_offset${offset}" \
        --config.progressive_goals=False \
        --config.stop_on_success=False \
        --config.num_train_models=1 \
        --config.allow_non_ascii=False \
        --config.n_train_data=$n \
        --config.n_test_data=1 \
        --config.n_steps=200 \
        --config.data_offset=$offset \
        --config.test_steps=10 \
        --config.batch_size=512 \
        --config.debug_mode=False \
        --config.random_seed_for_sampling_targets=$seed \
        --config.jf_notes="a40 cuda:${cuda_idx} model${model} jailflip individual with n${n} seed${seed} offset${offset}"
done