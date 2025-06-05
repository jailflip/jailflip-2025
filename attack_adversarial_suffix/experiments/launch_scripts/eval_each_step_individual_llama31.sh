#!/bin/bash
export model=llama31 # qwen25 or llama31


file_list=(
    # TODO: fill in your log files path, just like this
    # '../results_llama31_individual/offset0_seed2025_20250101-12:01:01/jailflip_result_llama31_n1_seed2025_offset0.json'
    # '../results_llama31_individual/offset1_seed2025_20250101-12:02:01/jailflip_result_llama31_n1_seed2025_offset0.json'
    # ...
)


# ------------------------------------------------------------------------------------------- #
# since our codebase operates in an 'individual' style, there is no need to specify test data #
# nevertheless, it is fine to play with additional testing data with it                       #
# e.g., by setting --config.n_test_data=100                                                   #
# ------------------------------------------------------------------------------------------- #
for file_path in "${file_list[@]}"
do
    python -u ../evaluate_each_step_individual_llama31.py \
        --config="../configs/transfer_${model}.py" \
        --config.train_data="../../data/jailflipbench.csv" \
        --config.n_test_data=0 \
        --config.eval_max_new_len=512 \
        --config.jf_notes="For each step, by generating 512 tokens, eval the adversarial suffix JailFlip attack. The log file path is ${file_path}" \
        --config.logfile="${file_path}"
done