#!/bin/bash

# ----------------------------------------------------------------------------------------- #
# WARNING:                                                                                  #
# To speed up the attack process, e.g. on the entire benchmark,                             #
# Here we adopt the multi-process version of implementation,                                #
# Which is likely to trigger undesired behavior if TOO MUCH processes are launched at once! #
# To prevent such, please consider deprecate the multi-process feature (removing &)         #
# Or strictly limiting the concurrency (e.g. 16/8=2 in our cases for each target model)     #
# ----------------------------------------------------------------------------------------- #

for ((i=0; i<16; i+=8))
do
    end=$((i+7))

    # attack claude-3-haiku model in sequential order
    echo "Running task for model claude-3-haiku, question index including from $i to $end..."
    python3 main.py \
        --attack-model "gemini-1.5" \
        --target-model "claude-3-haiku" \
        --judge-model "gemini-2.0-flash" \
        --target-dataset "jailflipbench" \
        --index $i \
        --index-end $end \
        --n-streams 5 \
        --n-iterations 5 \
        --topic-style-selection "all" &

    # attack gpt-4o model in sequential order
    echo "Running task for model gpt-4o, question index including from $i to $end..."
    python3 main.py \
        --attack-model "gemini-1.5" \
        --target-model "gpt" \
        --judge-model "gemini-2.0-flash" \
        --target-dataset "jailflipbench" \
        --index $i \
        --index-end $end \
        --n-streams 5 \
        --n-iterations 5 \
        --topic-style-selection "all" &

    # attack gemini-2.0 model in sequential order
    echo "Running task for model gemini-2.0, question index including from $i to $end..."
    python3 main.py \
        --attack-model "gemini-1.5" \
        --target-model "gemini" \
        --judge-model "gemini-2.0-flash" \
        --target-dataset "jailflipbench" \
        --index $i \
        --index-end $end \
        --n-streams 5 \
        --n-iterations 5 \
        --topic-style-selection "all" &

done
wait