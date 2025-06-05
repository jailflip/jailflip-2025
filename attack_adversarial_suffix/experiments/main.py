'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch.multiprocessing as mp
from absl import app
from ml_collections import config_flags
from pathlib import Path
import os
import math

from llm_attacks import get_goals_and_targets, get_workers

_CONFIG = config_flags.DEFINE_config_file('config')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def main(_):

    mp.set_start_method('spawn')
    params = _CONFIG.value
    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')
    print(params)
    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)

    # kind of regularization, to reformulate the affirmative response into more diverse pattern
    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
    test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]

    workers, test_workers = get_workers(params)
    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }
    
    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    seed = params.result_prefix[ params.result_prefix.index("_seed")+5 : params.result_prefix.index("_offset") ]
    offset = params.result_prefix[ params.result_prefix.index("_offset")+7 : ]
    
    logfile = f"{params.result_prefix}.json"
    path = Path(logfile)
    logfile = Path(*path.parts[:2], f'offset{offset}_seed{seed}_{timestamp}', *path.parts[2:])
    logfile = str(logfile)
    params.result_prefix = logfile
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    if params.transfer:
        # ---------------------------- for multiple attack and transfer attack --------------------------- #
        # ------------------------ this attack is in the path of llm_attacks/base ------------------------ #
        attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            logfile=logfile,
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            para=params,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps
        )
    else:
        raise ValueError("Only support the transfer mode...")

    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,                                       # temperature for sampling
        allow_non_ascii=params.allow_non_ascii,
        target_weight=params.target_weight,         # equals 1
        control_weight=params.control_weight,       # equals 0
        anneal=params.anneal,
        test_steps=getattr(params, 'test_steps', 1),
        incr_control=params.incr_control,           # whether to increase the control over time (set to False)
        stop_on_success=params.stop_on_success,     # whether to stop the attack upon success (set to False)
        verbose=params.verbose,                     # whether to print verbose output (default&set is True)
        filter_cand=params.filter_cand,             # whether to filter candidates (default&set is True)
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    app.run(main)