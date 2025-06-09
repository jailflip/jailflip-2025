# JailFlip Attack by Adversarial Suffix

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the adversarial suffix JailFlip attack, which is part of our work: 
[Beyond Jailbreaks: Revealing Stealthier and Broader LLM Security Risks Stemming from Alignment Failures](https://jailflip.github.io/).  
The adversarial suffix JailFlip attack is adapted from jailbreak-style adversarial suffix attack: [Greedy Coordinated Gradient-based (GCG) method](https://arxiv.org/abs/2307.15043).


Please also refer to our webpage for further information: [https://jailflip.github.io/](https://jailflip.github.io/)

## Installation

Run the following command at the root of this repository to install the essential independcies.
Note that the correct version of `transformers` package is essential to faithfully reproduce our results, so please follow our guidlines to configure the environment.

```bash
cd attack_adversarial_suffix
pip install -e .
```


## Models

Please first follow the instructions to download Llama-3.1-8B-Instruct or/and Qwen2.5-7B-Instruct. We use the weights converted by HuggingFace:
[Llama-3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
Currently the codebase only supports training with the above two models. 
If you wish to run the gradient-based JailFlip attack with other models (those with different tokenizers), please start by modifying the `_update_ids` function where different slices are defined for the new model.

Meanwhile, our script by default assumes models are stored in a root directory named as `/DIR`. 
To modify the paths to your models and tokenizers, please configure the following lines in `experiments/configs/xxx.py`. An example is given as follows.

```python
    config.model_paths = [
        "/DIR/meta-llama/Llama-3.1-8B-Instruct",
        ... # change to your model path
    ]
    config.tokenizer_paths = [
        "/DIR/meta-llama/Llama-3.1-8B-Instruct",
        ... # change to your tokenizer path
    ]
```

## Experiments 

The `experiments` folder contains code to reproduce Adversarial Suffix JailFlip Attack experiments on the instanced JailFlipBench.  
Notice that all hyper-parameters in our experiments are handled by the `ml_collections` package [here](https://github.com/google/ml_collections).
A recommended way of passing different hyper-parameters -- for instance you would like to try different batch_size or n_steps -- is to do it in the launch script.  
You can also specify your JailFlip-style questions in the data folder.
Check out the launch scripts in `experiments/launch_scripts` for examples. 

- To run experiments with Adversarial Suffix attack on our instanced JailFlipBench, run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_adv_flip_llama31.sh
```

- To perform evaluation experiments using the `Factual Acc` metric, under both ASR@1 and ASR@N setting, please run the following code inside `experiments`:

```bash
cd launch_scripts
bash eval_each_step_individual_llama31.sh
```

- To perform rigorous evaluation using the `Deep ASR` metric, e.g. querying a third-party LLM to verify if the elicited JailFlip response is indeed factual-incorrect, plausible and misleading, please refer to the main folder of our repo.

## Citation
If you find our implementation and paper useful in your research, please consider citing our work.

In addition, please also consider citing the `GCG` work for offering valuable insight.

```
@misc{zou2023universal,
      title={Universal and Transferable Adversarial Attacks on Aligned Language Models}, 
      author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
      year={2023},
      eprint={2307.15043},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgment
Our attack implementation is built upon modifying the `GCG` implementaion from [GCG github repo](https://github.com/llm-attacks/llm-attacks)

Sincere thanks to the authors of [Zou et al. 2023](https://arxiv.org/abs/2307.15043) for the valueable insight and open-sourced code.


## License
Our work is licensed under the terms of the MIT license. See LICENSE for more details.