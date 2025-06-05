# JailFlip Attack by LLM-as-an-attacker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the llm-as-an-attacker JailFlip attack, which is part of our work: 
[From Jailbreaks to JailFlip: Revealing Stealthier and Broader Implicit Harm Stemming from LLM Alignment Failures](https://jailflip.github.io/).  
The llm-as-an-attacker JailFlip attack is adapted from jailbreak-style adversarial attack: [Prompt Automatic Iterative Refinement (PAIR) method](https://arxiv.org/abs/2310.08419).

Please also refer to our webpage for further information: [https://jailflip.github.io/](https://jailflip.github.io/)

## Installation

Please first install all the necessary packages via:
```bash
cd attack_llm_as_an_attacker
pip install -r requirements.txt
```

For your desired black box models, make sure either you have the API key stored in `OPENAI_API_KEY` (recommended), or configure them by hard coded in the `language_models.py` (not recommended). For example, run the following code

```bash
export OPENAI_API_KEY=[YOUR_API_KEY_HERE]
```

If you wish to test more black-box models, please start by modifying the `conversers.py` file.

## Experiments

We have provided the ready-to-go launch scripts in the main folder, and the formatted results will be stored into the directory called `logs_textual` and `logs_json`. For reproducing our experiment results, run the following code:

```bash
bash launch_script.sh
```

As suggested by the original [PAIR implementation](https://github.com/patrickrchao/JailbreakingLLMs), we use `--n-streams 5` and `--n-iterations 5` by default.  
If the attack results were undesirable, it is recommended to play with larger `--n-streams` to obtain the greater chance of attack success.  
See `main.py` for all of the supported arguments and descriptions.

## Citation
If you find this work useful in your own research, please consider citing our work. 

In addition, please also consider citing the `PAIR` work for offering valuable insight.

```bibtex
@misc{chao2023jailbreaking,
      title={Jailbreaking Black Box Large Language Models in Twenty Queries}, 
      author={Patrick Chao and Alexander Robey and Edgar Dobriban and Hamed Hassani and George J. Pappas and Eric Wong},
      year={2023},
      eprint={2310.08419},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgment
Our attack implementation is built upon modifying the `PAIR` implementaion from [PAIR github repo](https://github.com/patrickrchao/JailbreakingLLMs).

Sincere thanks to the authors of [Chao et al. 2023](https://arxiv.org/abs/2310.08419) for the valueable insight and open-sourced code.

## License
Our work is licensed under the terms of the MIT license. See LICENSE for more details.