# From Jailbreaks to JailFlip: Revealing Stealthier and Broader Implicit Harm Stemming from LLM Alignment Failures

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the JailFlip attack:
[From Jailbreaks to JailFlip: Revealing Stealthier and Broader Implicit Harm Stemming from LLM Alignment Failures](https://jailflip.github.io/).
Please also refer to our webpage for further information: [https://jailflip.github.io/](https://jailflip.github.io/)

<p align="center">
  <img src="images/quadrants.png" width="70%">
</p>

<p align="center">
  <img src="images/Variants.png" width="70%">
</p>

## Experiments

In our paper, we propose a seires of `JailFlip` methods, spanning from the most trival `Direct Query`, `Direct Attack`, to more structed `Prompting Attack`, and to more advance jailbreak-style attacks: `llm-as-an-attacker` and `adversarial suffix` attack.

We have provided the codebase for each kind of attacks as well as the llm-as-a-judge protocol in their corresponding folder.
Specifically, `llm-as-an-attacker` and `adversarial suffix` attack are adapted from jailbreak-style attack methods, and please refer to the readme file within their corresponding folder to see more details.

## Dataset

Our proposed JailFlipBench could be categorized into three scenarios: single-modal, multi-modal, and factual extension. The intact multi-modal subset and instanced other subsets is included in the `data` folder. The full version of JailFlipBench will be released once our paper is accepted.

## Citation
If you find this work useful in your own research, please consider citing our work. 

## License
Our work is licensed under the terms of the MIT license. See LICENSE for more details.