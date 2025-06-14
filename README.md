# From Jailbreaks to JailFlip: Revealing Stealthier and Broader Implicit Harm Stemming from LLM Alignment Failures

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for our work:
[Beyond Jailbreaks: Revealing Stealthier and Broader LLM Security Risks Stemming from Alignment Failures](https://arxiv.org/abs/2506.07402).  
Please also refer to our [project webpage](https://jailflip.github.io/) for further information.

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

```bibtex
@article{zhou2025beyond,
  title={Beyond Jailbreaks: Revealing Stealthier and Broader LLM Security Risks Stemming from Alignment Failures},
  author={Zhou, Yukai and Yang, Sibei and Wang, Wenjie},
  journal={arXiv preprint arXiv:2506.07402},
  year={2025}
}
```

## License
Our work is licensed under the terms of the MIT license. See LICENSE for more details.