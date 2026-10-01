# Third-party source notices

Component-specific licenses and retained copyright headers apply in addition to the root license. Public upstream links below identify source origins and scientific references. No dataset, benchmark payload, pretrained model or cell library is distributed by this repository.

## REvolution population search

The population-search implementation derives from [REvolution](https://github.com/kmcho2019/REvolution), described by Kyungjun Min, Kyumin Cho, Junhwan Jang and Seokhyeong Kang in [REvolution: An Evolutionary Framework for RTL Generation driven by Large Language Models](https://arxiv.org/abs/2510.21407).

The original **Copyright (c) 2025 Kyumin Cho** and MIT license are retained verbatim in [LICENSE](LICENSE). Local modifications add PPA estimator/synthesis adapters, explicit configuration, benchmark interfaces and release organization. The internal `revolution` module name is retained for compatibility and provenance. This release does not imply endorsement by the original authors.

## Google DeepMind regress-lm

The numeric prediction backend and training workflow adapt [google-deepmind/regress-lm](https://github.com/google-deepmind/regress-lm). The exact imported upstream revision was not recorded. Retained adapted backend files carry **Copyright 2025 Google LLC.** and Apache-2.0 headers. The complete license is included in [licenses/regress-lm-Apache-2.0.txt](licenses/regress-lm-Apache-2.0.txt).

Local RTL/PPA modifications include constraint/RTL prompt construction, multimetric dataset wrappers and target formatting, Hugging Face training integration, standalone evaluation, sampled decoding/metric records, and historical ranking/loss objectives. Modified backend files carry modification notices. The project is research adaptation and is not an officially supported Google product. No upstream NOTICE file was present in the inspected regress-lm repository root.

## TorchRL distribution implementation

The discrete-distribution code excerpt in `src/text2text_regression/omnipred/modeling_omnipred.py` cites [TorchRL discrete.py at revision 3cf1df0cd1379dfe846b7a255be1cf359caa80e8](https://github.com/pytorch/rl/blob/3cf1df0cd1379dfe846b7a255be1cf359caa80e8/torchrl/modules/distributions/discrete.py). Its upstream MIT license and **Copyright (c) Meta Platforms, Inc. and affiliates.** are included in [licenses/TorchRL-MIT.txt](licenses/TorchRL-MIT.txt). The upstream excerpt remains subject to that notice.

## EvolVE paper inspiration and external benchmarks

The local EvolVE implementation is inspired by Wei-Po Hsin, Ren-Hao Deng, Yao-Ting Hsieh, En-Ming Huang and Shih-Hao Hung, [EvolVE: Evolutionary Search for LLM-based Verilog Generation and Optimization](https://arxiv.org/abs/2601.18067). It is a local paper-inspired implementation rather than a certified reproduction or a redistribution of the paper authors' claimed experimental results. Generation model, budget, scoring, tool flow and verification differences limit comparison. External IC-RTL, VerilogEval and CVDP benchmark payloads are not included.

## MetRex reference

[MetRex](https://github.com/scale-lab/MetRex) informed historical dataset/constraint formatting. Cite M. Abdelatty, J. Ma and S. Reda, *MetRex: A Benchmark for Verilog Code Metric Reasoning Using LLMs*, ASP-DAC 2025. The upstream README states BSD 3-Clause but its linked license file was absent at the inspected source revision and current branch. The original copied SDC template excerpt is excluded from this release; any released SDC compatibility formatting is an independently authored implementation of standard constraint commands. This reference supplies provenance and does not assert a license grant for excluded code or data.

## Installed dependencies

Other packages are dependencies installed separately and retain their own licenses. They are not vendored by this release. The optional Number Token Loss package is pinned to `ntloss==0.3.1` as recovered from the observed environment. Pretrained-model access and license terms remain separate from this code license.
