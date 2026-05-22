# 🪩 MLIP: Machine Learning Interatomic Potentials

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.11](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3110/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Tests and Linters 🧪](https://github.com/instadeepai/mlip/actions/workflows/tests_and_linters.yaml/badge.svg?branch=main)](https://github.com/instadeepai/mlip/actions/workflows/tests_and_linters.yaml)
![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mlipbot/b6e4bf384215e60775699a83c3c00aef/raw/pytest-coverage-comment.json)

> ## 🚀 *mlip* v2 is now available
>
> *mlip* v2 introduces a targeted API redesign focused on
> **greater modularity, flexibility, and user control**, alongside many new
> features and quality-of-life improvements across training, inference, and
> simulation workflows.
>
> ⚠️ **Please note:** v2 contains API changes that may require updates to
> existing codebases. We strongly recommend checking out the
> [Migration Guide](https://instadeepai.github.io/mlip/migration_guide/index.html)
> for upgrade instructions, breaking changes, and examples.

## 👀 Overview

*mlip* is a Python library for training and deploying
**Machine Learning Interatomic Potentials (MLIP)** written in JAX. It provides
the following functionalities:

- 🧠 Multiple model architectures (currently: MACE, NequIP, eSEN, and ViSNet) and
modular API for development of new architectures
- 🧩 Mixture-of-Experts (MoE) formalism for the eSEN architecture
- 📦 Dataset loading and preprocessing
- 🎯 Training and fine-tuning MLIP models
- 💨 Built-in distributed training for both GPU and TPU
- ⚡ Batched inference with trained MLIP models
- 🧪 MD simulations with MLIP models using multiple simulation backends
  (currently: JAX-MD and ASE)
- 🌡️ Support for both NVT and NPT ensembles in MD
- ⛰️ Energy minimizations using the same simulation backends as for MD
- 🚀 Batched MD simulations and energy minimizations with the JAX-MD backend
- 🔎 Transition state search with the nudged elastic band (NEB) method
- 🌐 Global charge conditioning, partial charge predictions, and
support for long-range interactions
- 📈 Training on Hessian labels
- ⚙️ Integration with the [e3j](https://github.com/instadeepai/e3j) backend
for accelerated equivariant operations (currently in beta)
- 🔬 To validate and benchmark your trained models, make sure to check out
[MLIPAudit](https://github.com/instadeepai/mlipaudit/)

The purpose of the library is to provide users with a toolbox
to deal with MLIP models in true end-to-end fashion.
Hereby we follow the key design principles of (1) **easy-of-use** also for non-expert
users that mainly care about applying pre-trained models to relevant biological or
material science applications, (2) **extensibility and flexibility** for users more
experienced with MLIP and JAX, and (3) a focus on **high inference speeds** that enable
running long MD simulations on large systems which we believe is necessary in order to
bring MLIP to large-scale industrial application.
See our [inference speed benchmark](#-inference-time-benchmarks) below.

🎙️ For further information on the design principles and story behind the *mlip* library,
also check out our [Let's Talk Research podcast episode](https://youtu.be/xsCclme6RmY)
on the topic.

See the [Installation](#-installation) section for details on how to install *mlip* and the
example Jupyter notebooks linked below for a quick way
to get started. For detailed instructions, visit our extensive
[code documentation](https://instadeepai.github.io/mlip/).

This repository currently supports implementations of:
- [MACE](https://arxiv.org/abs/2206.07697)
- [NequIP](https://www.nature.com/articles/s41467-022-29939-5)
- [ViSNet](https://www.nature.com/articles/s41467-023-43720-2)
- [eSEN](https://arxiv.org/abs/2502.12147)

## 📦 Installation

[JAX-install]: https://docs.jax.dev/en/latest/installation.html
[e3j]: https://github.com/instadeepai/e3j

To install the **regular CPU version** of *mlip* via pip, use this command:

```bash
pip install mlip
```

We however recommend that the library is run on GPU.
To install the **CUDA 13 version of JAX and e3j** binaries alongside *mlip*, run:

```bash
pip install "mlip[cuda13]"
```

To install the CUDA 12 version instead, run:

```bash
pip install "mlip[cuda12]"
```

*mlip* also defines `"mlip[cuda13_local]"` and `"mlip[cuda12_local]"`
following the JAX naming patterns of pip extras. For any other custom versions,
please install *mlip* without any CUDA flag, and refer to the installation guides
for [JAX][JAX-install] and [e3j].

We also support installation of the TPU version of JAX via this command:

```bash
pip install "mlip[tpu]"
```


## ⚡ Examples

In addition to the in-depth tutorials provided as part of our documentation
[here](https://instadeepai.github.io/mlip/user_guide/index.html#deep-dive-tutorials),
we also provide example Jupyter notebooks that can be used as
simple templates to build your own MLIP pipelines:

- [Inference and simulation](https://github.com/instadeepai/mlip/blob/main/tutorials/simulation_tutorial.ipynb)
- [Model training](https://github.com/instadeepai/mlip/blob/main/tutorials/model_training_tutorial.ipynb)
- [Addition of new models](https://github.com/instadeepai/mlip/blob/main/tutorials/model_addition_tutorial.ipynb)
- [Advanced simulation tutorial](https://github.com/instadeepai/mlip/blob/main/tutorials/advanced_simulation_tutorial.ipynb)
- [MoE training tutorial](https://github.com/instadeepai/mlip/blob/main/tutorials/moe_training_and_inference_tutorial.ipynb)
- [Hessian model training tutorial](https://github.com/instadeepai/mlip/blob/main/tutorials/hessian_model_training_tutorial.ipynb)

To run the tutorials, just install Jupyter notebooks via pip and launch it from
a directory that contains the notebooks:

```bash
pip install notebook && jupyter notebook
```

The installation of *mlip* itself is included within the notebooks. We recommend to
run these notebooks with GPU acceleration enabled.

Alternatively, we provide a `Dockerfile` in this repository that you can use to
run the tutorial notebooks. This can be achieved by executing the following lines
from any directory that contains the downloaded `Dockerfile`:

```bash
docker build . -t mlip_tutorials
docker run -p 8888:8888 --gpus all mlip_tutorials
```

Note that this will only work on machines with NVIDIA GPUs.
Once running, you can access the Jupyter notebook server by clicking on the URL
displayed in the console of the form "http[]()://127.0.0.1:8888/tree?token=abcdef...".

## 🤗 Pre-trained models (via HuggingFace)

We have prepared pre-trained models trained on a curated version of the
[SPICE2 subset of OMol25](https://arxiv.org/abs/2505.08762) for each of
the models included in
this repo. They can be accessed directly on [InstaDeep's MLIP collection](https://huggingface.co/collections/InstaDeepAI/ml-interatomic-potentials-68134208c01a954ede6dae42),
along with our curated dataset or directly through
the [huggingface-hub Python API](https://huggingface.co/docs/huggingface_hub/en/guides/download):

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="InstaDeepAI/mlip_models_organics_v2", filename="mace_organics_02.zip", local_dir="")
hf_hub_download(repo_id="InstaDeepAI/mlip_models_organics_v2", filename="visnet_organics_02.zip", local_dir="")
hf_hub_download(repo_id="InstaDeepAI/mlip_models_organics_v2", filename="nequip_organics_02.zip", local_dir="")
hf_hub_download(repo_id="InstaDeepAI/mlip_models_organics_v2", filename="esen_organics_02.zip", local_dir="")
hf_hub_download(repo_id="InstaDeepAI/SPICE2_curated_v2", filename="SPICE2_curated_v2.zip", local_dir="")
```
Note that the pre-trained models are released on a different license than this library,
please refer to the model cards of the relevant HuggingFace repos.

## 🚀 Inference time benchmarks

To showcase the runtime efficiency, we conducted benchmarks across all four
models on two different systems: Chignolin
([1UAO](https://www.rcsb.org/structure/1UAO), 138 atoms) and Alpha-bungarotoxin
([1ABT](https://www.rcsb.org/structure/1ABT), 1205 atoms), both run for 1 ns of
MD simulation using the NVT ensemble on a H100 NVIDIA GPU.
All these JAX-based model implementations are our own and should not be considered
representative of the performance of the code developed by the original authors of the
methods. In the table below, we compare our integrations with the JAX-MD and the ASE
simulation engines, respectively.
Further details can be found in our white paper (see [below](#-citing-our-work)).

**MACE (3,274,016 parameters):**
| Systems   | JAX-MD       | ASE          |
| --------- |-------------:|-------------:|
| 1UAO      | 2.4 ms/step  | 7.3 ms/step  |
| 1ABT      | 19.2 ms/step | 43.8 ms/step |

**ViSNet (1,172,676 parameters):**
| Systems   | JAX-MD       | ASE          |
| --------- |-------------:|-------------:|
| 1UAO      | 1.9 ms/step  | 7.1 ms/step  |
| 1ABT      | 13.7 ms/step | 30.2 ms/step |

**NequIP (1,327,792 parameters):**
| Systems   | JAX-MD       | ASE          |
| --------- |-------------:|-------------:|
| 1UAO      | 3.4 ms/step  | 8.9 ms/step  |
| 1ABT      | 22.0 ms/step | 44.6 ms/step |

**eSEN (3,210,498 parameters):**
| Systems   | JAX-MD       | ASE          |
| --------- |-------------:|-------------:|
| 1UAO      | 3.0 ms/step  | 8.9 ms/step  |
| 1ABT      | 22.8 ms/step | 46.7 ms/step |


## 🙏 Acknowledgments

This work was supported by Cloud TPUs from Google’s TPU Research Cloud (TRC).
We would also like to thank Bohan Cao from Nankai University / Zhongguancun
Academy  for the numerous suggestions and conversations.

## 📚 Citing our work

We kindly request that you to cite [our white paper](https://arxiv.org/abs/2605.22698)
when using this library:

C. Brunken, T. Cormier, L. Walewski, M. Carobene, Y. Khanfir, Z. Weller-Davies,
M. Bragança, A. Picard, A. Pichard, L. Wehrhan, H. Chomet, E. Varga-Umbrich,
M. Bluntzer, M. Bortone, V. Heyraud, S. Acosta-Gutiérrez,
J. Tilly, and O. Peltre, *Machine Learning Interatomic Potentials: Advancing
Open-Source Software for Efficient and Scalable Molecular Simulation*,
arXiv, 2026, arXiv:2605.22698.

The BibTeX formatted citation:

```
@misc{brunken2026machinelearninginteratomicpotentials,
      title={Machine Learning Interatomic Potentials: Advancing Open-Source Software for Efficient and Scalable Molecular Simulation},
      author={Christoph Brunken and Titouan Cormier and Lucien Walewski and Marco Carobene and Yessine Khanfir and Zachary Weller-Davies and Miguel Bragança and Armand Picard and Adrien Pichard and Leon Wehrhan and Heloise Chomet and Eszter Varga-Umbrich and Marie Bluntzer and Massimo Bortone and Valentin Heyraud and Silvia Acosta-Gutiérrez and Jules Tilly and Olivier Peltre},
      year={2026},
      eprint={2605.22698},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2605.22698},
}
```
