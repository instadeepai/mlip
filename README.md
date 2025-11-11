# 🪩 MLIP: Machine Learning Interatomic Potentials 🚀

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.11](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3110/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Tests and Linters 🧪](https://github.com/instadeepai/mlip/actions/workflows/tests_and_linters.yaml/badge.svg?branch=main)](https://github.com/instadeepai/mlip/actions/workflows/tests_and_linters.yaml)
![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mlipbot/b6e4bf384215e60775699a83c3c00aef/raw/pytest-coverage-comment.json)

## 👀 Overview

*mlip* is a Python library for training and deploying
**Machine Learning Interatomic Potentials (MLIP)** written in JAX. It provides
the following functionality:
- Multiple model architectures (for now: MACE, NequIP and ViSNet)
- Dataset loading and preprocessing
- Training and fine-tuning MLIP models
- Batched inference with trained MLIP models
- MD simulations with MLIP models using multiple simulation backends (for now: JAX-MD and ASE)
- Batched MD simulations and energy minimizations with the JAX-MD simulation backend.
- Energy minimizations with MLIP models using the same simulation backends as for MD.

The purpose of the library is to provide users with a toolbox
to deal with MLIP models in true end-to-end fashion.
Hereby we follow the key design principles of (1) **easy-of-use** also for non-expert
users that mainly care about applying pre-trained models to relevant biological or
material science applications, (2) **extensibility and flexibility** for users more
experienced with MLIP and JAX, and (3) a focus on **high inference speeds** that enable
running long MD simulations on large systems which we believe is necessary in order to
bring MLIP to large-scale industrial application.
See our [inference speed benchmark](#-inference-time-benchmarks) below.
With our library, we observe a 10x speedup on 138 atoms and up to 4x speed up
on 1205 atoms over equivalent implementations relying on Torch and ASE.

See the [Installation](#-installation) section for details on how to install
MLIP-JAX and the example Google Colab notebooks linked below for a quick way
to get started. For detailed instructions, visit our extensive
[code documentation](https://instadeepai.github.io/mlip/).

This repository currently supports implementations of:
- [MACE](https://arxiv.org/abs/2206.07697)
- [NequIP](https://www.nature.com/articles/s41467-022-29939-5)
- [ViSNet](https://www.nature.com/articles/s41467-023-43720-2)

As the backend for equivariant operations, the current version of the code relies
on the [e3nn](https://zenodo.org/records/6459381) library.

## 📦 Installation

*mlip* can be installed via pip like this:

```bash
pip install mlip
```

However, this command **only installs the regular CPU version** of JAX.
We recommend that the library is run on GPU.
Use this command instead to install the GPU-compatible version:

```bash
pip install "mlip[cuda]"
```

**This command installs the CUDA 12 version of JAX.** For different versions, please
install *mlip* without the `cuda` flag and install the desired JAX version via pip.

Note that using the TPU version of JAX is, in principle, also supported by
this library. You need to install it separately via pip. However, it has not been
thoroughly tested and should therefore be considered an experimental feature.

## ⚡ Examples

In addition to the in-depth tutorials provided as part of our documentation
[here](https://instadeepai.github.io/mlip/user_guide/index.html#deep-dive-tutorials),
we also provide example Jupyter notebooks that can be used as
simple templates to build your own MLIP pipelines:

- [Inference and simulation](https://github.com/instadeepai/mlip/blob/main/tutorials/simulation_tutorial.ipynb)
- [Model training](https://github.com/instadeepai/mlip/blob/main/tutorials/model_training_tutorial.ipynb)
- [Addition of new models](https://github.com/instadeepai/mlip/blob/main/tutorials/model_addition_tutorial.ipynb)

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

We have prepared pre-trained models trained on a subset of the
[SPICE2 dataset](https://zenodo.org/records/10975225) for each of the models included in
this repo. They can be accessed directly on [InstaDeep's MLIP collection](https://huggingface.co/collections/InstaDeepAI/ml-interatomic-potentials-68134208c01a954ede6dae42),
along with our curated dataset or directly through
the [huggingface-hub Python API](https://huggingface.co/docs/huggingface_hub/en/guides/download):

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="InstaDeepAI/mace-organics", filename="mace_organics_01.zip", local_dir="")
hf_hub_download(repo_id="InstaDeepAI/visnet-organics", filename="visnet_organics_01.zip", local_dir="")
hf_hub_download(repo_id="InstaDeepAI/nequip-organics", filename="nequip_organics_01.zip", local_dir="")
hf_hub_download(repo_id="InstaDeepAI/SPICE2-curated", filename="SPICE2_curated.zip", local_dir="")
```
Note that the pre-trained models are released on a different license than this library,
please refer to the model cards of the relevant HuggingFace repos.

## 🚀 Inference time benchmarks

To showcase the runtime efficiency, we conducted benchmarks across all three
models on two different systems: Chignolin
([1UAO](https://www.rcsb.org/structure/1UAO), 138 atoms) and Alpha-bungarotoxin
([1ABT](https://www.rcsb.org/structure/1ABT), 1205 atoms), both run for 1 ns of
MD simulation on a H100 NVIDIA GPU.
All these JAX-based model implementations are our own and should not be considered
representative of the performance of the code developed by the original authors of the
methods. In the table below, we compare our integrations with the JAX-MD and the ASE
simulation engines, respectively.
Further details can be found in our white paper (see [below](#-citing-our-work)).

**MACE (2,139,152 parameters):**
| Systems   | JAX-MD       | ASE          |
| --------- |-------------:|-------------:|
| 1UAO      | 6.3 ms/step  | 11.6 ms/step |
| 1ABT      | 66.8 ms/step | 99.5 ms/step |

**ViSNet (1,137,922 parameters):**
| Systems   | JAX-MD       | ASE          |
| --------- |-------------:|-------------:|
| 1UAO      | 2.9 ms/step  | 6.2 ms/step  |
| 1ABT      | 25.4 ms/step | 46.4 ms/step |

**NequIP (1,327,792 parameters):**
| Systems   | JAX-MD       | ASE          |
| --------- |-------------:|-------------:|
| 1UAO      | 3.8 ms/step  | 8.5 ms/step  |
| 1ABT      | 67.0 ms/step | 105.7 ms/step|

## 🙏 Acknowledgments

We would like to acknowledge beta testers for this library: Isabel Wilkinson,
Nick Venanzi, Hassan Sirelkhatim, Leon Wehrhan, Sebastien Boyer, Massimo Bortone,
Scott Cameron, Louis Robinson, Tom Barrett, and Alex Laterre.

## 📚 Citing our work

We kindly request that you to cite [our white paper](https://arxiv.org/abs/2505.22397)
when using this library:

C. Brunken, O. Peltre, H. Chomet, L. Walewski, M. McAuliffe, V. Heyraud,
S. Attias, M. Maarand, Y. Khanfir, E. Toledo, F. Falcioni, M. Bluntzer,
S. Acosta-Gutiérrez and J. Tilly, *Machine Learning Interatomic Potentials:
library for efficient training, model development and simulation of molecular systems*,
arXiv, 2025, arXiv:2505.22397.
