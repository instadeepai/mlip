# ⚛️ MLIP: SOTA Machine-Learning Interatomic Potentials in JAX 🚀

![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mlipbot/b6e4bf384215e60775699a83c3c00aef/raw/pytest-coverage-comment.json)

## ⚠️ Important note

The *mlip* library is currently available as a pre-release version only.
The release of the first stable version will follow later this month.

**The pre-release version of this library is not pushed to PyPI yet, but is expected to be soon.**

## 👀 Overview

*mlip* is a Python library for training and deploying
**Machine Learning Interatomic Potentials (MLIP)** written in JAX. It provides
the following functionality:
- Multiple model architectures (for now: MACE, NequIP and ViSNet)
- Dataset loading and preprocessing
- Training and fine-tuning MLIP models
- Batched inference with trained MLIP models
- MD simulations with MLIP models using multiple simulation backends (for now: JAX-MD and ASE)
- Energy minimizations with MLIP models using the same simulation backends as for MD.

The purpose of the library is to provide users with a toolbox
to deal with MLIP models in true end-to-end fashion.
Hereby we follow the key design principles of (1) **easy-of-use** also for non-expert
users that mainly care about applying pre-trained models to relevant biological or
material science applications, (2) **extensibility and flexibility** for users more
experienced with MLIP and JAX, and (3) a focus on **high inference speeds** that enable
running long MD simulations on large systems which we believe is necessary in order to
bring MLIP to large-scale industrial application.

See the [Installation](#-installation) section for details on how to install
MLIP-JAX and the example Google Colab notebooks linked below for a quick way
to get started. For detailed instructions, visit our extensive
[code documentation](https://instadeep.github.io/mlip/).

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
This requires also installing the necessary versions
of [jaxlib](https://pypi.org/project/jaxlib/) which can also be installed via pip. See
the [installation guide of JAX](https://docs.jax.dev/en/latest/installation.html) for
more information.
At time of release, the following install command is supported:

```bash
pip install -U "jax[cuda12]"
```

Note that using the TPU version of *jaxlib* is, in principle, also supported by
this library. However, it has not been thoroughly tested and should therefore be
considered an experimental feature.

Also, some tasks in *mlip* will
require [JAX-MD](https://github.com/jax-md/jax-md>) as a dependency. As the newest
version of JAX-MD is not available on PyPI yet, this dependency will not
be shipped with *mlip* automatically and instead must be installed
directly from the GitHub repository, like this:

```bash
pip install git+https://github.com/jax-md/jax-md.git
```

## ⚡ Examples

In addition to the in-depth tutorials provided as part of our documentation
[here](https://instadeep.github.io/mlip/user_guide/index.html#deep-dive-tutorials),
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

## 🤗 Foundation models (via HuggingFace)

We have prepared foundation models pre-trained on a subset of the
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
Note that the foundation models are released on a different license than this library,
please refer to the model cards of the relevant HuggingFace repos.

## 🙏 Acknowledgments

We would like to acknowledge beta testers for this library: Leon Wehrhan,
Sebastien Boyer, Massimo Bortone, Tom Barrett, and Alex Laterre.

## 📚 Citing our work

We kindly request to cite our white paper when using this library:

C. Brunken, O. Peltre, H. Chomet, L. Walewski, M. McAuliffe, V. Heyraud,
S. Attias, M. Maarand, Y. Khanfir, E. Toledo, F. Falcioni, M. Bluntzer,
S. Acosta-Gutiérrez and J. Tilly, *Machine Learning Interatomic Potentials:
library for efficient training, model development and simulation of molecular systems*,
uploaded to arXiv soon.
