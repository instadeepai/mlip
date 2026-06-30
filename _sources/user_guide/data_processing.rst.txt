.. _data_processing:

Data processing
===============

Set up the graph dataset builder
--------------------------------

To train a model (or optionally, to run batched inference), one needs to process
the data into objects of type
:py:class:`GraphDataset <mlip.data.graph_dataset.GraphDataset>`.
This can be achieved by using the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`
class, which can be instantiated from:

* its associated pydantic config,
* a builder mode (:py:class:`BuilderMode <mlip.data.graph_dataset_builder.BuilderMode>`),
* a dictionary of chemical systems readers that are derived from the
  :py:class:`ChemicalSystemsReader <mlip.data.chemical_systems_readers.chemical_systems_reader.ChemicalSystemsReader>`
  base class, and
* optionally, a :py:class:`DatasetInfo <mlip.data.dataset_info.DatasetInfo>` object
  that should be used instead of being computed by the builder
  (see :ref:`this <get_dataset_info>` section below for more information on that class),

and internally uses the
:py:class:`SingleGraphDatasetBuilder <mlip.data.single_graph_dataset_builder.SingleGraphDatasetBuilder>`
class that processes just a single dataset split. More information on the single-split
pathway can be found :ref:`below <single_split_builder>`.

Depending on the builder mode, the dictionary `readers` may have a different
structure. For the following, we assume the standard training case corresponding to the
`BuilderMode.TRAINING` mode. See sections below for a guide to the other builder modes
(:ref:`fine-tuning and multi-dataset training <multi_builder_mode>`,
and :ref:`custom <custom_builder_mode>`).

Below, we provide a simple code example:

.. code-block:: python

    from mlip.data import GraphDatasetBuilder, BuilderMode

    # The reader classes are placeholders for the moment
    readers = {
        "train": _get_chemical_systems_reader(),  # "train" key is mandatory
        "validation": _get_chemical_systems_reader(),
        "test": _get_chemical_systems_reader(),
    }

    builder_config = GraphDatasetBuilder.Config(
        graph_cutoff_angstrom=5.0,
        batch_size=32,
    )

    graph_dataset_builder = GraphDatasetBuilder(
        readers,
        builder_config,
        BuilderMode.TRAINING,  # could be omitted as it's the default
    )

In the example above, we set some example values for the settings in the
:py:class:`GraphDatasetBuilderConfig <mlip.data.configs.GraphDatasetBuilderConfig>`.
To simplify the coding effort, we allow access to this config object directly
via `GraphDatasetBuilder.Config`. Check out the API reference of the class to see the
full set of configurable values and what their defaults have been set to.

**Important:** The key `"train"` is a reserved key that must be present when using
`BuilderMode.TRAINING`. This signals to the builder that this is the dataset split to
base the dataset info computation on (see :ref:`here <get_dataset_info>`). The other
keys can be freely chosen.

The chemical systems reader is an instance of a
:py:class:`ChemicalSystemsReader <mlip.data.chemical_systems_readers.chemical_systems_reader.ChemicalSystemsReader>`
class.
This class allows to read a dataset into lists of
:py:class:`ChemicalSystem <mlip.data.chemical_system.ChemicalSystem>` objects via
its ``load()`` member function. You can either implement your own derived class to do
this for your custom dataset format, or you can employ one of the
:ref:`built-in implementations <chemical_systems_readers>`, for example, the
:py:class:`ExtxyzReader <mlip.data.chemical_systems_readers.extxyz_reader.ExtxyzReader>`
for datasets stored in extended XYZ format:

.. code-block:: python

    from mlip.data import ExtxyzReader

    # This can also be a list of files if multiple files make up one dataset
    dataset_file = "/path/to/dataset.xyz"

    # If data is stored locally
    reader = ExtxyzReader(dataset_file)

    # If data is on remote storage, one can also provide a data download function
    reader = ExtxyzReader(dataset_file, data_download_fun)


In the example above, the ``data_download_fun`` is a simple function that takes in
a source and a target path and performs the download operation. See the API reference
of :py:class:`ChemicalSystemsReader <mlip.data.chemical_systems_readers.chemical_systems_reader.ChemicalSystemsReader>`
for information on other configuration options.

Moreover, please note that we also provide some helper functions for splitting a
dataset which are documented :ref:`here <data_split>`.

Built-in graph dataset readers: data formats
--------------------------------------------

As mentioned above, three built-in core readers are currently provided:
:py:class:`ExtxyzReader <mlip.data.chemical_systems_readers.extxyz_reader.ExtxyzReader>`,
:py:class:`Hdf5Reader <mlip.data.chemical_systems_readers.hdf5_reader.Hdf5Reader>`.
and
:py:class:`ASEAtomsReader <mlip.data.chemical_systems_readers.ase_atoms_reader.ASEAtomsReader>`.

They each support their own data format.
To train an MLIP model, we need a dataset of atomic systems
with the following features per system with specific units:

* the positions (i.e., coordinates) of the atoms in the structure in Angstrom
* the element numbers of the atoms
* the forces of the atoms in eV / Angstrom
* the energy of the structure in eV
* (optional) the stress of the structure in eV / Angstrom\ :sup:`3`
* (optional) the periodic boundary conditions
* (optional) the energy Hessian in eV / Angstrom\ :sup:`2`
* (optional) the atomic partial charges
* (depending on model architecture) the total system charge and spin multiplicity

For a detailed description of the data format required by the
:py:class:`ExtxyzReader <mlip.data.chemical_systems_readers.extxyz_reader.ExtxyzReader>`,
see :ref:`here <extxyz_reader>`.

For a detailed description of the data format required by the
:py:class:`Hdf5Reader <mlip.data.chemical_systems_readers.hdf5_reader.Hdf5Reader>`,
see :ref:`here <hdf5_reader>`.

For a detailed description of the data format required by the
:py:class:`ASEAtomsReader <mlip.data.chemical_systems_readers.ase_atoms_reader.ASEAtomsReader>`,
see :ref:`here <ase_atoms_reader>`.

Start data pre-processing
-------------------------

Once you have the ``graph_dataset_builder`` set up, you can start the pre-processing of
the data and at the end fetch the resulting datasets:

.. code-block:: python

    splits = graph_dataset_builder.get_datasets()

    # Dictionary keys are same as in input readers dictionary
    train_set, validation_set, test_set = (
        splits["train"], splits["validation"], splits["test"]
    )

The resulting datasets are of type
:py:class:`GraphDataset <mlip.data.graph_dataset.GraphDataset>`
as mentioned above. For example, to process the batches in the training set, one
can execute:

.. code-block:: python

    num_graphs = len(train_set.graphs)
    num_batches = len(train_set)

    for batch in train_set:
        _process_batch_in_some_way(batch)

Note that the individual graphs and batches are of
type :py:class:`Graph <mlip.graph.Graph>`.


Also, you could use the `pickle <https://docs.python.org/3/library/pickle.html>`_
library to save the pre-processed splits to disk
in order to restore them later, for instance, when experimenting with
multiple training runs that do not differ in their datasets (to avoid re-processing).

Get sharded batches
-------------------

If one wants to generate batches that are sharded across devices and prefetched, the
arguments to the ``get_datasets()`` method of the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`
must be set to the following:

.. code-block:: python

    splits = graph_dataset_builder.get_datasets(prefetch=True)
    train_set, valid_set, test_set = splits

After this step, the resulting datasets are not of type
:py:class:`GraphDataset <mlip.data.graph_dataset.GraphDataset>` anymore,
but they become
:py:class:`PrefetchIterator <mlip.data.helpers.data_prefetching.PrefetchIterator>` type
instead, which implements batch prefetching on top of the
:py:class:`ParallelGraphDataset <mlip.data.helpers.data_prefetching.ParallelGraphDataset>`
class. It also implements an iterator that can be called to obtain sharded batches
(e.g., `for batch in parallel_graph_dataset: do_something(batch)`), however,
note that it does not have a `graphs` member that can be accessed directly.

Note that using `pickle <https://docs.python.org/3/library/pickle.html>`_ is not
possible to save these objects to disk due to some unhashable components in the
prefetching logic, however, it is possible to save the underlying
`PrefetchIterator.iterable` and restore the `PrefetchIterator` manually from it
upon loading.

.. _get_dataset_info:

Get dataset info
----------------

The builder class also populates a dataclass of type
:py:class:`DatasetInfo <mlip.data.dataset_info.DatasetInfo>`, which contains
metadata about the dataset that are relevant to the models while training and must be
stored together with the models for these to be usable. The populated instance of this
dataclass can be accessed easily like this:

.. code-block:: python

    # Note: this will raise an exception if accessed
    # before get_datasets() is run
    dataset_info = graph_dataset_builder.dataset_info

Usually, this object is computed by the builder based on the dataset(s), but it is also
possible to pass a pre-computed dataset info to the builder.

.. _pre_post_proc:

Custom system pre-processing and batch post-processing
------------------------------------------------------

The `get_datasets()` method of the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`
allows the user to specify custom system pre-processing and batch post-processing
functions.

These are provided as lists of simple `Callable` objects. Their purpose and signatures
are this:

* **System pre-processing:** These functions are of signature
  `list[ChemicalSystem] -> list[ChemicalSystem]` and are applied to the
  :py:class:`ChemicalSystem <mlip.data.chemical_system.ChemicalSystem>` lists (output
  of `load()` method of readers) before they are converted to
  :py:class:`Graph <mlip.graph.Graph>` objects.

* **Graph/batch post-processing:** These functions are of signature
  `Graph -> Graph` and are applied to the :py:class:`Graph <mlip.graph.Graph>` that
  represents the batches just before they are yielded by the iterator of the
  :py:class:`GraphDataset <mlip.data.graph_dataset.GraphDataset>` object.
  The builder itself only forwards these functions to the constructor of the
  :py:class:`GraphDataset <mlip.data.graph_dataset.GraphDataset>`.

An example of where this functionality is used can be found in the section
:ref:`hessian_training` below.

.. _multi_builder_mode:

Dataset processing for fine-tuning or multi-dataset training
------------------------------------------------------------

For multi-head fine-tuning (MHFT) or multi-dataset training (e.g., for an
eSEN Mixture-of-Experts model), we provide the `BuilderMode.MULTI` mode.
For an example of using multi-datasets to train an eSEN Mixture-of-Experts model,
see the
`MoE training and inference tutorial notebook <https://github.com/instadeepai/mlip/blob/main/tutorials/moe_training_and_inference_tutorial.ipynb>`_.

For this mode, the dictionary `readers` will be a nested dictionary of datasets
and splits. The only difference between the two use cases of MHFT and multi-dataset
training is that the reserved "replay" key needs to be given as the name for one of the
datasets in the MHFT case, which triggers that the builder uses a pre-computed dataset
info for that split (needs to be given and taken from the pre-trained model).

See the example below (code is provided for the MHFT case but differences
for the multi-dataset training case are explained in comments):

.. code-block:: python

    from mlip.data import GraphDatasetBuilder, BuilderMode

    # For multi-dataset training, the "replay" set would for instance
    # just be called "ds_0"
    readers = {
        "replay": {
            "train": _get_chemical_systems_reader(),
            "validation": _get_chemical_systems_reader(),
        },
        "ds_1": {
            "train": _get_chemical_systems_reader(),
            "validation": _get_chemical_systems_reader(),
        },
        "ds_2": {
            "train": _get_chemical_systems_reader(),
            "validation": _get_chemical_systems_reader(),
        },
    }

    # For multi-dataset training, the dataset info would just be
    # passed as None
    pretrained_ff = _get_pretrained_force_field()
    preset_dataset_info = pretrained_ff.dataset_info

    graph_dataset_builder = GraphDatasetBuilder(
        readers,
        GraphDatasetBuilder.Config(),
        BuilderMode.MULTI,
        dataset_info=preset_dataset_info,
    )

    # Datasets are combined for each split
    splits = graph_dataset_builder.get_datasets()
    train_set, validation_set = splits["train"], splits["validation"]

    # Single combined dataset info object (not a dictionary)
    dataset_info = graph_dataset_builder.dataset_info

.. _custom_builder_mode:

Builder mode `BuilderMode.CUSTOM`
---------------------------------

When using the custom builder mode, a flat dictionary `readers` can be passed, and each
split is processed independently (i.e., each split will get its independent dataset
info computed unless a pre-computed one is passed):

.. code-block:: python

    from mlip.data import GraphDatasetBuilder, BuilderMode

    readers = {
        "ds_1": _get_chemical_systems_reader(),
        "ds_2": _get_chemical_systems_reader(),
    }

    graph_dataset_builder = GraphDatasetBuilder(
        readers,
        GraphDatasetBuilder.Config(),
        BuilderMode.CUSTOM,
        dataset_info=None,  # if passed, this one is used for all splits
    )

    # Both, the splits and the dataset info are now dictionaries
    splits = graph_dataset_builder.get_datasets()
    dataset_info = graph_dataset_builder.dataset_info


.. _single_split_builder:

Single split dataset building
-----------------------------

When building just a single dataset split (e.g., for any inference task), the
:py:class:`SingleGraphDatasetBuilder <mlip.data.single_graph_dataset_builder.SingleGraphDatasetBuilder>`
should be used (it is also used internally by the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`).

It is straightforward to use as its API is analogous to the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`
API. The config class is of the identical type. However,
instead of accepting a dictionary of readers, it accepts a single reader
(or a list, which results in a simple concatenation of the datasets). The method
`get_dataset()` has an identical signature to the `get_datasets()` method of the
multi-split builder. The single split builder also allows passing a pre-defined
dataset info instead of computing one for the given dataset.

Note that this class is also used internally in the
:py:func:`run_batched_inference() <mlip.inference.batched_inference.run_batched_inference>`
function when the user passes a list of `ase.Atoms` to it.

.. _hessian_training:

Data processing for training on Hessian labels
----------------------------------------------

For training on Hessian labels, dataset files containing reference Hessians
are currently required to be in HDF5 format.
When reading HDF5 groups that correspond to systems with Hessians, the reader will
attempt to retrieve the property using ``"hessian"`` as the default key name.
If the provided files store
the Hessian label under a different key,
the default property name must be overwritten when instantiating the
:py:class:`Hdf5Reader <mlip.data.chemical_systems_readers.hdf5_reader.Hdf5Reader>`:

.. code-block:: python

    from mlip.data import Hdf5Reader

    reader = Hdf5Reader(
        filepath,
        data_download_fun,
        property_name_mapping={"hessian": "<hessian-property-key-name>"},
    )

When calling `get_datasets()` of the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`,
additional system pre-processing and batch post-processing functions must be applied.
See section :ref:`pre_post_proc` above for an introduction to the concept.

We provide these functions, as they are required for setting up Hessian training.
For system pre-processing, it is:
:py:func:`pad_systems_hessians <mlip.data.helper.hessian_utils.pad_systems_hessians>`.
This function pads all Hessian matrices to the maximum system size `N`,
transforming shapes from `(n, 3, n, 3)` to `(n, 3, N, 3)` (`n` is the number of atoms
for a given system)
to enable the batching of Hessians with heterogeneous shapes.
For batch post-processing, it is:
:py:func:`process_graph_hessian <mlip.data.helper.hessian_utils.process_graph_hessian>`.
It must be partially initialized with `num_hessian_rows`, indicating the number
of rows to be sampled from the full Hessian (i.e., the number of randomly
chosen force components to be differentiated with respect to all atomic coordinates).
This number is user-defined, and we recommend using a value between 4 and 16
depending on the computational cost that can be afforded.

These functions are passed during the building process:

.. code-block:: python

    from functools import partial

    post_proc = [partial(process_graph_hessian, num_rows=num_hessian_rows)]
    pre_proc = [pad_systems_hessians]

    splits = graph_dataset_builder.get_datasets(
        systems_preprocessing=pre_proc,
        graph_postprocessing=post_proc,
    )

In practice, generating Hessian references with quantum chemistry methods
is a computationally expensive process,
which limits the availability of large-scale Hessian-annotated data.
Therefore, when training Hessian-aware models
on large-scale molecular data, the dominant scenario is typically one where only a
small fraction of the dataset contains Hessian labels.

To avoid burdening the entire dataset with the restrictions and computational overhead
required for the Hessian-annotated
portion (e.g., padding operations and specific batching parameters), we provide the
:py:class:`CombinedGraphDataset <mlip.data.helpers.combined_graph_dataset.CombinedGraphDataset>` class.
This class allows to train on heterogeneous datasets, each maintaining its own
batching parameters and data processing steps.

In a nutshell, when training on large-scale data where only a subset is
annotated with Hessians, we recommend using the
:py:class:`CombinedGraphDataset <mlip.data.helpers.combined_graph_dataset.CombinedGraphDataset>`
class instead of the standard
:py:class:`GraphDataset <mlip.data.graph_dataset.GraphDataset>`.
Below is an example workflow:

.. code-block:: python

    from mlip.data import CombinedGraphDataset

    # Create dataset splits (train, validation, test) for non-Hessian dataset
    non_hessian_splits = _create_non_hessian_splits()

    # Create dataset splits (train, validation, test) for Hessian dataset
    hessian_splits = _create_hessian_splits()

    # Combine non-Hessian and Hessian splits into combined dataset objects
    splits = {
        split_name: CombinedGraphDataset.init(
            graph_datasets=[
                non_hessian_splits[split_name],
                hessian_splits[split_name]
            ],
            interleaving_method="regular",
        )
        for split_name in split_names
    }

The combined graph datasets can then be passed to the training loop in exactly the same
way as the normal graph datasets. During training, batches will be yielded from each
of the sub-datasets either in a random or regular interleaving fashion, depending on the
given `interleaving_method` argument.

See the API reference for
:py:class:`CombinedGraphDataset <mlip.data.helpers.combined_graph_dataset.CombinedGraphDataset>`
for more information on multi-host handling and different interleaving methods
between dataset types.
