.. _data_processing:

Data processing
===============

Set up graph dataset builder
----------------------------

In order to train a model or run batched inference, one needs to process the data
into objects of type
:py:class:`GraphDataset <mlip.data.helpers.graph_dataset.GraphDataset>`.
This can be achieved by using the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`
class, which can be instantiated from its associated pydantic config and a
chemical systems reader that is derived from the
:py:class:`ChemicalSystemsReader <mlip.data.chemical_systems_readers.chemical_systems_reader.ChemicalSystemsReader>`
base class:

.. code-block:: python

    from mlip.data import GraphDatasetBuilder

    reader = _get_chemical_systems_reader()  # this is a placeholder for the moment
    builder_config = GraphDatasetBuilder.Config(
        graph_cutoff_angstrom=5.0,
        max_n_node=None,
        max_n_edge=None,
        batch_size=16,
    )
    graph_dataset_builder = GraphDatasetBuilder(reader, builder_config)

In the example above, we set some example values for the settings in the
:py:class:`GraphDatasetBuilderConfig <mlip.data.configs.GraphDatasetBuilderConfig>`.
For simpler code, we allow to access this config object directly via
``GraphDatasetBuilder.Config``. Check out the API reference of the class to see the
full set of configurable values and for which values we have defaults available.

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

    reader_config = ExtxyzReader.Config(
        train_dataset_paths = "...",
        valid_dataset_paths = "...",
        test_dataset_paths = "...",
    )

    # If data is stored locally
    reader = ExtxyzReader(reader_config)

    # If data is on remote storage, one can also provide a data download function
    reader = ExtxyzReader(reader_config, data_download_fun)

The configuration object used here is the
:py:class:`ChemicalSystemsReaderConfig <mlip.data.configs.ChemicalSystemsReaderConfig>`,
again accessible via ``ExtxyzReader.Config`` to reduce the number of required imports.

In the example above, the ``data_download_fun`` is a simple function that takes in
a source and a target path and performs the download operation. Our helper functions
for splitting a dataset are documented :ref:`here <data_split>`.

If you have multiple datasets in different formats and would like to combine them,
you can do so by instead using the
:py:class:`CombinedReader <mlip.data.chemical_systems_readers.combined_reader.CombinedReader>`:

.. code-block:: python

    from mlip.data import CombinedReader

    readers = _get_list_of_individual_chemical_system_readers()  # placeholder
    combined_reader = CombinedReader(readers)

This combined reader can then also be used as an input to the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`.

Built-in graph dataset readers: data formats
--------------------------------------------

As mentioned above, two built-in core readers are currently provided:
:py:class:`ExtxyzReader <mlip.data.chemical_systems_readers.extxyz_reader.ExtxyzReader>`
and
:py:class:`Hdf5Reader <mlip.data.chemical_systems_readers.hdf5_reader.Hdf5Reader>`.

They each support their own data format.
To train an MLIP model, we need a dataset of atomic systems
with the following features per system with specific units:

* the positions (i.e., coordinates) of the atoms in the structure in Angstrom
* the element numbers of the atoms
* the forces of the atoms in eV / Angstrom
* the energy of the structure in eV
* (optional) the stress of the structure  in eV / Angstrom\ :sup:`3`
* (optional) the periodic boundary conditions

For a detailed description of the data format that the
:py:class:`ExtxyzReader <mlip.data.chemical_systems_readers.extxyz_reader.ExtxyzReader>`
requires, see
:ref:`here <extxyz_reader>`.

For a detailed description of the data format that the
:py:class:`Hdf5Reader <mlip.data.chemical_systems_readers.hdf5_reader.Hdf5Reader>`.
requires, see
:ref:`here <hdf5_reader>`.

Start preprocessing
-------------------

Once you have the ``graph_dataset_builder`` set up, you can start the preprocessing and
fetch the resulting datasets:

.. code-block:: python

    graph_dataset_builder.prepare_datasets()

    splits = graph_dataset_builder.get_splits()
    train_set, validation_set, test_set = splits

The resulting datasets are of type
:py:class:`GraphDataset <mlip.data.helpers.graph_dataset.GraphDataset>`
as mentioned above. For example, to process the batches in the training set, one
can execute:

.. code-block:: python

    num_graphs = len(train_set.graphs)
    num_batches = len(train_set)

    for batch in train_set:
        _process_batch_in_some_way(batch)

Get sharded batches
-------------------

If one wants to generate batches that are sharded across devices and prefetched, the
arguments to the ``get_splits()`` member of the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`
must be set to the following:

.. code-block:: python

    splits = graph_dataset_builder.get_datasets(
        prefetch=True, devices=jax.local_devices()
    )
    train_set, valid_set, test_set = splits

Now, the datasets are not of type
:py:class:`GraphDataset <mlip.data.helpers.graph_dataset.GraphDataset>` anymore,
but of type
:py:class:`PrefetchIterator <mlip.data.helpers.data_prefetching.PrefetchIterator>`
instead which implements batch prefetching on top of the
:py:class:`ParallelGraphDataset <mlip.data.helpers.data_prefetching.ParallelGraphDataset>`
class. It can be iterated over to obtain the sharded batches in the same way, however,
note that it does not have a ``graphs`` member that can be accessed directly.

.. _get_dataset_info:

Get dataset info
----------------

Furthermore, the builder class also populates a dataclass of type
:py:class:`DatasetInfo <mlip.data.dataset_info.DatasetInfo>`, which contains
metadata about the dataset which are relevant to the models while training and must be
stored together with the models for these to be usable. The populated instance of this
dataclass can be accessed easily like this:

.. code-block:: python

    # Note: this will raise an exception if accessed
    # before prepare_datasets() is run
    dataset_info = graph_dataset_builder.dataset_info
