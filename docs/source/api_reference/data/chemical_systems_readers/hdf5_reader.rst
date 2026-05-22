.. _hdf5_reader:

.. module:: mlip.data.chemical_systems_readers.hdf5_reader

HDF5 Reader
===========

This reader expects the data to be
in `HDF5 format <https://docs.h5py.org/en/>`_ and organized in the following way.
The data must be defined as groups by the structure name. The scalar properties will be
stored as attributes to the group and the array properties as arrays. Below, we provide
an example of how to read the data from such a compliant HDF5 file to demonstrate
how the data is organized:

.. code-block:: python

    with h5py.File(hdf5_dataset_path, "r") as h5file:
        # Get the identifiers for all structures in the dataset
        struct_names = list(h5file.keys())

        # Just loading the first one for the sake of an example
        structure = h5file[struct_names[0]]
        positions = structure["positions"][:]
        element_numbers = structure["elements"][:]
        forces = structure["forces"][:]
        # Stress could be optional if not needed during training
        if "stress" in structure:
            stress = structure["stress"][:]

        # Energy is a scalar
        energy = structure.attrs["energy"]

See below for the API reference to the associated loader class.

.. autoclass:: Hdf5Reader

    .. automethod:: __init__

    .. automethod:: load
