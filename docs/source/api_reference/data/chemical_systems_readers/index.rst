.. _chemical_systems_readers:

.. module:: mlip.data.chemical_systems_readers

Chemical System Readers
=======================

This module contains chemical systems readers.
The purpose of the readers is to load a
dataset into three lists containing
:py:class:`ChemicalSystem <mlip.data.chemical_system.ChemicalSystem>` objects,
the first list being for the training set, the second for the validation set,
and the third one for the test set.


.. toctree::
    :maxdepth: 1

    chemical_systems_reader
    extxyz_reader
    hdf5_reader
    combined_reader
    reader_utils
