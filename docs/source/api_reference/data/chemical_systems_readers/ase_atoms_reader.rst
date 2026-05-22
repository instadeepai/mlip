.. _ase_atoms_reader:

.. module:: mlip.data.chemical_systems_readers.ase_atoms_reader

``ase.Atoms`` Reader
====================

This reader expects the data to be a list of ``ase.Atoms`` objects, and converts it
to a list of :py:class:`ChemicalSystem <mlip.data.chemical_system.ChemicalSystem>`
objects.

.. code-block:: python

    with open(input_xyz_path) as xyz_file:
        atoms = ase_read(xyz_file, index=":")

    reader = ASEAtomsReader(atoms)
    systems = reader.load()

See below for the API reference to the associated loader class.

.. autoclass:: ASEAtomsReader

    .. automethod:: __init__

    .. automethod:: load
