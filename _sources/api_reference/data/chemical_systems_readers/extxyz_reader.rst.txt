.. _extxyz_reader:

.. module:: mlip.data.chemical_systems_readers.extxyz_reader

Extxyz Reader
=============

The features in the extxyz format are defined in the properties on the comment lines
of a concatenated XYZ file.
Each structure starts with the number of atoms, followed by the comment line and
then the elements, positions, and forces as specified in the properties.

Multiple structures are concatenated together, hence the whole set of training
structures can be in just one file, the validation structures in another, and
the test structures in a third file.

Here's a shortened example of the training data in the extxyz format:

.. code-block::

    21
    Properties=species:S:1:pos:R:3:forces:R:3 energy=-17617.63598758549 pbc="F F F"
    C  2.03112297  -1.10783801  -0.35158800   2.72979276  -1.55877755  -0.63202814
    C  0.68817554   0.94896126  -1.72487641  -0.92555477  -0.52119051   2.26082812
    C  2.47575017  -0.65064361  -1.63039847   1.67313734  -3.78441218   1.72467687
    ...

For loading the extxyz file, we internally use ``ase.io.read`` from the
`ASE library <https://wiki.fysik.dtu.dk/ase>`_.

See below for the API reference to the associated loader class.

.. autoclass:: ExtxyzReader

    .. automethod:: __init__

    .. automethod:: load
