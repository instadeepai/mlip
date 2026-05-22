.. _visnet:

ViSNet
======

.. module:: mlip.models.visnet.network

    .. autoclass:: Visnet

        .. automethod:: setup

        .. automethod:: __call__

.. module:: mlip.models.visnet.config

    .. autoclass:: VisnetConfig

.. module:: mlip.models.visnet.blocks

    .. autoclass:: VisnetEmbeddingBlock

        .. automethod:: setup

        .. automethod:: __call__

    .. autoclass:: VisnetNeighborEmbeddingBlock

        .. automethod:: setup

        .. automethod:: __call__

    .. autoclass:: VisnetEdgeEmbeddingBlock

        .. automethod:: setup

        .. automethod:: __call__

    .. autoclass:: VisnetMultiHeadReadoutBlock

        .. automethod:: setup

        .. automethod:: __call__

.. module:: mlip.models.visnet.layer

    .. autoclass:: VisnetLayer

        .. automethod:: setup

        .. automethod:: __call__
