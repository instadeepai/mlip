.. _force_field:

.. module:: mlip.models.force_field

Force Field
===========

.. autoclass:: ForceField

    .. automethod:: __call__

    .. automethod:: from_mlip_network

    .. automethod:: init

    .. automethod:: replace_config

    .. automethod:: replace_required_properties

    .. automethod:: replace_inference_context

    .. automethod:: calculate

    .. automethod:: predict

    .. automethod:: prepare_experts_for_inference

    .. autoproperty:: cutoff_distance

    .. autoproperty:: allowed_atomic_numbers

    .. autoproperty:: config

    .. autoproperty:: dataset_info

    .. automethod:: get_energy_head

    .. automethod:: get_predictor_class

    .. automethod:: validate_properties
