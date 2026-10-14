# -*- coding: utf-8 -*-
"""Architectures for spectrogram-domain speech enhancement.

Builders return plain :class:`torch.nn.Module` instances;
:mod:`kudio_enhance.train` owns the optimiser, the loss and the loop. The
builders import torch only when called, so listing the architectures does not
need it.
"""
from kudio_enhance.models.registry import (
    REGISTRY,
    ModelSpec,
    build_model,
    count_params,
    default_units,
    is_sequence,
    list_models,
    resolve_units,
)

__all__ = ["REGISTRY", "ModelSpec", "build_model", "count_params",
           "default_units", "is_sequence", "list_models", "resolve_units"]
