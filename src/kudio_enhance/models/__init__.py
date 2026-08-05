# -*- coding: utf-8 -*-
"""Architectures for spectrogram-domain speech enhancement.

Builders return *uncompiled* models; :mod:`kudio_enhance.train` owns the
optimiser, loss and callbacks.
"""
from kudio_enhance.models.registry import (
    REGISTRY,
    ModelSpec,
    build_model,
    is_sequence,
    list_models,
)

__all__ = ["REGISTRY", "ModelSpec", "build_model", "is_sequence", "list_models"]
