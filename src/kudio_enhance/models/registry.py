# -*- coding: utf-8 -*-
"""Model registry.

Adding an architecture is one entry here plus a builder function — no
``if/elif`` chain to extend in three separate places.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict

from kudio_enhance.config import Config

log = logging.getLogger(__name__)

__all__ = ["ModelSpec", "REGISTRY", "list_models", "is_sequence", "build_model"]


@dataclass(frozen=True)
class ModelSpec:
    """How to build an architecture, and what shape of data it wants.

    ``sequence`` decides the whole data path: sequence models are trained on
    ``(N, n_frames, bins)`` windows, frame-wise models on context-stacked
    ``(N, bins * (2 * context + 1))`` rows.
    """

    builder: Callable
    sequence: bool
    description: str


def _ddae(cfg: Config):
    from kudio_enhance.models.dense import build_ddae
    bins = cfg.audio.n_bins
    return build_ddae(input_dim=bins * (2 * cfg.model.context + 1),
                      output_dim=bins,
                      units=cfg.model.units,
                      dropout=cfg.model.dropout)


def _blstm(cfg: Config):
    from kudio_enhance.models.recurrent import build_blstm
    return build_blstm(n_bins=cfg.audio.n_bins,
                       units=cfg.model.units,
                       dropout=cfg.model.dropout)


def _conv_ae(cfg: Config):
    from kudio_enhance.models.convolutional import build_conv_ae
    return build_conv_ae(n_bins=cfg.audio.n_bins,
                         filters=cfg.model.units,
                         dropout=cfg.model.dropout)


REGISTRY: Dict[str, ModelSpec] = {
    "ddae": ModelSpec(_ddae, sequence=False,
                      description="dense denoising autoencoder, frame-wise"),
    "blstm": ModelSpec(_blstm, sequence=True,
                       description="stacked bidirectional LSTM"),
    "conv_ae": ModelSpec(_conv_ae, sequence=True,
                         description="dilated Conv1D autoencoder, residual"),
}


def list_models() -> Dict[str, str]:
    """``{name: description}`` for the CLI and the docs."""
    return {name: spec.description for name, spec in REGISTRY.items()}


def _spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}. Available: "
            f"{', '.join(sorted(REGISTRY))}") from None


def is_sequence(name: str) -> bool:
    return _spec(name).sequence


def build_model(cfg: Config):
    """Build the configured architecture, uncompiled.

    Compilation belongs to :mod:`kudio_enhance.train`, so a model can be built
    and inspected without committing to an optimiser.
    """
    model = _spec(cfg.model.name).builder(cfg)
    log.info("built %s: %d parameters", cfg.model.name, model.count_params())
    return model
