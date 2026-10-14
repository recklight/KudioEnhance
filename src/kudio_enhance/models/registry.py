# -*- coding: utf-8 -*-
"""Model registry.

Adding an architecture is one entry here plus a builder function — no
``if/elif`` chain to extend in three separate places.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from kudio_enhance.config import Config

log = logging.getLogger(__name__)

__all__ = ["ModelSpec", "REGISTRY", "list_models", "is_sequence",
           "default_units", "resolve_units", "build_model", "count_params"]


@dataclass(frozen=True)
class ModelSpec:
    """How to build an architecture, and what shape of data it wants.

    ``sequence`` decides the whole data path: sequence models are trained on
    ``(N, n_frames, bins)`` windows, frame-wise models on context-stacked
    ``(N, bins * (2 * context + 1))`` rows.

    ``default_units`` is what the builder is handed when ``model.units`` is
    left unset. A width means something different to each architecture — a
    dense layer's units, a 2-D convolution's channels, each of which is a
    whole time-frequency map — so the default belongs to the architecture
    rather than to the config. It falls back to the 1024-wide stack every
    architecture shared before it had one, so an entry written without it
    builds what it always did.
    """

    builder: Callable
    sequence: bool
    description: str
    default_units: Tuple[int, ...] = (1024, 1024, 1024)


def _ddae(cfg: Config):
    from kudio_enhance.models.dense import build_ddae
    bins = cfg.audio.n_bins
    return build_ddae(input_dim=bins * (2 * cfg.model.context + 1),
                      output_dim=bins,
                      units=cfg.model.units,
                      dropout=cfg.model.dropout,
                      mask=cfg.model.predicts_mask)


def _blstm(cfg: Config):
    from kudio_enhance.models.recurrent import build_blstm
    return build_blstm(n_bins=cfg.audio.n_bins,
                       units=cfg.model.units,
                       dropout=cfg.model.dropout,
                       mask=cfg.model.predicts_mask)


def _conv_ae(cfg: Config):
    from kudio_enhance.models.convolutional import build_conv_ae
    return build_conv_ae(n_bins=cfg.audio.n_bins,
                         filters=cfg.model.units,
                         dropout=cfg.model.dropout,
                         mask=cfg.model.predicts_mask)


def _crn(cfg: Config):
    from kudio_enhance.models.crn import build_crn
    return build_crn(n_bins=cfg.audio.n_bins,
                     channels=cfg.model.units,
                     dropout=cfg.model.dropout,
                     mask=cfg.model.predicts_mask)


def _unet(cfg: Config):
    from kudio_enhance.models.unet import build_unet
    return build_unet(channels=cfg.model.units,
                      dropout=cfg.model.dropout,
                      mask=cfg.model.predicts_mask)


REGISTRY: Dict[str, ModelSpec] = {
    "ddae": ModelSpec(_ddae, sequence=False,
                      description="dense denoising autoencoder, frame-wise"),
    "blstm": ModelSpec(_blstm, sequence=True,
                       description="stacked bidirectional LSTM"),
    "conv_ae": ModelSpec(_conv_ae, sequence=True,
                         description="dilated Conv1D autoencoder, residual"),
    # The CRN's LSTMs read every channel at every bin left after five
    # halvings, so their width is the last channel count times
    # ceil(bins / 32): 64 x 9 = 576 at 257 bins, 64 x 5 = 320 at 129, for
    # about 5.5M and 1.8M parameters, most of them in the LSTMs. The paper
    # doubles on to 256, which its unpadded convolutions can afford by
    # leaving 4 of 161 bins; here 256 x 9 would be an LSTM of some 85M.
    "crn": ModelSpec(_crn, sequence=True, default_units=(16, 32, 64, 64, 64),
                     description="causal conv-recurrent network "
                                 "(Tan & Wang 2018)"),
    # Nothing in a U-Net depends on the number of bins, so this is about
    # half a million parameters at any n_fft. Three halvings take the default
    # 64-frame training window down to 8 frames at the bottom, where each
    # position already sees 68 frames of input -- wider than the window, so a
    # fourth level would add parameters and no reach that training could use.
    "unet": ModelSpec(_unet, sequence=True, default_units=(16, 32, 64, 128),
                      description="2-D U-Net over time and frequency, "
                                  "non-causal"),
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


def default_units(name: str) -> List[int]:
    """The widths architecture *name* is built with when ``model.units`` is
    unset."""
    return list(_spec(name).default_units)


def resolve_units(cfg: Config) -> Config:
    """*cfg* with ``model.units`` spelled out.

    ``None`` means "the architecture's default", which lets one starter
    config switch between a 1024-wide dense model and a 16-channel
    convolution by changing the name alone. A run directory has to say what
    was actually built, though: a default is free to change between
    releases, and a ``config.yaml`` that still said ``None`` would then
    rebuild a network its own weights no longer fit.

    Returns *cfg* itself when the widths are already given, and otherwise a
    copy, so the caller's config is never changed underneath it.
    """
    if cfg.model.units is not None:
        return cfg
    resolved = copy.deepcopy(cfg)
    resolved.model.units = default_units(cfg.model.name)
    return resolved


def build_model(cfg: Config):
    """Build the configured architecture as a freshly initialised
    :class:`torch.nn.Module`.

    The optimiser and the loop belong to :mod:`kudio_enhance.train`, so a
    model can be built and inspected without committing to either — and
    :class:`~kudio_enhance.inference.Enhancer` rebuilds a trained one the same
    way before pouring its weights in.

    ``model.units`` left unset builds the architecture's default widths; see
    :func:`resolve_units`.
    """
    spec = _spec(cfg.model.name)
    cfg = resolve_units(cfg)
    model = spec.builder(cfg)
    log.info("built %s %s (%s target): %d parameters",
             cfg.model.name, cfg.model.units, cfg.model.target,
             count_params(model))
    return model


def count_params(model) -> int:
    """How many numbers the optimiser can move.

    BatchNorm's running mean and variance are buffers in torch rather than
    weights, so they are not counted; Keras counted them as non-trainable
    weights, and reported a slightly larger number for the same network.
    """
    return sum(param.numel() for param in model.parameters())
