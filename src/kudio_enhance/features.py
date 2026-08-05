# -*- coding: utf-8 -*-
"""Spectrogram features and their inverse.

Everything here now comes from `kudio` — this package no longer carries its own
copy of context stacking, windowing or feature normalisation. What is left is
the binding between a :class:`~kudio_enhance.config.Config` and the geometry it
describes.
"""
from __future__ import annotations

import logging

import numpy as np
from kudio import STFT, Standardizer, frame_windows, stack_context

from kudio_enhance.config import AudioConfig

log = logging.getLogger(__name__)

__all__ = ["spectrogram", "to_waveform", "stack_context", "frame_windows",
           "Standardizer", "STFT"]


def spectrogram(y: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Waveform -> log-power spectrogram, shaped ``(frames, bins)``."""
    return cfg.stft.forward(np.asarray(y, dtype=np.float32))


def to_waveform(y_ref: np.ndarray, spec: np.ndarray,
                cfg: AudioConfig) -> np.ndarray:
    """Log-power spectrogram -> waveform, borrowing the phase of *y_ref*.

    *y_ref* is the noisy input: phase is left untouched by these models, which
    is the standard magnitude-only enhancement assumption.
    """
    return cfg.stft.inverse(np.asarray(y_ref, dtype=np.float32),
                            np.asarray(spec))
