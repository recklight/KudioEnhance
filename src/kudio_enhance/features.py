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
           "Standardizer", "STFT", "ideal_ratio_mask", "apply_mask"]

#: Beyond this the mask is 0 or 1 to every decimal place anyone cares about,
#: and 10 ** 60 stops being a number worth computing.
_MASK_CLIP_DB = 60.0


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


def ideal_ratio_mask(clean_spec: np.ndarray,
                     noise_spec: np.ndarray) -> np.ndarray:
    """The training target for mask models, in ``[0, 1]``.

    ``M = sqrt(|S|^2 / (|S|^2 + |N|^2))`` — how much of each time-frequency bin
    is speech.

    Both arguments are the **log-power** spectrograms this package already
    computes, and the mask is derived from them rather than from a second pass
    over the waveform. A separate magnitude STFT would be a second framing of
    the same audio, and two framings that must agree are the failure this
    codebase keeps meeting.

    `kudio.waveform_to_spectrogram` returns ``log10(|X|^2)``, so
    ``|S|^2 / |N|^2`` is ``10 ** (Ls - Ln)`` and the whole thing collapses to
    ``1 / sqrt(1 + 10 ** (Ln - Ls))`` — no exponentials large enough to
    overflow once the difference is clipped.
    """
    clean_spec = np.asarray(clean_spec, dtype=np.float64)
    noise_spec = np.asarray(noise_spec, dtype=np.float64)
    difference = np.clip(noise_spec - clean_spec, -_MASK_CLIP_DB, _MASK_CLIP_DB)
    return (1.0 / np.sqrt(1.0 + np.power(10.0, difference))).astype(np.float32)


def apply_mask(noisy_spec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask x noisy log-power spectrogram -> estimated clean log-power.

    ``|S| = M |Y|`` in magnitudes, which in ``log10(|X|^2)`` is an addition:
    ``L_hat = L_noisy + 2 log10 M``. The level therefore comes from the input
    and the network only ever decides *how much to keep* — the reason a mask
    is easier to learn than a spectrogram.
    """
    mask = np.clip(np.asarray(mask, dtype=np.float64), 1e-6, 1.0)
    return (np.asarray(noisy_spec, dtype=np.float64)
            + 2.0 * np.log10(mask)).astype(np.float32)
