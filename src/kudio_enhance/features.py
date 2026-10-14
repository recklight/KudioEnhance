# -*- coding: utf-8 -*-
"""Spectrogram features and their inverse.

Everything here now comes from `kudio` — this package no longer carries its own
copy of context stacking, windowing or feature normalisation. What is left is
the binding between a :class:`~kudio_enhance.config.Config` and the geometry it
describes, and :func:`fit_standardizer`: kudio's normalisation fitted a chunk
at a time and summed in float64, which streamed training needs and
`kudio.Standardizer` cannot do, and which every training set is fitted with.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from kudio import STFT, Standardizer, frame_windows, stack_context

from kudio_enhance.config import AudioConfig

log = logging.getLogger(__name__)

__all__ = ["spectrogram", "to_waveform", "stack_context", "frame_windows",
           "Standardizer", "fit_standardizer", "STFT", "ideal_ratio_mask",
           "apply_mask"]

#: Beyond this the mask is 0 or 1 to every decimal place anyone cares about,
#: and 10 ** 60 stops being a number worth computing.
_MASK_CLIP_DB = 60.0

#: What `kudio.Standardizer.fit` adds to every standard deviation, so a column
#: that never changes divides by something. A test fits both ways and holds
#: this to it.
_STD_EPSILON = 1e-8


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


def fit_standardizer(chunks: Iterable[np.ndarray]) -> Standardizer:
    """``Standardizer().fit(np.concatenate(chunks))``, a chunk at a time.

    Fitting the normalisation is the one step of training that has to see
    every frame before the first batch can be built, and on a training set too
    large to hold, the concatenation is the thing that cannot be built. Here
    the frames go past a chunk at a time and only a count, a mean and a sum of
    squared deviations per column stay behind, merged chunk into chunk with
    Chan, Golub and LeVeque's update.

    The statistics are kudio's: the mean, the population standard deviation
    and the same ``1e-8`` added to it, returned as a `kudio.Standardizer` in
    float32. The sums are not: kudio's run in float32, along the frames, and
    their rounding error grows with the number of frames, where these run in
    float64. On a few hundred frames the two agree to float32 precision; on
    an hour of 16 kHz audio at a hop of 256, kudio's statistics drift from
    the true mean and deviation by a few parts in 10^5, and these stay within
    a part in 10^7. Training fits with this whether its set is held in memory
    or streamed, so the two normalise alike.

    Chunks without rows are skipped. Raises ``ValueError`` when no chunk had
    any, or when the chunks disagree on the number of columns.
    """
    count, mean, squares = 0, None, None
    for chunk in chunks:
        # cast to float32 first, as kudio's fit does, so both see the same
        # numbers whatever precision the caller handed over
        frames = np.asarray(chunk, dtype=np.float32)
        if frames.ndim != 2:
            raise ValueError(f"frames must be 2-D, got shape {frames.shape}")
        if not len(frames):
            continue
        if mean is not None and frames.shape[1] != len(mean):
            raise ValueError(f"a chunk of {frames.shape[1]} column(s) after "
                             f"chunks of {len(mean)}")
        frames = frames.astype(np.float64)
        n = len(frames)
        chunk_mean = frames.mean(axis=0)
        chunk_squares = np.square(frames - chunk_mean).sum(axis=0)
        if mean is None:
            mean, squares = chunk_mean, chunk_squares
        else:
            delta = chunk_mean - mean
            total = count + n
            mean = mean + delta * (n / total)
            squares = (squares + chunk_squares
                       + np.square(delta) * (count * n / total))
        count += n
    if not count:
        raise ValueError("no frames to fit a Standardizer on")
    # the epsilon goes onto the float32 deviation, as it does in kudio
    std = (np.sqrt(squares / count).astype(np.float32)
           + np.float32(_STD_EPSILON))
    return Standardizer(mean=mean.astype(np.float32), std=std)


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
