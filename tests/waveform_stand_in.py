# -*- coding: utf-8 -*-
"""A stand-in model for the waveform path's tests, and a toy corpus for it.

No architecture of that path is in the registry yet, so the trainer and the
chunked inference are tested on the smallest model that still learns: a
gate per frequency bin that opens with the bin's level. The tests that use
it register it for their own duration with :func:`register`.

Importing this module imports torch; the test modules that do so skip
first when it is missing.
"""
from __future__ import annotations

import kudio
import numpy as np
import torch
from torch import nn

from kudioenhance.data import Pair
from kudioenhance.models import REGISTRY, ModelSpec

NAME = "noise_gate"
SR = 8000


class NoiseGate(nn.Module):
    """``mask = sigmoid(slope_f (level - threshold_f))``, applied to the
    complex spectrum, phase kept.

    *level* is ``log10`` of the power averaged over three frames, so each
    output frame depends on its neighbours, as a real model's would. While
    training, dropout on the mask draws from torch's generator, so a resumed
    run has that generator to put back as well. 2 parameters per bin.
    """

    def __init__(self, n_bins: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.ones(n_bins))
        self.threshold = nn.Parameter(torch.full((n_bins,), -3.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        power = spec.real.square() + spec.imag.square()
        level = torch.log10(nn.functional.avg_pool1d(
            power, 3, stride=1, padding=1, count_include_pad=False) + 1e-8)
        mask = torch.sigmoid(self.slope[:, None]
                             * (level - self.threshold[:, None]))
        return spec * self.dropout(mask)


class Idle(NoiseGate):
    """A NoiseGate whose output its parameters do not change: the gradient
    is exactly zero, so the loss never moves and every epoch is a plateau."""

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return spec * (1 + 0 * self.slope.sum())


def _build(cfg):
    # n_fft // 2 + 1 rather than cfg.audio.n_bins, which asks kudio, and
    # kudio has no sqrt_hann
    return NoiseGate(cfg.audio.n_fft // 2 + 1, cfg.model.dropout)


def _build_idle(cfg):
    return Idle(cfg.audio.n_fft // 2 + 1)


def register(monkeypatch) -> None:
    """Put ``noise_gate`` and ``idle`` into the registry for one test."""
    for name, builder in ((NAME, _build), ("idle", _build_idle)):
        monkeypatch.setitem(REGISTRY, name, ModelSpec(
            builder, sequence=True, waveform=True, default_units=(1,),
            description="per-bin gate, a stand-in for the waveform path"))


def tones_in_noise(seconds: float, seed: int, snr_db: float = 0.0):
    """``(clean, noisy)``: a few low harmonics, and the same with white
    noise added at *snr_db*."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = rng.uniform(150, 300)
    clean = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k
                for k in range(1, 4))
    clean = 0.3 * clean / np.max(np.abs(clean))
    noise = rng.normal(0, 1, len(t))
    noise *= np.sqrt(np.mean(clean ** 2) / np.mean(noise ** 2)
                     / 10 ** (snr_db / 10))
    return clean.astype(np.float32), (clean + noise).astype(np.float32)


def write_pairs(folder, count: int, seconds: float, seed: int):
    """*count* clean/noisy pairs of :func:`tones_in_noise` under *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i in range(count):
        clean, noisy = tones_in_noise(seconds, seed + i)
        kudio.save_wave(folder / f"clean{i}.wav", clean, SR, subtype="FLOAT")
        kudio.save_wave(folder / f"noisy{i}.wav", noisy, SR, subtype="FLOAT")
        pairs.append(Pair(noisy=str(folder / f"noisy{i}.wav"),
                          clean=str(folder / f"clean{i}.wav")))
    return pairs
