# -*- coding: utf-8 -*-
"""The complex spectrum in torch, and power-law compression of its magnitude.

The log-power path's spectrogram is computed by kudio in numpy, and its phase
is thrown away (:mod:`kudioenhance.features`). The waveform path's models see
the complex spectrum and are trained through the inverse transform, so the
transform runs in torch, on the batch, where the gradient passes through it.

The framing is kudio's, which is librosa's: a centred STFT padded with
``n_fft // 2`` zeros on each side, and a periodic window of ``win_length``
centred in ``n_fft``. With the same ``audio`` section the frames here are the
log-power path's frames: this spectrum is ``kudio.STFT.analyse``'s to float32
precision, and ``log10(|X|^2)`` of it is the log-power spectrogram. A test
holds the two to that. ``pad_mode="reflect"`` mirrors the signal into the
padding instead, which is what ``torch.stft`` does unless told otherwise and
so what GTCRN's code does; only the frames within ``n_fft // 2`` of either
end differ.

Spectra are shaped ``(..., bins, frames)``, torch's order, which is the
transpose of kudio's ``(frames, bins)``.

Importing this module imports torch.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import nn

from kudioenhance.config import AudioConfig

__all__ = ["WINDOWS", "PAD_MODES", "make_window", "compress_magnitude",
           "expand_magnitude", "ComplexSTFT"]

#: Added to ``|X|^2`` before a magnitude is raised to a power below 1, where
#: the slope of ``|X|^c`` at a bin that is exactly zero would be infinite.
#: GTCRN's loss puts the same 1e-12 under its square root.
EPS = 1e-12

#: The windows a name can ask for. Every one but ``sqrt_hann`` is the
#: periodic window of the same name in scipy, which kudio's log-power path
#: takes through librosa. ``sqrt_hann`` is the square root of the periodic
#: Hann window, GTCRN's: applied at analysis and again at synthesis it
#: overlap-adds to a constant at half-window hops.
WINDOWS: Dict[str, Callable[[int], torch.Tensor]] = {
    "hann": lambda n: torch.hann_window(n, periodic=True),
    "hamming": lambda n: torch.hamming_window(n, periodic=True),
    "blackman": lambda n: torch.blackman_window(n, periodic=True),
    "bartlett": lambda n: torch.bartlett_window(n, periodic=True),
    "boxcar": lambda n: torch.ones(n),
    "sqrt_hann": lambda n: torch.hann_window(n, periodic=True).sqrt(),
}


#: What goes into the ``n_fft // 2`` samples a centred STFT adds at each
#: end: zeros, as kudio pads, or the signal mirrored, as torch pads by
#: default. The same names as ``torch.stft`` takes.
PAD_MODES = ("constant", "reflect")


def make_window(name: str, length: int) -> torch.Tensor:
    """The window *name* asks for, *length* samples of float32."""
    try:
        build = WINDOWS[name]
    except KeyError:
        raise ValueError(
            f"unknown window {name!r} for the complex spectrum; choose from "
            f"{', '.join(WINDOWS)}") from None
    return build(length).to(torch.float32)


def compress_magnitude(spec: torch.Tensor, power: Optional[float],
                       eps: float = EPS) -> torch.Tensor:
    """*spec* with its magnitude raised to *power* and its phase kept.

    ``X_c = |X|^c e^{j angle X}``, computed as ``X (|X|^2 + eps)^((c - 1) / 2)``.
    Away from silence that is ``X / |X|^(1 - c)``, GTCRN's form, and at a bin
    that is exactly zero it stays zero with a finite gradient, where
    ``|X|^c`` has an infinite slope. ``None`` or 1 returns *spec* as it is.
    """
    if power is None or power == 1:
        return spec
    energy = spec.real.square() + spec.imag.square()
    return spec * (energy + eps).pow((power - 1) / 2)


def expand_magnitude(spec: torch.Tensor, power: Optional[float],
                     eps: float = EPS) -> torch.Tensor:
    """Undo :func:`compress_magnitude`: the magnitude raised to ``1 / power``,
    the phase kept.

    The two invert each other to a relative error of about
    ``(1 - c) / (2c) * eps / |X|^2`` in each bin: 1.3e-3 at c = 0.3 for a
    bin of magnitude 3e-5, 1.2e-8 for one of 0.01. A test takes a waveform
    through both and back.
    """
    if power is None or power == 1:
        return spec
    energy = spec.real.square() + spec.imag.square()
    return spec * (energy + eps).pow((1 / power - 1) / 2)


def _check_overlap(window: torch.Tensor, n_fft: int, hop_length: int,
                   name: str) -> None:
    """Refuse a window the inverse STFT would divide by zero with.

    ``torch.istft`` divides by the squared window overlap-added at the hop,
    and fails on its first call where that sum touches zero. Checked here,
    the setting is refused when it is built, not at the first loss.
    """
    padded = torch.zeros(n_fft, dtype=torch.float64)
    start = (n_fft - len(window)) // 2
    padded[start:start + len(window)] = window.double().square()
    folded = torch.zeros(hop_length, dtype=torch.float64)
    for offset in range(0, n_fft, hop_length):
        piece = padded[offset:offset + hop_length]
        folded[:len(piece)] += piece
    if folded.min() < 1e-11:
        raise ValueError(
            f"the {name} window does not overlap-add at a hop of "
            f"{hop_length}: its squares sum to zero between frames, so the "
            f"inverse STFT has nothing to divide by; use a shorter hop")


class ComplexSTFT(nn.Module):
    """A complex STFT and its inverse, with optional power-law compression.

    :meth:`spectrum` and :meth:`waveform` are the plain transforms, on the
    linear scale. :meth:`forward` is what a model sees: the spectrum with its
    magnitude compressed to *compress*, or left linear when that is ``None``.
    :meth:`inverse` takes such a spectrum back to a waveform.

    The window is a buffer, so it moves with ``.to(device)`` and stays out of
    a checkpoint's weights. The transforms take any number of leading axes:
    ``(samples,)``, ``(batch, samples)`` and so on.

    :param win_length: defaults to *n_fft*; a shorter window is centred in
        the frame, as librosa and torch both do.
    :param window: a name from :data:`WINDOWS`.
    :param pad_mode: from :data:`PAD_MODES`. ``reflect`` needs a waveform
        longer than ``n_fft // 2``.
    :raises ValueError: also for a window whose squares do not overlap-add
        to something above zero at *hop_length* (Hann at a hop of the whole
        window, say), which the inverse cannot divide by.
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 256,
                 win_length: Optional[int] = None, window: str = "hann",
                 compress: Optional[float] = None,
                 pad_mode: str = "constant") -> None:
        super().__init__()
        win_length = n_fft if win_length is None else win_length
        if win_length > n_fft:
            raise ValueError(f"win_length ({win_length}) cannot exceed "
                             f"n_fft ({n_fft})")
        if not 1 <= hop_length <= win_length:
            raise ValueError(f"hop_length must be between 1 and win_length "
                             f"({win_length}), got {hop_length}")
        if compress is not None and not 0 < compress <= 1:
            raise ValueError(
                f"compress must be an exponent in (0, 1], got {compress!r}")
        if pad_mode not in PAD_MODES:
            raise ValueError(f"unknown pad_mode {pad_mode!r}; choose from "
                             f"{', '.join(PAD_MODES)}")
        window_values = make_window(window, win_length)
        _check_overlap(window_values, n_fft, hop_length, window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_name = window
        self.compress = compress
        self.pad_mode = pad_mode
        self.register_buffer("window", window_values, persistent=False)

    @classmethod
    def from_config(cls, audio: AudioConfig) -> "ComplexSTFT":
        """The geometry and compression an ``audio`` section describes."""
        return cls(n_fft=audio.n_fft, hop_length=audio.hop_length,
                   win_length=audio.window_length, window=audio.window,
                   compress=audio.compress, pad_mode=audio.pad_mode)

    def extra_repr(self) -> str:
        return (f"n_fft={self.n_fft}, hop_length={self.hop_length}, "
                f"win_length={self.win_length}, window={self.window_name!r}, "
                f"compress={self.compress}, pad_mode={self.pad_mode!r}")

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    def n_frames(self, n_samples: int) -> int:
        """Frames a waveform of *n_samples* gives, as kudio counts them."""
        return 1 + (n_samples + 2 * (self.n_fft // 2) - self.n_fft) \
            // self.hop_length

    # ------------------------------------------------------------ transforms

    def spectrum(self, wave: torch.Tensor) -> torch.Tensor:
        """Waveform ``(..., samples)`` -> complex spectrum
        ``(..., bins, frames)``, on the linear scale."""
        lead = wave.shape[:-1]
        spec = torch.stft(wave.reshape(-1, wave.shape[-1]), self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window.to(wave.dtype), center=True,
                          pad_mode=self.pad_mode, normalized=False,
                          onesided=True, return_complex=True)
        return spec.reshape(lead + spec.shape[-2:])

    def waveform(self, spec: torch.Tensor,
                 length: Optional[int] = None) -> torch.Tensor:
        """Linear-scale complex spectrum ``(..., bins, frames)`` -> waveform
        ``(..., samples)``.

        *length* is the number of samples the analysed waveform had; without
        it the waveform comes back ``samples % hop_length`` short, up to
        ``hop_length - 1`` samples.
        """
        lead = spec.shape[:-2]
        wave = torch.istft(spec.reshape((-1,) + spec.shape[-2:]), self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length,
                           window=self.window.to(spec.real.dtype),
                           center=True, normalized=False, onesided=True,
                           length=length)
        return wave.reshape(lead + wave.shape[-1:])

    def compress_spectrum(self, spec: torch.Tensor) -> torch.Tensor:
        """A linear-scale spectrum compressed as :meth:`forward` compresses."""
        return compress_magnitude(spec, self.compress)

    def expand_spectrum(self, spec: torch.Tensor) -> torch.Tensor:
        """A compressed spectrum back on the linear scale."""
        return expand_magnitude(spec, self.compress)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Waveform -> the spectrum a model sees: compressed to
        ``compress``, or linear."""
        return self.compress_spectrum(self.spectrum(wave))

    def inverse(self, spec: torch.Tensor,
                length: Optional[int] = None) -> torch.Tensor:
        """A spectrum as :meth:`forward` gives it -> waveform."""
        return self.waveform(self.expand_spectrum(spec), length)
