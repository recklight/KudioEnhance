# -*- coding: utf-8 -*-
"""Losses for the waveform path, as plain torch functions.

Each takes the estimate first and the reference second, batched along the
leading axes, and returns a 0-d tensor: the mean over everything, which for
equal-length examples is the mean over the batch of each example's loss.
Spectra are complex tensors on the linear scale, as
:meth:`~kudioenhance.spectral.ComplexSTFT.spectrum` gives them; waveforms are
real.

:func:`hybrid_loss` is the weighted sum the ``loss`` section of the config
describes, and :func:`loss_terms` its parts, unweighted, for the log.

References
----------
S. Braun and I. Tashev, "A consolidated view of loss functions for supervised
deep learning-based speech enhancement", 44th International Conference on
Telecommunications and Signal Processing (TSP), 2021, pp. 72-76,
doi:10.1109/TSP52935.2021.9522648.

A. Défossez, G. Synnaeve and Y. Adi, "Real time speech enhancement in the
waveform domain", Interspeech 2020, pp. 3291-3295,
doi:10.21437/Interspeech.2020-2409.

J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR - half-baked or
well done?", ICASSP 2019, pp. 626-630, doi:10.1109/ICASSP.2019.8683855.

Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing ideal time-frequency
magnitude masking for speech separation", IEEE/ACM Transactions on Audio,
Speech, and Language Processing 27(8), 2019, pp. 1256-1266,
doi:10.1109/TASLP.2019.2915167.

X. Rong, T. Sun, X. Zhang, Y. Hu, C. Zhu and J. Lu, "GTCRN: A speech
enhancement model requiring ultralow computational resources", ICASSP 2024,
pp. 971-975, doi:10.1109/ICASSP48485.2024.10448310; the loss it trains with
is ``loss.py`` in its repository, github.com/Xiaobin-Rong/gtcrn.

K. Wilson, M. Chinen, J. Thorpe, B. Patton, J. Hershey, R. A. Saurous,
J. Skoglund and R. F. Lyon, "Exploring tradeoffs in models for low-latency
speech enhancement", IWAENC 2018, pp. 366-370,
doi:10.1109/IWAENC.2018.8521347.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from kudioenhance.config import LossConfig
from kudioenhance.spectral import EPS, ComplexSTFT, compress_magnitude

__all__ = ["complex_mse", "magnitude_mse", "time_l1", "neg_si_snr",
           "loss_terms", "hybrid_loss"]

#: What keeps SI-SNR finite on silence, in the projection and in the ratio.
SI_SNR_EPS = 1e-8


def _magnitude(spec: torch.Tensor, power: Optional[float],
               eps: float) -> torch.Tensor:
    """``sqrt(|X|^2 + eps)``, raised to *power* when one is given."""
    energy = spec.real.square() + spec.imag.square() + eps
    return energy.sqrt() if power is None else energy.pow(power / 2)


def complex_mse(estimate: torch.Tensor, reference: torch.Tensor, *,
                compress: Optional[float] = None,
                eps: float = EPS) -> torch.Tensor:
    """Mean squared error between two complex spectra, after power-law
    compression.

    ``L = mean |X_c(est) - X_c(ref)|^2`` over every bin, where
    ``X_c = |X|^c e^{j angle X}`` (:func:`~kudioenhance.spectral.compress_magnitude`).
    It equals the MSE of the real parts plus the MSE of the imaginary parts.

    With ``compress=0.3`` it is the complex term of the compressed loss in
    Wilson et al. (2018) and Braun and Tashev (2021), and the term GTCRN's
    repository trains with, real and imaginary MSEs added. With ``None`` it
    is the plain complex MSE. The phase counts here, which the magnitude
    terms cannot see.
    """
    error = (compress_magnitude(estimate, compress, eps)
             - compress_magnitude(reference, compress, eps))
    return (error.real.square() + error.imag.square()).mean()


def magnitude_mse(estimate: torch.Tensor, reference: torch.Tensor, *,
                  compress: Optional[float] = None,
                  eps: float = EPS) -> torch.Tensor:
    """Mean squared error between two magnitude spectra, after power-law
    compression.

    ``L = mean (|X(est)|^c - |X(ref)|^c)^2`` over every bin, with
    ``|X| = sqrt(Re^2 + Im^2 + eps)``: GTCRN's magnitude, whose *eps* keeps
    the gradient finite where a bin is zero.

    The magnitude term of Wilson et al. (2018) and Braun and Tashev (2021),
    and GTCRN's at ``compress=0.3``. Compression is what keeps the loud bins
    from deciding it alone: double every bin, and a bin 40 dB down weighs
    10^-4 of a loud one uncompressed, 10^-1.2 at c = 0.3.
    """
    error = (_magnitude(estimate, compress, eps)
             - _magnitude(reference, compress, eps))
    return error.square().mean()


def time_l1(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between two waveforms.

    ``L = mean |s_est - s_ref|`` over every sample: the waveform term of
    Défossez et al. (2020), who train beside it on a multi-resolution STFT
    loss.
    """
    return (estimate - reference).abs().mean()


def neg_si_snr(estimate: torch.Tensor, reference: torch.Tensor, *,
               zero_mean: bool = True,
               eps: float = SI_SNR_EPS) -> torch.Tensor:
    """Negative scale-invariant signal-to-noise ratio, in dB.

    ::

        s_t = (<s_est, s_ref> / ||s_ref||^2) s_ref,    e = s_est - s_t
        SI-SNR = 10 log10(||s_t||^2 / ||e||^2)

    The target ``s_t`` is the reference scaled to its best fit to the
    estimate, Le Roux et al.'s (2019) SI-SDR. With *zero_mean* both signals
    lose their mean first, as Conv-TasNet's training objective does (Luo
    and Mesgarani 2019). Scaling the estimate by any factor other than 0
    leaves the value where it was, so a model trained on this alone is free
    to put out any level; the spectral terms are what hold it.

    GTCRN's repository uses ``-log10`` of the same ratio without removing the
    mean: a tenth of this at ``zero_mean=False``, hence the 0.1
    ``loss.neg_si_snr`` defaults to, beside ``loss.zero_mean: false``.
    *eps* goes on ``||s_ref||^2`` in the projection, on ``||e||^2`` and on
    the ratio inside the logarithm, so silence stays finite and an estimate
    equal to its reference scores about ``-10 log10(||s_ref||^2 / eps)``
    rather than minus infinity.

    Waveforms are ``(..., samples)``; the result is the mean over the
    leading axes.
    """
    if zero_mean:
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        reference = reference - reference.mean(dim=-1, keepdim=True)
    scale = ((estimate * reference).sum(dim=-1, keepdim=True)
             / (reference.square().sum(dim=-1, keepdim=True) + eps))
    target = scale * reference
    ratio = (target.square().sum(dim=-1)
             / ((estimate - target).square().sum(dim=-1) + eps))
    return -(10 * torch.log10(ratio + eps)).mean()


def loss_terms(estimate: torch.Tensor, reference: torch.Tensor,
               stft: ComplexSTFT, weights: LossConfig
               ) -> Dict[str, torch.Tensor]:
    """The terms *weights* gives a weight above 0, unweighted, by name.

    *reference* is the clean waveform, ``(..., samples)``. *estimate* is
    either a waveform of the same shape or a complex spectrum on *stft*'s
    linear scale, ``(..., bins, frames)``: a model that puts out a spectrum
    is scored on that spectrum, as GTCRN's loss scores it, and its waveform
    for the time-domain terms is ``stft.waveform`` of it. A model that works
    on compressed spectra hands over ``stft.expand_spectrum(output)``.

    The spectral terms compress to ``weights.compress``, which is the loss's
    own and need not be the front end's ``audio.compress``, and SI-SNR
    removes the mean when ``weights.zero_mean`` says so. Nothing is
    transformed that no term with a weight needs.
    """
    wanted = weights.weights
    spectral = {"complex_mse", "magnitude_mse"} & set(wanted)
    temporal = {"time_l1", "neg_si_snr"} & set(wanted)
    length = reference.shape[-1]

    if torch.is_complex(estimate):
        # a mismatch here would broadcast, or be zero-padded by the inverse
        # STFT, and score the wrong pairs without a word
        expected = tuple(reference.shape[:-1]) + (stft.n_bins,
                                                  stft.n_frames(length))
        if tuple(estimate.shape) != expected:
            raise ValueError(
                f"a spectrum estimate of shape {tuple(estimate.shape)} "
                f"against a reference of {tuple(reference.shape)}, whose "
                f"spectrum is {expected}")
        estimate_spec = estimate
        estimate_wave = stft.waveform(estimate, length) if temporal else None
    else:
        if estimate.shape != reference.shape:
            raise ValueError(f"an estimate of shape {tuple(estimate.shape)} "
                             f"against a reference of {tuple(reference.shape)}")
        estimate_wave = estimate
        estimate_spec = stft.spectrum(estimate) if spectral else None
    reference_spec = stft.spectrum(reference) if spectral else None

    terms: Dict[str, torch.Tensor] = {}
    for name in wanted:
        if name == "complex_mse":
            terms[name] = complex_mse(estimate_spec, reference_spec,
                                      compress=weights.compress)
        elif name == "magnitude_mse":
            terms[name] = magnitude_mse(estimate_spec, reference_spec,
                                        compress=weights.compress)
        elif name == "time_l1":
            terms[name] = time_l1(estimate_wave, reference)
        else:
            terms[name] = neg_si_snr(estimate_wave, reference,
                                     zero_mean=weights.zero_mean)
    return terms


def hybrid_loss(estimate: torch.Tensor, reference: torch.Tensor,
                stft: ComplexSTFT, weights: LossConfig) -> torch.Tensor:
    """The weighted sum the ``loss`` section describes.

    ``L = w_c complex_mse + w_m magnitude_mse + w_t time_l1 + w_s neg_si_snr``,
    each term as :func:`loss_terms` computes it. The defaults of
    :class:`~kudioenhance.config.LossConfig`, 30, 70, 0 and 0.1 with
    ``compress`` 0.3 and the mean left in, are GTCRN's loss (Rong et al.
    2024, its repository's ``loss.py``), whose 30/70 split is Braun and
    Tashev's (2021) mix scaled by 100, when *stft* is GTCRN's as well: 512
    points, a hop of 256, ``sqrt_hann`` and ``pad_mode="reflect"``.
    """
    terms = loss_terms(estimate, reference, stft, weights)
    return sum(getattr(weights, name) * value for name, value in terms.items())
