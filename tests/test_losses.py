# -*- coding: utf-8 -*-
"""The waveform path's losses against values worked out by hand.

Every case is built so the answer is known before the function runs: an
orthogonal error of known power, a phase turned by a known angle, every bin
doubled. A loss that only runs and decreases could still be the wrong loss.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs PyTorch",
                            exc_type=ImportError)

from kudioenhance.config import AudioConfig, LossConfig  # noqa: E402
from kudioenhance.losses import (  # noqa: E402
    complex_mse,
    hybrid_loss,
    loss_terms,
    magnitude_mse,
    neg_si_snr,
    time_l1,
)
from kudioenhance.spectral import ComplexSTFT  # noqa: E402

SR = 16000


def _tones(n_samples: int = 1600, batch: int = 3) -> torch.Tensor:
    """A batch of sines at whole numbers of cycles over the length, so a
    sine and a cosine of the same frequency are exactly orthogonal."""
    t = torch.arange(n_samples, dtype=torch.float64) / n_samples
    cycles = torch.tensor([5.0, 11.0, 23.0])[:batch, None]
    return 0.5 * torch.sin(2 * math.pi * cycles * t)


def _quadrature(n_samples: int = 1600, batch: int = 3) -> torch.Tensor:
    t = torch.arange(n_samples, dtype=torch.float64) / n_samples
    cycles = torch.tensor([5.0, 11.0, 23.0])[:batch, None]
    return 0.5 * torch.cos(2 * math.pi * cycles * t)


def _spectrum(seed: int = 0, shape=(2, 33, 20)) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return torch.from_numpy(values)


# ------------------------------------------------------- identical inputs

@pytest.mark.parametrize("compress", [None, 0.3])
def test_the_spectral_losses_are_zero_for_identical_spectra(compress):
    spec = _spectrum()
    assert complex_mse(spec, spec.clone(), compress=compress) == 0
    assert magnitude_mse(spec, spec.clone(), compress=compress) == 0


def test_time_l1_is_zero_for_identical_waveforms():
    wave = _tones()
    assert time_l1(wave, wave.clone()) == 0


def test_si_snr_of_a_perfect_estimate_is_bounded_by_its_eps():
    """Not zero but the far end of the scale: about 10 log10(||s||^2 / eps),
    the only thing that stops a perfect estimate scoring infinity."""
    wave = _tones()
    energy = wave.square().sum(dim=-1)
    expected = -(10 * torch.log10(energy / 1e-8)).mean()
    assert torch.isclose(neg_si_snr(wave, wave.clone()), expected, atol=1e-6)


def test_the_hybrid_of_a_perfect_estimate_is_only_its_si_snr_floor():
    stft = ComplexSTFT(256, 128, window="sqrt_hann")
    wave = _tones(n_samples=2048)
    weights = LossConfig()
    terms = loss_terms(wave, wave.clone(), stft, weights)
    assert terms["complex_mse"] == 0 and terms["magnitude_mse"] == 0
    assert torch.isclose(hybrid_loss(wave, wave.clone(), stft, weights),
                         0.1 * terms["neg_si_snr"])


# ------------------------------------------------------- known values

def test_complex_mse_of_a_constant_offset_is_its_squared_size():
    spec = _spectrum()
    offset = 0.3 - 0.4j                      # |offset|^2 = 0.25
    assert torch.isclose(complex_mse(spec + offset, spec),
                         torch.tensor(0.25, dtype=torch.float64))


def test_complex_mse_is_the_real_mse_plus_the_imaginary_mse():
    est, ref = _spectrum(1), _spectrum(2)
    by_parts = (torch.nn.functional.mse_loss(est.real, ref.real)
                + torch.nn.functional.mse_loss(est.imag, ref.imag))
    assert torch.isclose(complex_mse(est, ref), by_parts)


def test_complex_mse_sees_a_turned_phase_that_the_magnitude_cannot():
    """Every bin turned by phi: the magnitudes do not change, and the
    compressed complex error is |X|^(2c) |e^{j phi} - 1|^2 = |X|^(2c) (2 - 2
    cos phi)."""
    spec = _spectrum()
    phi = 0.7
    turned = spec * complex(math.cos(phi), math.sin(phi))
    expected = (spec.abs() ** 0.6).mean() * (2 - 2 * math.cos(phi))
    assert torch.isclose(complex_mse(turned, spec, compress=0.3, eps=0),
                         expected)
    assert magnitude_mse(turned, spec, compress=0.3) < 1e-20


def test_magnitude_mse_of_a_doubled_spectrum():
    """(2^c - 1)^2 mean |X|^(2c), and (2 - 1)^2 mean |X|^2 uncompressed."""
    spec = _spectrum()
    compressed = (2 ** 0.3 - 1) ** 2 * (spec.abs() ** 0.6).mean()
    assert torch.isclose(magnitude_mse(2 * spec, spec, compress=0.3, eps=0),
                         compressed)
    assert torch.isclose(magnitude_mse(2 * spec, spec, eps=0),
                         spec.abs().square().mean())


def test_compression_takes_the_loud_bins_down_to_size():
    """Two bins 40 dB apart, both doubled. Uncompressed the quiet one's error
    is 10^-4 of the loud one's, at c = 0.3 it is 10^-1.2: compression is what
    lets the quiet bins count."""
    spec = torch.tensor([[10.0 + 0j], [0.1 + 0j]], dtype=torch.complex128)

    def ratio(compress):
        quiet = magnitude_mse(2 * spec[1], spec[1], compress=compress, eps=0)
        loud = magnitude_mse(2 * spec[0], spec[0], compress=compress, eps=0)
        return (quiet / loud).item()

    assert ratio(None) == pytest.approx(1e-4)
    assert ratio(0.3) == pytest.approx(10 ** -1.2)
    complex_ratio = (complex_mse(2 * spec[1], spec[1], compress=0.3, eps=0)
                     / complex_mse(2 * spec[0], spec[0], compress=0.3, eps=0))
    assert complex_ratio.item() == pytest.approx(10 ** -1.2)


def test_time_l1_of_a_constant_offset():
    wave = _tones()
    assert torch.isclose(time_l1(wave + 0.1, wave),
                         torch.tensor(0.1, dtype=torch.float64))
    assert torch.isclose(time_l1(wave - 0.1, wave),
                         torch.tensor(0.1, dtype=torch.float64))


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_si_snr_of_an_orthogonal_error_is_its_power_ratio(snr_db):
    """s + n with n orthogonal to s: the projection is s itself, the error is
    n, and SI-SNR is 10 log10(||s||^2 / ||n||^2) exactly."""
    clean = _tones()
    noise = _quadrature() * 10 ** (-snr_db / 20)
    assert torch.isclose(neg_si_snr(clean + noise, clean),
                         torch.tensor(-snr_db, dtype=torch.float64),
                         atol=1e-6)


@pytest.mark.parametrize("scale", [0.01, 0.5, 3.0, -2.0])
def test_si_snr_ignores_the_level_of_the_estimate(scale):
    clean = _tones()
    estimate = clean + 0.2 * _quadrature() + 0.05
    assert torch.isclose(neg_si_snr(scale * estimate, clean),
                         neg_si_snr(estimate, clean), atol=1e-9)
    # and of the reference
    assert torch.isclose(neg_si_snr(estimate, 7 * clean),
                         neg_si_snr(estimate, clean), atol=1e-9)


def test_si_snr_removes_the_mean_unless_told_not_to():
    clean = _tones()
    estimate = clean + 0.1 * _quadrature()
    shifted = estimate + 0.3
    assert torch.isclose(neg_si_snr(shifted, clean),
                         neg_si_snr(estimate, clean), atol=1e-9)
    # GTCRN's form: an offset is part of the error
    assert (neg_si_snr(shifted, clean, zero_mean=False)
            > neg_si_snr(estimate, clean, zero_mean=False) + 1)


def test_si_snr_is_the_mean_over_the_batch():
    clean = _tones()
    estimate = clean + 0.1 * _quadrature()
    each = torch.stack([neg_si_snr(estimate[i], clean[i]) for i in range(3)])
    assert torch.isclose(neg_si_snr(estimate, clean), each.mean())


def test_si_snr_of_silence_is_finite():
    silence = torch.zeros(2, 800)
    assert torch.isfinite(neg_si_snr(silence, silence))
    assert torch.isfinite(neg_si_snr(_tones(800, 2).float(), silence))


# ------------------------------------------------------------- gradients

def test_every_loss_has_a_finite_gradient_from_a_silent_start():
    """A model's first output can be silence; |X|^0.3 has an infinite slope
    there, and the eps under the square root is what keeps it finite."""
    stft = ComplexSTFT(256, 128)
    clean = _tones(2048, 2).float()
    weights = LossConfig(time_l1=1.0)
    for start in (torch.zeros(2, 2048), torch.zeros(2, 129, 17,
                                                     dtype=torch.complex64)):
        estimate = start.clone().requires_grad_(True)
        hybrid_loss(estimate, clean, stft, weights).backward()
        assert torch.isfinite(torch.view_as_real(estimate.grad)
                              if estimate.is_complex()
                              else estimate.grad).all()


# ---------------------------------------------------------------- hybrid

def test_the_hybrid_is_the_weighted_sum_of_its_terms():
    stft = ComplexSTFT(256, 128, window="sqrt_hann")
    clean = _tones(2048).float()
    estimate = clean + 0.05 * torch.randn(clean.shape,
                                          generator=torch.Generator().manual_seed(0))
    weights = LossConfig(complex_mse=2.0, magnitude_mse=3.0, time_l1=5.0,
                         neg_si_snr=0.5, compress=0.5)
    terms = loss_terms(estimate, clean, stft, weights)
    assert list(terms) == ["complex_mse", "magnitude_mse", "time_l1",
                           "neg_si_snr"]
    spec, ref_spec = stft.spectrum(estimate), stft.spectrum(clean)
    assert torch.isclose(terms["complex_mse"],
                         complex_mse(spec, ref_spec, compress=0.5))
    assert torch.isclose(terms["magnitude_mse"],
                         magnitude_mse(spec, ref_spec, compress=0.5))
    assert torch.isclose(terms["time_l1"], time_l1(estimate, clean))
    assert torch.isclose(terms["neg_si_snr"],
                         neg_si_snr(estimate, clean, zero_mean=False))
    expected = (2 * terms["complex_mse"] + 3 * terms["magnitude_mse"]
                + 5 * terms["time_l1"] + 0.5 * terms["neg_si_snr"])
    assert torch.isclose(hybrid_loss(estimate, clean, stft, weights),
                         expected)


@pytest.mark.parametrize("shape", [(3, 1, 129, 17), (3, 129, 16),
                                   (2, 129, 17), (3, 128, 17)])
def test_a_spectrum_of_the_wrong_shape_is_refused(shape):
    """An extra axis would broadcast into every estimate against every
    reference, and a frame short would be zero-padded by the inverse STFT;
    both would score something, and the wrong thing."""
    stft = ComplexSTFT(256, 128)
    clean = _tones(2048).float()
    assert stft.spectrum(clean).shape == (3, 129, 17)
    with pytest.raises(ValueError, match="spectrum estimate of shape"):
        loss_terms(torch.zeros(shape, dtype=torch.complex64), clean, stft,
                   LossConfig(time_l1=1.0))


def test_a_term_weighted_zero_is_left_out():
    stft = ComplexSTFT(256, 128)
    clean = _tones(2048).float()
    terms = loss_terms(clean * 0.9, clean, stft, LossConfig(
        complex_mse=0, magnitude_mse=0, time_l1=1.0, neg_si_snr=0))
    assert list(terms) == ["time_l1"]


def test_a_spectrum_estimate_is_scored_as_the_spectrum_it_is():
    """GTCRN's loss scores the network's spectrum itself, and the waveform
    the inverse STFT makes of it; a waveform estimate gives the same numbers
    through its own spectrum, since the transform comes back exactly."""
    stft = ComplexSTFT(256, 128, window="sqrt_hann")
    clean = _tones(2048).float()
    estimate = 0.8 * clean + 0.05 * _quadrature(2048).float()
    weights = LossConfig(time_l1=1.0)
    from_wave = loss_terms(estimate, clean, stft, weights)
    from_spec = loss_terms(stft.spectrum(estimate), clean, stft, weights)
    for name in from_wave:
        assert torch.isclose(from_wave[name], from_spec[name], rtol=1e-4,
                             atol=1e-6), name


def test_the_defaults_are_gtcrns_loss():
    """30 x (real MSE + imaginary MSE) + 70 x magnitude MSE at c = 0.3, plus
    -log10 of the SI-SNR ratio with the mean left in, worked out here the
    way GTCRN's loss.py writes it, on spectra from the torch.stft call its
    code makes. The estimate carries an offset, so a default that removed
    the mean would not match, and an end that is not silent, so zero
    padding would not either."""
    audio = AudioConfig(n_fft=512, hop_length=256, window="sqrt_hann",
                        pad_mode="reflect")
    stft = ComplexSTFT.from_config(audio)
    clean = _tones(4096).float() + 0.2 * _quadrature(4096).float()
    estimate = 0.9 * clean + 0.03 * _quadrature(4096).float() + 0.02
    window = torch.hann_window(512).pow(0.5)
    spec = torch.stft(estimate, 512, 256, 512, window, return_complex=True)
    ref = torch.stft(clean, 512, 256, 512, window, return_complex=True)

    def squeezed(x):
        magnitude = (x.real ** 2 + x.imag ** 2 + 1e-12).sqrt()
        return x.real / magnitude ** 0.7, x.imag / magnitude ** 0.7, \
            magnitude ** 0.3

    (er, ei, em), (rr, ri, rm) = squeezed(spec), squeezed(ref)
    mse = torch.nn.functional.mse_loss
    y_pred = torch.istft(spec, 512, 256, 512, window=window)
    y_true = torch.istft(ref, 512, 256, 512, window=window)
    target = ((y_true * y_pred).sum(-1, keepdim=True)
              / (y_true.square().sum(-1, keepdim=True) + 1e-8)) * y_true
    sisnr = -torch.log10(target.norm(dim=-1) ** 2
                         / ((y_pred - target).norm(dim=-1) ** 2 + 1e-8)
                         + 1e-8).mean()
    gtcrn = 30 * (mse(er, rr) + mse(ei, ri)) + 70 * mse(em, rm) + sisnr
    assert torch.isclose(hybrid_loss(estimate, clean, stft, LossConfig()),
                         gtcrn, rtol=1e-4)
    # scored on the network's spectrum, as GTCRN's loss is
    assert torch.isclose(hybrid_loss(spec, clean, stft, LossConfig()),
                         gtcrn, rtol=1e-4)
    centred = hybrid_loss(estimate, clean, stft, LossConfig(zero_mean=True))
    assert not torch.isclose(centred, gtcrn, rtol=1e-3)
    padded_with_zeros = ComplexSTFT(512, 256, window="sqrt_hann")
    assert not torch.isclose(
        hybrid_loss(estimate, clean, padded_with_zeros, LossConfig()), gtcrn,
        rtol=1e-3)


def test_zero_mean_reaches_the_si_snr_term():
    stft = ComplexSTFT(256, 128)
    clean = _tones(2048).float()
    estimate = clean + 0.1 * _quadrature(2048).float() + 0.3
    only_si_snr = dict(complex_mse=0, magnitude_mse=0, neg_si_snr=1.0)
    for zero_mean in (False, True):
        terms = loss_terms(estimate, clean, stft,
                           LossConfig(zero_mean=zero_mean, **only_si_snr))
        assert torch.isclose(terms["neg_si_snr"],
                             neg_si_snr(estimate, clean, zero_mean=zero_mean))
