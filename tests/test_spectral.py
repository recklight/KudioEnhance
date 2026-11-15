# -*- coding: utf-8 -*-
"""The complex spectrum in torch: there and back to float precision, framed
the way kudio frames the log-power path, and compression that inverts."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs PyTorch",
                            exc_type=ImportError)

from kudioenhance.config import AudioConfig  # noqa: E402
from kudioenhance.spectral import (  # noqa: E402
    WINDOWS,
    ComplexSTFT,
    compress_magnitude,
    expand_magnitude,
    make_window,
)

SR = 16000


def _speechy(n_samples: int = 4000, seed: int = 0) -> torch.Tensor:
    """Harmonics under an envelope, with a little noise, so no bin is zero."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / SR
    wave = sum(np.sin(2 * np.pi * 180 * k * t) / k for k in range(1, 8))
    wave = wave * np.hanning(n_samples) + 0.01 * rng.standard_normal(n_samples)
    return torch.from_numpy((0.3 * wave).astype(np.float32))


# ------------------------------------------------------------- round trips

@pytest.mark.parametrize("window", sorted(WINDOWS))
@pytest.mark.parametrize("n_fft, hop, win", [(512, 256, None), (512, 128, 400),
                                             (320, 160, None)])
def test_a_waveform_comes_back_from_its_spectrum(window, n_fft, hop, win):
    """Every window and a window shorter than the frame: the inverse divides
    by the overlap-added squared window, so any of them comes back exact."""
    stft = ComplexSTFT(n_fft, hop, win, window=window)
    wave = _speechy(4001)                     # not a multiple of any hop
    back = stft.waveform(stft.spectrum(wave), length=len(wave))
    assert back.shape == wave.shape
    assert torch.allclose(back, wave, atol=1e-6)


def test_a_batch_keeps_its_leading_axes():
    stft = ComplexSTFT(256, 128)
    batch = torch.stack([_speechy(2000, seed) for seed in range(6)]
                        ).reshape(2, 3, 2000)
    spec = stft.spectrum(batch)
    assert spec.shape == (2, 3, 129, stft.n_frames(2000))
    assert spec.dtype == torch.complex64
    assert torch.allclose(stft.waveform(spec, 2000), batch, atol=1e-6)
    # each example is transformed on its own
    assert torch.equal(spec[1, 2], stft.spectrum(batch[1, 2]))


def test_reflect_padding_comes_back_and_moves_only_the_edge_frames():
    """Mirrored padding changes what the end frames see and nothing else;
    the inverse trims the padding either way."""
    wave = _speechy(4096)
    zeros = ComplexSTFT(512, 256, window="sqrt_hann")
    mirror = ComplexSTFT(512, 256, window="sqrt_hann", pad_mode="reflect")
    a, b = zeros.spectrum(wave), mirror.spectrum(wave)
    assert torch.allclose(mirror.waveform(b, 4096), wave, atol=1e-6)
    # frame t spans t * 256 - 256 to t * 256 + 256: only the first and the
    # last reach past the signal
    assert torch.allclose(a[:, 1:-1], b[:, 1:-1])
    assert not torch.allclose(a[:, 0], b[:, 0])
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_reflect_is_torchs_own_default():
    """GTCRN calls torch.stft with its defaults; pad_mode='reflect' is that."""
    wave = _speechy(4096)
    window = torch.hann_window(512).pow(0.5)
    theirs = torch.stft(wave, 512, 256, 512, window, return_complex=True)
    ours = ComplexSTFT(512, 256, 512, "sqrt_hann",
                       pad_mode="reflect").spectrum(wave)
    assert torch.allclose(ours, theirs)


@pytest.mark.parametrize("power", [0.3, 0.5, 1.0])
def test_compression_comes_back_too(power):
    stft = ComplexSTFT(512, 256, window="sqrt_hann", compress=power)
    wave = _speechy()
    back = stft.inverse(stft(wave), length=len(wave))
    assert torch.allclose(back, wave, atol=1e-5)


def test_float64_stays_float64():
    stft = ComplexSTFT(256, 128)
    wave = _speechy(1000).double()
    spec = stft.spectrum(wave)
    assert spec.dtype == torch.complex128
    assert torch.allclose(stft.waveform(spec, 1000), wave, atol=1e-12)


# ------------------------------------------------------------- compression

def test_compression_raises_the_magnitude_and_keeps_the_phase():
    spec = torch.tensor([3 + 4j, -0.6j, 0.05 + 0j], dtype=torch.complex128)
    squeezed = compress_magnitude(spec, 0.3)
    assert torch.allclose(squeezed.abs(), spec.abs() ** 0.3, rtol=1e-9)
    assert torch.allclose(torch.angle(squeezed), torch.angle(spec))
    assert torch.allclose(expand_magnitude(squeezed, 0.3), spec, rtol=1e-9)


def test_no_compression_is_the_identity():
    spec = torch.tensor([3 + 4j, 0j])
    assert compress_magnitude(spec, None) is spec
    assert compress_magnitude(spec, 1) is spec
    assert expand_magnitude(spec, None) is spec


def test_a_silent_bin_stays_silent_with_a_finite_gradient():
    """``|X|^0.3`` has an infinite slope at 0; with the eps under the square
    root, a model whose output starts at silence can still learn."""
    spec = torch.zeros(4, dtype=torch.complex64, requires_grad=True)
    squeezed = compress_magnitude(spec, 0.3)
    assert torch.equal(squeezed, torch.zeros_like(squeezed))
    (squeezed.real.sum() + squeezed.imag.sum()).backward()
    assert torch.isfinite(torch.view_as_real(spec.grad)).all()


# --------------------------------------------------- framing against kudio

@pytest.mark.parametrize("window, win", [("hamming", None), ("hann", 400)])
def test_the_frames_are_the_log_power_paths_frames(window, win):
    """The spectrum is kudio's complex spectrum to float32 precision, and
    ``log10(|X|^2)`` of it kudio's spectrogram, frame for frame: the two paths
    frame the audio alike, so they can be compared or mixed."""
    pytest.importorskip("librosa", reason="kudio's spectrogram needs librosa")
    audio = AudioConfig(sr=SR, n_fft=512, hop_length=256, win_length=win,
                        window=window)
    wave = _speechy(8000)
    ours = ComplexSTFT.from_config(audio).spectrum(wave)
    complex_theirs = audio.stft.analyse(wave.numpy()).T
    assert ours.shape == complex_theirs.shape
    peak = np.abs(complex_theirs).max()
    np.testing.assert_allclose(ours.numpy(), complex_theirs, rtol=0,
                               atol=1e-6 * peak)
    # the quietest bins here are about 100 dB under the peak, where the same
    # absolute rounding moves the logarithm by a few parts in 10^4
    log_power = torch.log10(ours.abs().square()).T.numpy()
    np.testing.assert_allclose(log_power, audio.stft.forward(wave.numpy()),
                               atol=2e-3)


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman", "bartlett",
                                  "boxcar"])
def test_the_windows_are_scipys_periodic_ones(name):
    signal = pytest.importorskip("scipy.signal")
    np.testing.assert_allclose(make_window(name, 400).numpy(),
                               signal.get_window(name, 400, fftbins=True),
                               atol=1e-6)


def test_the_square_root_hann_window_overlap_adds_to_one_at_half_hops():
    window = make_window("sqrt_hann", 512)
    squared = window.square()
    np.testing.assert_allclose((squared[:256] + squared[256:]).numpy(), 1.0,
                               atol=1e-6)


def test_the_config_decides_the_geometry_and_the_compression():
    audio = AudioConfig(sr=SR, n_fft=512, hop_length=128, win_length=400,
                        window="sqrt_hann", compress=0.3)
    stft = ComplexSTFT.from_config(audio)
    assert (stft.n_fft, stft.hop_length, stft.win_length, stft.compress) == (
        512, 128, 400, 0.3)
    assert stft.pad_mode == "constant"
    audio.pad_mode = "reflect"
    assert ComplexSTFT.from_config(audio).pad_mode == "reflect"
    assert stft.window_name == "sqrt_hann"
    assert stft.n_bins == audio.n_bins
    assert stft.n_frames(4000) == audio.stft.n_frames(4000)


def test_the_window_is_not_a_weight():
    """A buffer, so it follows ``.to(device)``, and kept out of the
    state_dict, so a model holding a front end checkpoints only weights."""
    stft = ComplexSTFT(256, 128)
    assert "window" in dict(stft.named_buffers())
    assert stft.state_dict() == {}


@pytest.mark.parametrize("kwargs, match", [
    ({"window": "kaiser"}, "unknown window"),
    ({"win_length": 600}, "cannot exceed"),
    ({"hop_length": 600}, "hop_length"),
    ({"compress": 1.5}, "compress"),
    ({"compress": 0}, "compress"),
    ({"pad_mode": "replicate"}, "pad_mode"),
    # a periodic Hann window is zero at its first sample, so frames a whole
    # window apart leave a sample covered by nothing
    ({"hop_length": 512, "window": "hann"}, "does not overlap-add"),
    ({"hop_length": 512, "window": "sqrt_hann"}, "does not overlap-add"),
])
def test_impossible_settings_are_refused(kwargs, match):
    """Refused when built, rather than at the first inverse transform."""
    settings = {"n_fft": 512, "hop_length": 256, **kwargs}
    with pytest.raises(ValueError, match=match):
        ComplexSTFT(**settings)


def test_a_flat_window_at_a_hop_of_the_whole_window_is_allowed():
    """Frames side by side cover every sample once, which is enough."""
    stft = ComplexSTFT(256, 256, window="boxcar")
    wave = _speechy(2048)
    assert torch.allclose(stft.waveform(stft.spectrum(wave), 2048), wave,
                          atol=1e-6)
