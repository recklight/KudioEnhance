# -*- coding: utf-8 -*-
"""Chunked inference on the waveform path: a long file is enhanced a fixed
length at a time, and away from the seams between chunks the result is the
whole file's, its two ends included."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs PyTorch",
                            exc_type=ImportError)

from waveform_stand_in import (  # noqa: E402
    NAME,
    SR,
    NoiseGate,
    register,
    tones_in_noise,
)

from kudioenhance.config import Config  # noqa: E402
from kudioenhance.inference import Enhancer  # noqa: E402

N_FFT, HOP = 64, 32
#: 8000 and 2048 samples, 250 and 64 hops
CHUNK, OVERLAP = 1.0, 0.256
#: How far an end of a chunk reaches into the gate's output: half a window
#: into the frames that see it, a frame either side through the gate's
#: three-frame average, and half a window back out.
REACH = N_FFT + HOP
#: How far chunked and whole-file output may differ further than that from
#: a seam, as a fraction of the output's peak. There a chunk's frames are
#: the whole file's, the same samples at the same places, and the two
#: chunks a cross-fade mixes agree: the difference is float32 rounding, at
#: most 2e-8 over five lengths and four seeds of gate.
TOLERANCE = 1e-5


@pytest.fixture(autouse=True)
def stand_in(monkeypatch):
    register(monkeypatch)


def _config(pad_mode="constant", window="hann") -> Config:
    return Config.from_dict({
        "audio": {"sr": SR, "n_fft": N_FFT, "hop_length": HOP,
                  "window": window, "pad_mode": pad_mode},
        "model": {"name": NAME, "units": [1]},
    })


def _gate(seed=3) -> NoiseGate:
    """A gate whose bins all differ, so no two frames are treated alike."""
    generator = torch.Generator().manual_seed(seed)
    gate = NoiseGate(N_FFT // 2 + 1)
    with torch.no_grad():
        gate.slope.copy_(0.5 + torch.rand(gate.slope.shape,
                                          generator=generator))
        gate.threshold.copy_(-4 + 2 * torch.rand(gate.threshold.shape,
                                                 generator=generator))
    return gate


def _enhancers(cfg, model, **chunking):
    return (Enhancer(model, None, cfg, chunk_seconds=None),
            Enhancer(model, None, cfg, **chunking))


def _near_a_seam(n, chunk, overlap, margin) -> np.ndarray:
    """True within *margin* samples of where a chunk after the first starts,
    or one before the last ends. The file's own ends are not seams: the
    first chunk starts and the last one stops where the file does."""
    near = np.zeros(n, dtype=bool)
    start = 0
    while start + chunk < n:
        following = start + chunk - overlap
        for edge in (following, start + chunk):
            near[max(edge - margin, 0):edge + margin] = True
        start = following
    return near


@pytest.mark.parametrize("seconds", [3.7, 5.13])
@pytest.mark.parametrize("pad_mode, window", [("constant", "hann"),
                                              ("reflect", "sqrt_hann")])
def test_chunks_agree_with_the_whole_file_away_from_the_seams(pad_mode,
                                                              window,
                                                              seconds):
    cfg = _config(pad_mode, window)
    _, noisy = tones_in_noise(seconds, seed=8)
    whole, chunked = _enhancers(cfg, _gate(), chunk_seconds=CHUNK,
                                overlap_seconds=OVERLAP)
    reference, result = whole.enhance(noisy), chunked.enhance(noisy)

    assert result.shape == noisy.shape and result.dtype == np.float32
    near = _near_a_seam(len(noisy), chunked.chunk, chunked.overlap, REACH)
    peak = np.max(np.abs(reference))
    away = np.max(np.abs(result - reference)[~near]) / peak
    assert away < TOLERANCE
    # at a seam the frames that see a chunk's padding are faded down to a
    # weight below 0.006; the difference there was at most 8e-6 of the peak
    # over the same lengths and seeds
    assert np.max(np.abs(result - reference)) / peak < 1e-4
    assert np.mean(near) < 0.25


def test_the_model_never_sees_more_than_a_chunk():
    cfg = _config()
    gate = _gate()
    shapes = []
    gate.register_forward_hook(
        lambda module, inputs, output: shapes.append(tuple(inputs[0].shape)))
    _, chunked = _enhancers(cfg, gate, chunk_seconds=CHUNK,
                            overlap_seconds=OVERLAP)
    _, noisy = tones_in_noise(3.7, seed=8)
    chunked.enhance(noisy)

    frames = 1 + chunked.chunk // HOP
    # 3.7 s in chunks of 1 s that step 0.744 s: the fifth reaches the end,
    # and holds the 29,600 - 4 x 5,952 = 5,792 samples left
    bins = N_FFT // 2 + 1
    assert shapes == [(1, bins, frames)] * 4 + [(1, bins, 1 + 5792 // HOP)]


@pytest.mark.parametrize("pad_mode, window", [("constant", "hann"),
                                              ("reflect", "sqrt_hann")])
def test_a_long_file_ends_as_it_does_when_enhanced_whole(pad_mode, window):
    """The last chunk stops where the file does, so its last frames are
    padded the way the whole file's are, mirrored or with zeros, and not
    with the silence a full-length chunk would add."""
    cfg = _config(pad_mode, window)
    whole, chunked = _enhancers(cfg, _gate(), chunk_seconds=CHUNK,
                                overlap_seconds=OVERLAP)
    for seconds in (3.7, 5.13):
        _, noisy = tones_in_noise(seconds, seed=9)
        reference, result = whole.enhance(noisy), chunked.enhance(noisy)
        peak = np.max(np.abs(reference))
        tail = slice(len(noisy) - chunked.overlap, None)
        assert np.max(np.abs(result[tail] - reference[tail])) \
            < TOLERANCE * peak


def test_a_file_that_fits_in_one_chunk_is_enhanced_whole():
    cfg = _config()
    whole, chunked = _enhancers(cfg, _gate(), chunk_seconds=CHUNK,
                                overlap_seconds=OVERLAP)
    for seconds in (CHUNK, 0.4, 0.002):
        _, noisy = tones_in_noise(seconds, seed=2)
        assert np.array_equal(chunked.enhance(noisy), whole.enhance(noisy))


def test_a_file_one_sample_past_a_chunk_still_comes_back_whole_length():
    cfg = _config("reflect", "sqrt_hann")
    _, chunked = _enhancers(cfg, _gate(), chunk_seconds=CHUNK,
                            overlap_seconds=OVERLAP)
    whole = Enhancer(_gate(), None, cfg, chunk_seconds=None)
    # cut to length: int(1.000125 * 8000) would round back down to a chunk
    noisy = tones_in_noise(2 * CHUNK, seed=4)[1][:chunked.chunk + 1]
    result, reference = chunked.enhance(noisy), whole.enhance(noisy)
    assert len(result) == len(noisy) == chunked.chunk + 1
    near = _near_a_seam(len(noisy), chunked.chunk, chunked.overlap, REACH)
    assert np.max(np.abs(result - reference)[~near]) \
        < TOLERANCE * np.max(np.abs(reference))
    assert len(chunked.enhance(np.zeros(0, dtype=np.float32))) == 0


def test_without_overlap_a_last_chunk_of_one_sample_still_comes_back():
    """The one case the last chunk can be too short for reflect padding:
    it is padded with zeros, as a file that short would be."""
    cfg = _config("reflect", "sqrt_hann")
    _, chunked = _enhancers(cfg, _gate(), chunk_seconds=CHUNK,
                            overlap_seconds=0)
    noisy = tones_in_noise(2 * CHUNK, seed=4)[1][:chunked.chunk + 1]
    result = chunked.enhance(noisy)
    assert len(result) == len(noisy) == chunked.chunk + 1
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("pad_mode, window", [("constant", "hann"),
                                              ("reflect", "sqrt_hann")])
def test_a_model_that_changes_nothing_gives_the_input_back(pad_mode, window):
    """Through every chunk, its padding and each cross-fade: the STFT and
    its inverse undo each other, ends included, so nothing is left to show
    where a chunk began. A model without weights runs on the CPU."""
    _, chunked = _enhancers(_config(pad_mode, window), torch.nn.Identity(),
                            chunk_seconds=CHUNK, overlap_seconds=OVERLAP)
    _, noisy = tones_in_noise(3.7, seed=5)
    assert np.max(np.abs(chunked.enhance(noisy) - noisy)) < 1e-5


def test_without_overlap_the_chunks_are_butted_together():
    cfg = _config()
    whole, chunked = _enhancers(cfg, _gate(), chunk_seconds=CHUNK,
                                overlap_seconds=0)
    _, noisy = tones_in_noise(2.5, seed=6)
    reference, result = whole.enhance(noisy), chunked.enhance(noisy)

    near = _near_a_seam(len(noisy), chunked.chunk, 0, REACH)
    assert np.max(np.abs(result - reference)[~near]) \
        < TOLERANCE * np.max(np.abs(reference))


def test_chunks_and_overlaps_are_whole_hops():
    enhancer = Enhancer(_gate(), None, _config(), chunk_seconds=1.003,
                        overlap_seconds=0.2509)
    assert enhancer.chunk % HOP == 0 and enhancer.overlap % HOP == 0
    assert (enhancer.chunk, enhancer.overlap) == (8032, 2016)


@pytest.mark.parametrize("chunk, overlap, message", [
    (1.0, 0.6, "half a chunk"),
    (0.004, 0.0, "shorter than one frame"),
])
def test_impossible_chunking_is_refused(chunk, overlap, message):
    with pytest.raises(ValueError, match=message):
        Enhancer(_gate(), None, _config(), chunk_seconds=chunk,
                 overlap_seconds=overlap)


def test_the_log_power_path_still_needs_its_statistics():
    cfg = Config.from_dict({"model": {"name": "ddae", "units": [8]}})
    with pytest.raises(ValueError, match="statistics"):
        Enhancer(torch.nn.Linear(1, 1), None, cfg)
