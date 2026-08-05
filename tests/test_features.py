# -*- coding: utf-8 -*-
"""Feature extraction, context stacking, windowing and normalisation."""
from __future__ import annotations

import numpy as np
import pytest

from kudio_enhance.features import (
    Standardizer,
    frame_windows,
    spectrogram,
    stack_context,
    to_waveform,
)


def test_spectrogram_shape_follows_the_stft_geometry(config, tone):
    spec = spectrogram(tone, config.audio)
    assert spec.ndim == 2
    assert spec.shape[1] == config.audio.n_bins


def test_reconstruction_keeps_the_input_length(config, tone):
    spec = spectrogram(tone, config.audio)
    restored = to_waveform(tone, spec, config.audio)
    assert len(restored) == len(tone)
    # the phase is the original's, so an unmodified spectrogram round-trips
    assert np.corrcoef(restored, tone)[0, 1] > 0.99


def test_stack_context_widens_rows_but_keeps_their_count():
    frames = np.arange(20, dtype=np.float32).reshape(5, 4)

    assert stack_context(frames, 0).shape == (5, 4)
    stacked = stack_context(frames, 2)
    assert stacked.shape == (5, 4 * 5)
    # the middle block is the frame itself
    assert np.allclose(stacked[:, 8:12], frames)


def test_stack_context_edge_pads_rather_than_truncating():
    frames = np.arange(12, dtype=np.float32).reshape(3, 4)
    stacked = stack_context(frames, 1)
    # first row's "previous frame" repeats row 0
    assert np.allclose(stacked[0, 0:4], frames[0])
    # last row's "next frame" repeats the last row
    assert np.allclose(stacked[-1, 8:12], frames[-1])


@pytest.mark.parametrize("n, n_frames, expected", [
    (64, 16, 4),
    (60, 16, 4),     # tail padded, not dropped
    (5, 16, 1),      # shorter than one window still yields one
])
def test_frame_windows_pads_the_tail(n, n_frames, expected):
    frames = np.random.default_rng(0).normal(size=(n, 9)).astype(np.float32)
    windows = frame_windows(frames, n_frames)
    assert windows.shape == (expected, n_frames, 9)


def test_standardizer_round_trips():
    rng = np.random.default_rng(1)
    frames = rng.normal(3.0, 2.0, size=(200, 9)).astype(np.float32)

    std = Standardizer().fit(frames)
    normalised = std.transform(frames)
    assert np.allclose(normalised.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(normalised.std(axis=0), 1, atol=1e-3)
    assert np.allclose(std.inverse(normalised), frames, atol=1e-3)


def test_standardizer_survives_a_save_load(tmp_path):
    frames = np.random.default_rng(2).normal(size=(50, 9)).astype(np.float32)
    std = Standardizer().fit(frames)
    reloaded = Standardizer.load(std.save(tmp_path / "stats.npz"))

    assert np.allclose(reloaded.mean, std.mean)
    assert np.allclose(reloaded.transform(frames), std.transform(frames))


def test_unfitted_standardizer_refuses_to_transform():
    """kudio raises from its own hierarchy, not a bare RuntimeError."""
    from kudio.exceptions import FeatureError

    with pytest.raises(FeatureError, match="not been fitted"):
        Standardizer().transform(np.zeros((3, 4), dtype=np.float32))
