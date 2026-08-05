# -*- coding: utf-8 -*-
"""Mixing, manifests, splits and the two array layouts."""
from __future__ import annotations

import numpy as np
import pytest

from kudio_enhance.data import (
    Pair,
    build_arrays,
    load_manifest,
    save_manifest,
    split_pairs,
    synthesize,
)


def test_synthesize_writes_pairs_that_exist(config):
    pairs = synthesize(config, "exp1")

    assert pairs
    assert all(p.exists() for p in pairs)
    assert {p.noise for p in pairs} <= {"white", "hum"}
    assert {p.snr_db for p in pairs} <= {0, 5}


def test_mixtures_are_written_at_the_configured_rate(config):
    """The dataset on disk must match the rate the model is trained at."""
    import soundfile as sf

    pairs = synthesize(config, "exp1")
    rates = {sf.info(p.noisy).samplerate for p in pairs}
    assert rates == {config.audio.sr}


def test_manifest_round_trips(tmp_path):
    pairs = [Pair("a.wav", "b.wav", "white", 5),
             Pair("c.wav", "d.wav", "hum", -5)]
    reloaded = load_manifest(save_manifest(tmp_path / "m.json", pairs))
    assert reloaded == pairs


def test_split_is_disjoint_and_covers_everything():
    pairs = [Pair(f"{i}.wav", f"c{i}.wav", "white", 0) for i in range(20)]
    train, val, test = split_pairs(pairs, 0.2, 0.1, seed=0)

    assert (len(train), len(val), len(test)) == (14, 4, 2)
    assert set(train) | set(val) | set(test) == set(pairs)
    assert not (set(train) & set(val))
    assert not (set(val) & set(test))


def test_split_is_reproducible_for_a_given_seed():
    pairs = [Pair(f"{i}.wav", f"c{i}.wav", "white", 0) for i in range(20)]
    assert split_pairs(pairs, 0.2, 0.1, seed=7) == split_pairs(pairs, 0.2, 0.1, seed=7)
    assert split_pairs(pairs, 0.2, 0.1, seed=7) != split_pairs(pairs, 0.2, 0.1, seed=8)


def test_tiny_sets_keep_everything_in_training():
    pairs = [Pair("a.wav", "b.wav", "white", 0)]
    train, val, test = split_pairs(pairs, 0.5, 0.5, seed=0)
    assert (len(train), len(val), len(test)) == (1, 0, 0)


def test_frame_wise_arrays_are_context_stacked(config):
    pairs = synthesize(config, "exp1")
    x, y, std = build_arrays(pairs, config, sequence=False, fit=True)

    bins = config.audio.n_bins
    assert x.shape[1] == bins * (2 * config.model.context + 1)
    assert y.shape[1] == bins
    assert len(x) == len(y)
    assert std.fitted


def test_sequence_arrays_are_windowed(config):
    pairs = synthesize(config, "exp1")
    x, y, _ = build_arrays(pairs, config, sequence=True, fit=True)

    assert x.shape[1:] == (config.model.n_frames, config.audio.n_bins)
    assert x.shape == y.shape


def test_a_fitted_standardizer_is_reused_not_refitted(config):
    pairs = synthesize(config, "exp1")
    _, _, std = build_arrays(pairs, config, sequence=False, fit=True)
    mean_before = std.mean.copy()

    build_arrays(pairs[:1], config, sequence=False, standardizer=std)
    assert np.allclose(std.mean, mean_before)


def test_build_arrays_needs_pairs(config):
    with pytest.raises(ValueError, match="no pairs"):
        build_arrays([], config, sequence=False)
