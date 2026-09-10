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
    """Percentages on two examples would leave nothing to fit; the splits are
    given up instead. An empty validation set is visible, a one-example
    training set looks like it worked."""
    pairs = [Pair("a.wav", "b.wav", "white", 0),
             Pair("c.wav", "d.wav", "white", 5)]
    train, val, test = split_pairs(pairs, 0.4, 0.4, seed=0)
    assert (len(train), len(val), len(test)) == (2, 0, 0)


def test_splits_that_leave_no_training_data_are_rejected():
    """Since 3.5.0 these helpers come from kudio, which refuses the request
    rather than silently handing everything to training: asking for a 50/50
    val/test split is a mistake on any real dataset."""
    pairs = [Pair(f"{i}.wav", f"c{i}.wav", "white", 0) for i in range(10)]
    with pytest.raises(ValueError, match="leave something to train on"):
        split_pairs(pairs, 0.5, 0.5, seed=0)


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


def test_history_helpers_round_trip(tmp_path):
    """The curves used to be returned and dropped, which made "did it converge
    or stop early?" unanswerable once the terminal closed."""
    from kudio_enhance import load_history, save_history, summarise_history

    curves = {"loss": [1.0, 0.6, 0.4, 0.45], "val_loss": [1.1, 0.7, 0.5, 0.55]}
    path = save_history(tmp_path / "history.json", curves)
    assert load_history(path) == curves

    summary = summarise_history(curves)
    assert summary["epochs"] == 4
    assert summary["monitor"] == "val_loss"
    assert summary["best"] == pytest.approx(0.5)
    assert summary["best_epoch"] == 3
    assert summary["improved"] is True


def test_a_run_that_learned_nothing_says_so():
    """The best epoch being the first one is what that looks like."""
    from kudio_enhance import summarise_history

    summary = summarise_history({"loss": [0.5, 0.6, 0.7]})
    assert summary["monitor"] == "loss"          # falls back when val is absent
    assert summary["best_epoch"] == 1
    assert summary["improved"] is False


def test_summarising_nothing_does_not_crash():
    from kudio_enhance import summarise_history
    assert summarise_history({})["best"] is None


def test_save_history_accepts_a_keras_history_object(tmp_path):
    from kudio_enhance import load_history, save_history

    class FakeHistory:
        history = {"loss": [np.float32(0.5), np.float32(0.25)]}

    path = save_history(tmp_path / "h.json", FakeHistory())
    assert load_history(path) == {"loss": [0.5, 0.25]}
