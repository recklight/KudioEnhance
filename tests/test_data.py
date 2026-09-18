# -*- coding: utf-8 -*-
"""Mixing, manifests, splits and the two array layouts."""
from __future__ import annotations

import numpy as np
import pathlib

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


# ============================================================ reverberation

def test_rooms_are_off_by_default(config):
    assert config.data.rt60 == []
    assert config.data.reverberant is False


def test_an_impossible_room_is_refused_at_load():
    from kudio_enhance.config import DataConfig

    with pytest.raises(ValueError, match="rt60"):
        DataConfig(rt60=[0.5, 0.0])
    with pytest.raises(ValueError, match="rt60"):
        DataConfig(rt60=[-1.0])


def test_synthesizing_in_rooms_records_which_room(config):
    """The manifest is where the room lives: `syn` returns four-tuples and a
    fifth element that appeared only with `rt60=` would be a return shape that
    depends on a keyword."""
    config.data.rt60 = [0.4]
    config.data.drr_db = 3.0
    assert config.data.reverberant is True

    pairs = synthesize(config, "exp1")
    assert pairs
    assert all(p.rt60 == 0.4 for p in pairs)
    assert all("rt400ms" in p.noisy for p in pairs)
    assert all(p.exists() for p in pairs)


def test_the_room_is_a_fourth_axis(config):
    plain = len(synthesize(config, "exp1"))

    config.data.rt60 = [0.3, 0.6, 0.9]
    config.data.mode = "inc"
    with_rooms = len(synthesize(config, "exp2"))
    assert with_rooms > plain


def test_a_reverberant_manifest_round_trips(config, tmp_path):
    from kudio_enhance.data import load_manifest, save_manifest

    config.data.rt60 = [0.5]
    pairs = synthesize(config, "exp1")
    path = save_manifest(tmp_path / "m.json", pairs)
    assert load_manifest(path) == pairs


def test_a_manifest_written_before_rooms_still_loads(tmp_path):
    """`rt60` is optional and last, so the field is simply absent."""
    import json

    from kudio_enhance.data import load_manifest
    path = tmp_path / "old.json"
    path.write_text(json.dumps([
        {"noisy": "a.wav", "clean": "b.wav", "noise": "white", "snr_db": 0}
    ]), encoding="utf-8")

    pairs = load_manifest(path)
    assert len(pairs) == 1 and pairs[0].rt60 is None


def test_reverberant_features_are_still_the_right_shape(config):
    """The room changes what the model hears, not the geometry it hears it
    in."""
    config.data.rt60 = [0.5]
    pairs = synthesize(config, "exp1")
    x, y, std = build_arrays(pairs, config, sequence=False, fit=True)

    bins = config.audio.n_bins
    assert x.shape[1] == bins * (2 * config.model.context + 1)
    assert y.shape[1] == bins
    assert len(x) == len(y)


# =================================================== the reverberant target

def test_the_reverberant_target_is_off_by_default(config):
    assert config.data.reverberant_target is False
    assert config.data.channel is None


def test_a_target_without_a_room_is_refused_at_load():
    from kudio_enhance.config import DataConfig

    with pytest.raises(ValueError, match="already is"):
        DataConfig(reverberant_target=True)
    DataConfig(reverberant_target=True, rt60=[0.5])       # fine with a room


def test_an_unknown_link_is_refused_at_load_rather_than_an_hour_in():
    from kudio_enhance.config import DataConfig

    with pytest.raises(Exception, match="unknown channel"):
        DataConfig(channel="opus")
    DataConfig(channel="telephone")


def test_training_against_the_room_uses_the_file_the_room_produced(config):
    from kudio_enhance.data import reference_for

    config.data.rt60 = [0.5]
    config.data.reverberant_target = True
    pairs = synthesize(config, "exp1")

    assert pairs and all(p.target for p in pairs)
    for pair in pairs:
        assert pathlib.Path(pair.target).is_file()
        assert reference_for(pair, config) == pair.target
        assert pathlib.Path(pair.target).name == pathlib.Path(pair.noisy).name

    config.data.reverberant_target = False
    assert reference_for(pairs[0], config) == pairs[0].clean


def test_an_older_manifest_without_targets_still_trains(config):
    """A run made before targets existed has no `target` key, and asking for
    one should fall back rather than fail on the missing file."""
    from kudio_enhance.data import reference_for

    config.data.rt60 = [0.5]
    pairs = synthesize(config, "exp1")
    config.data.reverberant_target = True
    assert all(p.target is None for p in pairs)
    assert reference_for(pairs[0], config) == pairs[0].clean


def test_the_mask_is_only_correct_against_what_was_actually_mixed(config):
    """The real reason this matters. The mask is built from
    ``noisy - reference``; in a reverberant dataset the dry clean file is not
    what was mixed, so that subtraction hands the room's tail to the noise
    and the model is asked to remove the reverberation as if it were noise.
    Against the reverberant target the subtraction is the noise again.
    """
    import kudio
    import numpy as np

    config.data.rt60 = [0.7]
    config.data.reverberant_target = True
    pair = synthesize(config, "exp1")[0]

    noisy, sr = kudio.file_load(pair.noisy, sr=config.audio.sr)
    dry, _ = kudio.file_load(pair.clean, sr=config.audio.sr)
    wet, _ = kudio.file_load(pair.target, sr=config.audio.sr)

    def residual(reference):
        n = min(len(noisy), len(reference))
        return noisy[:n] - reference[:n]

    # the residual against the wet target is the noise that was mixed in;
    # against the dry file it is the noise *plus* the room, so it is louder
    assert np.mean(residual(wet) ** 2) < np.mean(residual(dry) ** 2)


def test_a_link_reaches_the_mixture_and_not_the_target(config):
    import kudio

    config.data.rt60 = [0.4]
    config.data.reverberant_target = True
    config.data.channel = "telephone"
    pair = synthesize(config, "exp1")[0]

    assert pair.channel == "telephone"
    noisy, sr = kudio.file_load(pair.noisy, sr=config.audio.sr)
    assert kudio.audio_report(noisy, sr).band_limited


def test_features_are_the_same_shape_against_either_target(config):
    config.data.rt60 = [0.5]
    config.data.reverberant_target = True
    pairs = synthesize(config, "exp1")

    wet_x, wet_y, _ = build_arrays(pairs, config, sequence=False, fit=True)
    config.data.reverberant_target = False
    dry_x, dry_y, _ = build_arrays(pairs, config, sequence=False, fit=True)

    assert wet_x.shape == dry_x.shape and wet_y.shape == dry_y.shape
