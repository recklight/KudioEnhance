# -*- coding: utf-8 -*-
"""End-to-end: mix, train one epoch, enhance, score."""
from __future__ import annotations

import numpy as np
import pytest

from kudioenhance import pipeline
from kudioenhance.inference import Enhancer

# the stages that need no model run without PyTorch; everything that trains
# asks for the `torch` fixture, and skips cleanly when it is not installed


@pytest.fixture
def trained(config, torch):
    """A one-epoch run: enough to prove the wiring, not the model."""
    pipeline.synthesize(config, "exp1")
    pipeline.train(config, "exp1", epochs=1, verbose=0)
    return config, "exp1"


def test_synthesize_freezes_the_splits(config):
    pipeline.synthesize(config, "exp1")
    run = config.run_dir("exp1")

    for part in ("train", "val", "test"):
        assert (run / f"{part}.json").is_file()
    assert (run / "manifest.json").is_file()


def test_train_writes_everything_inference_needs(trained):
    config, name = trained
    assert config.model_path(name).is_file()
    assert config.model_path(name).suffix == ".pt"
    assert config.stats_path(name).is_file()
    assert (config.run_dir(name) / "config.yaml").is_file()


def test_train_keeps_the_curves(trained):
    from kudioenhance import load_history

    config, name = trained
    history = load_history(config.history_path(name))
    assert set(history) == {"loss", "mae", "val_loss", "val_mae", "lr"}
    assert history["lr"] == [config.train.learning_rate]


def test_training_the_same_config_twice_gives_the_same_model(config, torch):
    """The seed reaches the initial weights, not only the shuffle, so a run
    can be repeated from its config alone."""
    from kudioenhance.checkpoint import load_checkpoint

    config.model.dropout = 0.2
    pipeline.synthesize(config, "exp1")
    first = pipeline.train(config, "exp1", epochs=2, verbose=0)
    weights = load_checkpoint(config.model_path("exp1"))["state_dict"]

    assert pipeline.train(config, "exp1", epochs=2, verbose=0) == first
    again = load_checkpoint(config.model_path("exp1"))["state_dict"]
    assert all(torch.equal(weights[key], again[key]) for key in weights)


def test_train_refuses_a_batch_size_its_model_cannot_take_before_reading(
        config):
    """Before the splits are even looked for, let alone a pass over the
    audio to build the training set: a ``ddae`` at ``batch_size: 1`` is a
    config mistake, and it says so first. No torch needed to find out."""
    config.train.batch_size = 1
    with pytest.raises(ValueError, match="batch_size to 2 or more"):
        pipeline.train(config, "exp1", epochs=1, verbose=0)

    # a sequence model gets past the check, to the missing splits
    config.model.name = "conv_ae"
    with pytest.raises(FileNotFoundError, match="synthesize"):
        pipeline.train(config, "exp1", epochs=1, verbose=0)


def test_train_without_synthesize_says_so(config):
    with pytest.raises(FileNotFoundError, match="synthesize stage first"):
        pipeline.train(config, "never-mixed", epochs=1, verbose=0)


def test_enhancer_round_trips_through_the_run_directory(trained, tone):
    config, name = trained
    enhancer = Enhancer.load(config.run_dir(name))

    enhanced = enhancer.enhance(tone)
    assert enhanced.shape == tone.shape
    assert np.isfinite(enhanced).all()


def test_enhancer_rejects_the_wrong_sample_rate(trained, tone):
    config, name = trained
    enhancer = Enhancer.load(config.run_dir(name))
    with pytest.raises(ValueError, match="trained at"):
        enhancer.enhance(tone, sr=44100)


def test_enhance_file_writes_audio(trained, tmp_path, corpus):
    config, name = trained
    enhancer = Enhancer.load(config.run_dir(name))
    source = next(corpus["clean"].glob("*.wav"))

    out = tmp_path / "enhanced.wav"
    enhancer.enhance_file(source, out)
    assert out.is_file()


def test_evaluate_reports_before_and_after(trained):
    config, name = trained
    rows, summary = pipeline.evaluate(config, name)

    assert rows
    assert config.report_path(name).is_file()
    for metric in ("snr", "si_sdr", "seg_snr"):
        assert f"noisy_{metric}" in summary
        assert f"enhanced_{metric}" in summary
        assert summary[f"delta_{metric}"] == pytest.approx(
            summary[f"enhanced_{metric}"] - summary[f"noisy_{metric}"], abs=1e-6)


def test_report_csv_ends_with_a_mean_row(trained):
    import csv
    config, name = trained
    pipeline.evaluate(config, name)

    with open(config.report_path(name), encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[-1][0] == "MEAN"


@pytest.mark.parametrize("model_name, target", [
    ("crn", "spectrum"), ("crn", "irm"), ("unet", "spectrum"), ("unet", "irm"),
])
def test_the_2d_models_train_load_and_enhance(config, torch, tone,
                                              model_name, target):
    """At their default widths, and on an utterance whose length is neither
    the training window nor a multiple of anything the U-Net halves by."""
    from kudioenhance.models import default_units

    config.model.name = model_name
    config.model.target = target
    config.model.units = None
    pipeline.synthesize(config, "exp1")
    pipeline.train(config, "exp1", epochs=1, verbose=0)

    enhancer = Enhancer.load(config.run_dir("exp1"))
    assert enhancer.cfg.model.units == default_units(model_name)
    assert enhancer.mask is (target == "irm")
    enhanced = enhancer.enhance(tone)
    assert enhanced.shape == tone.shape
    assert np.isfinite(enhanced).all()
    if target == "irm":
        # a mask only ever keeps part of what is there
        assert np.max(np.abs(enhanced)) <= np.max(np.abs(tone)) * 1.05


def test_a_run_records_the_widths_it_was_built_with(config, torch,
                                                    monkeypatch):
    """``units: null`` means "the default", and a default can change in a
    later release. The run directory has to keep loading when it does, so
    it records the widths that were built — without writing them into the
    caller's config."""
    import dataclasses

    import yaml

    from kudioenhance.models import REGISTRY, default_units

    config.model.name = "unet"
    config.model.units = None
    built = default_units("unet")
    pipeline.synthesize(config, "exp1")
    pipeline.train(config, "exp1", epochs=1, verbose=0)

    assert config.model.units is None
    saved = yaml.safe_load(
        (config.run_dir("exp1") / "config.yaml").read_text(encoding="utf-8"))
    assert saved["model"]["units"] == built

    monkeypatch.setitem(REGISTRY, "unet", dataclasses.replace(
        REGISTRY["unet"], default_units=(4, 8)))
    enhancer = Enhancer.load(config.run_dir("exp1"))
    assert enhancer.cfg.model.units == built


@pytest.mark.parametrize("model_name, target, rooms", [
    ("ddae", "spectrum", False),
    ("ddae", "irm", True),
    ("blstm", "spectrum", True),
    ("conv_ae", "irm", False),
])
def test_a_streamed_run_trains_loads_and_enhances(config, torch, tone,
                                                  model_name, target, rooms):
    """Frame-wise and sequence models, both targets, and the reverberant
    target that `reference_for` picks, all read from disk every epoch. The
    run directory is the same kind an in-memory run leaves."""
    from kudioenhance import load_history

    config.model.name = model_name
    config.model.target = target
    if rooms:
        config.data.rt60 = [0.5]
        config.data.reverberant_target = True
    config.train.streaming = True
    config.train.shuffle_buffer = 64
    pipeline.synthesize(config, "exp1")
    pipeline.train(config, "exp1", epochs=1, verbose=0)

    history = load_history(config.history_path("exp1"))
    assert set(history) == {"loss", "mae", "val_loss", "val_mae", "lr"}
    assert all(np.isfinite(history[key]).all() for key in history)
    enhancer = Enhancer.load(config.run_dir("exp1"))
    assert enhancer.cfg.train.streaming is True
    enhanced = enhancer.enhance(tone)
    assert enhanced.shape == tone.shape
    assert np.isfinite(enhanced).all()


def test_a_streamed_run_repeats_exactly_from_its_seed(config, torch):
    """The files come in a fresh order every epoch and through a shuffle
    buffer, all of it from the training seed, so the run still repeats to
    the last bit — and another seed is another run."""
    from kudioenhance.checkpoint import load_checkpoint

    config.data.mode = "inc"
    config.model.dropout = 0.2
    config.train.streaming = True
    config.train.shuffle_buffer = 100
    pipeline.synthesize(config, "exp1")
    first = pipeline.train(config, "exp1", epochs=3, verbose=0)
    weights = load_checkpoint(config.model_path("exp1"))["state_dict"]

    assert pipeline.train(config, "exp1", epochs=3, verbose=0) == first
    again = load_checkpoint(config.model_path("exp1"))["state_dict"]
    assert all(torch.equal(weights[key], again[key]) for key in weights)

    config.train.seed = 4
    assert pipeline.train(config, "exp1", epochs=3, verbose=0) != first


def test_streaming_without_a_buffer_says_what_that_costs(config, torch,
                                                         caplog):
    import logging

    config.train.streaming = True
    pipeline.synthesize(config, "exp1")
    with caplog.at_level(logging.WARNING, logger="kudioenhance"):
        pipeline.train(config, "exp1", epochs=1, verbose=0)
    assert any("shuffle_buffer 0" in record.getMessage()
               for record in caplog.records)


@pytest.mark.parametrize("model_name", ["ddae", "conv_ae"])
def test_run_covers_both_data_layouts(config, torch, model_name):
    config.model.name = model_name
    config.model.units = [16, 16]
    summary = pipeline.run(config, f"exp-{model_name}", epochs=1, verbose=0)

    assert "enhanced_si_sdr" in summary
    assert np.isfinite(summary["enhanced_si_sdr"])
