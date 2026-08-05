# -*- coding: utf-8 -*-
"""End-to-end: mix, train one epoch, enhance, score."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="pipeline tests need TensorFlow")

from kudio_enhance import pipeline  # noqa: E402
from kudio_enhance.inference import Enhancer  # noqa: E402


@pytest.fixture
def trained(config):
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
    assert config.stats_path(name).is_file()
    assert (config.run_dir(name) / "config.yaml").is_file()


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


@pytest.mark.parametrize("model_name", ["ddae", "conv_ae"])
def test_run_covers_both_data_layouts(config, model_name):
    config.model.name = model_name
    config.model.units = [16, 16]
    summary = pipeline.run(config, f"exp-{model_name}", epochs=1, verbose=0)

    assert "enhanced_si_sdr" in summary
    assert np.isfinite(summary["enhanced_si_sdr"])
