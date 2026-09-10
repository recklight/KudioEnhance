# -*- coding: utf-8 -*-
"""Typed config: defaults, round trips, and failing loudly on typos."""
from __future__ import annotations

from pathlib import Path

import pytest

from kudio_enhance import AudioConfig, Config


def test_defaults_are_usable():
    cfg = Config()
    assert cfg.audio.sr == 16000
    assert cfg.audio.n_bins == 257
    assert cfg.model.name == "ddae"
    assert cfg.data.snr_db == [-5, 0, 5]


def test_yaml_round_trip(tmp_path, config):
    path = config.to_yaml(tmp_path / "config.yaml")
    reloaded = Config.from_yaml(path)
    assert reloaded.to_dict() == config.to_dict()


def test_unknown_key_names_the_offender():
    with pytest.raises(ValueError, match="hop_lenght"):
        Config.from_dict({"audio": {"hop_lenght": 128}})


def test_unknown_section_is_rejected():
    with pytest.raises(ValueError, match="trainning"):
        Config.from_dict({"trainning": {"epochs": 3}})


def test_section_must_be_a_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        Config.from_dict({"audio": [1, 2, 3]})


@pytest.mark.parametrize("section, values, match", [
    ("audio", {"n_fft": 256, "win_length": 512}, "win_length"),
    ("audio", {"n_fft": 256, "hop_length": 300}, "hop_length"),
    ("data", {"mode": "sideways"}, "unknown synthesis mode"),
    ("data", {"val_split": 0.7, "test_split": 0.5}, "must be in"),
    ("model", {"context": -1}, "context"),
    ("train", {"epochs": 0}, "epochs"),
])
def test_validation_rejects_bad_values(section, values, match):
    with pytest.raises(ValueError, match=match):
        Config.from_dict({section: values})


def test_n_bins_follows_n_fft():
    assert AudioConfig(n_fft=256).n_bins == 129
    assert AudioConfig(n_fft=1024).n_bins == 513


def test_win_length_follows_n_fft_unless_given():
    assert AudioConfig(n_fft=256).window_length == 256
    assert AudioConfig(n_fft=512, win_length=400).window_length == 400
    # ...and keeps following it, rather than being frozen at construction:
    # resolving it eagerly left a stale 512 behind after n_fft was lowered,
    # and the geometry then refused to build at all
    cfg = AudioConfig()
    cfg.n_fft = 256
    assert cfg.window_length == 256
    assert cfg.stft.win_length == 256


def test_run_paths_live_under_the_experiment(config):
    run = config.run_dir("exp1")
    assert config.model_path("exp1") == run / "ddae.keras"
    assert config.stats_path("exp1") == run / "stats.npz"
    assert config.report_path("exp1") == run / "report.csv"
    assert Path(config.train.out_dir).name == "runs"
