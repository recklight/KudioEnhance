# -*- coding: utf-8 -*-
"""Typed config: defaults, round trips, and failing loudly on typos."""
from __future__ import annotations

from pathlib import Path

import pytest

from kudioenhance import AudioConfig, Config, LossConfig


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
    ("model", {"units": [64, 0]}, "units must be"),
    ("model", {"units": 64}, "units must be"),
    ("train", {"epochs": 0}, "epochs"),
    ("train", {"batch_size": 0}, "batch_size"),
    ("train", {"batch_size": 2.5}, "batch_size"),
    ("train", {"batch_size": True}, "batch_size"),
    ("train", {"batch_size": None}, "batch_size"),
    ("train", {"device": "tpu"}, "unknown device"),
    ("train", {"device": "cpu:0"}, "unknown device"),
    ("train", {"device": "cuda:first"}, "unknown device"),
    ("train", {"shuffle_buffer": -1}, "shuffle_buffer"),
    ("train", {"shuffle_buffer": 2.5}, "shuffle_buffer"),
    # what YAML hands over for `streaming: "false"`, which is true
    ("train", {"streaming": "false"}, "streaming must be true or false"),
    ("train", {"num_workers": -1}, "num_workers"),
    ("train", {"num_workers": 1.5}, "num_workers"),
    ("train", {"num_workers": True}, "num_workers"),
    ("audio", {"compress": 0}, "audio.compress"),
    ("audio", {"compress": 2}, "audio.compress"),
    ("audio", {"compress": "0.3"}, "audio.compress"),
    ("audio", {"pad_mode": "replicate"}, "audio.pad_mode"),
    ("data", {"crop_seconds": 0}, "crop_seconds"),
    ("data", {"crop_seconds": None}, "crop_seconds"),
    ("data", {"crop_seconds": float("inf")}, "crop_seconds"),
    ("loss", {"time_l1": -1}, "loss.time_l1"),
    ("loss", {"neg_si_snr": float("nan")}, "loss.neg_si_snr"),
    ("loss", {"complex_mse": True}, "loss.complex_mse"),
    ("loss", {"complex_mse": 0, "magnitude_mse": 0, "neg_si_snr": 0},
     "nothing to train on"),
    ("loss", {"compress": 1.2}, "loss.compress"),
    ("loss", {"zero_mean": "false"}, "loss.zero_mean"),
])
def test_validation_rejects_bad_values(section, values, match):
    with pytest.raises(ValueError, match=match):
        Config.from_dict({section: values})


def test_training_runs_wherever_torch_finds_a_device_by_default():
    """One config file for the laptop and the training box; the seed is
    borrowed from the data section unless it is set here."""
    cfg = Config()
    assert cfg.train.device == "auto"
    assert cfg.train.seed is None
    assert Config.from_dict({"train": {"device": "cuda:1"}}).train.device \
        == "cuda:1"


def test_training_data_is_held_in_memory_unless_streaming_is_asked_for(
        tmp_path):
    """Streaming trades a pass over the audio every epoch for memory, which
    is the wrong trade for the corpora most runs use."""
    cfg = Config()
    assert cfg.train.streaming is False
    assert cfg.train.shuffle_buffer == 0

    cfg.train.streaming = True
    cfg.train.shuffle_buffer = 4096
    reloaded = Config.from_yaml(cfg.to_yaml(tmp_path / "config.yaml"))
    assert reloaded.train.streaming is True
    assert reloaded.train.shuffle_buffer == 4096


def test_the_widths_are_the_architectures_own_unless_given():
    """1024 suits a dense layer and not a 2-D convolution, so an unset
    ``units`` follows the model's name — and the three architectures that
    shared the 1024-wide stack keep it."""
    from kudioenhance.models import default_units, resolve_units

    cfg = Config()
    assert cfg.model.units is None
    for name in ("ddae", "blstm", "conv_ae"):
        cfg.model.name = name
        assert resolve_units(cfg).model.units == [1024, 1024, 1024]
    for name in ("crn", "unet"):
        cfg.model.name = name
        assert resolve_units(cfg).model.units == default_units(name)
        assert max(default_units(name)) <= 128
    # resolving hands back a copy and leaves the caller's config alone, so
    # switching the name afterwards still follows the new architecture
    assert cfg.model.units is None

    cfg.model.units = [64, 64]
    assert resolve_units(cfg) is cfg


def test_unset_widths_survive_a_round_trip_through_yaml(tmp_path):
    path = Config().to_yaml(tmp_path / "config.yaml")
    assert Config.from_yaml(path).model.units is None


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
    assert config.model_path("exp1") == run / "ddae.pt"
    assert config.stats_path("exp1") == run / "stats.npz"
    assert config.report_path("exp1") == run / "report.csv"
    assert Path(config.train.out_dir).name == "runs"


# ------------------------------------------------------- the waveform path

#: A run's config.yaml with none of the waveform path's settings: a crn
#: baseline on a prepared corpus.
RUN_CONFIG_LOG_POWER = """\
audio:
  sr: 16000
  n_fft: 512
  hop_length: 256
  win_length: null
  window: hamming
data:
  clean_dir: data/clean
  noise_dir: data/noise
  mixed_dir: runs/mixed
  snr_db: [-5, 0, 5]
  rt60: []
  drr_db: 0.0
  reverberant_target: false
  channel: null
  mode: regular
  seed: 17
  val_split: 0.1
  test_split: 0.1
  max_files: null
  prepared: data/vbd16k
model:
  name: crn
  target: irm
  context: 2
  n_frames: 64
  units: [16, 32, 64, 64, 64]
  dropout: 0.1
train:
  epochs: 40
  batch_size: 16
  learning_rate: 0.001
  patience: 8
  out_dir: runs
  shuffle_buffer: 3000
  device: cpu
  seed: 17
  streaming: true
"""


def test_a_config_without_the_waveform_settings_loads_unchanged(tmp_path):
    """Every value it names comes back as it was, and nothing it did not
    name stands in the way: the new keys and the loss section take their
    defaults."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(RUN_CONFIG_LOG_POWER, encoding="utf-8")
    cfg = Config.from_yaml(path)
    written = yaml.safe_load(RUN_CONFIG_LOG_POWER)
    loaded = cfg.to_dict()
    for section, values in written.items():
        for key, value in values.items():
            assert loaded[section][key] == value, (section, key)
    assert set(loaded) - set(written) == {"loss"}
    assert {key for section in written
            for key in set(loaded[section]) - set(written[section])} == {
        "compress", "pad_mode", "crop_seconds", "num_workers"}
    assert cfg.audio.compress is None
    assert cfg.audio.pad_mode == "constant"
    assert cfg.data.crop_seconds == 2.0
    assert cfg.train.num_workers == 0
    assert cfg.loss == LossConfig()


def test_the_waveform_settings_survive_a_round_trip(tmp_path):
    cfg = Config.from_dict({
        "audio": {"window": "sqrt_hann", "compress": 0.5,
                  "pad_mode": "reflect"},
        "data": {"crop_seconds": 1.5},
        "train": {"num_workers": 4},
        "loss": {"complex_mse": 1, "magnitude_mse": 0, "time_l1": 0.2,
                 "neg_si_snr": 0, "compress": None, "zero_mean": True},
    })
    reloaded = Config.from_yaml(cfg.to_yaml(tmp_path / "config.yaml"))
    assert reloaded.to_dict() == cfg.to_dict()
    assert reloaded.loss.weights == {"complex_mse": 1.0, "time_l1": 0.2}
    assert reloaded.loss.compress is None


def test_the_default_loss_is_gtcrns():
    """30 x complex + 70 x magnitude at c = 0.3, and its -log10 SI-SNR term,
    a tenth of the negative SI-SNR in dB, with the mean left in."""
    loss = LossConfig()
    assert loss.weights == {"complex_mse": 30.0, "magnitude_mse": 70.0,
                            "neg_si_snr": 0.1}
    assert loss.compress == 0.3
    assert loss.zero_mean is False


def test_an_unknown_loss_key_names_the_offender():
    with pytest.raises(ValueError, match=r"'loss': si_snr\."):
        Config.from_dict({"loss": {"si_snr": 1.0}})
