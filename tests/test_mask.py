# -*- coding: utf-8 -*-
"""The ideal ratio mask: the target, and what training against it changes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kudio
from kudio_enhance.config import AudioConfig, Config, ModelConfig
from kudio_enhance.features import (
    apply_mask,
    ideal_ratio_mask,
    spectrogram,
    to_waveform,
)

SR = 16000


@pytest.fixture
def audio_cfg():
    return AudioConfig(sr=SR, n_fft=256, hop_length=128)


@pytest.fixture
def speech_and_noise():
    t = np.arange(SR) / SR
    speech = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    noise = (0.4 * np.random.default_rng(1).standard_normal(SR)).astype(np.float32)
    return speech, noise


# --------------------------------------------------------------- the target

def test_the_mask_is_bounded(audio_cfg, speech_and_noise):
    """A sigmoid output can reach every value the target takes — which is the
    reason the target is a mask and not a spectrogram."""
    speech, noise = speech_and_noise
    mask = ideal_ratio_mask(spectrogram(speech, audio_cfg),
                            spectrogram(noise, audio_cfg))
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_the_mask_keeps_speech_and_rejects_noise(audio_cfg, speech_and_noise):
    speech, noise = speech_and_noise
    mask = ideal_ratio_mask(spectrogram(speech, audio_cfg),
                            spectrogram(noise, audio_cfg))
    assert mask.max() > 0.9          # the 200 Hz bin is almost all speech
    assert mask.min() < 0.5          # bins with no speech in them


def test_all_speech_and_all_noise_are_the_two_extremes(audio_cfg):
    """Sanity on the formula itself: sqrt(S / (S + N))."""
    quiet = np.full((4, 8), -12.0)          # log10(|X|^2) = -12
    loud = np.full((4, 8), 0.0)
    assert ideal_ratio_mask(loud, quiet) == pytest.approx(1.0, abs=1e-5)
    assert ideal_ratio_mask(quiet, loud) == pytest.approx(0.0, abs=1e-5)
    # equal power in both -> keep half the energy, i.e. 1/sqrt(2)
    assert ideal_ratio_mask(loud, loud) == pytest.approx(1 / np.sqrt(2), abs=1e-6)


def test_the_mask_does_not_overflow_on_extreme_ratios():
    """10 ** (huge difference) would; the difference is clipped instead."""
    mask = ideal_ratio_mask(np.array([[-300.0, 300.0]]),
                            np.array([[300.0, -300.0]]))
    assert np.all(np.isfinite(mask))
    assert mask[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert mask[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_applying_a_mask_is_an_addition_in_the_log_domain(audio_cfg):
    """log10(|MY|^2) = log10(|Y|^2) + 2 log10 M."""
    noisy = np.full((3, 5), 2.0)
    half = np.full((3, 5), 0.5)
    assert apply_mask(noisy, half) == pytest.approx(
        2.0 + 2.0 * np.log10(0.5), abs=1e-5)
    assert apply_mask(noisy, np.ones_like(half)) == pytest.approx(2.0)


def test_the_oracle_mask_actually_denoises(audio_cfg, speech_and_noise):
    """The ceiling this target aims at: a perfect mask, applied perfectly."""
    speech, noise = speech_and_noise
    noisy = speech + noise
    mask = ideal_ratio_mask(spectrogram(speech, audio_cfg),
                            spectrogram(noise, audio_cfg))
    recovered = to_waveform(noisy, apply_mask(spectrogram(noisy, audio_cfg), mask),
                            audio_cfg)
    n = min(len(recovered), len(speech))
    assert kudio.si_sdr(speech[:n], recovered[:n]) > \
        kudio.si_sdr(speech[:n], noisy[:n]) + 3.0


# ---------------------------------------------------------------- the config

def test_the_target_is_declared_and_validated():
    assert ModelConfig().target == "spectrum"
    assert ModelConfig().predicts_mask is False
    assert ModelConfig(target="irm").predicts_mask is True
    with pytest.raises(ValueError, match="unknown target"):
        ModelConfig(target="magic")


def test_the_target_survives_a_round_trip_through_yaml(tmp_path):
    """Inference reads it back to decide what the prediction *means*, so it
    has to be in the run directory, not in the caller's head."""
    cfg = Config()
    cfg.model.target = "irm"
    path = cfg.to_yaml(tmp_path / "config.yaml")
    assert Config.from_yaml(path).model.predicts_mask is True


# ---------------------------------------------------------------- the arrays

def test_a_mask_target_is_not_standardised(config):
    """It is already on a bounded scale; putting the input's mean and standard
    deviation through it would produce a target the sigmoid cannot reach."""
    from kudio_enhance.data import build_arrays
    from kudio_enhance.pipeline import synthesize

    config.model.target = "irm"
    pairs = synthesize(config, "exp_mask")
    _, y, _ = build_arrays(pairs, config, sequence=False, fit=True)

    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_a_spectrum_target_is_standardised(config):
    from kudio_enhance.data import build_arrays
    from kudio_enhance.pipeline import synthesize

    config.model.target = "spectrum"
    pairs = synthesize(config, "exp_spec")
    _, y, _ = build_arrays(pairs, config, sequence=False, fit=True)
    # normalised log-power goes well outside [0, 1] in both directions
    assert y.min() < 0.0


def test_both_targets_produce_the_same_input_shape(config):
    from kudio_enhance.data import build_arrays
    from kudio_enhance.pipeline import synthesize

    shapes = {}
    for target in ("spectrum", "irm"):
        config.model.target = target
        pairs = synthesize(config, f"exp_{target}")
        x, y, _ = build_arrays(pairs, config, sequence=False, fit=True)
        shapes[target] = (x.shape, y.shape)
    assert shapes["spectrum"][0] == shapes["irm"][0]
    assert shapes["spectrum"][1] == shapes["irm"][1]


# ---------------------------------------------------------------- the model

@pytest.mark.parametrize("name", ["ddae", "blstm", "conv_ae"])
def test_a_mask_model_can_only_output_zero_to_one(config, name):
    from kudio_enhance.models import build_model

    config.model.name = name
    config.model.target = "irm"
    model = build_model(config)

    bins = config.audio.n_bins
    if name == "ddae":
        probe = np.random.default_rng(0).standard_normal(
            (4, bins * (2 * config.model.context + 1))).astype(np.float32) * 10
    else:
        probe = np.random.default_rng(0).standard_normal(
            (2, 8, bins)).astype(np.float32) * 10

    out = model.predict(probe, verbose=0)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_a_spectrum_model_is_unbounded(config):
    """...which is the cost: it has to learn the level as well as the shape."""
    from kudio_enhance.models import build_model

    config.model.name = "ddae"
    config.model.target = "spectrum"
    model = build_model(config)
    probe = np.random.default_rng(0).standard_normal(
        (8, config.audio.n_bins * (2 * config.model.context + 1))
    ).astype(np.float32) * 10
    out = model.predict(probe, verbose=0)
    assert out.min() < 0.0 or out.max() > 1.0


def test_the_conv_residual_shortcut_is_dropped_for_a_mask(config):
    """'add this to the input' and 'keep this fraction of it' are different
    parameterisations; combining them makes the sigmoid an offset."""
    from kudio_enhance.models import build_model

    config.model.name = "conv_ae"
    config.model.target = "spectrum"
    assert any(layer.name == "clean_spec" for layer in build_model(config).layers)

    config.model.target = "irm"
    layers = build_model(config).layers
    assert any(layer.name == "mask_spec" for layer in layers)
    assert not any(layer.name == "clean_spec" for layer in layers)


# ------------------------------------------------------------- end to end

def test_a_mask_model_trains_and_enhances(config, tmp_path):
    from kudio_enhance import pipeline
    from kudio_enhance.inference import Enhancer

    config.model.name = "ddae"
    config.model.target = "irm"
    config.model.units = [64]
    summary = pipeline.run(config, "mask_run", epochs=1, verbose=0)
    assert "enhanced_si_sdr" in summary

    enhancer = Enhancer.load(config.run_dir("mask_run"))
    assert enhancer.mask is True

    y, _ = kudio.file_load(sorted(Path(config.data.mixed_dir).rglob("*.wav"))[0],
                           sr=config.audio.sr)
    out = enhancer.enhance(y)
    assert len(out) == len(y)
    assert np.all(np.isfinite(out))
    # a mask can only attenuate, so the output cannot be louder than the input
    assert np.max(np.abs(out)) <= np.max(np.abs(y)) * 1.05

