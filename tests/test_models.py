# -*- coding: utf-8 -*-
"""Architectures build with the shapes the data path produces."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="model tests need TensorFlow")

from kudio_enhance.models import REGISTRY, build_model, is_sequence, list_models  # noqa: E402


def test_registry_is_described():
    listed = list_models()
    assert set(listed) == set(REGISTRY)
    assert all(text for text in listed.values())


def test_unknown_model_lists_the_alternatives(config):
    config.model.name = "transformer"
    with pytest.raises(ValueError, match="Available: blstm, conv_ae, ddae"):
        build_model(config)


def test_ddae_is_frame_wise_with_context_stacked_input(config):
    config.model.name = "ddae"
    model = build_model(config)

    bins = config.audio.n_bins
    assert is_sequence("ddae") is False
    assert model.input_shape == (None, bins * (2 * config.model.context + 1))
    assert model.output_shape == (None, bins)


@pytest.mark.parametrize("name", ["blstm", "conv_ae"])
def test_sequence_models_accept_any_length(config, name):
    config.model.name = name
    config.model.units = [16, 16]
    model = build_model(config)

    bins = config.audio.n_bins
    assert is_sequence(name) is True
    assert model.input_shape == (None, None, bins)
    assert model.output_shape == (None, None, bins)

    # the point of the undefined time axis: train short, infer long
    for frames in (16, 57):
        out = model.predict(np.zeros((1, frames, bins), dtype=np.float32),
                            verbose=0)
        assert out.shape == (1, frames, bins)


def test_conv_ae_starts_as_a_pass_through(config):
    """The residual connection means an untrained net barely changes the input."""
    config.model.name = "conv_ae"
    config.model.units = [8]
    model = build_model(config)

    x = np.random.default_rng(0).normal(
        size=(1, 20, config.audio.n_bins)).astype(np.float32)
    out = model.predict(x, verbose=0)
    assert np.corrcoef(out.ravel(), x.ravel())[0, 1] > 0.5
