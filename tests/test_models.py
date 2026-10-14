# -*- coding: utf-8 -*-
"""Architectures build with the shapes the data path produces."""
from __future__ import annotations

import numpy as np
import pytest

from kudio_enhance.models import (
    REGISTRY,
    build_model,
    count_params,
    is_sequence,
    list_models,
)
from kudio_enhance.train import seeded


def test_registry_is_described():
    listed = list_models()
    assert set(listed) == set(REGISTRY)
    assert all(text for text in listed.values())


def test_unknown_model_lists_the_alternatives(config):
    config.model.name = "transformer"
    with pytest.raises(ValueError,
                       match="Available: blstm, conv_ae, crn, ddae, unet"):
        build_model(config)


def test_ddae_is_frame_wise_with_context_stacked_input(config, torch, predict):
    config.model.name = "ddae"
    model = build_model(config)

    bins = config.audio.n_bins
    width = bins * (2 * config.model.context + 1)
    assert is_sequence("ddae") is False
    assert predict(model, np.zeros((7, width))).shape == (7, bins)
    # the context is part of the architecture: an unstacked frame is refused,
    # not silently treated as something else
    with pytest.raises(RuntimeError):
        predict(model, np.zeros((7, bins)))


@pytest.mark.parametrize("name", ["blstm", "conv_ae", "crn", "unet"])
def test_sequence_models_accept_any_length(config, torch, predict, name):
    config.model.name = name
    config.model.units = [16, 16]
    model = build_model(config)

    bins = config.audio.n_bins
    assert is_sequence(name) is True
    # the point of leaving the time axis open: train short, infer long
    for frames in (1, 16, 57):
        out = predict(model, np.zeros((2, frames, bins)))
        assert out.shape == (2, frames, bins)


def test_conv_ae_starts_as_a_pass_through(config, torch, predict):
    """The residual connection means an untrained net barely changes the input."""
    config.model.name = "conv_ae"
    config.model.units = [8]
    model = build_model(config)

    x = np.random.default_rng(0).normal(
        size=(1, 20, config.audio.n_bins)).astype(np.float32)
    out = predict(model, x)
    assert np.corrcoef(out.ravel(), x.ravel())[0, 1] > 0.5


def test_conv_ae_dilation_doubles_the_reach_of_every_layer(config, torch,
                                                          predict):
    """Three layers of kernel 5 dilated 1, 2 and 4 see 2 * (1 + 2 + 4) = 14
    frames either side — more than twice what the same stack would see
    undilated — and nothing beyond that."""
    config.model.name = "conv_ae"
    config.model.units = [8, 8, 8]
    with seeded(0):
        model = build_model(config)

    bins = config.audio.n_bins
    silence = np.zeros((1, 101, bins), dtype=np.float32)
    impulse = silence.copy()
    impulse[0, 50] = 1.0
    changed = np.flatnonzero(np.any(
        predict(model, impulse) != predict(model, silence), axis=-1)[0])

    assert changed.min() >= 50 - 14 and changed.max() <= 50 + 14
    assert changed.max() - 50 > 2 * 3            # beyond an undilated stack


def test_the_parameter_count_is_the_architecture(config, torch):
    """Dense, BatchNorm scale and shift, dense, BatchNorm, head — and no
    running statistics, which are buffers rather than parameters."""
    config.model.name = "ddae"
    config.model.units = [32, 32]
    width = config.audio.n_bins * (2 * config.model.context + 1)
    bins = config.audio.n_bins

    expected = ((width * 32 + 32) + 2 * 32
                + (32 * 32 + 32) + 2 * 32
                + (32 * bins + bins))
    assert count_params(build_model(config)) == expected


@pytest.mark.parametrize("name", ["ddae", "conv_ae"])
def test_batch_norm_keeps_a_slow_running_average(config, torch, name):
    """Keras's momentum of 0.99 moves the running mean 1% of the way per
    batch; torch's own default would move it 10%, and inference would use
    statistics ten times noisier than the ones these configs were tuned on."""
    config.model.name = name
    config.model.units = [8]
    model = build_model(config).train()
    norm = next(m for m in model.modules()
                if isinstance(m, torch.nn.BatchNorm1d))

    seen = {}
    norm.register_forward_hook(
        lambda module, inputs, output: seen.setdefault("x", inputs[0]))
    bins = config.audio.n_bins
    shape = ((16, bins * (2 * config.model.context + 1)) if name == "ddae"
             else (4, 10, bins))
    with torch.no_grad():
        model(torch.randn(*shape) + 5.0)

    batch_mean = seen["x"].mean(dim=[d for d in range(seen["x"].dim())
                                     if d != 1])
    assert torch.allclose(norm.running_mean, 0.01 * batch_mean, atol=1e-6)
    assert norm.eps == pytest.approx(1e-3)


def test_the_lstm_forget_gate_starts_open(config, torch):
    """Keras's ``unit_forget_bias``: a cell that starts out forgetting has to
    learn to remember before it can learn anything else."""
    config.model.name = "blstm"
    config.model.units = [8, 6]
    lstms = [m for m in build_model(config).modules()
             if isinstance(m, torch.nn.LSTM)]
    assert len(lstms) == 2

    for lstm in lstms:
        hidden = lstm.hidden_size
        for suffix in ("l0", "l0_reverse"):
            bias = (getattr(lstm, f"bias_ih_{suffix}")
                    + getattr(lstm, f"bias_hh_{suffix}"))
            gates = bias.detach().reshape(4, hidden)       # i, f, g, o
            assert torch.all(gates[1] == 1.0)
            assert torch.all(gates[[0, 2, 3]] == 0.0)


@pytest.mark.parametrize("name", ["ddae", "blstm", "conv_ae", "crn", "unet"])
def test_dropout_is_only_built_when_asked_for(config, torch, name):
    config.model.name = name
    config.model.units = [8, 8]

    config.model.dropout = 0.0
    assert not any(isinstance(m, torch.nn.Dropout)
                   for m in build_model(config).modules())

    config.model.dropout = 0.3
    rates = {m.p for m in build_model(config).modules()
             if isinstance(m, torch.nn.Dropout)}
    assert rates == {0.3}

