# -*- coding: utf-8 -*-
"""The 2-D architectures: any spectrogram in, the same shape out — and the
one that calls itself causal is."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from kudio_enhance.models import (
    REGISTRY,
    build_model,
    count_params,
    default_units,
    is_sequence,
)
from kudio_enhance.train import seeded

TWO_D = ["crn", "unet"]


def _set_bins(config, n_fft: int) -> int:
    """Re-shape the conftest STFT for *n_fft*, returning the bin count."""
    config.audio.n_fft = n_fft
    config.audio.win_length = None
    config.audio.hop_length = n_fft // 2
    return config.audio.n_bins


def _probe(frames: int, bins: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(
        (2, frames, bins)).astype(np.float32)


# ------------------------------------------------------------------ shapes

def test_both_are_sequence_models_built_small_by_default():
    """A 2-D convolution's channel is a whole time-frequency map, so the
    widths that suit a dense layer would make these enormous."""
    for name in TWO_D:
        assert is_sequence(name) is True
        assert max(default_units(name)) <= 128
    # the dense and recurrent models keep the stack they always had
    for name in ("ddae", "blstm", "conv_ae"):
        assert default_units(name) == [1024, 1024, 1024]


def test_the_builders_default_to_what_the_registry_builds(torch):
    """Two copies of one default must not drift apart."""
    from kudio_enhance.models.crn import build_crn
    from kudio_enhance.models.unet import build_unet

    for name, builder in (("crn", build_crn), ("unet", build_unet)):
        default = inspect.signature(builder).parameters["channels"].default
        assert list(default) == default_units(name)


@pytest.mark.parametrize("name", TWO_D)
@pytest.mark.parametrize("n_fft", [32, 256, 512])
def test_any_spectrogram_comes_back_the_same_shape(config, torch, predict,
                                                   name, n_fft):
    """Trained on fixed windows, run on whole utterances: one frame, a
    length that no halving divides, and everything between. 17, 129 and 257
    bins are none of them a power of two."""
    config.model.name = name
    config.model.units = None
    bins = _set_bins(config, n_fft)
    model = build_model(config)

    for frames in (1, 7, 64, 101):
        out = predict(model, _probe(frames, bins))
        assert out.shape == (2, frames, bins)
        assert np.isfinite(out).all()


@pytest.mark.parametrize("bins", [1, 2, 3, 64, 100])
def test_odd_and_even_bin_counts_are_rebuilt_exactly(torch, predict, bins):
    """Halving rounds up, so 2n - 1 and 2n bins meet at n on the way down;
    the way back up has to know which it came from."""
    from kudio_enhance.models.crn import build_crn
    from kudio_enhance.models.unet import build_unet

    for model in (build_crn(bins, channels=[4, 8, 8]),
                  build_unet(channels=[4, 8, 8])):
        assert predict(model, _probe(9, bins)).shape == (2, 9, bins)


@pytest.mark.parametrize("name", TWO_D)
def test_a_training_step_reaches_every_weight(config, torch, name):
    """Every layer is on the path from the input to the loss — the skip
    connections included — even when the cropping has to undo padding."""
    config.model.name = name
    config.model.units = [8, 16, 16]
    model = build_model(config).train()

    x = torch.from_numpy(_probe(13, config.audio.n_bins))
    model(x).square().mean().backward()
    starved = [key for key, param in model.named_parameters()
               if param.grad is None or not torch.any(param.grad != 0)]
    assert starved == []


@pytest.mark.parametrize("name", TWO_D)
def test_a_mask_is_bounded_and_a_spectrum_is_not(config, torch, predict, name):
    config.model.name = name
    config.model.units = [8, 16]
    probe = _probe(20, config.audio.n_bins) * 10

    config.model.target = "irm"
    mask = predict(build_model(config), probe)
    assert mask.min() >= 0.0 and mask.max() <= 1.0

    config.model.target = "spectrum"
    spectrum = predict(build_model(config), probe)
    assert spectrum.min() < 0.0 or spectrum.max() > 1.0


# --------------------------------------------------------------- causality

def _future_sensitivity(predict, model, bins: int, t: int):
    """``(changed before t, changed at t, changed after t)`` when only frame
    *t* of the input changes."""
    x = _probe(40, bins)
    moved = x.copy()
    moved[:, t] += 3.0
    changed = np.any(predict(model, moved) != predict(model, x), axis=(0, 2))
    return changed[:t].any(), bool(changed[t]), changed[t + 1:].any()


def test_the_crn_output_never_depends_on_the_future(config, torch, predict):
    """Changing every frame after *t* leaves frames up to *t* bit for bit
    alone, while the frames after it do move — so the test is not passing
    on a model that ignores its input."""
    config.model.name = "crn"
    config.model.units = None
    with seeded(0):
        model = build_model(config)
    bins = config.audio.n_bins

    x = _probe(40, bins)
    before = predict(model, x)
    for t in (0, 13, 38):
        future = x.copy()
        future[:, t + 1:] = _probe(40 - t - 1, bins, seed=t + 1)
        after = predict(model, future)
        assert np.array_equal(after[:, :t + 1], before[:, :t + 1])
        assert not np.allclose(after[:, t + 1:], before[:, t + 1:])


def test_the_crn_uses_the_present_frame(config, torch, predict):
    """Causal without a frame of latency: a change at frame *t* already
    shows at frame *t*, and nowhere before it."""
    config.model.name = "crn"
    config.model.units = None
    with seeded(0):
        model = build_model(config)

    earlier, now, later = _future_sensitivity(predict, model,
                                              config.audio.n_bins, t=20)
    assert not earlier
    assert now and later


def test_the_unet_is_not_causal(config, torch, predict):
    """Its description says so, and this is what that means."""
    assert "non-causal" in REGISTRY["unet"].description
    config.model.name = "unet"
    config.model.units = None
    with seeded(0):
        model = build_model(config)

    earlier, now, later = _future_sensitivity(predict, model,
                                              config.audio.n_bins, t=20)
    assert earlier and now and later


# -------------------------------------------------------------- parameters

def test_the_crn_parameter_count_is_the_architecture(config, torch):
    """Encoder convolutions of 2 x 3 with BatchNorm; two LSTMs as wide as
    the channels times the bins left after halving; decoder layers that read
    twice their encoder layer's channels, the skip included; and no
    BatchNorm on the output layer."""
    config.model.name = "crn"
    config.model.units = [8, 16]
    bins = config.audio.n_bins                       # 129 -> 65 -> 33
    width = 16 * math.ceil(math.ceil(bins / 2) / 2)
    kernel = 2 * 3

    encoder = (1 * 8 * kernel + 8 + 2 * 8) + (8 * 16 * kernel + 16 + 2 * 16)
    lstm = 2 * (4 * width * width * 2 + 2 * 4 * width)
    decoder = (2 * 16 * 8 * kernel + 8 + 2 * 8) + (2 * 8 * 1 * kernel + 1)
    model = build_model(config)
    assert model.lstm.input_size == width
    assert count_params(model) == encoder + lstm + decoder


def test_the_unet_parameter_count_is_the_architecture(config, torch):
    """Two 3 x 3 convolutions with BatchNorm per block; a 2 x 2 up-convolution
    and a block reading twice the channels on the way back; a 1 x 1 head —
    and nothing that depends on the number of bins."""
    def block(width_in, width_out):
        return ((width_in * width_out * 9 + width_out) + 2 * width_out
                + (width_out * width_out * 9 + width_out) + 2 * width_out)

    def up(width_in, width_out):
        return width_in * width_out * 4 + width_out

    expected = (block(1, 8) + block(8, 16) + block(16, 32)
                + up(32, 16) + block(32, 16) + up(16, 8) + block(16, 8)
                + (8 + 1))
    config.model.name = "unet"
    config.model.units = [8, 16, 32]
    for n_fft in (256, 512, 1024):
        _set_bins(config, n_fft)
        assert count_params(build_model(config)) == expected


def test_the_default_widths_stay_a_few_million_parameters(config, torch):
    """At the package's default n_fft of 512, against the dense baseline's
    3.7M. The CRN's LSTM reads every bin left after five halvings — 9 of
    257 — at 64 channels, and holds most of the network; the paper's own
    widths, doubling on to 256, would make it four times as wide and sixteen
    times the size, which is why they are not the default."""
    _set_bins(config, 512)
    config.model.units = None

    config.model.name = "crn"
    crn = build_model(config)
    lstm = sum(param.numel() for param in crn.lstm.parameters())
    assert crn.lstm.input_size == 64 * 9
    assert count_params(crn) < 6_000_000
    assert lstm > 0.8 * count_params(crn)

    config.model.name = "unet"
    assert count_params(build_model(config)) < 1_000_000


# ------------------------------------------------------------ the defaults

@pytest.mark.parametrize("name", TWO_D)
def test_they_normalise_and_initialise_like_the_others(config, torch, name):
    """One config trains every architecture with one learning rate, so they
    all start alike: Keras's slow BatchNorm average, zero biases, and an
    LSTM forget gate that starts open."""
    config.model.name = name
    config.model.units = [8, 8]
    model = build_model(config)

    norms = [m for m in model.modules()
             if isinstance(m, torch.nn.BatchNorm2d)]
    assert norms
    assert all(m.momentum == 0.01 and m.eps == pytest.approx(1e-3)
               for m in norms)
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            assert torch.all(module.bias == 0)

    for lstm in (m for m in model.modules()
                 if isinstance(m, torch.nn.LSTM)):
        hidden = lstm.hidden_size
        for layer in range(lstm.num_layers):
            bias = (getattr(lstm, f"bias_ih_l{layer}")
                    + getattr(lstm, f"bias_hh_l{layer}"))
            assert torch.all(bias.detach().reshape(4, hidden)[1] == 1.0)


@pytest.mark.parametrize("name", TWO_D)
def test_an_empty_width_list_is_refused_by_name(config, torch, name):
    config.model.name = name
    config.model.units = []
    with pytest.raises(ValueError, match="at least one"):
        build_model(config)
