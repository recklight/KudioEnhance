# -*- coding: utf-8 -*-
"""GTCRN: the paper's network weight for weight, no look-ahead to the last
bit, and trained end to end on the waveform path."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs PyTorch",
                            exc_type=ImportError)

import kudio  # noqa: E402

from kudioenhance import progress  # noqa: E402
from kudioenhance.cli import main  # noqa: E402
from kudioenhance.config import Config  # noqa: E402
from kudioenhance.data import Pair  # noqa: E402
from kudioenhance.inference import Enhancer  # noqa: E402
from kudioenhance.models import (  # noqa: E402
    REGISTRY,
    build_model,
    count_macs,
    count_params,
    default_units,
    is_waveform,
    model_bibtex,
)
from kudioenhance.models.gtcrn import GTCRN, build_gtcrn, erb_filters  # noqa: E402
from kudioenhance.spectral import ComplexSTFT  # noqa: E402
from kudioenhance.train import seeded, training_seed  # noqa: E402
from kudioenhance.waveform import waveform_loader  # noqa: E402
from kudioenhance.waveform_train import (  # noqa: E402
    estimate_spectrum,
    fit_waveform,
)

SR, N_FFT, HOP = 16000, 512, 256
BINS = N_FFT // 2 + 1
#: GTCRN's STFT, as its paper and code compute it.
AUDIO = {"sr": SR, "n_fft": N_FFT, "hop_length": HOP, "win_length": N_FFT,
         "window": "sqrt_hann", "pad_mode": "reflect"}


def _config(**sections) -> Config:
    values = {"audio": dict(AUDIO), "model": {"name": "gtcrn"}}
    for name, section in sections.items():
        values.setdefault(name, {}).update(section)
    return Config.from_dict(values)


def _spectrum(frames: int, seed: int = 0, bins: int = BINS,
              batch: int = 2) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    parts = rng.standard_normal((2, batch, bins, frames))
    return torch.from_numpy((parts[0] + 1j * parts[1]).astype(np.complex64))


def _run(model, spec: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(spec)


def _mask(model, spec: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model.mask(spec)


def _model(seed: int = 0, **kwargs) -> GTCRN:
    with seeded(seed):
        return GTCRN(BINS, SR, **kwargs)


# ---------------------------------------------------------------- registry

def test_it_is_registered_on_the_waveform_path_with_its_paper():
    assert is_waveform("gtcrn")
    assert "causal" in REGISTRY["gtcrn"].description
    assert "10.1109/ICASSP48485.2024.10448310" in model_bibtex("gtcrn")
    assert default_units("gtcrn") == [16]


def test_the_builder_defaults_to_what_the_registry_builds():
    default = inspect.signature(build_gtcrn).parameters["units"].default
    assert list(default) == default_units("gtcrn")


def test_the_registry_reads_the_bins_from_n_fft_not_from_kudio():
    """kudio's STFT has no sqrt_hann, so ``cfg.audio.n_bins`` would fail
    on GTCRN's own config."""
    model = build_model(_config())
    assert model.bands.filters.shape == (64, BINS - 65)


@pytest.mark.parametrize("units", [[6], [2], [16, 16]])
def test_widths_it_cannot_split_are_refused_by_name(units):
    with pytest.raises(ValueError, match="GTCRN"):
        build_model(_config(model={"units": units}))


# ------------------------------------------------------------------ shapes

@pytest.mark.parametrize("frames", [1, 7, 63, 101])
def test_a_spectrum_comes_back_complex_and_the_same_shape(frames):
    out = _run(_model(), _spectrum(frames))
    assert out.shape == (2, BINS, frames)
    assert out.dtype == torch.complex64
    assert torch.isfinite(torch.view_as_real(out)).all()


def test_other_ffts_widths_and_an_even_count_of_folded_bins():
    """1024 points fold 448 bins into the 64 bands; 64 low bins leave 128
    folded bins, which halve to 64 and 32 and come back with the bin a
    halving rounds away."""
    for model, bins in ((GTCRN(N_FFT + 1, SR), N_FFT + 1),
                        (GTCRN(BINS, SR, low_bins=64), BINS),
                        (GTCRN(BINS, SR, width=8), BINS)):
        assert _run(model, _spectrum(9, bins=bins)).shape == (2, bins, 9)


def test_too_few_bins_for_the_bands_is_refused():
    """At 8 kHz and 256 points, 64 bins above bin 65 cannot hold 64 bands
    with distinct centres."""
    with pytest.raises(ValueError, match="too few"):
        GTCRN(129, 8000)


@pytest.mark.parametrize("samples", [N_FFT, SR, SR + 77, 3 * SR + 255])
def test_a_waveform_comes_back_as_long_as_it_went_in(samples):
    stft = ComplexSTFT.from_config(_config().audio)
    model = _model().eval()
    noisy = 0.1 * torch.from_numpy(np.random.default_rng(1).standard_normal(
        (2, samples)).astype(np.float32))
    with torch.no_grad():
        spec = estimate_spectrum(model, stft, noisy)
        wave = stft.waveform(spec, length=samples)
    assert spec.shape == (2, BINS, stft.n_frames(samples))
    assert wave.shape == noisy.shape
    assert torch.isfinite(wave).all()


# --------------------------------------------------------------- causality

def test_the_mask_never_depends_on_the_future():
    """Changing every frame after *t* leaves the mask up to *t*, and so the
    output, bit for bit alone. The mask after *t* has to move, which a model
    that ignored its input, or only scaled it, would not do. The mask is
    what is compared, because the output at a frame is that frame's input
    times the mask and moves with the input whatever the mask does."""
    model = _model()
    x = _spectrum(40)
    mask, out = _mask(model, x), _run(model, x)
    for t in (0, 13, 38):
        future = x.clone()
        future[..., t + 1:] = _spectrum(40 - t - 1, seed=t + 1)
        assert torch.equal(_mask(model, future)[..., :t + 1],
                           mask[..., :t + 1])
        assert torch.equal(_run(model, future)[..., :t + 1],
                           out[..., :t + 1])
        assert not torch.allclose(_mask(model, future)[..., t + 1:],
                                  mask[..., t + 1:])


def test_the_mask_reads_the_present_frame_at_once():
    """No frame of latency: a change at frame *t* moves the mask at *t* and
    after, and nowhere before."""
    model = _model()
    x = _spectrum(40)
    moved = x.clone()
    moved[..., 20] = _spectrum(1, seed=7)[..., 0]
    changed = (_mask(model, moved) != _mask(model, x)).any(dim=1).any(dim=0)
    assert not changed[:20].any()
    assert changed[20] and changed[21:].any()


def test_the_mask_is_bounded_and_multiplies_the_input():
    model = _model()
    x = 10 * _spectrum(30)
    mask = _mask(model, x)
    assert mask.real.abs().max() <= 1 and mask.imag.abs().max() <= 1
    assert torch.allclose(_run(model, x), x * mask)


# -------------------------------------------------------------- ERB bands

def test_the_erb_bands_are_the_authors():
    """Centres evenly spaced on the ERB-rate scale from bin 65 (2031 Hz) to
    bin 256 (8 kHz), as the authors' filters place them, and every folded
    bin shared out between neighbouring bands in full."""
    filters = erb_filters(BINS, SR)
    assert filters.shape == (64, 192)
    centres = (filters.argmax(dim=1) + 65).tolist()
    assert centres[:5] == [65, 66, 68, 70, 71]
    assert centres[-5:] == [235, 240, 245, 251, 256]
    assert all(b > a for a, b in zip(centres, centres[1:]))
    assert torch.allclose(filters.sum(dim=0), torch.ones(192))
    # a band sums its bins rather than averaging them
    assert filters[0].sum() == 1.0 and filters[-1].sum() == pytest.approx(3.0)


def test_a_mask_the_same_in_every_band_is_the_same_in_every_bin():
    model = _model()
    folded = torch.full((1, 2, 3, 129), 0.5)
    assert torch.allclose(model.bands.spread(folded),
                          torch.full((1, 2, 3, BINS), 0.5))


# ------------------------------------------------------------- the weights

def _gru(inputs: int, hidden: int) -> int:
    """Weights of a one-layer, one-direction GRU: three gates, each with an
    input and a recurrent matrix and a bias for both."""
    return 3 * (hidden * inputs + hidden * hidden + 2 * hidden)


def _expected_params(width: int, quarter_bins: int) -> int:
    """Conv blocks: weights, bias, BatchNorm's scale and shift, one PReLU
    slope. A grouped temporal convolution block on half the channels: two
    pointwise convolutions round a depthwise 3 x 3 one, and the attention's
    GRU, twice as wide as the half, with its linear layer. A dual-path
    GRNN: two bidirectional GRUs on half the channels at a quarter of the
    width each way, two forward GRUs on half the channels, a linear layer
    and a layer norm over the frame's bins and channels after each. The
    last layer has a tanh, which has no weights."""
    c, half = width, width // 2

    def conv_block(c_in: int, c_out: int, groups: int) -> int:
        return c_in // groups * c_out * 5 + c_out + 2 * c_out + 1

    temporal = ((3 * half * c + c) + 2 * c + 1
                + (9 * c + c) + 2 * c + 1
                + (c * half + half) + 2 * half
                + _gru(half, 2 * half) + (2 * half * half + half))
    dual_path = (2 * 2 * _gru(half, c // 4) + (c * c + c)
                 + 2 * quarter_bins * c
                 + 2 * _gru(half, half) + (c * c + c)
                 + 2 * quarter_bins * c)
    encoder = conv_block(9, c, 1) + conv_block(c, c, 2) + 3 * temporal
    decoder = (3 * temporal + conv_block(c, c, 2)
               + (c * 2 * 5 + 2) + 2 * 2)
    return encoder + 2 * dual_path + decoder


def test_the_parameter_count_is_the_papers():
    """23,669 trained weights, the paper's 23.7K. The authors' repository
    reports 48.2K because it keeps the ERB filters, and their transpose, as
    two frozen linear layers of 64 x 192."""
    model = _model()
    assert _expected_params(16, 33) == 23_669
    assert count_params(model) == 23_669
    assert count_params(model) + 2 * model.bands.filters.numel() == 48_245
    assert all(param.requires_grad for param in model.parameters())
    assert count_params(_model(width=8)) == _expected_params(8, 33)


def test_the_filters_are_saved_with_the_weights():
    """They are a buffer, so the optimiser leaves them alone, but a weights
    file still says everything the network computes."""
    state = _model().state_dict()
    assert torch.equal(state["bands.filters"], erb_filters(BINS, SR))


def test_a_training_step_reaches_every_weight():
    model = _model().train()
    loss = torch.view_as_real(model(_spectrum(13))).square().mean()
    loss.backward()
    starved = [name for name, param in model.named_parameters()
               if param.grad is None or not torch.any(param.grad != 0)]
    assert starved == []


def _gru_macs(inputs: int, hidden: int) -> int:
    return 3 * hidden * (inputs + hidden)


def _expected_macs_per_frame(width: int = 16) -> int:
    """Multiply-accumulates for one frame at 257 bins: the ERB fold of three
    channels and the spread of two, every convolution, linear layer and GRU
    step. A transposed convolution costs its input positions times its
    kernel and output channels per group."""
    c, half = width, width // 2
    f2, f4 = 65, 33
    erb = (3 + 2) * 192 * 64
    temporal = (f4 * c * 3 * half + f4 * c * 9 + f4 * half * c
                + _gru_macs(half, 2 * half) + 2 * half * half)
    dual_path = (f4 * 2 * 2 * _gru_macs(half, c // 4) + f4 * c * c
                 + f4 * 2 * _gru_macs(half, half) + f4 * c * c)
    encoder = f2 * c * 9 * 5 + f4 * c * half * 5 + 3 * temporal
    decoder = 3 * temporal + f4 * c * half * 5 + f2 * c * 2 * 5
    return erb + encoder + 2 * dual_path + decoder


def test_the_multiply_accumulates_are_counted_frame_by_frame():
    """26.3M a second at 62.5 frames a second; 26.5M on the 63 frames of
    one second's STFT, where ptflops' rules, which also count biases,
    norms, activations and the GRUs' gate arithmetic, give the 33.0M the
    authors report. Every layer costs the same for every frame."""
    model = _model()
    per_frame = _expected_macs_per_frame()
    assert per_frame == 420_944
    for frames in (1, 63, 100):
        assert count_macs(model, _spectrum(frames, batch=1)) \
            == frames * per_frame
    assert per_frame * SR / HOP == pytest.approx(26.31e6, rel=1e-3)


def test_counting_leaves_the_model_as_it_was():
    """It counts on a copy, so every module keeps its own mode."""
    model = _model().train()
    model.encoder[0].eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    count_macs(model, _spectrum(3, batch=1))
    assert model.training and not model.encoder[0].training
    assert all(module.training for module in model.decoder.modules())
    assert all(torch.equal(before[key], value)
               for key, value in model.state_dict().items())


@pytest.mark.parametrize("layer, gates", [(torch.nn.LSTM, 4),
                                          (torch.nn.GRU, 3)])
def test_recurrent_steps_are_counted_on_any_backend(layer, gates):
    """oneDNN runs an LSTM as one fused operation the counter has no formula
    for, so the count is taken without it: every step is an input and a
    recurrent matrix product per gate."""
    onednn = torch.backends.mkldnn.enabled
    rnn = layer(8, 16, batch_first=True)
    assert count_macs(rnn, torch.randn(1, 10, 8)) == 10 * gates * 16 * (8 + 16)
    assert torch.backends.mkldnn.enabled == onednn


# ----------------------------------------------------- trained end to end

def _tones(seconds: float, seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = rng.uniform(120, 240)
    clean = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k
                for k in range(1, 6))
    clean = 0.3 * clean / np.max(np.abs(clean))
    noise = rng.normal(0, 1, len(t))
    noise *= np.sqrt(np.mean(clean ** 2) / np.mean(noise ** 2))
    return clean.astype(np.float32), (clean + noise).astype(np.float32)


def _pairs(folder, count: int, seconds: float, seed: int):
    folder.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i in range(count):
        clean, noisy = _tones(seconds, seed + i)
        kudio.save_wave(folder / f"clean{i}.wav", clean, SR, subtype="FLOAT")
        kudio.save_wave(folder / f"noisy{i}.wav", noisy, SR, subtype="FLOAT")
        pairs.append(Pair(noisy=str(folder / f"noisy{i}.wav"),
                          clean=str(folder / f"clean{i}.wav")))
    return pairs


def test_the_trainer_and_the_chunked_enhancer_run_it(tmp_path):
    cfg = _config(data={"seed": 5, "crop_seconds": 0.5},
                  train={"epochs": 4, "batch_size": 2, "learning_rate": 5e-3,
                         "patience": 4, "grad_clip": 3.0,
                         "out_dir": str(tmp_path / "runs")})
    cfg.model.units = default_units("gtcrn")
    train = _pairs(tmp_path / "train", 6, 0.8, seed=1)
    val = _pairs(tmp_path / "val", 2, 0.8, seed=50)
    run = tmp_path / "runs" / "g"
    with seeded(training_seed(cfg)):
        model = build_model(cfg)
    history = fit_waveform(model, waveform_loader(train, cfg),
                           waveform_loader(val, cfg, shuffle=False), cfg,
                           run, verbose=0)

    assert history["loss"][-1] < history["loss"][0]
    assert history["val_loss"][-1] < history["val_loss"][0]
    state = progress.read_progress(run)
    assert state["status"] == "finished" and state["model"] == "gtcrn"

    enhancer = Enhancer.load(run, chunk_seconds=1.0, overlap_seconds=0.25)
    _, noisy = _tones(2.7, seed=99)
    chunked = enhancer.enhance(noisy)
    whole = Enhancer.load(run, chunk_seconds=None).enhance(noisy)
    assert chunked.shape == whole.shape == noisy.shape
    assert np.all(np.isfinite(chunked))
    # away from the chunk ends the two agree in the first half second,
    # before the first chunk ends and its GRUs are started afresh
    assert np.allclose(chunked[:SR // 2], whole[:SR // 2], atol=1e-5)


@pytest.fixture
def mixed_corpus(tmp_path):
    """Clean tones and white noise at 16 kHz, for ``synth`` to mix."""
    clean_dir, noise_dir = tmp_path / "clean", tmp_path / "noise"
    clean_dir.mkdir()
    noise_dir.mkdir()
    for i in range(4):
        kudio.save_wave(clean_dir / f"utt{i}.wav", _tones(0.8, i)[0], SR)
    kudio.save_wave(noise_dir / "white.wav", (0.3 * np.random.default_rng(
        9).normal(0, 1, SR)).astype(np.float32), SR)
    return tmp_path


def test_it_trains_and_resumes_from_the_command_line(mixed_corpus):
    root = mixed_corpus
    cfg = _config(
        data={"clean_dir": str(root / "clean"),
              "noise_dir": str(root / "noise"),
              "mixed_dir": str(root / "mixed"), "snr_db": [0, 5],
              "seed": 3, "val_split": 0.25, "test_split": 0.25,
              "crop_seconds": 0.4},
        train={"epochs": 1, "batch_size": 2, "patience": 2,
               "out_dir": str(root / "runs")})
    path = str(cfg.to_yaml(root / "gtcrn.yaml"))

    assert main(["synth", "-c", path, "-n", "g"]) == 0
    assert main(["train", "-c", path, "-n", "g"]) == 0
    assert main(["train", "-c", path, "-n", "g", "--resume",
                 "--epochs", "2"]) == 0
    assert progress.read_progress(cfg.progress_path("g"))["epoch"] == 2
    run = cfg.run_dir("g")
    assert Config.from_yaml(run / "config.yaml").model.units == [16]

    tone = _tones(0.6, seed=7)[1]
    out = Enhancer.load(run).enhance(tone)
    assert len(out) == len(tone) and np.all(np.isfinite(out))
    assert math.isfinite(progress.read_progress(run)["val_loss"])
