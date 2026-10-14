# -*- coding: utf-8 -*-
"""The training loop: what it keeps, when it stops, and that it repeats."""
from __future__ import annotations

import copy
import logging

import numpy as np
import pytest

from kudio_enhance.models import build_model
from kudio_enhance.train import (
    check_batch_size,
    compile_model,
    fit,
    load_history,
    save_history,
    seeded,
    training_seed,
)


@pytest.fixture
def arrays(config):
    """Frame-wise training and validation rows for the conftest ``ddae``."""
    rng = np.random.default_rng(0)
    width = config.audio.n_bins * (2 * config.model.context + 1)
    x = rng.standard_normal((200, width)).astype(np.float32)
    x_val = rng.standard_normal((40, width)).astype(np.float32)
    # a target the network can actually learn: the frame at the centre
    centre = slice(config.audio.n_bins * config.model.context,
                   config.audio.n_bins * (config.model.context + 1))
    return x, 0.5 * x[:, centre], x_val, 0.5 * x_val[:, centre]


def _val_loss(torch, model, x, y) -> float:
    model.eval()
    with torch.no_grad():
        return float(((model(torch.from_numpy(x))
                       - torch.from_numpy(y)) ** 2).mean())


def test_fit_returns_the_curves(config, arrays, tmp_path, torch):
    x, y, x_val, y_val = arrays
    history = fit(build_model(config), x, y, config, tmp_path / "m.pt",
                  validation_data=(x_val, y_val), epochs=3, verbose=0)

    assert list(history) == ["loss", "mae", "val_loss", "val_mae", "lr"]
    assert all(len(values) == 3 for values in history.values())
    assert all(np.isfinite(values).all() for values in history.values())
    assert history["loss"][-1] < history["loss"][0]

    without = fit(build_model(config), x, y, config, tmp_path / "n.pt",
                  epochs=2, verbose=0)
    assert list(without) == ["loss", "mae", "lr"]


def test_the_curves_survive_a_round_trip_through_json(config, arrays,
                                                      tmp_path, torch):
    x, y, _, _ = arrays
    history = fit(build_model(config), x, y, config, tmp_path / "m.pt",
                  epochs=2, verbose=0)
    assert load_history(save_history(tmp_path / "h.json", history)) == history


def test_early_stopping_puts_the_best_weights_back(config, tmp_path, torch):
    """Training pulls the output towards 2 while validation wants 0.5, so the
    validation loss falls while the output passes 0.5 and climbs after. The
    model handed back has to be the one from the bottom of that curve, not
    the last one trained — and so does the file on disk."""
    from kudio_enhance.checkpoint import load_checkpoint

    rng = np.random.default_rng(0)
    width = config.audio.n_bins * (2 * config.model.context + 1)
    bins = config.audio.n_bins
    x = rng.standard_normal((256, width)).astype(np.float32)
    x_val = rng.standard_normal((64, width)).astype(np.float32)
    y = np.full((256, bins), 2.0, dtype=np.float32)
    y_val = np.full((64, bins), 0.5, dtype=np.float32)

    config.model.units = [8]
    config.train.learning_rate = 2e-3
    config.train.patience = 2
    with seeded(0):
        model = build_model(config)
    path = tmp_path / "m.pt"
    history = fit(model, x, y, config, path, validation_data=(x_val, y_val),
                  epochs=40, verbose=0)

    curve = history["val_loss"]
    best = int(np.argmin(curve))
    assert 0 < best < len(curve) - 1                 # it fell, then rose
    assert len(curve) == best + 1 + config.train.patience < 40
    assert _val_loss(torch, model, x_val, y_val) == pytest.approx(
        curve[best], rel=1e-5)

    reloaded = build_model(config)
    reloaded.load_state_dict(load_checkpoint(path)["state_dict"])
    assert _val_loss(torch, reloaded, x_val, y_val) == pytest.approx(
        curve[best], rel=1e-5)


def test_patience_counts_epochs_without_improvement(config, arrays,
                                                     tmp_path, torch):
    """Not epochs in total: with no patience at all, a run that keeps
    improving still runs to the end, and stops at the first epoch that
    does not."""
    x, y, x_val, y_val = arrays
    config.train.patience = 0
    history = fit(build_model(config), x, y, config, tmp_path / "m.pt",
                  validation_data=(x_val, y_val), epochs=3, verbose=0)

    curve = history["val_loss"]
    assert len(curve) == 3
    assert curve[0] > curve[1] > curve[2]


def _stuck(torch, bins: int):
    """A model that cannot learn — its only weight is multiplied by zero —
    so every epoch's loss is exactly the first one's: a plateau on demand."""
    class Stuck(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return 0.0 * self.weight * x[:, :bins]
    return Stuck()


@pytest.mark.parametrize("learning_rate, patience, expected", [
    # halved after patience // 2 flat epochs, stopped after patience of them
    (1e-3, 4, [1e-3, 1e-3, 1e-3, 5e-4, 5e-4]),
    # ...and never taken below 1e-6
    (2e-6, 3, [2e-6, 2e-6, 1e-6, 1e-6]),
])
def test_the_learning_rate_is_halved_on_a_plateau(config, tmp_path, torch,
                                                  learning_rate, patience,
                                                  expected):
    """Keras's schedule, epoch for epoch: torch's own ReduceLROnPlateau waits
    one epoch longer for the same ``patience``."""
    bins = config.audio.n_bins
    x = np.ones((64, bins * (2 * config.model.context + 1)), dtype=np.float32)
    y = np.zeros((64, bins), dtype=np.float32)
    config.train.learning_rate = learning_rate
    config.train.patience = patience

    history = fit(_stuck(torch, bins), x, y, config, tmp_path / "m.pt",
                  epochs=50, verbose=0)
    assert history["loss"] == [0.0] * len(expected)
    assert history["lr"] == pytest.approx(expected)


def test_a_seeded_run_reproduces_its_history_exactly(config, arrays,
                                                     tmp_path, torch):
    """Same starting weights, same seed: the same shuffles and the same
    dropout masks, so the same numbers to the last bit. A different seed has
    to change them, or the seed is not what is driving it."""
    x, y, x_val, y_val = arrays
    config.model.dropout = 0.2
    with seeded(0):
        start = build_model(config)

    def train(seed):
        config.train.seed = seed
        return fit(copy.deepcopy(start), x, y, config, tmp_path / "m.pt",
                   validation_data=(x_val, y_val), epochs=3, verbose=0)

    first = train(5)
    assert train(5) == first
    assert train(6) != first


def test_fit_leaves_the_callers_random_state_alone(config, arrays, tmp_path,
                                                   torch):
    x, y, _, _ = arrays
    model = build_model(config)
    before = torch.get_rng_state()
    fit(model, x, y, config, tmp_path / "m.pt", epochs=2, verbose=0)
    assert torch.equal(torch.get_rng_state(), before)


def test_the_training_seed_falls_back_to_the_data_seed(config):
    assert config.train.seed is None
    assert training_seed(config) == config.data.seed == 3
    config.train.seed = 11
    assert training_seed(config) == 11


def test_a_lone_last_example_does_not_break_batch_norm(config, arrays,
                                                       tmp_path, torch):
    """33 rows in batches of 32 would leave BatchNorm one example to take a
    variance from, which torch refuses in training mode."""
    x, y, _, _ = arrays
    history = fit(build_model(config), x[:33], y[:33], config,
                  tmp_path / "m.pt", epochs=1, verbose=0)
    assert np.isfinite(history["loss"][0])


def test_a_frame_wise_model_refuses_batches_of_one_by_name(config, arrays,
                                                           tmp_path, torch):
    """Every batch would be a lone example, which BatchNorm cannot take a
    variance from. Refused before a weight moves or a file is written, and
    in words, not as torch's error from inside the first batch."""
    x, y, _, _ = arrays
    config.train.batch_size = 1
    with pytest.raises(ValueError, match="train.batch_size to 2 or more"):
        check_batch_size(config)
    with pytest.raises(ValueError, match="ddae normalises every batch"):
        fit(build_model(config), x[:10], y[:10], config, tmp_path / "m.pt",
            epochs=1, verbose=0)
    assert not (tmp_path / "m.pt").exists()

    config.train.batch_size = 2
    check_batch_size(config)


def test_a_training_set_of_one_example_is_refused_by_name(config, arrays,
                                                          tmp_path, torch):
    """No batch before it for the lone example to join."""
    x, y, _, _ = arrays
    with pytest.raises(ValueError, match="single example"):
        fit(build_model(config), x[:1], y[:1], config, tmp_path / "m.pt",
            epochs=1, verbose=0)
    assert not (tmp_path / "m.pt").exists()


def test_a_sequence_model_still_trains_on_batches_of_one(config, tmp_path,
                                                         torch):
    """Its BatchNorm takes the frames of the window too, so one window is
    enough, as it was under Keras."""
    config.model.name = "conv_ae"
    config.model.units = [8]
    config.train.batch_size = 1
    check_batch_size(config)

    rng = np.random.default_rng(0)
    shape = (3, config.model.n_frames, config.audio.n_bins)
    x = rng.standard_normal(shape).astype(np.float32)
    history = fit(build_model(config), x, 0.5 * x, config, tmp_path / "m.pt",
                  epochs=1, verbose=0)
    assert np.isfinite(history["loss"][0])


def test_verbose_logs_one_line_per_epoch_and_zero_logs_nothing(
        config, arrays, tmp_path, torch, caplog):
    x, y, _, _ = arrays
    config.train.patience = 10
    with caplog.at_level(logging.INFO, logger="kudio_enhance"):
        fit(build_model(config), x, y, config, tmp_path / "m.pt", epochs=3,
            verbose=0)
    assert not [r for r in caplog.records if r.name == "kudio_enhance.train"]

    with caplog.at_level(logging.INFO, logger="kudio_enhance"):
        fit(build_model(config), x, y, config, tmp_path / "m.pt", epochs=3,
            verbose=1)
    lines = [r.getMessage() for r in caplog.records
             if r.name == "kudio_enhance.train"
             and r.getMessage().startswith("epoch")]
    assert [line.split()[1] for line in lines] == ["1/3", "2/3", "3/3"]
    assert all("loss" in line and "lr" in line for line in lines)


def test_auto_picks_the_cpu_without_a_gpu(torch, monkeypatch):
    """Only CUDA counts: a Mac whose GPU torch can drive through ``mps``
    still trains on the CPU, as the README says."""
    from kudio_enhance.train import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError, match="cannot see a GPU"):
        resolve_device("cuda")


def test_a_dataset_trains_exactly_as_its_arrays_do(config, tmp_path, torch):
    """`fit` on the in-memory dataset — context stacked a batch at a time —
    is the same run as `fit` on the stacked matrix: the same shuffles, the
    same curves, the same weights to the last bit."""
    from kudio_enhance.data import build_arrays, build_dataset, synthesize

    config.model.dropout = 0.2
    pairs = synthesize(config, "exp1")
    train, val = pairs[:3], pairs[3:]
    x, y, std = build_arrays(train, config, sequence=False, fit=True)
    x_val, y_val, _ = build_arrays(val, config, sequence=False,
                                   standardizer=std)
    train_set, _ = build_dataset(train, config, sequence=False, fit=True)
    val_set, _ = build_dataset(val, config, sequence=False, standardizer=std)
    with seeded(0):
        start = build_model(config)

    def run(training, targets, validation, path):
        model = copy.deepcopy(start)
        history = fit(model, training, targets, config, path,
                      validation_data=validation, epochs=3, verbose=0)
        return history, model.state_dict()

    arrays, weights = run(x, y, (x_val, y_val), tmp_path / "a.pt")
    dataset, dataset_weights = run(train_set, None, val_set,
                                   tmp_path / "d.pt")
    assert dataset == arrays
    assert all(torch.equal(weights[key], dataset_weights[key])
               for key in weights)


def test_fit_wants_targets_beside_arrays_and_none_beside_a_dataset(
        config, arrays, tmp_path, torch):
    from kudio_enhance.datasets import InMemoryDataset

    x, y, _, _ = arrays
    model = build_model(config)
    with pytest.raises(TypeError, match="targets beside them"):
        fit(model, x, None, config, tmp_path / "m.pt", verbose=0)
    with pytest.raises(TypeError, match="carries its own targets"):
        fit(model, InMemoryDataset(x, y), y, config, tmp_path / "m.pt",
            verbose=0)


def test_compile_model_is_a_deprecated_no_op(config):
    """Kept so 0.1-era scripts still run; the optimiser now lives in fit."""
    model = object()
    with pytest.warns(DeprecationWarning, match="fit"):
        assert compile_model(model, config) is model
