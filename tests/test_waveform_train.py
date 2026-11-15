# -*- coding: utf-8 -*-
"""The waveform path's trainer: it learns, keeps its best epoch, stops when
patience runs out, and a run interrupted anywhere carries on to the same
last bit as one that never was. progress.json says where a run stands."""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs PyTorch",
                            exc_type=ImportError)

from waveform_stand_in import NAME, SR, register, write_pairs  # noqa: E402

from kudioenhance import pipeline, progress  # noqa: E402
from kudioenhance import waveform_train as wt  # noqa: E402
from kudioenhance.checkpoint import (  # noqa: E402
    load_checkpoint,
    load_training_state,
)
from kudioenhance.cli import main  # noqa: E402
from kudioenhance.config import Config  # noqa: E402
from kudioenhance.inference import Enhancer  # noqa: E402
from kudioenhance.losses import hybrid_loss  # noqa: E402
from kudioenhance.models import build_model, resolve_units  # noqa: E402
from kudioenhance.spectral import ComplexSTFT  # noqa: E402
from kudioenhance.train import seeded, training_seed  # noqa: E402
from kudioenhance.waveform import waveform_loader  # noqa: E402

EPOCHS = 5


class Interrupted(Exception):
    """Stands for the process being killed."""


@pytest.fixture(autouse=True)
def stand_in(monkeypatch):
    register(monkeypatch)


@pytest.fixture(scope="module")
def pairs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tones")
    return (write_pairs(folder / "train", 6, 0.5, seed=1),
            write_pairs(folder / "val", 2, 0.5, seed=50))


def _config(tmp_path, name=NAME, **train) -> Config:
    return resolve_units(Config.from_dict({
        "audio": {"sr": SR, "n_fft": 64, "hop_length": 32},
        "data": {"seed": 11, "crop_seconds": 0.25},
        "model": {"name": name, "dropout": 0.2},
        "train": {"epochs": EPOCHS, "batch_size": 2, "learning_rate": 0.05,
                  "patience": 4, "out_dir": str(tmp_path / "runs"),
                  **train},
    }))


def _fit(cfg, pairs, run_dir, *, validate=True, **kwargs):
    """Build everything as a fresh process would, then train: the model is
    initialised from the training seed, the loaders from their own."""
    train_pairs, val_pairs = pairs
    with seeded(training_seed(cfg)):
        model = build_model(cfg)
    val_loader = (waveform_loader(val_pairs, cfg, shuffle=False)
                  if validate else None)
    history = wt.fit_waveform(model, waveform_loader(train_pairs, cfg),
                              val_loader, cfg, run_dir, verbose=0, **kwargs)
    return model, history


def _without_seconds(history):
    """The curves but the wall time, which no two runs share."""
    return {key: values for key, values in history.items()
            if key != "seconds"}


def _same_weights(first, second) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[key], second[key]) for key in first)


def _same_tree(first, second) -> bool:
    """Equal to the last bit, tensors included, however deeply nested."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return (isinstance(second, dict) and first.keys() == second.keys()
                and all(_same_tree(first[k], second[k]) for k in first))
    if isinstance(first, (list, tuple)):
        return (type(first) is type(second) and len(first) == len(second)
                and all(_same_tree(a, b) for a, b in zip(first, second)))
    return first == second


def _interrupt_after(monkeypatch, epoch: int) -> None:
    """Kill the run once *epoch* is saved and its progress written."""
    real = wt.write_json

    def write(path, payload, **kwargs):
        written = real(path, payload, **kwargs)
        if path.name == progress.PROGRESS_FILE and payload["epoch"] == epoch:
            raise Interrupted
        return written

    monkeypatch.setattr(wt, "write_json", write)


# ------------------------------------------------------------------ training

def test_the_loss_falls_and_the_best_epoch_is_kept(tmp_path, pairs):
    cfg = _config(tmp_path)
    run = tmp_path / "run"
    model, history = _fit(cfg, pairs, run)

    terms = ["complex_mse", "magnitude_mse", "neg_si_snr"]
    assert set(history) == {"loss", *terms, "val_loss",
                            *(f"val_{name}" for name in terms), "lr",
                            "seconds"}
    assert all(len(values) == EPOCHS for values in history.values())
    assert history["val_loss"][-1] < history["val_loss"][0]

    best = int(np.argmin(history["val_loss"])) + 1
    saved = load_checkpoint(run / f"{NAME}.pt")
    assert (saved["model"], saved["target"]) == (NAME, cfg.model.target)
    assert _same_weights(saved["state_dict"], model.state_dict())
    assert progress.read_progress(run)["best_epoch"] == best
    assert not model.training


def test_the_logged_validation_loss_is_the_hybrid_loss(tmp_path, pairs):
    """What the curves say about the kept epoch, recomputed from its
    weights: the terms were weighted and averaged over the examples."""
    cfg = _config(tmp_path)
    model, history = _fit(cfg, pairs, tmp_path / "run")

    stft = ComplexSTFT.from_config(cfg.audio)
    total, count = 0.0, 0
    with torch.no_grad():
        for noisy, clean in waveform_loader(pairs[1], cfg, shuffle=False):
            estimate = wt.estimate_spectrum(model, stft, noisy)
            total += hybrid_loss(estimate, clean, stft, cfg.loss).item() \
                * len(noisy)
            count += len(noisy)
    assert total / count == pytest.approx(min(history["val_loss"]), rel=1e-6)


def test_training_stops_once_patience_runs_out(tmp_path, pairs):
    cfg = _config(tmp_path, name="idle", epochs=20, patience=3)
    _, history = _fit(cfg, pairs, tmp_path / "run")

    assert len(history["val_loss"]) == 1 + 3
    assert len(set(history["val_loss"])) == 1
    report = progress.read_progress(tmp_path / "run")
    assert report["status"] == "stopped"
    assert report["epochs_without_improvement"] == 3
    assert report["eta_seconds"] == 0.0


def test_the_learning_rate_halves_on_a_plateau(tmp_path, pairs):
    """patience 4 halves the rate after two epochs without progress."""
    cfg = _config(tmp_path, name="idle", epochs=20, patience=4,
                  learning_rate=0.01)
    _, history = _fit(cfg, pairs, tmp_path / "run")

    assert history["lr"] == pytest.approx([0.01] * 3 + [0.005] * 2)


def test_without_validation_the_training_loss_is_watched(tmp_path, pairs):
    cfg = _config(tmp_path)
    _, history = _fit(cfg, pairs, tmp_path / "run", validate=False)

    assert "val_loss" not in history
    report = progress.read_progress(tmp_path / "run")
    assert report["monitor"] == "loss"
    assert report["val_loss"] is None and report["val_terms"] is None


def test_the_callers_random_state_is_left_alone(tmp_path, pairs,
                                                monkeypatch):
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    states = (torch.get_rng_state(), np.random.get_state()[1].copy(),
              random.getstate())
    _interrupt_after(monkeypatch, 2)
    with pytest.raises(Interrupted):
        _fit(_config(tmp_path), pairs, tmp_path / "run")
    monkeypatch.setattr(wt, "write_json", progress.write_json)
    _fit(_config(tmp_path), pairs, tmp_path / "run", resume=True)

    assert torch.equal(torch.get_rng_state(), states[0])
    assert np.array_equal(np.random.get_state()[1], states[1])
    assert random.getstate() == states[2]


# -------------------------------------------------------------------- resume

@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory, pairs):
    """The run every resumed one has to equal: EPOCHS epochs in one go."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        register(monkeypatch)
        tmp_path = tmp_path_factory.mktemp("uninterrupted")
        model, history = _fit(_config(tmp_path), pairs, tmp_path / "run")
        return {"model": model.state_dict(), "history": history,
                "state": load_training_state(tmp_path / "run" / "resume.pt"),
                "best": load_checkpoint(tmp_path / "run" / f"{NAME}.pt")}


def _assert_same_run(run, model, history, uninterrupted) -> None:
    assert _without_seconds(history) == \
        _without_seconds(uninterrupted["history"])
    assert _same_weights(model.state_dict(), uninterrupted["model"])
    assert _same_weights(load_checkpoint(run / f"{NAME}.pt")["state_dict"],
                         uninterrupted["best"]["state_dict"])
    state, reference = (load_training_state(run / "resume.pt"),
                        uninterrupted["state"])
    for key in ("epoch", "state_dict", "optimizer", "scheduler", "rng",
                "best", "best_epoch", "best_state", "wait"):
        assert _same_tree(state[key], reference[key]), key


@pytest.mark.parametrize("after", [1, 2, 4])
def test_a_resumed_run_is_the_uninterrupted_one_to_the_last_bit(
        tmp_path, pairs, monkeypatch, uninterrupted, after):
    """Dropout, the order of the crops and where each is cut, Adam's
    moments and the schedule all carry on from where they were."""
    run = tmp_path / "run"
    _interrupt_after(monkeypatch, after)
    with pytest.raises(Interrupted):
        _fit(_config(tmp_path), pairs, run)
    assert load_training_state(run / "resume.pt")["epoch"] == after
    monkeypatch.setattr(wt, "write_json", progress.write_json)

    model, history = _fit(_config(tmp_path), pairs, run, resume=True)
    _assert_same_run(run, model, history, uninterrupted)


def test_an_epoch_cut_short_leaves_no_trace(tmp_path, pairs, monkeypatch,
                                            uninterrupted):
    """Killed in the middle of epoch 3, the run resumes from epoch 2's end
    and the half-trained steps are as if they never happened."""
    run = tmp_path / "run"
    real, calls = wt.loss_terms, []

    def loss_terms(*args, **kwargs):
        calls.append(None)
        # three training and one validation batch an epoch
        if len(calls) == 2 * 4 + 2:
            raise Interrupted
        return real(*args, **kwargs)

    monkeypatch.setattr(wt, "loss_terms", loss_terms)
    with pytest.raises(Interrupted):
        _fit(_config(tmp_path), pairs, run)
    assert load_training_state(run / "resume.pt")["epoch"] == 2
    monkeypatch.setattr(wt, "loss_terms", real)

    model, history = _fit(_config(tmp_path), pairs, run, resume=True)
    _assert_same_run(run, model, history, uninterrupted)


def test_workers_change_nothing_across_a_resume(tmp_path, pairs, monkeypatch,
                                                uninterrupted):
    """Two persistent workers, killed and resumed: still the run that one
    process trained in one go. Slow on Windows, where each worker is a new
    process that imports torch."""
    run = tmp_path / "run"
    _interrupt_after(monkeypatch, 2)
    with pytest.raises(Interrupted):
        _fit(_config(tmp_path, num_workers=2), pairs, run)
    monkeypatch.setattr(wt, "write_json", progress.write_json)

    model, history = _fit(_config(tmp_path, num_workers=2), pairs, run,
                          resume=True)
    _assert_same_run(run, model, history, uninterrupted)


def test_a_finished_run_is_extended_by_more_epochs(tmp_path, pairs,
                                                   uninterrupted):
    run = tmp_path / "run"
    _fit(_config(tmp_path), pairs, run, epochs=3)
    assert progress.read_progress(run)["status"] == "finished"

    model, history = _fit(_config(tmp_path), pairs, run, resume=True)
    _assert_same_run(run, model, history, uninterrupted)


def test_resuming_with_nothing_left_changes_nothing(tmp_path, pairs):
    run = tmp_path / "run"
    first_model, first = _fit(_config(tmp_path), pairs, run)
    model, history = _fit(_config(tmp_path), pairs, run, resume=True)

    assert history == first
    assert _same_weights(model.state_dict(), first_model.state_dict())
    assert progress.read_progress(run)["status"] == "finished"


def test_a_run_out_of_patience_stays_stopped_unless_given_more(tmp_path,
                                                               pairs):
    run = tmp_path / "run"
    cfg = _config(tmp_path, name="idle", epochs=20, patience=2)
    _, first = _fit(cfg, pairs, run)
    _, again = _fit(cfg, pairs, run, resume=True)
    assert again == first
    assert progress.read_progress(run)["status"] == "stopped"

    more = _config(tmp_path, name="idle", epochs=20, patience=4)
    _, longer = _fit(more, pairs, run, resume=True)
    assert len(longer["loss"]) == len(first["loss"]) + 2


def test_a_patience_of_zero_resumes_like_any_other(tmp_path, pairs,
                                                   monkeypatch):
    """Zero stops at the first epoch without progress; an epoch that
    improved, which the first always does, is not one."""
    _, straight = _fit(_config(tmp_path / "a", patience=0), pairs,
                       tmp_path / "a" / "run")
    assert len(straight["loss"]) >= 2

    run = tmp_path / "b" / "run"
    _interrupt_after(monkeypatch, 1)
    with pytest.raises(Interrupted):
        _fit(_config(tmp_path / "b", patience=0), pairs, run)
    monkeypatch.setattr(wt, "write_json", progress.write_json)
    _, history = _fit(_config(tmp_path / "b", patience=0), pairs, run,
                      resume=True)
    assert _without_seconds(history) == _without_seconds(straight)


def test_a_setting_added_since_the_run_started_takes_its_default(tmp_path):
    cfg = _config(tmp_path)
    saved = cfg.to_dict()
    del saved["loss"]["zero_mean"]
    wt._check_resumable(saved, cfg, tmp_path / "resume.pt")

    cfg.loss.zero_mean = True
    with pytest.raises(ValueError, match=r"loss\.zero_mean False -> True"):
        wt._check_resumable(saved, cfg, tmp_path / "resume.pt")


def test_resuming_needs_a_state_to_resume(tmp_path, pairs):
    with pytest.raises(FileNotFoundError, match="resume"):
        _fit(_config(tmp_path), pairs, tmp_path / "run", resume=True)


def test_a_run_is_not_started_over_by_accident(tmp_path, pairs):
    run = tmp_path / "run"
    _fit(_config(tmp_path), pairs, run, epochs=1)
    before = (run / "resume.pt").read_bytes()

    with pytest.raises(FileExistsError, match="resume it"):
        _fit(_config(tmp_path), pairs, run)
    assert (run / "resume.pt").read_bytes() == before


def test_resuming_under_another_config_says_what_changed(tmp_path, pairs):
    run = tmp_path / "run"
    _fit(_config(tmp_path), pairs, run, epochs=1)

    with pytest.raises(ValueError, match=r"train\.learning_rate 0\.05 -> 0\.01"):
        _fit(_config(tmp_path, learning_rate=0.01), pairs, run, resume=True)
    with pytest.raises(ValueError, match="resumed without one"):
        _fit(_config(tmp_path), pairs, run, resume=True, validate=False)
    # stopping later, or with more patience, is still the same run
    _fit(_config(tmp_path, patience=6), pairs, run, resume=True, epochs=2)


def test_the_training_state_loads_with_weights_only(tmp_path, pairs):
    run = tmp_path / "run"
    _fit(_config(tmp_path), pairs, run, epochs=2)

    state = torch.load(run / "resume.pt", weights_only=True)
    assert state["format"] == "kudioenhance/training-state"
    assert state["epoch"] == 2
    assert set(state["rng"]) >= {"torch", "numpy", "python", "sampler"}
    assert not list(run.glob("*.partial"))


def test_a_running_run_loads_into_an_enhancer(tmp_path, pairs, monkeypatch):
    """Its best epoch so far: config.yaml is written before the first one."""
    run = tmp_path / "run"
    _interrupt_after(monkeypatch, 2)
    with pytest.raises(Interrupted):
        _fit(_config(tmp_path), pairs, run)

    enhancer = Enhancer.load(run)
    clean = enhancer.enhance(np.zeros(SR // 2, dtype=np.float32) + 0.1)
    assert len(clean) == SR // 2 and np.all(np.isfinite(clean))


# ------------------------------------------------------------------ progress

def test_progress_is_rewritten_after_every_epoch(tmp_path, pairs,
                                                 monkeypatch):
    seen = []
    real = wt.write_json

    def write(path, payload, **kwargs):
        written = real(path, payload, **kwargs)
        if path.name == progress.PROGRESS_FILE:
            seen.append(progress.read_progress(path))
        return written

    monkeypatch.setattr(wt, "write_json", write)
    run = tmp_path / "run"
    _, history = _fit(_config(tmp_path), pairs, run)

    assert [report["epoch"] for report in seen] == list(range(EPOCHS + 1))
    assert [report["status"] for report in seen] == \
        ["training"] * EPOCHS + ["finished"]
    before = seen.pop(0)
    assert before["loss"] is None and before["best"] is None
    assert before["eta_seconds"] is None and before["pid"] == os.getpid()
    for k, report in enumerate(seen, start=1):
        assert report["loss"] == history["loss"][k - 1]
        assert report["terms"] == {
            name: history[name][k - 1]
            for name in ("complex_mse", "magnitude_mse", "neg_si_snr")}
        assert report["val_loss"] == history["val_loss"][k - 1]
        assert report["val_terms"]["neg_si_snr"] == \
            history["val_neg_si_snr"][k - 1]
        assert report["best"] == min(history["val_loss"][:k])
        assert report["elapsed_seconds"] == pytest.approx(
            sum(history["seconds"][:k]))
        assert report["eta_seconds"] == pytest.approx(
            progress.remaining_seconds(history["seconds"][:k], EPOCHS - k))
        assert report["eta_seconds_min"] <= report["eta_seconds"]
        assert report["run"] == "run" and report["model"] == NAME
    assert seen[-1] == progress.read_progress(run)
    assert json.loads((run / "history.json").read_text()) == history
    assert not list(run.glob("*.partial"))


def test_a_progress_file_that_stays_locked_does_not_stop_training(
        tmp_path, pairs, monkeypatch, caplog):
    real = progress.os.replace

    def replace(src, dst):
        if str(dst).endswith(progress.PROGRESS_FILE):
            raise PermissionError(13, "in use")
        real(src, dst)

    monkeypatch.setattr(progress, "REPLACE_WAIT", 0)
    monkeypatch.setattr(progress.os, "replace", replace)
    run = tmp_path / "run"
    _, history = _fit(_config(tmp_path), pairs, run, epochs=2)

    assert len(history["loss"]) == 2
    assert load_training_state(run / "resume.pt")["epoch"] == 2
    assert not (run / progress.PROGRESS_FILE).exists()
    assert "could not replace" in caplog.text


def test_a_resumed_sitting_says_so_before_its_first_epoch(tmp_path, pairs,
                                                         monkeypatch):
    """Not the finished run it extends, and not the dead process."""
    run = tmp_path / "run"
    _, first = _fit(_config(tmp_path), pairs, run, epochs=3)
    (run / "history.json").write_text("{}")
    seen = []
    real = wt.write_json

    def write(path, payload, **kwargs):
        if path.name == progress.PROGRESS_FILE and not seen:
            seen.append(dict(payload))
            seen.append(json.loads((run / "history.json").read_text()))
        return real(path, payload, **kwargs)

    monkeypatch.setattr(wt, "write_json", write)
    _fit(_config(tmp_path), pairs, run, resume=True)

    report, history = seen
    assert (report["status"], report["epoch"], report["epochs"]) == \
        ("training", 3, EPOCHS)
    assert report["pid"] == os.getpid()
    assert report["eta_seconds"] == pytest.approx(
        progress.remaining_seconds(first["seconds"], EPOCHS - 3))
    assert history == first


def test_nothing_left_still_brings_the_curves_up_to_date(tmp_path, pairs):
    """A run that died after its last state but before its last curves."""
    run = tmp_path / "run"
    _, first = _fit(_config(tmp_path), pairs, run)
    short = {key: values[:-1] for key, values in first.items()}
    (run / "history.json").write_text(json.dumps(short))

    _, history = _fit(_config(tmp_path), pairs, run, resume=True)
    assert history == first
    assert json.loads((run / "history.json").read_text()) == first


def test_a_loss_that_is_not_a_number_is_null_in_progress(tmp_path):
    cfg = _config(tmp_path)
    history = {"loss": [math.nan], "complex_mse": [math.inf],
               "magnitude_mse": [0.5], "neg_si_snr": [-3.0], "lr": [0.05],
               "seconds": [2.0]}
    report = wt._progress(
        run_dir=tmp_path, cfg=cfg, status="training", epoch=1, epochs=3,
        history=history, monitor="loss", best=math.inf, best_epoch=0, wait=1,
        patience=4, elapsed=2.0, started=progress.timestamp())

    assert report["loss"] is None and report["best"] is None
    assert report["terms"] == {"complex_mse": None, "magnitude_mse": 0.5,
                               "neg_si_snr": -3.0}
    json.dumps(report, allow_nan=False)


def test_a_best_checkpoint_held_open_by_a_reader_is_retried(tmp_path,
                                                            monkeypatch):
    """Loading a run that is still training must not kill it."""
    from kudioenhance.checkpoint import save_checkpoint

    real, refused = progress.os.replace, []

    def replace(src, dst):
        if str(dst).endswith(".pt") and len(refused) < 3:
            refused.append(dst)
            raise PermissionError(13, "in use")
        real(src, dst)

    monkeypatch.setattr(progress, "REPLACE_WAIT", 0)
    monkeypatch.setattr(progress.os, "replace", replace)
    path = save_checkpoint(tmp_path / "m.pt", {"w": torch.ones(2)},
                           model=NAME, target="spectrum")
    assert torch.equal(load_checkpoint(path)["state_dict"]["w"],
                       torch.ones(2))
    assert len(refused) == 3


# ---------------------------------------------------------- pipeline and CLI

@pytest.fixture
def waveform_config(config):
    """conftest's synthesized corpus, trained on the waveform path."""
    config.model.name = NAME
    config.model.units = None
    config.model.dropout = 0.1
    config.data.crop_seconds = 0.3
    config.train.batch_size = 2
    config.train.epochs = 2
    return config


def test_the_pipeline_trains_and_resumes_a_waveform_model(waveform_config,
                                                          tone):
    cfg = waveform_config
    pipeline.synthesize(cfg, "exp1")
    pipeline.train(cfg, "exp1", verbose=0)
    run = cfg.run_dir("exp1")
    for name in ("config.yaml", f"{NAME}.pt", "resume.pt", "progress.json",
                 "history.json"):
        assert (run / name).is_file(), name
    assert not cfg.stats_path("exp1").exists()
    assert Config.from_yaml(run / "config.yaml").model.units == [1]

    history = pipeline.train(cfg, "exp1", epochs=3, verbose=0, resume=True)
    assert len(history["loss"]) == 3
    assert progress.read_progress(cfg.progress_path("exp1"))["epoch"] == 3

    clean = Enhancer.load(run).enhance(tone)
    assert len(clean) == len(tone) and np.all(np.isfinite(clean))


def test_a_waveform_run_is_scored_like_any_other(waveform_config):
    cfg = waveform_config
    pipeline.synthesize(cfg, "exp1")
    pipeline.train(cfg, "exp1", epochs=1, verbose=0)
    rows, summary = pipeline.evaluate(cfg, "exp1")

    assert rows and "delta_si_sdr" in summary
    assert len(list(cfg.enhanced_dir("exp1").glob("*.wav"))) == len(rows)


def test_run_does_not_mix_over_a_run_in_progress(waveform_config):
    cfg = waveform_config
    pipeline.synthesize(cfg, "exp1")
    pipeline.train(cfg, "exp1", epochs=1, verbose=0)
    manifest = cfg.run_dir("exp1") / "train.json"
    before = (manifest.read_bytes(), manifest.stat().st_mtime_ns)

    with pytest.raises(FileExistsError, match="resume it"):
        pipeline.run(cfg, "exp1")
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before


def test_the_log_power_path_cannot_resume(config):
    with pytest.raises(ValueError, match="cannot resume"):
        pipeline.train(config, "exp1", resume=True)


def test_train_resume_on_the_command_line(tmp_path, waveform_config,
                                          capsys):
    path = str(waveform_config.to_yaml(tmp_path / "config.yaml"))
    assert main(["synth", "-c", path, "-n", "exp1"]) == 0
    assert main(["train", "-c", path, "-n", "exp1", "--epochs", "1"]) == 0

    assert main(["train", "-c", path, "-n", "exp1"]) == 1
    assert "resume it" in capsys.readouterr().err
    assert main(["train", "-c", path, "-n", "exp1", "--resume"]) == 0
    assert progress.read_progress(
        waveform_config.progress_path("exp1"))["epoch"] == 2
    assert main(["train", "-c", path, "-n", "exp2", "--resume"]) == 1

    waveform_config.train.learning_rate = 0.5
    changed = str(waveform_config.to_yaml(tmp_path / "changed.yaml"))
    capsys.readouterr()
    assert main(["train", "-c", changed, "-n", "exp1", "--resume"]) == 1
    assert "train.learning_rate" in capsys.readouterr().err
