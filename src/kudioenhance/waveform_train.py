# -*- coding: utf-8 -*-
"""The waveform path's training loop, which can stop and carry on.

The log-power path's :func:`kudioenhance.train.fit` trains in one sitting,
and a run stopped halfway leaves only its best weights. The models of the
waveform path are trained for days on the CPU, so :func:`fit_waveform`
saves everything it needs after every epoch and ``resume=True`` carries on
from there: on the CPU the resumed run is the uninterrupted one to the last
bit, and a test holds it to that.

What a model of this path does is fixed by :func:`estimate_spectrum`: it is
handed the noisy spectrum as ``ComplexSTFT.forward`` gives it, ``(batch,
bins, frames)`` complex and compressed to ``audio.compress``, and gives back
a spectrum of the same shape on the same scale. The STFT stays outside the
model, so the loss is taken on the spectrum the model put out, as GTCRN's
is (:func:`kudioenhance.losses.loss_terms`), and inference can cut a long
file into chunks without the model knowing.

A run directory holds, beside ``config.yaml``:

``<model>.pt``
    the best epoch's weights so far, in the format of
    :func:`~kudioenhance.checkpoint.save_checkpoint`, so
    :class:`~kudioenhance.inference.Enhancer` loads a run that is still
    training.
``resume.pt``
    the latest epoch's training state
    (:func:`~kudioenhance.checkpoint.save_training_state`).
``history.json``, ``progress.json``
    the curves, and where the run stands (:mod:`kudioenhance.progress`).

Importing this module imports torch.
"""
from __future__ import annotations

import contextlib
import copy
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch

from kudioenhance.checkpoint import (
    load_training_state,
    save_checkpoint,
    save_training_state,
)
from kudioenhance.config import Config
from kudioenhance.losses import loss_terms, weighted_sum
from kudioenhance.progress import (
    PROGRESS_FILE,
    RESUME_FILE,
    remaining_seconds,
    timestamp,
    write_json,
)
from kudioenhance.spectral import ComplexSTFT
from kudioenhance.train import (
    LR_MIN_DELTA,
    MIN_LR,
    resolve_device,
    training_seed,
)

log = logging.getLogger(__name__)

__all__ = ["estimate_spectrum", "fit_waveform", "RESUMABLE_CHANGES"]

#: The settings a resumed run may have changed. Anything else in the config
#: has to be what the run started with: a resumed run that trained on other
#: data, another loss or another learning rate would be a new run started
#: from these weights, and its ``config.yaml`` would describe neither.
#: Stopping later, or with more patience, and running on other hardware are
#: still the same run.
RESUMABLE_CHANGES = ("train.epochs", "train.patience", "train.num_workers",
                     "train.device", "train.out_dir")


def estimate_spectrum(model, stft: ComplexSTFT,
                      noisy: torch.Tensor) -> torch.Tensor:
    """The enhanced spectrum, on the linear scale, of *noisy* waveforms
    ``(batch, samples)``.

    ``stft.expand_spectrum(model(stft(noisy)))``: the model sees the spectrum
    compressed to ``audio.compress`` and answers on that scale. Training
    and inference both go through here, so they cannot disagree on it.
    """
    return stft.expand_spectrum(model(stft(noisy)))


# ------------------------------------------------------------------ random

def _numpy_state() -> Dict[str, Any]:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"name": name, "keys": torch.from_numpy(keys.astype(np.int64)),
            "position": int(position), "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached)}


def _set_numpy_state(state: Dict[str, Any]) -> None:
    np.random.set_state((state["name"],
                         state["keys"].numpy().astype(np.uint32),
                         state["position"], state["has_gauss"],
                         state["cached_gaussian"]))


def _rng_state(sampler_generator: Optional[torch.Generator]
               ) -> Dict[str, Any]:
    """Every generator a training epoch can draw from.

    torch's global one draws dropout masks, the sampler's draws the order
    and where each crop is cut, and NumPy's and Python's are kept as well,
    for anything a model of the caller's might draw from them. A GPU's are
    kept when there is one, but GPU kernels are not bitwise reproducible by
    default, so a run on one resumes with the same dropout masks and not
    necessarily the same weights.

    The loader's own generator is left out. A loader with persistent
    workers draws their base seed from it once, when its first epoch
    starts, and the fresh loader of a resumed run draws the same seed from
    the same initial state. Put back as it was after epoch k, it would hand
    the resumed workers a different one. Without workers that draw is not
    used. The workers' own random state cannot be saved from here; they
    draw nothing at present, and only read and cut where they are told.
    """
    state = {"torch": torch.get_rng_state(), "numpy": _numpy_state(),
             "python": random.getstate(),
             "sampler": (None if sampler_generator is None
                         else sampler_generator.get_state())}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Dict[str, Any],
                   sampler_generator: Optional[torch.Generator]) -> None:
    torch.set_rng_state(state["torch"])
    _set_numpy_state(state["numpy"])
    random.setstate(state["python"])
    if sampler_generator is not None and state["sampler"] is not None:
        sampler_generator.set_state(state["sampler"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextlib.contextmanager
def _forked_rng() -> Iterator[None]:
    """Give the caller's torch, NumPy and Python generators back as they
    were. A resumed run sets all three, and a library call that leaves
    them changed is a surprise waiting for someone."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


# ------------------------------------------------------------------ resume

def _flatten(tree: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def _check_resumable(saved: Dict[str, Any], cfg: Config, path: Path) -> None:
    """Refuse to resume under a config that differs from the run's in
    anything but :data:`RESUMABLE_CHANGES`.

    A setting the run's config does not have was added after it started,
    and its default is what the run trained with; one that has gone since
    cannot be compared, and is not.
    """
    before, now = _flatten(saved), _flatten(cfg.to_dict())
    defaults = _flatten(Config().to_dict())
    for key in set(now) - set(before):
        before[key] = defaults.get(key)
    changed = sorted(key for key in now
                     if before[key] != now[key]
                     and key not in RESUMABLE_CHANGES)
    if changed:
        detail = ", ".join(f"{key} {before.get(key)!r} -> {now.get(key)!r}"
                           for key in changed)
        raise ValueError(
            f"{path} was trained under another config ({detail}); only "
            f"{', '.join(RESUMABLE_CHANGES)} may change when a run is "
            f"resumed. Put the rest back, or start a new run")


def _check_loaders(state: Dict[str, Any],
                   sampler_generator: Optional[torch.Generator], val_loader,
                   path: Path) -> None:
    """Refuse loaders that do not draw, or validate, as the run's did."""
    seeded_before = state["rng"]["sampler"] is not None
    if seeded_before != (sampler_generator is not None):
        raise ValueError(
            f"{path} was trained from a loader "
            f"{'with' if seeded_before else 'without'} a seeded sampler, "
            f"and this one has {'one' if sampler_generator else 'none'}")
    validated_before = "val_loss" in state["history"]
    if validated_before != (val_loader is not None):
        raise ValueError(
            f"{path} was trained {'with' if validated_before else 'without'} "
            f"a validation split, and is being resumed "
            f"{'without' if validated_before else 'with'} one")


def _snapshot(model) -> Dict[str, torch.Tensor]:
    """A copy of the weights that later training steps cannot touch."""
    return {key: value.detach().to("cpu", copy=True)
            for key, value in model.state_dict().items()}


def _scheduler(optimiser, patience: int):
    """Halve the learning rate on a plateau, as the log-power path does.

    The threshold is absolute: the waveform loss is negative once its
    SI-SNR term outweighs the rest, and torch's relative threshold then
    counts a slightly worse epoch as progress.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5,
        patience=_plateau_patience(patience), threshold=LR_MIN_DELTA,
        threshold_mode="abs", min_lr=MIN_LR)


def _plateau_patience(patience: int) -> int:
    """torch halves after its ``patience`` epochs without progress plus
    one, so this halves after ``patience // 2``, at least 1."""
    return max(1, patience // 2) - 1


# ------------------------------------------------------------------- epochs

def _run_epoch(model, batches, stft: ComplexSTFT, cfg: Config, device,
               optimiser=None) -> Dict[str, float]:
    """One pass over *batches*: a training epoch with *optimiser*, a
    validation pass without. Returns the mean of the loss and of each term
    over the examples, ``loss`` first."""
    training = optimiser is not None
    model.train(training)
    totals: Dict[str, float] = {}
    count = 0
    with torch.set_grad_enabled(training):
        for noisy, clean in batches:
            noisy, clean = noisy.to(device), clean.to(device)
            if training:
                optimiser.zero_grad(set_to_none=True)
            terms = loss_terms(estimate_spectrum(model, stft, noisy), clean,
                               stft, cfg.loss)
            loss = weighted_sum(terms, cfg.loss)
            if training:
                loss.backward()
                if cfg.train.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.train.grad_clip)
                optimiser.step()
            n = len(noisy)
            count += n
            for name, value in [("loss", loss)] + list(terms.items()):
                totals[name] = totals.get(name, 0.0) + value.item() * n
    if count == 0:
        raise ValueError("the loader gave no batches")
    return {name: total / count for name, total in totals.items()}


def _progress(*, run_dir: Path, cfg: Config, status: str, epoch: int,
              epochs: int, history: Dict[str, List[float]], monitor: str,
              best: float, best_epoch: int, wait: int, patience: int,
              elapsed: float, started: str) -> Dict[str, Any]:
    """What ``progress.json`` says after *epoch*, 0 before the first.

    ``status`` is ``training``, ``finished`` (the epochs ran out) or
    ``stopped`` (patience did). ``terms`` and ``val_terms`` are the loss's
    parts before weighting. ``elapsed_seconds`` adds up the epochs of every
    sitting, not the time the run lay interrupted. ``eta_seconds`` is the
    time to the last epoch at the pace of the latest few, and
    ``eta_seconds_min`` the earliest patience could stop it, if the
    monitored loss never improves again; both are ``None`` until an epoch
    has been timed. The record is rewritten when a sitting starts as well,
    so a ``training`` run whose ``updated`` is far older than its
    ``epoch_seconds`` has died; ``pid`` says which process to look for. A
    loss that is not a finite number is ``null``, so strict JSON readers
    take the file too.
    """
    terms = list(cfg.loss.weights)
    last = {key: _finite(values[-1]) for key, values in history.items()
            if values}
    left = epochs - epoch if status == "training" else 0
    timed = history["seconds"]
    return {
        "status": status,
        "run": run_dir.name,
        "model": cfg.model.name,
        "epoch": epoch,
        "epochs": epochs,
        "loss": last.get("loss"),
        "terms": {name: last.get(name) for name in terms},
        "val_loss": last.get("val_loss"),
        "val_terms": ({name: last.get(f"val_{name}") for name in terms}
                      if "val_loss" in history else None),
        "monitor": monitor,
        "best": _finite(best),
        "best_epoch": best_epoch or None,
        "epochs_without_improvement": wait,
        "patience": patience,
        "lr": last.get("lr"),
        "epoch_seconds": last.get("seconds"),
        "elapsed_seconds": elapsed,
        "eta_seconds": (remaining_seconds(timed, left) if timed else None),
        "eta_seconds_min": (remaining_seconds(
            timed, min(left, max(patience - wait, 0))) if timed else None),
        "started": started,
        "updated": timestamp(),
        "pid": os.getpid(),
    }


def _finite(value: Optional[float]) -> Optional[float]:
    return value if value is not None and math.isfinite(value) else None


def fit_waveform(model, train_loader, val_loader, cfg: Config, run_dir, *,
                 epochs: Optional[int] = None, resume: bool = False,
                 verbose: int = 1) -> Dict[str, List[float]]:
    """Train *model* on waveform crops, keeping its best epoch in *run_dir*.

    *train_loader* and *val_loader* are :func:`kudioenhance.waveform.waveform_loader`'s,
    a training one and a validation one (``shuffle=False``), or
    *val_loader* is ``None`` and the training loss is watched instead. The
    loss is the ``loss`` section's (:func:`kudioenhance.losses.hybrid_loss`),
    on the STFT the ``audio`` section describes, and *model* follows
    :func:`estimate_spectrum`.

    Adam at ``train.learning_rate``, the rate halved after ``patience // 2``
    epochs without progress, and training stopped once ``train.patience``
    epochs pass without one, as the log-power path does. With
    ``train.grad_clip`` set, the gradients are clipped to that norm before
    every step. Whenever the
    monitored loss falls, ``<model>.pt`` is rewritten with the new best
    weights, and at the end the best weights are put back into *model*.

    After every epoch, ``resume.pt`` gets the model, the optimiser, the
    schedule, every random generator (:func:`_rng_state`), the curves and
    the best weights, then ``history.json`` and ``progress.json`` are
    rewritten. However it was interrupted, ``resume=True`` carries the run on
    from the last epoch that finished, under the same config give or take
    :data:`RESUMABLE_CHANGES`, and a larger *epochs* extends a run that
    finished. *model* and the loaders must be built as they were for the
    first start (:func:`kudioenhance.pipeline.train` does); the weights and
    the draws are put back from the file.

    Weight initialisation is the caller's. The epochs run with torch's
    generator seeded by :func:`~kudioenhance.train.training_seed`, and the
    caller's torch, NumPy and Python generators come back as they were.

    :raises FileExistsError: without *resume*, for a *run_dir* that holds a
        ``resume.pt`` already: days of training are not overwritten by a
        command run twice.
    :raises FileNotFoundError: with *resume*, for one that does not.
    :raises ValueError: with *resume*, for a config that differs from the
        run's in anything else.
    :returns: the curves: ``loss`` and each weighted term per epoch, the
        same with ``val_`` in front when validated, ``lr`` and ``seconds``.
    """
    run_dir = Path(run_dir)
    resume_path = run_dir / RESUME_FILE
    model_path = run_dir / f"{cfg.model.name}.pt"
    device = resolve_device(cfg.train.device)
    epochs = epochs or cfg.train.epochs
    patience = cfg.train.patience
    monitor = "val_loss" if val_loader is not None else "loss"
    sampler_generator = getattr(train_loader.sampler, "generator", None)

    state = None
    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"no {RESUME_FILE} in {run_dir} to resume from; train it "
                f"without resuming first")
        state = load_training_state(resume_path)
        _check_resumable(state["config"], cfg, resume_path)
        _check_loaders(state, sampler_generator, val_loader, resume_path)
    elif resume_path.is_file():
        raise FileExistsError(
            f"{run_dir} holds a run already ({RESUME_FILE}); resume it, or "
            f"remove that folder or pick another name to start afresh")
    cfg.to_yaml(run_dir / "config.yaml")

    model.to(device)
    stft = ComplexSTFT.from_config(cfg.audio).to(device)
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=cfg.train.learning_rate)
    scheduler = _scheduler(optimiser, patience)

    terms = list(cfg.loss.weights)
    keys = ["loss"] + terms
    if val_loader is not None:
        keys += ["val_loss"] + [f"val_{name}" for name in terms]
    history: Dict[str, List[float]] = {key: [] for key in keys + ["lr",
                                                                  "seconds"]}
    best, best_epoch, best_state, wait = math.inf, 0, None, 0
    first_epoch, elapsed, started = 1, 0.0, timestamp()
    if state is not None:
        model.load_state_dict(state["state_dict"])
        optimiser.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        # the state carries the schedule's patience, and a resumed run may
        # have been given more
        scheduler.patience = _plateau_patience(patience)
        history = copy.deepcopy(state["history"])
        best, best_epoch, wait = state["best"], state["best_epoch"], \
            state["wait"]
        best_state = state["best_state"]
        first_epoch, elapsed = state["epoch"] + 1, state["elapsed"]
        started = state["started"]
        if best_state is not None:
            # the best file may be a step ahead of the state, if the run
            # died between writing one and the other
            save_checkpoint(model_path, best_state, model=cfg.model.name,
                            target=cfg.model.target)
        if verbose:
            log.info("resuming %s after epoch %d (best %s %.5f at epoch %d)",
                     run_dir.name, state["epoch"], monitor, best, best_epoch)

    # a run that ran out of patience stays stopped unless it is given more;
    # a wait of 0 means the last epoch improved, which never stops a run
    stopped = (state is not None and state["wait"] > 0
               and state["wait"] >= patience)
    trained = 0
    if state is not None:
        # the last write of the run that died may not have happened
        write_json(run_dir / "history.json", history, required=False)
    if not stopped and first_epoch <= epochs:
        # what was there describes the previous sitting until this one
        # finishes an epoch, which can be hours away
        write_json(run_dir / PROGRESS_FILE, _progress(
            run_dir=run_dir, cfg=cfg, status="training",
            epoch=first_epoch - 1, epochs=epochs, history=history,
            monitor=monitor, best=best, best_epoch=best_epoch, wait=wait,
            patience=patience, elapsed=elapsed, started=started),
            required=False)
    seed = training_seed(cfg)
    with _forked_rng():
        if state is not None:
            _set_rng_state(state["rng"], sampler_generator)
        elif seed is not None:
            torch.manual_seed(seed)

        for epoch in range(first_epoch, epochs + 1):
            if stopped:
                break
            trained += 1
            tic = time.monotonic()
            row = {"lr": optimiser.param_groups[0]["lr"]}
            row.update(_run_epoch(model, train_loader, stft, cfg, device,
                                  optimiser))
            if val_loader is not None:
                row.update({f"val_{name}": value for name, value in
                            _run_epoch(model, val_loader, stft, cfg,
                                       device).items()})
            row["seconds"] = time.monotonic() - tic
            elapsed += row["seconds"]
            for key in history:
                history[key].append(float(row[key]))

            current = row[monitor]
            improved = current < best
            if improved:
                best, best_epoch, wait = current, epoch, 0
                best_state = _snapshot(model)
                save_checkpoint(model_path, best_state, model=cfg.model.name,
                                target=cfg.model.target)
            else:
                wait += 1
            scheduler.step(current)
            stopped = not improved and wait >= patience

            save_training_state(resume_path, {
                "model": cfg.model.name, "target": cfg.model.target,
                "epoch": epoch, "state_dict": _snapshot(model),
                "optimizer": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": _rng_state(sampler_generator),
                "history": history, "best": best, "best_epoch": best_epoch,
                "best_state": best_state, "wait": wait,
                "elapsed": elapsed, "started": started,
                "config": cfg.to_dict()})
            write_json(run_dir / "history.json", history, required=False)
            status = ("stopped" if stopped
                      else "finished" if epoch == epochs else "training")
            write_json(run_dir / PROGRESS_FILE, _progress(
                run_dir=run_dir, cfg=cfg, status=status, epoch=epoch,
                epochs=epochs, history=history, monitor=monitor, best=best,
                best_epoch=best_epoch, wait=wait, patience=patience,
                elapsed=elapsed, started=started), required=False)

            if verbose:
                log.info("epoch %d/%d  %s", epoch, epochs, "  ".join(
                    f"{key} {row[key]:.3g}" if key in ("lr", "seconds")
                    else f"{key} {row[key]:.5f}" for key in history))
                if stopped:
                    log.info("%s has not improved for %d epoch(s); stopping "
                             "at epoch %d", monitor, wait, epoch)

    if state is not None and not trained:
        # resumed with nothing left to do: say so where a reader looks
        write_json(run_dir / PROGRESS_FILE, _progress(
            run_dir=run_dir, cfg=cfg,
            status="stopped" if stopped else "finished",
            epoch=state["epoch"], epochs=max(epochs, state["epoch"]),
            history=history, monitor=monitor, best=best,
            best_epoch=best_epoch, wait=wait, patience=patience,
            elapsed=elapsed, started=started), required=False)
        if verbose:
            log.info("%s has nothing left to train after epoch %d",
                     run_dir.name, state["epoch"])

    if best_state is None:
        log.warning("%s was never finite; saving the final weights", monitor)
        save_checkpoint(model_path, _snapshot(model), model=cfg.model.name,
                        target=cfg.model.target)
    else:
        model.load_state_dict(best_state)
        if verbose:
            log.info("kept epoch %d (%s %.5f); saved model -> %s",
                     best_epoch, monitor, best, model_path)
    model.eval()
    return history
