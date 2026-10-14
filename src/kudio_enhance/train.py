# -*- coding: utf-8 -*-
"""The training loop, and the curves it leaves behind.

Kept separate from :mod:`kudio_enhance.models` so architectures stay pure
builders and every experiment shares one optimisation setup: Adam on MSE, the
best epoch checkpointed, the learning rate halved on a plateau, and training
stopped once the plateau outlasts its patience.

torch is imported inside the functions that train rather than at the top. The
history helpers below are re-exported from the package, and reading a
finished run's curves must not need the framework that produced them.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from kudio_enhance.config import Config

log = logging.getLogger(__name__)

__all__ = ["fit", "check_batch_size", "resolve_device", "training_seed",
           "seeded", "compile_model", "save_history", "load_history",
           "summarise_history", "plot_history"]

#: The learning rate is never halved below this.
MIN_LR = 1e-6
#: How far the monitored loss has to fall before the learning-rate schedule
#: counts it as progress — Keras's ``min_delta`` for ReduceLROnPlateau, which
#: the checkpoint and early stopping do not use: for them any fall counts.
LR_MIN_DELTA = 1e-4


def resolve_device(device: str = "auto"):
    """``'auto'`` -> an NVIDIA GPU when torch can see one, the CPU otherwise.

    Only CUDA counts: on a Mac, where torch can also drive Apple's GPU through
    ``mps``, training runs on the CPU.

    Asking for ``'cuda'`` on a machine without one fails here, by name, rather
    than as a driver error from somewhere inside the first batch.
    """
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"train.device is {device!r} but torch cannot see a GPU; install "
            f"a CUDA build of torch, or set it to 'auto'")
    return torch.device(device)


def _frame_wise(cfg: Config) -> bool:
    """Is ``cfg.model.name`` an architecture that trains on single frames?

    Its BatchNorm then takes a mean and a variance over the batch and nothing
    else, and one example has no variance to take: torch refuses it in
    training mode. Keras let it through and normalised every unit to its
    offset, which no gradient flows back through, so in a 0.1 ``ddae`` at
    ``batch_size: 1`` nothing below the last BatchNorm ever moved off its
    initial weights. A sequence model's normalisation also spans the frames
    of its window, and one window is enough for it.

    A name the registry does not know is a model of the caller's own, and
    left to the caller.
    """
    from kudio_enhance.models.registry import REGISTRY
    spec = REGISTRY.get(cfg.model.name)
    return spec is not None and not spec.sequence


def check_batch_size(cfg: Config) -> None:
    """Refuse a ``train.batch_size`` the configured model cannot train on.

    :func:`fit` asks this before the first batch, and the pipeline before it
    spends a pass over the audio building the training set, so a ``ddae`` at
    ``batch_size: 1`` fails at once and says why.
    """
    if cfg.train.batch_size < 2 and _frame_wise(cfg):
        raise ValueError(
            f"{cfg.model.name} normalises every batch with BatchNorm, which "
            f"needs at least two examples to take a variance from; set "
            f"train.batch_size to 2 or more (it is {cfg.train.batch_size})")


def _never_alone(batches: Iterable, name: str) -> Iterator:
    """*batches*, refusing one of a single example, for a frame-wise model.

    With ``batch_size`` 2 or more the datasets only ever hand one out when
    the whole training set is one example — a lone example left over at the
    end of an epoch joins the batch before it — so this fails at the first
    batch of the first epoch, before any weight has moved.
    """
    for inputs, targets in batches:
        if len(inputs) < 2:
            raise ValueError(
                f"a training batch holds a single example, and {name} cannot "
                f"train on one: its BatchNorm needs two to take a variance "
                f"from. The training set has to hold at least two examples")
        yield inputs, targets


def training_seed(cfg: Config) -> Optional[int]:
    """``train.seed``, or ``data.seed`` when that is not set."""
    return cfg.train.seed if cfg.train.seed is not None else cfg.data.seed


@contextlib.contextmanager
def seeded(seed: Optional[int]) -> Iterator[None]:
    """Run a block with torch's global RNG seeded, then put the caller's back.

    Weight initialisation and dropout both draw from the global generator, so
    a reproducible run has to seed it — but a library call that quietly
    reseeds its caller's RNG is a surprise waiting for someone. Forking gives
    both. ``None`` leaves the RNG alone.
    """
    import torch
    if seed is None:
        yield
        return
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        yield


def compile_model(model, cfg: Config):
    """Deprecated since 0.2: returns *model* unchanged.

    Keras needed a model compiled before it could be fitted. The optimiser,
    loss and metric now live in :func:`fit`, so there is nothing left to do
    here; it is kept so ``fit(compile_model(build_model(cfg), cfg), ...)``
    written against 0.1 still runs.
    """
    warnings.warn(
        "compile_model() does nothing since kudio-enhance 0.2; fit() sets up "
        "the optimiser and loss itself, so pass the model straight to it",
        DeprecationWarning, stacklevel=2)
    return model


def fit(model, x, y: Optional[np.ndarray], cfg: Config, model_path,
        validation_data=None, epochs: Optional[int] = None,
        verbose: int = 1) -> Dict[str, List[float]]:
    """Train *model* in place, checkpointing its best epoch to *model_path*.

    *x* and *y* are the training inputs and targets as arrays — or *x* is a
    dataset from :func:`kudio_enhance.data.build_dataset`, held in memory or
    streamed, and *y* is ``None``, because a dataset carries its own targets.
    *validation_data* is likewise an ``(x, y)`` pair of arrays or a dataset.
    Arrays are trained on as the in-memory dataset they make, so the two give
    the same run to the last digit.

    Adam on MSE, with MAE reported alongside. The training set is reshuffled
    every epoch from a generator seeded by :func:`training_seed` — the order
    of the examples in memory, the order of the files and the draws from the
    shuffle buffer when streaming — and dropout draws from a seeded RNG too,
    so the same config and the same starting weights give the same run.
    Without validation data everything watches training loss instead, so a
    smoke run on a handful of files still works.

    A frame-wise model normalises over the batch, so it needs two examples
    or more in every batch: a ``batch_size`` of 1, or a training set of a
    single example, raises ``ValueError`` before any weight moves. See
    :func:`check_batch_size`.

    The checkpoint is rewritten whenever the monitored loss improves. When
    training ends, because patience ran out or the epochs did, the best
    epoch's weights are put back into *model* — so the model handed back and
    the file on disk are always the same weights, and always the best ones.

    Returns the curves: ``loss`` and ``mae`` per epoch, ``val_loss`` and
    ``val_mae`` when validated, and the ``lr`` each epoch ran at.

    :param verbose: ``0`` is silent; ``1`` or more logs one line per epoch.
    """
    import torch
    from kudio_enhance.checkpoint import save_checkpoint
    from kudio_enhance.datasets import InMemoryDataset, as_dataset

    check_batch_size(cfg)
    frame_wise = _frame_wise(cfg)
    device = resolve_device(cfg.train.device)
    seed = training_seed(cfg)
    epochs = epochs or cfg.train.epochs
    patience = cfg.train.patience
    train_set = as_dataset(x, y)
    validation = (as_dataset(*validation_data)
                  if isinstance(validation_data, (tuple, list))
                  else validation_data)
    if isinstance(validation, InMemoryDataset) and len(validation) == 0:
        validation = None
    monitor = "val_loss" if validation is not None else "loss"

    model.to(device)

    # Keras's epsilon, ten times torch's: it damps the steps of parameters
    # whose gradients are tiny
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=cfg.train.learning_rate, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5,
        # torch halves after `patience` epochs without progress *plus one*,
        # Keras after `patience`; one fewer keeps the schedule this config
        # was tuned with
        patience=max(1, patience // 2) - 1,
        threshold=LR_MIN_DELTA, threshold_mode="abs", min_lr=MIN_LR)
    # the shuffle has a generator of its own, so the order the batches come
    # in does not depend on how many numbers dropout happened to draw
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    history: Dict[str, List[float]] = {"loss": [], "mae": []}
    if validation is not None:
        history.update(val_loss=[], val_mae=[])
    history["lr"] = []

    best, best_epoch, best_state, wait = math.inf, 0, None, 0
    with seeded(seed):
        for epoch in range(1, epochs + 1):
            row = {"lr": optimiser.param_groups[0]["lr"]}
            batches = train_set.shuffled_batches(cfg.train.batch_size,
                                                 generator)
            if frame_wise:
                batches = _never_alone(batches, cfg.model.name)
            row["loss"], row["mae"] = _train_epoch(model, optimiser, batches,
                                                   device)
            if validation is not None:
                row["val_loss"], row["val_mae"] = _evaluate(
                    model, validation.ordered_batches(cfg.train.batch_size),
                    device)
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

            if verbose:
                log.info("epoch %d/%d  %s", epoch, epochs, "  ".join(
                    f"{key} {row[key]:.3g}" if key == "lr"
                    else f"{key} {row[key]:.5f}" for key in history))
            if not improved and wait >= patience:
                if verbose:
                    log.info("%s has not improved for %d epoch(s); stopping "
                             "at epoch %d", monitor, wait, epoch)
                break

    if best_state is None:
        # the monitor was never a number: keep the last weights anyway, so
        # the run directory is complete and the curves can say what happened
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


def _snapshot(model) -> Dict[str, object]:
    """A copy of the weights that later training steps cannot touch."""
    return {key: value.detach().to("cpu", copy=True)
            for key, value in model.state_dict().items()}


def _train_epoch(model, optimiser, batches: Iterable, device
                 ) -> Tuple[float, float]:
    """One pass over the training *batches*; returns their mean
    ``(mse, mae)``."""
    import torch
    model.train()
    squared = absolute = 0.0
    count = 0
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        optimiser.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimiser.step()
        squared += loss.item() * targets.numel()
        absolute += (outputs.detach() - targets).abs().sum().item()
        count += targets.numel()
    return squared / count, absolute / count


def _evaluate(model, batches: Iterable, device) -> Tuple[float, float]:
    """Mean ``(mse, mae)`` over held-out *batches*, with inference behaviour:
    no dropout, and BatchNorm on its running statistics."""
    import torch
    model.eval()
    squared = absolute = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in batches:
            error = model(inputs.to(device)) - targets.to(device)
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            count += targets.numel()
    return squared / count, absolute / count


# --------------------------------------------------------------------- curves

def save_history(path, history) -> Path:
    """Write the training curves next to the model.

    A training loop hands them back and almost every script drops them on the
    floor, which means "did it converge, or did it stop early because the
    validation loss was already climbing?" becomes unanswerable the moment the
    terminal is closed. A checkpoint without its curves is a number with no
    working.

    Accepts the dict :func:`fit` returns, any other mapping, or an object
    carrying one as ``.history`` — a Keras ``History`` from a 0.1 run
    included.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    curves = getattr(history, "history", history) or {}
    plain = {key: [float(v) for v in values] for key, values in curves.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plain, fh, indent=2)
    return path


def load_history(path) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def summarise_history(history, monitor: str = "val_loss") -> Dict[str, object]:
    """The three things worth knowing, without opening a plot.

    ``epochs`` is how many actually ran — shorter than configured means early
    stopping fired — ``best`` and ``best_epoch`` say where the kept weights came
    from, and ``improved`` is False when the best epoch was the *first* one,
    which is what "this did not learn anything" looks like.
    """
    curves = getattr(history, "history", history) or {}
    key = monitor if monitor in curves else ("loss" if "loss" in curves else None)
    if key is None:
        return {"epochs": 0, "monitor": monitor, "best": None,
                "best_epoch": None, "improved": False}

    values = [float(v) for v in curves[key]]
    best_index = int(np.argmin(values))
    return {
        "epochs": len(values),
        "monitor": key,
        "best": values[best_index],
        "best_epoch": best_index + 1,
        "first": values[0],
        "last": values[-1],
        "improved": bool(best_index > 0 and values[best_index] < values[0]),
    }


def plot_history(history, path, title: str = "training") -> Optional[Path]:
    """Draw the curves, if matplotlib is around.

    Returns ``None`` rather than raising when it is not: a missing plotting
    library must never cost you a finished training run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed; skipping the training plot "
                 "(the curves are still in history.json)")
        return None

    curves = getattr(history, "history", history) or {}
    if not curves:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(7, 4))
    for key, values in curves.items():
        if key.endswith("loss"):
            axes.plot(range(1, len(values) + 1), values, label=key)
    axes.set_xlabel("epoch")
    axes.set_ylabel("loss")
    axes.set_title(title)
    axes.grid(alpha=0.3)
    if axes.get_legend_handles_labels()[0]:
        axes.legend()
    figure.tight_layout()
    figure.savefig(str(path), dpi=120)
    plt.close(figure)
    return path
