# -*- coding: utf-8 -*-
"""Compilation, callbacks and fitting.

Kept separate from :mod:`kudio_enhance.models` so architectures stay pure
builders and every experiment shares one optimisation setup.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from kudio_enhance.config import Config

log = logging.getLogger(__name__)

__all__ = ["compile_model", "build_callbacks", "fit",
           "save_history", "load_history", "summarise_history", "plot_history"]


def compile_model(model, cfg: Config):
    """Adam + MSE on log-power frames; MAE reported alongside."""
    import keras
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_callbacks(cfg: Config, model_path, monitor: str = "val_loss") -> List:
    """Checkpoint the best epoch, stop when it plateaus, decay on the way."""
    import keras
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(str(model_path), monitor=monitor,
                                        save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor=monitor, patience=cfg.train.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                          patience=max(1, cfg.train.patience // 2),
                                          min_lr=1e-6, verbose=0),
    ]


def fit(model, x: np.ndarray, y: np.ndarray, cfg: Config, model_path,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: Optional[int] = None, verbose: int = 1):
    """Train *model*, checkpointing to *model_path*.

    Without validation data the callbacks watch training loss instead, so a
    smoke run on a handful of files still works.
    """
    monitor = "val_loss" if validation_data else "loss"
    callbacks = build_callbacks(cfg, model_path, monitor=monitor)
    history = model.fit(
        x, y,
        validation_data=validation_data,
        epochs=epochs or cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )
    # EarlyStopping restores the best weights, but it only fires when it
    # triggers -- save explicitly so the file always matches the returned model
    model.save(str(model_path))
    log.info("saved model -> %s", model_path)
    return history


# --------------------------------------------------------------------- curves

def save_history(path, history) -> Path:
    """Write the training curves next to the model.

    Keras returns them and almost every script drops them on the floor, which
    means "did it converge, or did it stop early because the validation loss
    was already climbing?" becomes unanswerable the moment the terminal is
    closed. A checkpoint without its curves is a number with no working.

    Accepts a Keras ``History``, its ``.history`` dict, or any mapping.
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
