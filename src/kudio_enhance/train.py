# -*- coding: utf-8 -*-
"""Compilation, callbacks and fitting.

Kept separate from :mod:`kudio_enhance.models` so architectures stay pure
builders and every experiment shares one optimisation setup.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from kudio_enhance.config import Config

log = logging.getLogger(__name__)

__all__ = ["compile_model", "build_callbacks", "fit"]


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
