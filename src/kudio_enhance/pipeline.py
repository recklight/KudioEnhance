# -*- coding: utf-8 -*-
"""The three stages, and the command that runs all of them.

    synthesize  ->  train  ->  evaluate

Each stage reads what the previous one wrote, so any of them can be re-run on
its own: re-train without re-mixing, re-evaluate without re-training.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from kudio_enhance.config import Config
from kudio_enhance.data import (
    Pair,
    build_arrays,
    load_manifest,
    save_manifest,
    split_pairs,
    synthesize as synthesize_data,
)
from kudio_enhance.evaluate import evaluate_pairs, write_report
from kudio_enhance.features import Standardizer
from kudio_enhance.inference import Enhancer

log = logging.getLogger(__name__)

__all__ = ["synthesize", "train", "evaluate", "run"]

SPLIT_FILE = "splits.json"


def _split_paths(cfg: Config, name: str) -> Dict[str, Path]:
    run = cfg.run_dir(name)
    return {part: run / f"{part}.json" for part in ("train", "val", "test")}


# ------------------------------------------------------------------ stage 1

def synthesize(cfg: Config, name: str) -> List[Pair]:
    """Mix the dataset and freeze the train/val/test split."""
    pairs = synthesize_data(cfg, name)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, cfg.data.val_split, cfg.data.test_split, cfg.data.seed)

    paths = _split_paths(cfg, name)
    for part, subset in (("train", train_pairs), ("val", val_pairs),
                         ("test", test_pairs)):
        save_manifest(paths[part], subset)
    log.info("split: %d train / %d val / %d test",
             len(train_pairs), len(val_pairs), len(test_pairs))
    return pairs


def _load_split(cfg: Config, name: str, part: str) -> List[Pair]:
    path = _split_paths(cfg, name)[part]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — run the synthesize stage first")
    return load_manifest(path)


# ------------------------------------------------------------------ stage 2

def train(cfg: Config, name: str, *, epochs: Optional[int] = None,
          verbose: int = 1):
    """Build the arrays, fit the model, and save everything needed to reuse it."""
    from kudio_enhance.models import build_model, is_sequence
    from kudio_enhance.train import (
        compile_model,
        fit,
        plot_history,
        save_history,
        summarise_history,
    )

    sequence = is_sequence(cfg.model.name)
    train_pairs = _load_split(cfg, name, "train")
    val_pairs = _load_split(cfg, name, "val")

    x_train, y_train, standardizer = build_arrays(
        train_pairs, cfg, sequence=sequence, fit=True)
    log.info("train arrays: X%s Y%s", x_train.shape, y_train.shape)

    validation = None
    if val_pairs:
        x_val, y_val, _ = build_arrays(val_pairs, cfg, sequence=sequence,
                                       standardizer=standardizer)
        validation = (x_val, y_val)

    model = compile_model(build_model(cfg), cfg)
    model_path = cfg.model_path(name)
    history = fit(model, x_train, y_train, cfg, model_path,
                  validation_data=validation, epochs=epochs, verbose=verbose)

    standardizer.save(cfg.stats_path(name))
    cfg.to_yaml(cfg.run_dir(name) / "config.yaml")

    # the curves used to be returned and dropped, which made "did it converge,
    # or stop early because validation was already climbing?" unanswerable once
    # the terminal closed
    save_history(cfg.history_path(name), history)
    plot_history(history, cfg.run_dir(name) / "history.png", title=name)
    summary = summarise_history(history)
    if summary["best"] is not None:
        log.info("%s: %d epoch(s), best %s %.5f at epoch %d%s",
                 name, summary["epochs"], summary["monitor"], summary["best"],
                 summary["best_epoch"],
                 "" if summary["improved"]
                 else " — no improvement over the first epoch")
    return history


# ------------------------------------------------------------------ stage 3

def evaluate(cfg: Config, name: str, *, optional: bool = False
             ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Score the held-out split with the trained model."""
    test_pairs = _load_split(cfg, name, "test")
    if not test_pairs:
        raise ValueError(
            "the test split is empty — raise data.test_split and re-synthesize")
    enhancer = Enhancer.load(cfg.run_dir(name))
    rows, summary = evaluate_pairs(enhancer, test_pairs, optional=optional)
    write_report(cfg.report_path(name), rows, summary)
    return rows, summary


# --------------------------------------------------------------------- all

def run(cfg: Config, name: str, *, epochs: Optional[int] = None,
        optional: bool = False, verbose: int = 1) -> Dict[str, float]:
    """synthesize -> train -> evaluate, returning the summary scores."""
    synthesize(cfg, name)
    train(cfg, name, epochs=epochs, verbose=verbose)
    _, summary = evaluate(cfg, name, optional=optional)
    return summary
