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
    build_dataset,
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
          verbose: int = 1) -> Dict[str, List[float]]:
    """Build the datasets, fit the model, and save everything to reuse it.

    The training and validation sets are held in memory, or streamed from
    the audio every epoch when ``train.streaming`` is set; see
    :func:`~kudio_enhance.data.build_dataset`. The weights are initialised
    from the training seed as well, so the same config trains the same model
    twice over. Returns the curves.

    The run's ``config.yaml`` records the layer widths that were built, even
    when *cfg* left them to the architecture's default, so the run can be
    rebuilt exactly whatever the default later becomes. *cfg* itself is left
    as it was.
    """
    from kudio_enhance.models import build_model, is_sequence, resolve_units
    from kudio_enhance.train import (
        check_batch_size,
        fit,
        plot_history,
        save_history,
        seeded,
        summarise_history,
        training_seed,
    )

    cfg = resolve_units(cfg)
    # before the training set is built, which can be a pass over the audio
    check_batch_size(cfg)
    sequence = is_sequence(cfg.model.name)
    train_pairs = _load_split(cfg, name, "train")
    val_pairs = _load_split(cfg, name, "val")

    if cfg.train.streaming and cfg.train.shuffle_buffer == 0:
        log.warning("streaming with shuffle_buffer 0: every batch is drawn "
                    "from the file being read and less than a batch left "
                    "over from earlier ones; a buffer as many examples deep "
                    "as a hundred files hold mixes the corpus properly")
    train_set, standardizer = build_dataset(
        train_pairs, cfg, sequence=sequence, fit=True)
    log.info("train set: %r", train_set)

    validation = None
    if val_pairs:
        validation, _ = build_dataset(val_pairs, cfg, sequence=sequence,
                                      standardizer=standardizer)

    with seeded(training_seed(cfg)):
        model = build_model(cfg)
    model_path = cfg.model_path(name)
    history = fit(model, train_set, None, cfg, model_path,
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
             , pesq_mode: str = "auto",
             pesq_scale: str = "lqo") -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Score the held-out split with the trained model."""
    test_pairs = _load_split(cfg, name, "test")
    if not test_pairs:
        raise ValueError(
            "the test split is empty — raise data.test_split and re-synthesize")
    enhancer = Enhancer.load(cfg.run_dir(name))
    rows, summary = evaluate_pairs(enhancer, test_pairs, optional=optional,
                                   pesq_mode=pesq_mode, pesq_scale=pesq_scale)
    write_report(cfg.report_path(name), rows, summary)
    return rows, summary


# --------------------------------------------------------------------- all

def run(cfg: Config, name: str, *, epochs: Optional[int] = None,
        optional: bool = False, verbose: int = 1, pesq_mode: str = "auto",
             pesq_scale: str = "lqo") -> Dict[str, float]:
    """synthesize -> train -> evaluate, returning the summary scores."""
    synthesize(cfg, name)
    train(cfg, name, epochs=epochs, verbose=verbose)
    _, summary = evaluate(cfg, name, optional=optional,
                          pesq_mode=pesq_mode, pesq_scale=pesq_scale)
    return summary
