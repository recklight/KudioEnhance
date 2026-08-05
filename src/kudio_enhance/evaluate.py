# -*- coding: utf-8 -*-
"""Score enhancement against the clean reference.

Every metric is reported for the noisy input as well as the enhanced output.
An absolute SI-SDR means little on its own; the improvement over the mixture
is the number that says whether the model did anything.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import kudio
from kudio_enhance.config import Config
from kudio_enhance.data import Pair
from kudio_enhance.inference import Enhancer

log = logging.getLogger(__name__)

__all__ = ["score_pair", "evaluate_pairs", "write_report"]

#: metrics computed from numpy alone, always available
BASE_METRICS = ("snr", "si_sdr", "seg_snr")
#: metrics needing kudio[eval]
OPTIONAL_METRICS = ("pesq", "stoi", "sdr")


def _align(*signals: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = min(len(s) for s in signals)
    return tuple(np.asarray(s[:n], dtype=np.float32) for s in signals)


def _base_scores(ref: np.ndarray, deg: np.ndarray) -> Dict[str, float]:
    return {
        "snr": float(kudio.snr(ref, deg)),
        "si_sdr": float(kudio.si_sdr(ref, deg)),
        "seg_snr": float(kudio.segmental_snr(ref, deg)),
    }


def score_pair(enhancer: Enhancer, pair: Pair, *,
               optional: bool = False) -> Dict[str, object]:
    """Score one mixture: noisy vs clean, then enhanced vs clean."""
    sr = enhancer.cfg.audio.sr
    noisy, _ = kudio.file_load(pair.noisy, sr=sr)
    clean, _ = kudio.file_load(pair.clean, sr=sr)
    enhanced = enhancer.enhance(noisy)
    clean, noisy, enhanced = _align(clean, noisy, enhanced)

    row: Dict[str, object] = {
        "file": Path(pair.noisy).name,
        "noise": pair.noise,
        "snr_db": pair.snr_db,
    }
    for name, value in _base_scores(clean, noisy).items():
        row[f"noisy_{name}"] = value
    for name, value in _base_scores(clean, enhanced).items():
        row[f"enhanced_{name}"] = value
    for name in BASE_METRICS:
        row[f"delta_{name}"] = row[f"enhanced_{name}"] - row[f"noisy_{name}"]

    if optional:
        noisy_opt = kudio.eval_metrics(sr, clean, noisy)
        enhanced_opt = kudio.eval_metrics(sr, clean, enhanced)
        for name, n_val, e_val in zip(OPTIONAL_METRICS, noisy_opt, enhanced_opt):
            row[f"noisy_{name}"] = n_val
            row[f"enhanced_{name}"] = e_val
            row[f"delta_{name}"] = (None if n_val is None or e_val is None
                                    else e_val - n_val)
    return row


def evaluate_pairs(enhancer: Enhancer, pairs: Sequence[Pair], *,
                   optional: bool = False
                   ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Score every pair and average the numeric columns."""
    if not pairs:
        raise ValueError("nothing to evaluate")

    rows = []
    for pair in pairs:
        try:
            rows.append(score_pair(enhancer, pair, optional=optional))
        except Exception as e:
            log.warning("skipping %s — %s: %s", pair.noisy, type(e).__name__, e)
    if not rows:
        raise RuntimeError("every pair failed to score")

    summary = {}
    for key in rows[0]:
        values = [r[key] for r in rows
                  if isinstance(r.get(key), (int, float)) and r[key] is not None]
        if values and key != "snr_db":
            summary[key] = float(np.mean(values))
    log.info("evaluated %d file(s): SI-SDR %.2f -> %.2f dB (%+.2f)",
             len(rows), summary.get("noisy_si_sdr", float("nan")),
             summary.get("enhanced_si_sdr", float("nan")),
             summary.get("delta_si_sdr", float("nan")))
    return rows, summary


def write_report(path, rows: Sequence[Dict[str, object]],
                 summary: Optional[Dict[str, float]] = None) -> Path:
    """Per-file scores as CSV, with a trailing MEAN row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        if summary:
            mean_row = {k: summary.get(k, "") for k in fieldnames}
            mean_row[fieldnames[0]] = "MEAN"
            writer.writerow(mean_row)
    log.info("report -> %s", path)
    return path
