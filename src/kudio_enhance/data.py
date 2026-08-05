# -*- coding: utf-8 -*-
"""Dataset construction: mixing, manifests, splits and feature matrices.

`kudio.Synthesizer` returns the ``(mixed, clean, noise, snr)`` tuples it wrote,
so pairing is recorded at generation time instead of being re-derived from
filenames later.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import kudio
from kudio_enhance.config import Config
from kudio_enhance.features import (
    Standardizer,
    frame_windows,
    spectrogram,
    stack_context,
)

log = logging.getLogger(__name__)

__all__ = ["Pair", "synthesize", "save_manifest", "load_manifest",
           "split_pairs", "build_arrays"]


@dataclass(frozen=True)
class Pair:
    """One training example: a mixture and the clean signal behind it.

    Frozen so a manifest can be put in a set — splits are checked for overlap.
    """

    noisy: str
    clean: str
    noise: str
    snr_db: int

    def exists(self) -> bool:
        return Path(self.noisy).is_file() and Path(self.clean).is_file()


# --------------------------------------------------------------------- mixing

def synthesize(cfg: Config, name: str) -> List[Pair]:
    """Mix clean × noise at the configured SNRs and record the pairing."""
    data = cfg.data
    syx = kudio.Synthesizer(data.clean_dir, data.noise_dir,
                            out_path=data.mixed_dir, snr_ratio=data.snr_db)
    # be explicit: the mixture on disk should be at the rate the model trains
    # on, whatever the source corpus happens to be
    produced = syx.syn(mode=data.mode, seed=data.seed, overwrite=True,
                       target_sr=cfg.audio.sr)
    if not produced:
        raise RuntimeError(
            f"synthesis produced nothing — check {data.clean_dir!r} and "
            f"{data.noise_dir!r}")

    pairs = [Pair(noisy=str(noisy), clean=str(clean), noise=Path(noise).stem,
                  snr_db=int(snr))
             for noisy, clean, noise, snr in produced]
    missing = [p for p in pairs if not p.exists()]
    if missing:
        log.warning("%d/%d mixtures missing on disk, dropping them",
                    len(missing), len(pairs))
        pairs = [p for p in pairs if p.exists()]

    if data.max_files is not None:
        pairs = pairs[:data.max_files]

    save_manifest(cfg.manifest_path(name), pairs)
    log.info("synthesized %d pair(s) -> %s", len(pairs), data.mixed_dir)
    return pairs


# ------------------------------------------------------------------ manifests

def save_manifest(path, pairs: Sequence[Pair]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(p) for p in pairs], fh, indent=2, ensure_ascii=False)
    return path


def load_manifest(path) -> List[Pair]:
    with open(path, "r", encoding="utf-8") as fh:
        return [Pair(**row) for row in json.load(fh)]


def split_pairs(pairs: Sequence[Pair], val_split: float, test_split: float,
                seed: Optional[int] = None
                ) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    """Shuffle once, then cut into train / validation / test."""
    items = list(pairs)
    rng = np.random.default_rng(seed)
    rng.shuffle(items)

    n = len(items)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    # never let the splits starve training when the set is tiny
    if n - n_val - n_test < 1:
        n_val = n_test = 0
    return (items[n_val + n_test:], items[:n_val], items[n_val:n_val + n_test])


# ------------------------------------------------------------------- features

def build_arrays(pairs: Sequence[Pair], cfg: Config, *, sequence: bool,
                 standardizer: Optional[Standardizer] = None,
                 fit: bool = False) -> Tuple[np.ndarray, np.ndarray, Standardizer]:
    """Turn pairs into ``(X, Y, standardizer)`` ready for ``model.fit``.

    Frame-wise models get ``(N, bins * (2 * context + 1))`` inputs against
    ``(N, bins)`` targets; sequence models get ``(N, n_frames, bins)`` for both.

    Everything is held in memory — fine for the tens of hours these models are
    normally trained on, but a `tf.data` pipeline is the answer for more.
    """
    if not pairs:
        raise ValueError("no pairs to build arrays from")

    noisy_specs, clean_specs = [], []
    for pair in pairs:
        noisy, _ = kudio.file_load(pair.noisy, sr=cfg.audio.sr)
        clean, _ = kudio.file_load(pair.clean, sr=cfg.audio.sr)
        n_spec = spectrogram(noisy, cfg.audio)
        c_spec = spectrogram(clean, cfg.audio)
        frames = min(len(n_spec), len(c_spec))
        if frames == 0:
            log.warning("skipping empty pair: %s", pair.noisy)
            continue
        noisy_specs.append(n_spec[:frames])
        clean_specs.append(c_spec[:frames])

    if not noisy_specs:
        raise ValueError("every pair was empty or unreadable")

    if standardizer is None:
        standardizer = Standardizer()
    if fit or not standardizer.fitted:
        standardizer.fit(np.concatenate(noisy_specs, axis=0))

    xs, ys = [], []
    for n_spec, c_spec in zip(noisy_specs, clean_specs):
        n_norm = standardizer.transform(n_spec)
        c_norm = standardizer.transform(c_spec)
        if sequence:
            xs.append(frame_windows(n_norm, cfg.model.n_frames))
            ys.append(frame_windows(c_norm, cfg.model.n_frames))
        else:
            xs.append(stack_context(n_norm, cfg.model.context))
            ys.append(c_norm)

    return (np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), standardizer)
