# -*- coding: utf-8 -*-
"""Dataset construction: mixing, manifests, splits and feature matrices.

`kudio.Synthesizer` returns the ``(mixed, clean, noise, snr)`` tuples it wrote,
so pairing is recorded at generation time instead of being re-derived from
filenames later.

**`Pair`, `save_manifest`, `load_manifest` and `split_pairs` now live in
kudio** — nothing in them was specific to denoising, and two copies of "what
came from what" is one copy too many. They are re-exported here so existing
imports keep working, and the JSON on disk is byte-identical to what this
module used to write.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import kudio
from kudio import Pair, load_manifest, save_manifest, split_pairs
from kudio_enhance.config import Config
from kudio_enhance.features import (
    Standardizer,
    frame_windows,
    ideal_ratio_mask,
    spectrogram,
    stack_context,
)

log = logging.getLogger(__name__)

__all__ = ["Pair", "synthesize", "save_manifest", "load_manifest",
           "split_pairs", "build_arrays"]


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


# ------------------------------------------------------------------- features

def build_arrays(pairs: Sequence[Pair], cfg: Config, *, sequence: bool,
                 standardizer: Optional[Standardizer] = None,
                 fit: bool = False) -> Tuple[np.ndarray, np.ndarray, Standardizer]:
    """Turn pairs into ``(X, Y, standardizer)`` ready for ``model.fit``.

    Frame-wise models get ``(N, bins * (2 * context + 1))`` inputs against
    ``(N, bins)`` targets; sequence models get ``(N, n_frames, bins)`` for both.

    **The target depends on `cfg.model.target`.** For ``spectrum`` it is the
    normalised clean spectrogram. For ``irm`` it is a mask in ``[0, 1]``, which
    is **not** standardised — a mask is already on its own bounded scale, and
    putting the input's mean and standard deviation through it would produce a
    target the sigmoid output cannot even reach.

    Everything is held in memory — fine for the tens of hours these models are
    normally trained on, but a `tf.data` pipeline is the answer for more.
    """
    if not pairs:
        raise ValueError("no pairs to build arrays from")

    mask_target = cfg.model.predicts_mask
    noisy_specs, target_specs = [], []
    for pair in pairs:
        noisy, _ = kudio.file_load(pair.noisy, sr=cfg.audio.sr)
        clean, _ = kudio.file_load(pair.clean, sr=cfg.audio.sr)
        n_spec = spectrogram(noisy, cfg.audio)
        c_spec = spectrogram(clean, cfg.audio)
        frames = min(len(n_spec), len(c_spec))
        if frames == 0:
            log.warning("skipping empty pair: %s", pair.noisy)
            continue

        if mask_target:
            # the noise is what the mixture has that the clean file does not.
            # kudio.Synthesizer writes mixed = clean + scaled noise, so this is
            # exact bar the 16-bit quantisation of the files themselves
            length = min(len(noisy), len(clean))
            noise_spec = spectrogram(noisy[:length] - clean[:length], cfg.audio)
            target = ideal_ratio_mask(c_spec[:frames],
                                      noise_spec[:frames])
        else:
            target = c_spec[:frames]

        noisy_specs.append(n_spec[:frames])
        target_specs.append(target)

    if not noisy_specs:
        raise ValueError("every pair was empty or unreadable")

    if standardizer is None:
        standardizer = Standardizer()
    if fit or not standardizer.fitted:
        standardizer.fit(np.concatenate(noisy_specs, axis=0))

    xs, ys = [], []
    for n_spec, target in zip(noisy_specs, target_specs):
        n_norm = standardizer.transform(n_spec)
        y_target = target if mask_target else standardizer.transform(target)
        if sequence:
            xs.append(frame_windows(n_norm, cfg.model.n_frames))
            ys.append(frame_windows(y_target, cfg.model.n_frames))
        else:
            xs.append(stack_context(n_norm, cfg.model.context))
            ys.append(y_target)

    return (np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), standardizer)
