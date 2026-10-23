# -*- coding: utf-8 -*-
"""Dataset construction: mixing, manifests, splits and training sets.

`kudio.Synthesizer` returns the ``(mixed, clean, noise, snr)`` tuples it wrote,
so pairing is recorded at generation time instead of being re-derived from
filenames later.

**A training set is held in memory or streamed from disk.**
:func:`build_dataset` is what training uses, and ``train.streaming`` picks.
Held in memory, the features are computed once, and a frame-wise model's
context is stacked a batch at a time as the batch is drawn rather than for
every frame up front, which multiplied the memory by ``2 * context + 1``.
Streamed, nothing is held: the features are computed again from the audio, a
file at a time, every epoch, for a corpus larger than the memory there is. The
rows are the same either way, and the same as :func:`build_arrays`, which still
builds the whole stacked matrix for code that wants it.

**`Pair`, `save_manifest`, `load_manifest` and `split_pairs` now live in
kudio** — nothing in them was specific to denoising, and two copies of "what
came from what" is one copy too many. They are re-exported here so existing
imports keep working, and the JSON on disk is byte-identical to what this
module used to write.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

import kudio
from kudio import Pair, load_manifest, save_manifest, split_pairs
from kudioenhance.config import Config
from kudioenhance.features import (
    Standardizer,
    fit_standardizer,
    frame_windows,
    ideal_ratio_mask,
    spectrogram,
    stack_context,
)

if TYPE_CHECKING:  # torch is only imported once a dataset is built
    from kudioenhance.datasets import InMemoryDataset, StreamingDataset

log = logging.getLogger(__name__)

__all__ = ["Pair", "synthesize", "save_manifest", "load_manifest",
           "split_pairs", "build_dataset", "build_arrays", "pair_examples",
           "reference_for"]


# --------------------------------------------------------------------- mixing

def synthesize(cfg: Config, name: str) -> List[Pair]:
    """Mix clean × noise × SNR (× room) and record the pairing."""
    data = cfg.data
    syx = kudio.Synthesizer(data.clean_dir, data.noise_dir,
                            out_path=data.mixed_dir, snr_ratio=data.snr_db,
                            rt60=data.rt60 or None, drr_db=data.drr_db,
                            channel=data.channel,
                            write_targets=data.reverberant_target)
    # be explicit: the mixture on disk should be at the rate the model trains
    # on, whatever the source corpus happens to be
    produced = syx.syn(mode=data.mode, seed=data.seed, overwrite=True,
                       target_sr=cfg.audio.sr)
    if not produced:
        raise RuntimeError(
            f"synthesis produced nothing — check {data.clean_dir!r} and "
            f"{data.noise_dir!r}")

    # `syn` returns four-tuples and always has -- the room travels in the
    # manifest, where it is attached to the output path it belongs to rather
    # than held in a list that has to stay in step
    pairs = syx.manifest()
    missing = [p for p in pairs if not p.exists()]
    if missing:
        log.warning("%d/%d mixtures missing on disk, dropping them",
                    len(missing), len(pairs))
        pairs = [p for p in pairs if p.exists()]

    if data.max_files is not None:
        pairs = pairs[:data.max_files]

    save_manifest(cfg.manifest_path(name), pairs)
    if data.reverberant:
        log.info("synthesized %d pair(s) in rooms of %s s -> %s",
                 len(pairs), data.rt60, data.mixed_dir)
        if data.reverberant_target:
            log.info("training against the reverberant target, so the model "
                     "is being asked to remove the noise and leave the room")
        else:
            log.info("the clean file is still the target, so the model is "
                     "being asked to undo the room as well as the noise")
    else:
        log.info("synthesized %d pair(s) -> %s", len(pairs), data.mixed_dir)
    if syx.channel_tag:
        log.info("every mixture went down a %s link, and the target did not, "
                 "so the model is being asked to undo that too",
                 syx.channel_tag)
    return pairs


def reference_for(pair: Pair, cfg: Config) -> str:
    """The file the model is being trained to produce, for *pair*.

    Normally the dry clean file. With ``reverberant_target`` it is the
    reverberant-clean signal the Synthesizer wrote beside the mixture -- the
    same speech with the room left on -- and the room stops being something
    the model has to undo.

    Falls back to the clean file when a manifest has no target, so an older
    run still trains rather than failing on a missing key.
    """
    if cfg.data.reverberant_target and pair.target:
        return pair.target
    return pair.clean


# ------------------------------------------------------------------- features

def _spectra(pair: Pair, cfg: Config, *, target: bool = True
             ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """*pair*'s noisy spectrogram and the target it trains towards, frame for
    frame; ``None`` for a pair with no frames, which callers skip.

    With ``target=False`` the target is not worked out and ``None`` stands in
    for it: fitting the normalisation needs only the noisy half. The reference
    is still read, because the shorter of the two decides which frames count.
    """
    noisy, _ = kudio.file_load(pair.noisy, sr=cfg.audio.sr)
    clean, _ = kudio.file_load(reference_for(pair, cfg), sr=cfg.audio.sr)
    n_spec = spectrogram(noisy, cfg.audio)
    c_spec = spectrogram(clean, cfg.audio)
    frames = min(len(n_spec), len(c_spec))
    if frames == 0:
        return None
    if not target:
        return n_spec[:frames], None

    if cfg.model.predicts_mask:
        # the noise is what the mixture has that the reference does not.
        # kudio.Synthesizer writes mixed = (reverberant) clean + scaled
        # noise, so this is exact bar the 16-bit quantisation -- as long
        # as the reference really is what was mixed, which in a
        # reverberant dataset means `reverberant_target`
        length = min(len(noisy), len(clean))
        noise_spec = spectrogram(noisy[:length] - clean[:length], cfg.audio)
        return n_spec[:frames], ideal_ratio_mask(c_spec[:frames],
                                                 noise_spec[:frames])
    return n_spec[:frames], c_spec[:frames]


def _load_spectra(pairs: Sequence[Pair], cfg: Config
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Every pair's spectra, in order, with the empty pairs left out."""
    noisy_specs, target_specs = [], []
    for pair in pairs:
        spectra = _spectra(pair, cfg)
        if spectra is None:
            log.warning("skipping empty pair: %s", pair.noisy)
            continue
        noisy_specs.append(spectra[0])
        target_specs.append(spectra[1])
    if not noisy_specs:
        raise ValueError("every pair was empty or unreadable")
    return noisy_specs, target_specs


def _noisy_spectra(pairs: Sequence[Pair], cfg: Config) -> Iterator[np.ndarray]:
    """Each pair's noisy spectrogram in turn: what the Standardizer is fitted
    on, a file at a time.

    Empty pairs are passed over without a word here; the streamed dataset
    reports each one the first time an epoch meets it.
    """
    produced = False
    for pair in pairs:
        spectra = _spectra(pair, cfg, target=False)
        if spectra is not None:
            produced = True
            yield spectra[0]
    if not produced:
        raise ValueError("every pair was empty or unreadable")


def _fitted(standardizer: Optional[Standardizer], fit: bool,
            noisy_specs: Iterable[np.ndarray]) -> Standardizer:
    """*standardizer* fitted on *noisy_specs* when *fit* asks for it or it
    has never been fitted, and left as it is otherwise.

    Fitted in place, a spectrogram at a time, by
    :func:`~kudioenhance.features.fit_standardizer` -- held in memory or
    streamed, from a list or from a generator that reads the files as it
    goes -- so the two modes normalise with the same statistics to the bit.
    `kudio.Standardizer.fit` on the concatenated frames would be a second
    copy of every frame, and sums in float32: on an hour of frames it drifts
    from the true mean and deviation by a few parts in 10^5, where this
    stays within a part in 10^7.
    """
    if standardizer is None:
        standardizer = Standardizer()
    if fit or not standardizer.fitted:
        fitted = fit_standardizer(noisy_specs)
        standardizer.mean, standardizer.std = fitted.mean, fitted.std
    return standardizer


def _examples(noisy_spec: np.ndarray, target: np.ndarray, cfg: Config,
              standardizer: Standardizer, *,
              sequence: bool) -> Tuple[np.ndarray, np.ndarray]:
    """One pair's spectra as the rows the model trains on."""
    n_norm = standardizer.transform(noisy_spec)
    y_target = (target if cfg.model.predicts_mask
                else standardizer.transform(target))
    if sequence:
        return (frame_windows(n_norm, cfg.model.n_frames),
                frame_windows(y_target, cfg.model.n_frames))
    return stack_context(n_norm, cfg.model.context), y_target


def build_arrays(pairs: Sequence[Pair], cfg: Config, *, sequence: bool,
                 standardizer: Optional[Standardizer] = None,
                 fit: bool = False) -> Tuple[np.ndarray, np.ndarray, Standardizer]:
    """Turn pairs into ``(X, Y, standardizer)`` as plain arrays.

    Frame-wise models get ``(N, bins * (2 * context + 1))`` inputs against
    ``(N, bins)`` targets; sequence models get ``(N, n_frames, bins)`` for both.

    **The target depends on `cfg.model.target`.** For ``spectrum`` it is the
    normalised clean spectrogram. For ``irm`` it is a mask in ``[0, 1]``, which
    is **not** standardised — a mask is already on its own bounded scale, and
    putting the input's mean and standard deviation through it would produce a
    target the sigmoid output cannot even reach.

    Everything is held in memory, and a frame-wise X holds every frame
    ``2 * context + 1`` times over. Training no longer goes through here:
    :func:`build_dataset` gives the same rows without that, or without
    holding anything at all. This is kept for code that wants the matrices.
    """
    if not pairs:
        raise ValueError("no pairs to build arrays from")

    noisy_specs, target_specs = _load_spectra(pairs, cfg)
    standardizer = _fitted(standardizer, fit, noisy_specs)

    xs, ys = [], []
    for n_spec, target in zip(noisy_specs, target_specs):
        x, y = _examples(n_spec, target, cfg, standardizer, sequence=sequence)
        xs.append(x)
        ys.append(y)
    return (np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), standardizer)


def pair_examples(pair: Pair, cfg: Config, standardizer: Standardizer, *,
                  sequence: bool) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The ``(X, Y)`` rows *pair* contributes: exactly its rows of
    :func:`build_arrays`, or ``None`` when it has no frames.

    What a streamed epoch computes for each file as it reads it.
    """
    spectra = _spectra(pair, cfg)
    if spectra is None:
        return None
    return _examples(*spectra, cfg, standardizer, sequence=sequence)


def build_dataset(pairs: Sequence[Pair], cfg: Config, *, sequence: bool,
                  standardizer: Optional[Standardizer] = None,
                  fit: bool = False
                  ) -> Tuple[Union["InMemoryDataset", "StreamingDataset"],
                             Standardizer]:
    """Turn pairs into ``(dataset, standardizer)`` ready for ``train.fit``.

    The examples are :func:`build_arrays`'s — the same normalisation, the same
    targets — and what changes is where they are kept. ``cfg.train.streaming``
    chooses:

    **In memory**, the default. The features are computed once and held once.
    A frame-wise model's ±``context`` neighbours are stacked by
    `kudio.ContextFrames` for the rows a batch asks for, as it asks, instead of
    for the whole set up front, so the set takes the memory of its features
    rather than ``2 * context + 1`` times that: eleven times less at
    ``context: 5``. Row *i* is row *i* of ``build_arrays``'s X, and a training
    run is the same to the last digit. Sequence models' windows were never
    larger than their features, and are held as ``build_arrays`` builds them.

    **Streamed**, with ``train.streaming: true``. Nothing is held: every epoch
    reads the files again, in a fresh order, and mixes each one's examples
    through a buffer of ``train.shuffle_buffer`` of them — see
    :class:`~kudioenhance.datasets.StreamingDataset`. Fitting the
    Standardizer costs one more pass first, a file at a time. A validation
    set is streamed too, in a fixed order, so nothing held in memory grows
    with the corpus.

    Either way the Standardizer is fitted a spectrogram at a time by
    :func:`~kudioenhance.features.fit_standardizer`, so the two modes
    normalise with the same statistics, bit for bit, and so does
    ``build_arrays``.

    A *standardizer* passed in is fitted in place when *fit* is set or it has
    never been fitted, and reused as it is otherwise, as with ``build_arrays``.
    """
    from kudioenhance.datasets import InMemoryDataset, StreamingDataset

    if not pairs:
        raise ValueError("no pairs to build a dataset from")

    if cfg.train.streaming:
        # a generator, so the files are only read when there is fitting to do
        standardizer = _fitted(standardizer, fit, _noisy_spectra(pairs, cfg))
        return (StreamingDataset(pairs, cfg, standardizer, sequence=sequence),
                standardizer)

    if sequence:
        x, y, standardizer = build_arrays(pairs, cfg, sequence=True,
                                          standardizer=standardizer, fit=fit)
        return InMemoryDataset(x, y), standardizer

    noisy_specs, target_specs = _load_spectra(pairs, cfg)
    standardizer = _fitted(standardizer, fit, noisy_specs)
    # normalised in place, so each raw spectrogram is let go as soon as its
    # normalised copy exists
    for i, n_spec in enumerate(noisy_specs):
        noisy_specs[i] = standardizer.transform(n_spec)
        if not cfg.model.predicts_mask:
            target_specs[i] = standardizer.transform(target_specs[i])
    # ContextFrames pads every utterance at its own edges by repeating the
    # edge frame, which is what stack_context does by default
    inputs = kudio.ContextFrames(noisy_specs, cfg.model.context)
    targets = np.concatenate(target_specs, axis=0)
    return InMemoryDataset(inputs, targets), standardizer
