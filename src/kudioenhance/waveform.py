# -*- coding: utf-8 -*-
"""Waveform crops: the training examples of the models that see the complex
spectrum.

The log-power path cuts its examples out of spectrograms it computes up front
(:mod:`kudioenhance.data`) and throws the phase away. A model that works on
the complex spectrum, and is trained on losses that need the waveform back,
is fed the audio itself instead: ``data.crop_seconds`` cut from each
clean/noisy pair, at the same place in both, and at a fresh random place
every epoch. The STFT then runs on the batch, in torch, where the gradient
can pass through it (:mod:`kudioenhance.spectral`).

Where each crop is cut is drawn by :class:`CropSampler` in the process that
iterates the loader, from a generator seeded by the training seed. A worker
only reads the two files and cuts where it is told, so the batches are the
same with no workers or with eight, and the same again from the same seed.
The workers' own generators (torch, ``random`` and NumPy's global one) are
seeded by torch from the loader's generator, which is seeded as well, so
anything random a worker might do later repeats too.

The pairs come from wherever the stages read them: a prepared corpus
(``data.prepared``, through :func:`kudioenhance.data.load_prepared`) or a
synthesized run's split manifests.

Importing this module imports torch.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import kudio
import numpy as np
import torch

from kudioenhance.config import Config
from kudioenhance.data import Pair, reference_for
from kudioenhance.train import training_seed

__all__ = ["crop_length", "WaveformCrops", "CropSampler", "waveform_loader"]

#: A crop's draw is a number in [0, 2 ** 62), and its offset the draw modulo
#: the number of places the crop fits: uniform to better than a part in
#: 10 ** 10 for any file shorter than an hour at 16 kHz.
_DRAWS = 2 ** 62


def crop_length(cfg: Config) -> int:
    """``data.crop_seconds`` in samples at ``audio.sr``."""
    samples = int(round(cfg.data.crop_seconds * cfg.audio.sr))
    if samples < 1:
        raise ValueError(
            f"data.crop_seconds ({cfg.data.crop_seconds}) is less than one "
            f"sample at {cfg.audio.sr} Hz")
    return samples


def _offset(n_samples: int, length: int, draw: Optional[int]) -> int:
    """Where a crop of *length* starts in a pair of *n_samples*: the place
    *draw* picks, or the middle when it is ``None``. A pair no longer than
    the crop starts at 0 and is padded."""
    if n_samples <= length:
        return 0
    if draw is None:
        return (n_samples - length) // 2
    return int(draw) % (n_samples - length + 1)


def _cut(wave: np.ndarray, start: int, length: int) -> np.ndarray:
    """*length* samples of *wave* from *start*, zeros after its end."""
    piece = wave[start:start + length]
    if len(piece) < length:
        piece = np.pad(piece, (0, length - len(piece)))
    return np.ascontiguousarray(piece, dtype=np.float32)


class WaveformCrops(torch.utils.data.Dataset):
    """Clean/noisy pairs as crops of :func:`crop_length` samples.

    An item is ``(noisy, clean)``, two 1-D float32 tensors cut from the same
    place. The reference is :func:`~kudioenhance.data.reference_for` the
    pair, so ``data.reverberant_target`` means here what it means to the
    log-power path. Both files are read at ``audio.sr``, resampled when they
    are not at it, and the longer is cut to the shorter.

    ``crops[(i, draw)]`` is pair *i* cut where *draw*, a number from
    :class:`CropSampler`, says. ``crops[i]`` is the crop from the middle of
    pair *i*, the same every time: what validation reads, so its loss means
    the same from one epoch to the next.

    A pair shorter than a crop is padded with zeros at its end, noisy and
    clean alike, so the losses compare silence with silence there. An empty
    pair comes out as a crop of silence.
    """

    def __init__(self, pairs: Sequence[Pair], cfg: Config) -> None:
        if not pairs:
            raise ValueError("no pairs to crop")
        self.pairs = list(pairs)
        self.references = [reference_for(pair, cfg) for pair in self.pairs]
        self.sr = cfg.audio.sr
        self.length = crop_length(cfg)

    def __len__(self) -> int:
        return len(self.pairs)

    def __repr__(self) -> str:
        return (f"<WaveformCrops {len(self)} pair(s), crops of "
                f"{self.length} samples at {self.sr} Hz>")

    def __getitem__(self, key) -> Tuple[torch.Tensor, torch.Tensor]:
        index, draw = key if isinstance(key, tuple) else (key, None)
        index = int(index)
        noisy, _ = kudio.file_load(self.pairs[index].noisy, sr=self.sr)
        clean, _ = kudio.file_load(self.references[index], sr=self.sr)
        n_samples = min(len(noisy), len(clean))
        start = _offset(n_samples, self.length, draw)
        return (torch.from_numpy(_cut(noisy[:n_samples], start, self.length)),
                torch.from_numpy(_cut(clean[:n_samples], start, self.length)))


class CropSampler(torch.utils.data.Sampler):
    """One training epoch of :class:`WaveformCrops`: every pair once, in an
    order drawn from *generator*, each with a fresh draw for where its crop
    is cut.

    It yields ``(index, draw)`` keys, so everything random about an epoch is
    drawn here, in the process iterating the loader, and neither the number
    of workers nor which of them reads which file can change a batch.

    *generator* carries on from one epoch to the next: the second epoch is
    not the first again, and both repeat from the same seed. Its state
    (``generator.get_state()``) is what a resumed run has to put back.
    """

    def __init__(self, size: int, generator: torch.Generator) -> None:
        if size < 1:
            raise ValueError("nothing to sample from")
        self.size = size
        self.generator = generator

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        order = torch.randperm(self.size, generator=self.generator).tolist()
        draws = torch.randint(_DRAWS, (self.size,),
                              generator=self.generator).tolist()
        return iter(zip(order, draws))


def _generators(seed: Optional[int]) -> Tuple[torch.Generator, torch.Generator]:
    """Two generators from one seed: the sampler's, and the one the loader
    seeds its workers from.

    Kept apart because a loader draws its workers' seed once per iterator,
    and with persistent workers it builds only one: sharing a generator
    would make the second epoch's crops depend on the number of workers.
    Their seeds are drawn from a torch generator seeded with *seed*, so any
    seed the log-power path's shuffle takes is taken here too. ``None``
    seeds both afresh.
    """
    sampler, workers = torch.Generator(), torch.Generator()
    if seed is None:
        sampler.seed()
        workers.seed()
    else:
        parent = torch.Generator()
        parent.manual_seed(seed)
        first, second = torch.randint(_DRAWS, (2,), generator=parent).tolist()
        sampler.manual_seed(first)
        workers.manual_seed(second)
    return sampler, workers


def waveform_loader(pairs: Sequence[Pair], cfg: Config, *,
                    shuffle: bool = True, batch_size: Optional[int] = None,
                    num_workers: Optional[int] = None
                    ) -> torch.utils.data.DataLoader:
    """A ``DataLoader`` of ``(noisy, clean)`` batches of crops of *pairs*,
    each ``(batch, crop_length)`` float32.

    ``shuffle=True`` makes a training loader: every pair once an epoch, in a
    fresh order, each cut at a fresh place (see :class:`CropSampler`, which
    is ``loader.sampler``). ``shuffle=False`` reads the pairs in order, each
    its middle crop, the same every epoch: for validation.

    The draws are seeded by :func:`~kudioenhance.train.training_seed`;
    ``None`` there draws a fresh seed. *batch_size* and *num_workers* default
    to ``train.batch_size`` and ``train.num_workers``. Workers are kept from
    one epoch to the next: on Windows each is a new process that imports
    torch again, which costs seconds. A training and a validation loader
    each keep their own, so a validation loader with fewer, or none, saves
    the memory of processes that sit idle while the model trains.
    """
    dataset = WaveformCrops(pairs, cfg)
    sampler_generator, worker_generator = _generators(training_seed(cfg))
    workers = cfg.train.num_workers if num_workers is None else num_workers
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size or cfg.train.batch_size,
        sampler=(CropSampler(len(dataset), sampler_generator)
                 if shuffle else None),
        num_workers=workers,
        persistent_workers=workers > 0,
        generator=worker_generator,
    )
