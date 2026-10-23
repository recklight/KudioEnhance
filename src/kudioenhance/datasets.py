# -*- coding: utf-8 -*-
"""The training set as torch sees it: held in memory, or streamed from disk.

:func:`kudioenhance.data.build_dataset` returns one of the two, and
:func:`kudioenhance.train.fit` trains on either. All the loop asks of a
training set is two ways through it:

``shuffled_batches(batch_size, generator)``
    one training epoch, in an order drawn from *generator*, with a single
    example left over at the end folded into the batch before it;
``ordered_batches(batch_size)``
    the whole set in its stored order, which is how validation reads it.

Both hand out whole batches. A batch of frames is one gather from an array,
where the per-example ``__getitem__`` a ``torch.utils.data.DataLoader`` calls
costs a Python call and a tensor for every frame — and frame-wise training
draws hundreds of thousands of them an epoch. They are still ordinary torch
datasets, and work in a ``DataLoader`` for anyone who wants one.

Importing this module imports torch, which is why :mod:`kudioenhance.data`
only reaches for it once a dataset is actually built.
"""
from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from kudioenhance.config import Config
from kudioenhance.data import Pair, pair_examples
from kudioenhance.features import Standardizer
from kudioenhance.train import training_seed

log = logging.getLogger(__name__)

__all__ = ["InMemoryDataset", "StreamingDataset", "as_dataset"]

#: ``(inputs, targets)``, as float32 tensors on the CPU.
Batch = Tuple[torch.Tensor, torch.Tensor]


def as_dataset(x, y: Optional[np.ndarray] = None):
    """*x* if it already is a training set, or arrays *x* and *y* as one.

    A dataset carries its own targets, so *y* has to be ``None`` with one and
    has to be given with arrays; mixing the two up is refused rather than
    guessed at.
    """
    if hasattr(x, "shuffled_batches"):
        if y is not None:
            raise TypeError("a dataset carries its own targets; pass y=None "
                            "with it")
        return x
    if y is None:
        raise TypeError("training inputs given as an array need their "
                        "targets beside them")
    return InMemoryDataset(x, y)


def _tensors(x: np.ndarray, y: np.ndarray) -> Batch:
    return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)))


def _never_a_lone_last(batches: Iterable[Batch]) -> Iterator[Batch]:
    """*batches*, with a last batch of one example folded into the one before.

    BatchNorm cannot take a variance from a single example in training mode;
    Keras tolerated it and torch raises, so the straggler that ``batch_size``
    leaves at the end of an epoch joins the batch before it instead. Only
    the last batch is looked at. Every batch is a single example at a
    ``batch_size`` of 1, and a set of one example has no batch before it to
    join; :func:`~kudioenhance.train.fit` refuses both for the frame-wise
    model that cannot train on them, and the sequence models can. Two
    batches are held back, which is what it takes to know which one is last.
    """
    held: List[Batch] = []
    for batch in batches:
        held.append(batch)
        if len(held) > 2:
            yield held.pop(0)
    if len(held) == 2 and len(held[1][0]) == 1:
        (x0, y0), (x1, y1) = held
        held = [(torch.cat([x0, x1]), torch.cat([y0, y1]))]
    yield from held


# ------------------------------------------------------------------ in memory

class InMemoryDataset(torch.utils.data.Dataset):
    """Training examples held in memory, gathered a batch at a time.

    *inputs* is anything that answers ``inputs[indices]`` with a float32 array
    of those rows: a plain array, or the `kudio.ContextFrames` a frame-wise
    model's features are kept in, which stacks a row's neighbours only when
    the row is asked for. *targets* is an array with a row for every input.

    Indexed with an int it gives one ``(x, y)`` example, as any torch dataset
    does; with an index array, an index tensor or a slice, the whole batch in
    one gather.
    """

    def __init__(self, inputs, targets) -> None:
        if isinstance(inputs, np.ndarray):
            inputs = np.ascontiguousarray(inputs, dtype=np.float32)
        targets = np.ascontiguousarray(targets, dtype=np.float32)
        if len(inputs) != len(targets):
            raise ValueError(f"{len(inputs)} input row(s) against "
                             f"{len(targets)} target row(s)")
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> Batch:
        if isinstance(index, torch.Tensor):
            index = index.numpy()
        return _tensors(self.inputs[index], self.targets[index])

    def __repr__(self) -> str:
        return (f"<InMemoryDataset {len(self)} example(s) of "
                f"X{tuple(self.inputs.shape[1:])} Y{self.targets.shape[1:]}>")

    def shuffled_batches(self, batch_size: int,
                         generator: torch.Generator) -> Iterator[Batch]:
        """One training epoch: every example once, reshuffled from
        *generator*."""
        order = torch.randperm(len(self), generator=generator)
        return _never_a_lone_last(
            self[index] for index in torch.split(order, batch_size))

    def ordered_batches(self, batch_size: int) -> Iterator[Batch]:
        """Every example once, in the order the set holds them."""
        for start in range(0, len(self), batch_size):
            yield self[start:start + batch_size]


# ------------------------------------------------------------------ streamed

class StreamingDataset(torch.utils.data.IterableDataset):
    """Training examples computed from the audio a file at a time, every epoch.

    Nothing is held but the file being read and the shuffle buffer, so the
    memory a run takes does not grow with the corpus. The price is reading,
    resampling and transforming every file again each epoch.

    A shuffled epoch takes the files in a fresh order drawn from the
    generator and works out each one's examples with
    :func:`~kudioenhance.data.pair_examples` — exactly its rows of
    ``build_arrays``, so a frame-wise row is stacked from its own file's
    frames and never reaches into the next. They join a pool, and every
    batch is drawn at random, without replacement, from everything in the
    pool, for as long as at least ``cfg.train.shuffle_buffer`` examples would
    be left waiting. When the files run out, what is left goes out in random
    order. A buffer of ``0`` makes the pool the file being read and what
    earlier draws left over, which is less than a batch.

    The pool keeps examples as the model sees them, so a buffer costs
    ``shuffle_buffer`` times one input and one target: for a frame-wise model
    ``bins * (2 * context + 2)`` float32s an example, for a sequence model
    ``2 * n_frames * bins``.

    An ordered pass reads the files in manifest order and cuts its batches
    exactly where slicing the concatenated set would, so streamed validation
    scores what the in-memory set would score.

    *standardizer* has to be fitted already; ``build_dataset`` fits it in a
    streaming pass of its own before building one of these.
    """

    def __init__(self, pairs: Sequence[Pair], cfg: Config,
                 standardizer: Standardizer, *, sequence: bool) -> None:
        if not pairs:
            raise ValueError("no pairs to stream")
        if not standardizer.fitted:
            raise ValueError("the Standardizer has to be fitted before the "
                             "files can be streamed through it")
        self.pairs = list(pairs)
        self.cfg = cfg
        self.standardizer = standardizer
        self.sequence = sequence
        self._reported: Set[str] = set()
        # for iterating the dataset directly; fit shuffles from a generator of
        # its own, seeded the same way
        self.generator = torch.Generator()
        seed = training_seed(cfg)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

    def __repr__(self) -> str:
        return (f"<StreamingDataset {len(self.pairs)} file(s), shuffle buffer "
                f"{self.cfg.train.shuffle_buffer}>")

    def shuffled_batches(self, batch_size: int,
                         generator: torch.Generator) -> Iterator[Batch]:
        """One training epoch: every file read once, in an order drawn from
        *generator*, its examples mixed through the shuffle buffer."""
        order = torch.randperm(len(self.pairs), generator=generator).tolist()
        # numpy draws k of n without replacement in O(k), where
        # torch.randperm would cost the whole pool for every batch; seeded
        # from the same generator, so the one seed still fixes the epoch
        rng = np.random.default_rng(
            int(torch.randint(2 ** 62, (), generator=generator)))
        return _never_a_lone_last(
            _tensors(x, y) for x, y in self._drawn(order, batch_size, rng))

    def ordered_batches(self, batch_size: int) -> Iterator[Batch]:
        """Every example once, file by file in manifest order, with batches
        that run across file boundaries."""
        carried: Optional[Tuple[np.ndarray, np.ndarray]] = None
        for x, y in self._examples(self.pairs):
            if carried is not None:
                x = np.concatenate([carried[0], x])
                y = np.concatenate([carried[1], y])
            whole = len(x) - len(x) % batch_size
            for start in range(0, whole, batch_size):
                yield _tensors(x[start:start + batch_size],
                               y[start:start + batch_size])
            carried = (x[whole:], y[whole:]) if whole < len(x) else None
        if carried is not None:
            yield _tensors(*carried)

    def __iter__(self) -> Iterator[Batch]:
        """One shuffled epoch an example at a time, for a
        ``torch.utils.data.DataLoader``; every pass reshuffles.

        It reads its files in the process iterating it. A ``DataLoader`` with
        workers would hand every worker every file, and each would repeat the
        epoch, so that is refused rather than quietly training on duplicates.
        """
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("StreamingDataset reads its files in one "
                               "process; iterate it with num_workers=0")
        for inputs, targets in self.shuffled_batches(
                self.cfg.train.batch_size, self.generator):
            yield from zip(inputs, targets)

    # -------------------------------------------------------------- inside

    def _examples(self, pairs: Iterable[Pair]
                  ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Each pair's examples in turn, reporting an empty pair the first
        time it is met rather than every epoch."""
        produced = False
        for pair in pairs:
            examples = pair_examples(pair, self.cfg, self.standardizer,
                                     sequence=self.sequence)
            if examples is None:
                if pair.noisy not in self._reported:
                    self._reported.add(pair.noisy)
                    log.warning("skipping empty pair: %s", pair.noisy)
                continue
            produced = True
            yield examples
        if not produced:
            raise ValueError("every pair was empty or unreadable")

    def _drawn(self, order: List[int], batch_size: int,
               rng: np.random.Generator
               ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """The batches of one shuffled epoch, as arrays."""
        keep = self.cfg.train.shuffle_buffer
        pool = _ShufflePool(rng, reserve=keep + batch_size)
        for x, y in self._examples(self.pairs[i] for i in order):
            pool.add(x, y)
            while len(pool) >= keep + batch_size:
                yield pool.draw(batch_size)
        if len(pool):
            x, y = pool.draw(len(pool))
            for start in range(0, len(x), batch_size):
                yield x[start:start + batch_size], y[start:start + batch_size]


class _ShufflePool:
    """Examples waiting to be drawn, each draw uniform over all of them.

    The rows live in one array, grown when a file brings more than there is
    room for. A draw takes *k* rows at random and moves the undrawn rows from
    the top *k* places into the holes, so a draw costs the rows drawn rather
    than the rows waiting.
    """

    def __init__(self, rng: np.random.Generator, reserve: int) -> None:
        self._rng = rng
        self._reserve = reserve
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        size, k = self._size, len(x)
        if self._x is None or size + k > len(self._x):
            capacity = max(size + k, self._reserve + k)
            self._x = _regrown(self._x, size, capacity, x)
            self._y = _regrown(self._y, size, capacity, y)
        self._x[size:size + k] = x
        self._y[size:size + k] = y
        self._size = size + k

    def draw(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        size = self._size
        picked = self._rng.choice(size, size=k, replace=False)
        x, y = self._x[picked], self._y[picked]
        holes = picked[picked < size - k]
        undrawn = np.setdiff1d(np.arange(size - k, size), picked,
                               assume_unique=True)
        self._x[holes] = self._x[undrawn]
        self._y[holes] = self._y[undrawn]
        self._size = size - k
        return x, y


def _regrown(rows: Optional[np.ndarray], size: int, capacity: int,
             like: np.ndarray) -> np.ndarray:
    """Room for *capacity* rows shaped like *like*'s, the first *size* of
    them copied over from *rows*."""
    grown = np.empty((capacity,) + like.shape[1:], dtype=np.float32)
    if rows is not None:
        grown[:size] = rows[:size]
    return grown
