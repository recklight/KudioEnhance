# -*- coding: utf-8 -*-
"""Training sets: held in memory without the context blow-up, or streamed.

Both have to hand the model the rows `build_arrays` would, and the tests hold
them to that row for row, not just shape for shape: a training set that is
cheaper but subtly different is a different experiment.
"""
from __future__ import annotations

import logging
import tracemalloc

import numpy as np
import pytest

import kudio
from kudio_enhance.data import (
    build_arrays,
    build_dataset,
    pair_examples,
    synthesize,
)
from kudio_enhance.features import fit_standardizer


@pytest.fixture
def pairs(config):
    """Every clean file under every noise at every SNR: sixteen files of
    about forty frames, enough for a shuffle to have somewhere to go."""
    config.data.mode = "inc"
    return synthesize(config, "exp1")


def _everything(dataset):
    """A whole in-memory set as two arrays."""
    x, y = dataset[np.arange(len(dataset))]
    return x.numpy(), y.numpy()


def _concatenated(batches):
    xs, ys = zip(*batches)
    return (np.concatenate([x.numpy() for x in xs]),
            np.concatenate([y.numpy() for y in ys]))


def _sorted_rows(array: np.ndarray) -> np.ndarray:
    """The rows of *array* in a canonical order, to compare as multisets."""
    flat = array.reshape(len(array), -1)
    return flat[np.lexsort(flat.T[::-1])]


# ================================================================ in memory

@pytest.mark.parametrize("context", [0, 1, 3])
@pytest.mark.parametrize("target", ["spectrum", "irm"])
def test_the_in_memory_set_holds_exactly_the_rows_build_arrays_stacks(
        config, pairs, torch, context, target):
    """Row for row and bit for bit, standardizer included: the training run
    it feeds is the one the stacked matrix used to feed."""
    config.model.context = context
    config.model.target = target
    dataset, std = build_dataset(pairs, config, sequence=False, fit=True)
    x, y, std_arrays = build_arrays(pairs, config, sequence=False, fit=True)

    x_set, y_set = _everything(dataset)
    assert np.array_equal(x_set, x)
    assert np.array_equal(y_set, y)
    assert np.array_equal(std.mean, std_arrays.mean)
    assert np.array_equal(std.std, std_arrays.std)


def test_reverberant_target_rows_match_too(config, pairs, torch):
    """The reference comes from `reference_for`, the same as the arrays'."""
    config.data.rt60 = [0.5]
    config.data.reverberant_target = True
    config.model.target = "irm"
    wet = synthesize(config, "exp-wet")
    assert all(pair.target for pair in wet)

    dataset, _ = build_dataset(wet, config, sequence=False, fit=True)
    x, y, _ = build_arrays(wet, config, sequence=False, fit=True)
    x_set, y_set = _everything(dataset)
    assert np.array_equal(x_set, x) and np.array_equal(y_set, y)


def test_a_sequence_set_holds_the_windows_build_arrays_cuts(config, pairs,
                                                            torch):
    dataset, _ = build_dataset(pairs, config, sequence=True, fit=True)
    x, y, _ = build_arrays(pairs, config, sequence=True, fit=True)

    x_set, y_set = _everything(dataset)
    assert x_set.shape == (len(x), config.model.n_frames, config.audio.n_bins)
    assert np.array_equal(x_set, x) and np.array_equal(y_set, y)


def test_a_frame_wise_set_keeps_each_frame_once_not_once_per_neighbour(
        config, pairs, torch):
    """The reason it exists. At ±10 frames the stacked matrix is twenty-one
    frames wide; the set keeps the features and stacks a batch as it is
    drawn, and it never builds the whole matrix even for a moment."""
    config.model.context = 10
    x, _, _ = build_arrays(pairs, config, sequence=False, fit=True)

    tracemalloc.start()
    try:
        dataset, _ = build_dataset(pairs, config, sequence=False, fit=True)
        held, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert isinstance(dataset.inputs, kudio.ContextFrames)
    assert held < x.nbytes / 4
    assert peak < x.nbytes / 2
    assert dataset[np.arange(5)][0].shape == (5, x.shape[1])


def test_one_example_at_a_time_is_what_a_data_loader_gets(config, pairs,
                                                          torch):
    dataset, _ = build_dataset(pairs, config, sequence=False, fit=True)
    x_row, y_row = dataset[3]
    assert x_row.shape == (config.audio.n_bins
                           * (2 * config.model.context + 1),)

    loader = torch.utils.data.DataLoader(dataset, batch_size=8)
    x_batch, y_batch = next(iter(loader))
    x_first, y_first = dataset[np.arange(8)]
    assert torch.equal(x_batch, x_first) and torch.equal(y_batch, y_first)


# ============================================================ standardizer

def test_a_standardizer_fitted_a_chunk_at_a_time_matches_kudios_fit():
    """Same formula — mean, population deviation, the same epsilon — so on
    a set this small, where kudio's float32 sums have not drifted yet, the
    same statistics to float32 precision. Chunks of every awkward size are
    in there: one row, none at all, and one much larger than the rest."""
    rng = np.random.default_rng(0)
    sizes = [37, 1, 0, 400, 2, 63]
    chunks = [(rng.standard_normal((n, 9)) * 2.5 - 4.0).astype(np.float32)
              for n in sizes]

    streamed = fit_standardizer(iter(chunks))
    whole = kudio.Standardizer().fit(np.concatenate(chunks))

    assert streamed.mean.dtype == whole.mean.dtype == np.float32
    assert streamed.std.dtype == whole.std.dtype == np.float32
    np.testing.assert_allclose(streamed.mean, whole.mean, rtol=1e-6)
    np.testing.assert_allclose(streamed.std, whole.std, rtol=1e-6)


def test_a_column_that_never_changes_gets_kudios_epsilon_exactly():
    """Not a division by zero, and not an epsilon of this module's own
    choosing."""
    chunks = [np.full((n, 3), 0.5, dtype=np.float32) for n in (4, 7, 1)]
    streamed = fit_standardizer(chunks)
    whole = kudio.Standardizer().fit(np.concatenate(chunks))

    assert np.array_equal(streamed.mean, whole.mean)
    assert np.array_equal(streamed.std, whole.std)
    assert streamed.std[0] == np.float32(1e-8)


def test_the_chunked_fit_does_not_drift_with_the_number_of_frames():
    """kudio sums in float32, along the frames, so its error grows with the
    corpus; summed in float64 this stays at float32's own precision. Here
    are 200,000 frames, under an hour of 16 kHz audio, in file-sized chunks,
    against the statistics worked out in float64."""
    rng = np.random.default_rng(0)
    frames = (rng.standard_normal((200_000, 8)) * 2.0 - 6.0).astype(
        np.float32)
    truth = frames.astype(np.float64)
    mean, std = truth.mean(axis=0), truth.std(axis=0)

    chunked = fit_standardizer(np.array_split(frames, 800))
    np.testing.assert_allclose(chunked.mean, mean, rtol=1e-6)
    np.testing.assert_allclose(chunked.std, std, rtol=1e-6)

    # ...which is the reason training does not fit with kudio's own
    whole = kudio.Standardizer().fit(frames)
    assert np.max(np.abs(whole.std - std) / std) > 1e-6


def test_fitting_on_nothing_or_on_mismatched_chunks_is_refused():
    with pytest.raises(ValueError, match="no frames"):
        fit_standardizer([np.zeros((0, 4), dtype=np.float32)])
    with pytest.raises(ValueError, match="column"):
        fit_standardizer([np.zeros((3, 4)), np.zeros((3, 5))])


@pytest.mark.parametrize("sequence", [False, True])
def test_held_and_streamed_sets_are_normalised_with_the_same_statistics(
        config, pairs, torch, sequence):
    """Bit for bit, and the same as ``build_arrays``: one config trains on
    the same normalisation in either mode, whatever the size of the corpus."""
    _, std_held = build_dataset(pairs, config, sequence=sequence,
                                fit=True)
    _, _, std_arrays = build_arrays(pairs, config, sequence=sequence,
                                    fit=True)
    config.train.streaming = True
    _, std_streamed = build_dataset(pairs, config, sequence=sequence,
                                    fit=True)

    for std in (std_arrays, std_streamed):
        assert np.array_equal(std.mean, std_held.mean)
        assert np.array_equal(std.std, std_held.std)


def test_a_fitted_standardizer_is_reused_by_a_streamed_set(config, pairs,
                                                           torch):
    """A validation set has to be normalised with the training set's
    statistics, not its own."""
    config.train.streaming = True
    _, std = build_dataset(pairs[:4], config, sequence=False, fit=True)
    mean = std.mean.copy()
    _, same = build_dataset(pairs[4:], config, sequence=False,
                            standardizer=std)
    assert same is std and np.array_equal(std.mean, mean)


# ================================================================ streamed

@pytest.mark.parametrize("name, target, rooms", [
    ("ddae", "spectrum", False),
    ("ddae", "irm", False),
    ("blstm", "spectrum", False),
    ("conv_ae", "irm", True),
])
def test_a_streamed_pass_in_order_gives_build_arrays_rows(
        config, pairs, torch, name, target, rooms):
    """Stream the files in manifest order and the rows are the stacked
    matrix's, cut into batches where slicing it would cut them, so streamed
    validation scores exactly what the held set scores."""
    from kudio_enhance.models import is_sequence

    config.model.name = name
    config.model.target = target
    if rooms:
        config.data.rt60 = [0.5]
        config.data.reverberant_target = True
        pairs = synthesize(config, "exp-wet")
    sequence = is_sequence(name)
    held, std = build_dataset(pairs, config, sequence=sequence, fit=True)
    x, y, _ = build_arrays(pairs, config, sequence=sequence, standardizer=std)

    config.train.streaming = True
    streamed, _ = build_dataset(pairs, config, sequence=sequence,
                                standardizer=std)
    batches = list(streamed.ordered_batches(config.train.batch_size))
    x_streamed, y_streamed = _concatenated(batches)
    assert np.array_equal(x_streamed, x)
    assert np.array_equal(y_streamed, y)

    held_batches = list(held.ordered_batches(config.train.batch_size))
    assert len(batches) == len(held_batches)
    for (xs, ys), (xh, yh) in zip(batches, held_batches):
        assert torch.equal(xs, xh) and torch.equal(ys, yh)


@pytest.mark.parametrize("shuffle_buffer", [0, 50, 100_000])
def test_a_shuffled_epoch_is_every_example_exactly_once(config, pairs, torch,
                                                        shuffle_buffer):
    config.train.streaming = True
    config.train.shuffle_buffer = shuffle_buffer
    streamed, std = build_dataset(pairs, config, sequence=False, fit=True)
    x, y, _ = build_arrays(pairs, config, sequence=False, standardizer=std)

    generator = torch.Generator().manual_seed(0)
    x_epoch, y_epoch = _concatenated(streamed.shuffled_batches(32, generator))
    assert not np.array_equal(x_epoch, x)
    assert np.array_equal(_sorted_rows(np.hstack([x_epoch, y_epoch])),
                          _sorted_rows(np.hstack([x, y])))


def test_every_epoch_reshuffles_and_the_seed_repeats_them(config, pairs,
                                                          torch, monkeypatch):
    """The order the files are read in and the draws from the buffer both
    come from the one generator: the same seed gives the same epochs, file
    for file and batch for batch, and the next epoch from the same
    generator reads the files in a different order."""
    import kudio_enhance.datasets as datasets

    config.train.streaming = True
    config.train.shuffle_buffer = 50
    streamed, _ = build_dataset(pairs, config, sequence=False, fit=True)
    read = []
    real = datasets.pair_examples

    def reading(pair, *args, **kwargs):
        read.append(pair.noisy)
        return real(pair, *args, **kwargs)
    monkeypatch.setattr(datasets, "pair_examples", reading)

    def epochs(seed, count=2):
        generator = torch.Generator().manual_seed(seed)
        runs = []
        for _ in range(count):
            read.clear()
            runs.append((list(streamed.shuffled_batches(32, generator)),
                         list(read)))
        return runs

    def same(a, b):
        return len(a) == len(b) and all(
            torch.equal(xa, xb) and torch.equal(ya, yb)
            for (xa, ya), (xb, yb) in zip(a, b))

    (first, first_files), (second, second_files) = epochs(7)
    (again, again_files), (again_second, _) = epochs(7)
    assert sorted(first_files) == sorted(p.noisy for p in pairs)
    assert first_files == again_files and same(first, again)
    assert same(second, again_second)
    assert second_files != first_files and not same(first, second)
    (other, other_files), = epochs(8, count=1)
    assert other_files != first_files and not same(first, other)


def _owners(pairs, config, std):
    """Which file every input row came from."""
    owner = {}
    for index, pair in enumerate(pairs):
        x, _ = pair_examples(pair, config, std, sequence=False)
        for row in x:
            owner.setdefault(row.tobytes(), set()).add(index)
    return owner


def test_a_buffer_of_zero_mixes_within_a_file_and_a_big_one_across_files(
        config, pairs, torch):
    """What ``shuffle_buffer`` means. At 0 the first batch is drawn the
    moment the first file is read, from that file alone; with room for the
    whole corpus it waits until every file is in and draws on all of them."""
    config.train.streaming = True
    streamed, std = build_dataset(pairs, config, sequence=False, fit=True)
    owner = _owners(pairs, config, std)
    frames_per_file = min(
        len(pair_examples(pair, config, std, sequence=False)[0])
        for pair in pairs)
    assert frames_per_file >= 32

    def files_in_first_batch(shuffle_buffer):
        config.train.shuffle_buffer = shuffle_buffer
        generator = torch.Generator().manual_seed(0)
        x, _ = next(iter(streamed.shuffled_batches(32, generator)))
        return set().union(*(owner[row.tobytes()] for row in x.numpy()))

    assert len(files_in_first_batch(0)) == 1
    assert len(files_in_first_batch(100_000)) > 4


def test_a_streamed_epoch_never_ends_on_a_lone_example(config, pairs, torch):
    """BatchNorm cannot train on a batch of one, held or streamed."""
    config.train.streaming = True
    streamed, std = build_dataset(pairs, config, sequence=False, fit=True)
    total = len(build_arrays(pairs, config, sequence=False,
                             standardizer=std)[0])

    generator = torch.Generator().manual_seed(0)
    sizes = [len(x) for x, _ in streamed.shuffled_batches(total - 1,
                                                          generator)]
    assert sizes == [total]


def test_an_empty_file_is_reported_once_however_many_epochs(
        config, pairs, torch, monkeypatch, caplog):
    """Reported every epoch it would bury everything else in the log."""
    import kudio_enhance.data as data

    config.train.streaming = True
    streamed, _ = build_dataset(pairs, config, sequence=False, fit=True)
    empty = pairs[2].noisy
    real = data._spectra

    def spectra(pair, cfg, **kwargs):
        return None if pair.noisy == empty else real(pair, cfg, **kwargs)
    monkeypatch.setattr(data, "_spectra", spectra)

    generator = torch.Generator().manual_seed(0)
    with caplog.at_level(logging.WARNING, logger="kudio_enhance"):
        for _ in range(3):
            list(streamed.shuffled_batches(32, generator))
        list(streamed.ordered_batches(32))
    reports = [r for r in caplog.records if "empty pair" in r.getMessage()]
    assert len(reports) == 1 and empty in reports[0].getMessage()


def test_a_streamed_set_iterates_like_any_torch_iterable_dataset(
        config, pairs, torch):
    """One epoch per pass, every example once, a new order each time; and
    it refuses worker processes, each of which would read every file."""
    config.train.streaming = True
    config.train.shuffle_buffer = 50
    streamed, std = build_dataset(pairs, config, sequence=False, fit=True)
    x, _, _ = build_arrays(pairs, config, sequence=False, standardizer=std)

    loader = torch.utils.data.DataLoader(streamed, batch_size=16)
    first = torch.cat([batch for batch, _ in loader]).numpy()
    second = torch.cat([batch for batch, _ in loader]).numpy()
    assert np.array_equal(_sorted_rows(first), _sorted_rows(x))
    assert np.array_equal(_sorted_rows(second), _sorted_rows(x))
    assert not np.array_equal(first, second)

    with pytest.raises(RuntimeError, match="num_workers=0"):
        next(iter(torch.utils.data.DataLoader(streamed, num_workers=1)))


def test_a_streamed_set_needs_pairs_and_fitted_statistics(config, torch):
    from kudio_enhance.data import Pair
    from kudio_enhance.datasets import StreamingDataset
    from kudio_enhance.features import Standardizer

    config.train.streaming = True
    with pytest.raises(ValueError, match="no pairs"):
        build_dataset([], config, sequence=False, fit=True)
    with pytest.raises(ValueError, match="fitted"):
        StreamingDataset([Pair("a.wav", "b.wav", "white", 0)], config,
                         Standardizer(), sequence=False)
