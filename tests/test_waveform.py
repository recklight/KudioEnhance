# -*- coding: utf-8 -*-
"""Waveform crops: cut at the same place in both halves, padded when short,
fresh every epoch, and the same batches from the same seed whatever the
number of workers."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs PyTorch",
                            exc_type=ImportError)

import kudio  # noqa: E402

from kudioenhance.config import Config  # noqa: E402
from kudioenhance.data import Pair, load_prepared  # noqa: E402
from kudioenhance.waveform import (  # noqa: E402
    CropSampler,
    WaveformCrops,
    crop_length,
    waveform_loader,
)

SR = 8000
CROP = 400
#: Sample k of clean file i is 0.2 i + STEP (k + 1), so a crop says which
#: file it came from and where it was cut, all from its first sample.
STEP = 1e-4
BASE = 0.2
#: noisy = clean + OFFSET, so noisy - clean shows the two were cut together.
OFFSET = 0.25
LENGTHS = {"long": 1000, "middling": 700, "exact": CROP, "short": 250}


def _write(path, wave) -> None:
    kudio.save_wave(path, wave.astype(np.float32), SR, subtype="FLOAT")


@pytest.fixture
def prepared(tmp_path):
    """A folder shaped like `kudioenhance prepare`'s: the wave files under
    the split, and a manifest whose paths are relative to the folder."""
    folder = tmp_path / "prepared"
    pairs = []
    for i, (name, n_samples) in enumerate(LENGTHS.items()):
        clean = BASE * i + STEP * np.arange(1, n_samples + 1)
        for kind, wave in (("clean", clean), ("noisy", clean + OFFSET)):
            (folder / "train" / kind).mkdir(parents=True, exist_ok=True)
            _write(folder / "train" / kind / f"{name}.wav", wave)
        pairs.append(Pair(noisy=f"train/noisy/{name}.wav",
                          clean=f"train/clean/{name}.wav"))
    kudio.save_manifest(folder / "train.json", pairs)
    return folder


def _config(folder, seed=5, **train) -> Config:
    return Config.from_dict({
        "audio": {"sr": SR, "n_fft": 64, "hop_length": 32},
        "data": {"prepared": str(folder), "seed": seed,
                 "crop_seconds": CROP / SR},
        "train": {"batch_size": 2, **train},
    })


def _origin(clean_crop: torch.Tensor):
    """``(file index, start)`` of *clean_crop*, read off its first sample."""
    first = clean_crop[0].item()
    index = int(first // BASE)
    return index, int(round((first - BASE * index) / STEP)) - 1


def _start(clean_crop: torch.Tensor) -> int:
    return _origin(clean_crop)[1]


def _epoch(loader):
    return [(noisy.clone(), clean.clone()) for noisy, clean in loader]


def _same(first, second) -> bool:
    return len(first) == len(second) and all(
        torch.equal(a, c) and torch.equal(b, d)
        for (a, b), (c, d) in zip(first, second))


# ------------------------------------------------------------------- crops

def test_the_crop_length_is_in_seconds(prepared):
    cfg = _config(prepared)
    assert crop_length(cfg) == CROP
    cfg.data.crop_seconds = 1e-5
    with pytest.raises(ValueError, match="less than one sample"):
        crop_length(cfg)


def test_a_crop_is_cut_at_the_same_place_in_both_halves(prepared):
    cfg = _config(prepared)
    crops = WaveformCrops(load_prepared(prepared, "train"), cfg)
    names = list(LENGTHS)
    for index, draw in CropSampler(len(crops), torch.Generator().manual_seed(1)):
        noisy, clean = crops[(index, draw)]
        assert noisy.shape == clean.shape == (CROP,)
        assert noisy.dtype == clean.dtype == torch.float32
        n_samples = LENGTHS[names[index]]
        came_from, start = _origin(clean)
        assert came_from == index
        assert 0 <= start <= max(n_samples - CROP, 0)
        valid = min(CROP, n_samples)
        expected = BASE * index + STEP * np.arange(start + 1, start + valid + 1)
        np.testing.assert_allclose(clean[:valid].numpy(), expected, rtol=1e-6)
        np.testing.assert_allclose((noisy - clean)[:valid].numpy(), OFFSET,
                                   atol=1e-6)


def test_a_short_pair_is_padded_with_silence_at_its_end(prepared):
    cfg = _config(prepared)
    crops = WaveformCrops(load_prepared(prepared, "train"), cfg)
    short = list(LENGTHS).index("short")
    for key in (short, (short, 12345)):
        noisy, clean = crops[key]
        assert _start(clean) == 0
        assert torch.count_nonzero(clean[:250]) == 250
        assert torch.count_nonzero(clean[250:]) == 0
        assert torch.count_nonzero(noisy[250:]) == 0


def test_an_index_alone_gives_the_middle_crop_every_time(prepared):
    cfg = _config(prepared)
    crops = WaveformCrops(load_prepared(prepared, "train"), cfg)
    for index, n_samples in enumerate(LENGTHS.values()):
        first, second = crops[index], crops[index]
        assert torch.equal(first[1], second[1])
        assert _start(first[1]) == max(n_samples - CROP, 0) // 2


def test_the_reference_follows_reverberant_target(prepared):
    """With ``reverberant_target`` the clean half is the pair's target, as
    it is for the log-power path."""
    cfg = _config(prepared)
    cfg.data.rt60, cfg.data.reverberant_target = [0.3], True
    target = prepared / "train" / "target.wav"
    _write(target, np.full(1000, 0.125))
    pair = load_prepared(prepared, "train")[0]
    pair = Pair(noisy=pair.noisy, clean=pair.clean, target=str(target))
    _, clean = WaveformCrops([pair], cfg)[0]
    assert torch.allclose(clean, torch.tensor(0.125))


# ------------------------------------------------------------------ epochs

def test_every_pair_once_an_epoch_cut_somewhere_new_each_epoch(prepared):
    cfg = _config(prepared)
    loader = waveform_loader(load_prepared(prepared, "train"), cfg)
    assert isinstance(loader.sampler, CropSampler)
    starts = []
    for _ in range(3):
        batches = _epoch(loader)
        assert [len(noisy) for noisy, _ in batches] == [2, 2]
        epoch = dict(_origin(crop) for _, clean in batches for crop in clean)
        assert sorted(epoch) == list(range(len(LENGTHS)))
        starts.append(tuple(epoch[i] for i in range(len(LENGTHS))))
    # the long file fits 601 places, so three epochs on one would be a fluke
    assert len(set(starts)) > 1


def test_the_same_seed_gives_the_same_batches_and_another_does_not(prepared):
    pairs = load_prepared(prepared, "train")
    first = waveform_loader(pairs, _config(prepared))
    again = waveform_loader(pairs, _config(prepared))
    other = waveform_loader(pairs, _config(prepared, seed=6))
    one, two = _epoch(first), _epoch(first)
    assert not _same(one, two)                       # epochs differ...
    assert _same(one, _epoch(again))                 # ...and repeat
    assert _same(two, _epoch(again))
    assert not _same(one, _epoch(other))


def test_train_seed_overrides_data_seed(prepared):
    """``train.seed`` wins over ``data.seed``, as it does for the log-power
    path's shuffle."""
    pairs = load_prepared(prepared, "train")
    borrowed = waveform_loader(pairs, _config(prepared, seed=5))
    cfg = _config(prepared, seed=99)
    cfg.train.seed = 5
    assert _same(_epoch(borrowed), _epoch(waveform_loader(pairs, cfg)))


def test_validation_reads_the_middle_crops_in_order(prepared):
    cfg = _config(prepared)
    loader = waveform_loader(load_prepared(prepared, "train"), cfg,
                             shuffle=False, batch_size=4)
    (_, clean), = _epoch(loader)
    assert [_origin(crop) for crop in clean] == [
        (i, max(n - CROP, 0) // 2) for i, n in enumerate(LENGTHS.values())]
    assert _same(_epoch(loader), _epoch(loader))


def test_the_batches_do_not_depend_on_the_number_of_workers(prepared):
    """Where every crop is cut is drawn in this process, so two worker
    processes, kept between epochs, hand back what reading here does, epoch
    after epoch."""
    pairs = load_prepared(prepared, "train")
    here = waveform_loader(pairs, _config(prepared, num_workers=0))
    workers = waveform_loader(pairs, _config(prepared, num_workers=2))
    assert workers.num_workers == 2 and workers.persistent_workers
    for _ in range(2):
        assert _same(_epoch(here), _epoch(workers))
    del workers


def test_a_negative_seed_works_as_it_does_for_the_log_power_path(prepared):
    pairs = load_prepared(prepared, "train")
    first = _epoch(waveform_loader(pairs, _config(prepared, seed=-3)))
    assert _same(first, _epoch(waveform_loader(pairs,
                                               _config(prepared, seed=-3))))
