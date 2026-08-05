# -*- coding: utf-8 -*-
"""A tiny synthetic corpus, so the whole pipeline is exercised in seconds."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

SR = 8000


def _harmonic(seed: int, seconds: float = 0.6) -> np.ndarray:
    """A crude voiced sound: a few harmonics under an envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    f0 = rng.uniform(110, 220)
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    y *= np.hanning(len(t))
    return (0.6 * y / np.max(np.abs(y))).astype(np.float32)


@pytest.fixture
def corpus(tmp_path) -> dict:
    """``{'clean': dir, 'noise': dir, 'mixed': dir, 'out': dir}``."""
    import kudio

    clean_dir = tmp_path / "clean"
    noise_dir = tmp_path / "noise"
    clean_dir.mkdir()
    noise_dir.mkdir()

    for i in range(4):
        kudio.save_wave(clean_dir / f"utt{i}.wav", _harmonic(i), SR)

    rng = np.random.default_rng(99)
    kudio.save_wave(noise_dir / "white.wav",
                    (0.3 * rng.normal(0, 1, SR)).astype(np.float32), SR)
    kudio.save_wave(noise_dir / "hum.wav",
                    (0.3 * np.sin(2 * np.pi * 60
                                  * np.arange(SR) / SR)).astype(np.float32), SR)

    return {"clean": clean_dir, "noise": noise_dir,
            "mixed": tmp_path / "mixed", "out": tmp_path / "runs"}


@pytest.fixture
def config(corpus):
    """A configuration small enough to train in a couple of seconds."""
    from kudio_enhance import Config

    return Config.from_dict({
        "audio": {"sr": SR, "n_fft": 256, "hop_length": 128, "win_length": 256},
        "data": {
            "clean_dir": str(corpus["clean"]),
            "noise_dir": str(corpus["noise"]),
            "mixed_dir": str(corpus["mixed"]),
            "snr_db": [0, 5],
            "mode": "regular",
            "seed": 3,
            "val_split": 0.25,
            "test_split": 0.25,
        },
        "model": {"name": "ddae", "context": 1, "n_frames": 16,
                  "units": [32, 32], "dropout": 0.0},
        "train": {"epochs": 1, "batch_size": 32, "learning_rate": 1e-3,
                  "patience": 2, "out_dir": str(corpus["out"])},
    })


@pytest.fixture
def tone() -> np.ndarray:
    return _harmonic(seed=42)
