# -*- coding: utf-8 -*-
"""score_pair: the numpy scores on the three cut to the shortest, PESQ, STOI
and SDR on them as loaded, so that PESQ gets the pair whole."""
from types import SimpleNamespace

import kudio
import numpy as np

from kudioenhance.evaluate import score_pair

SR = 16000


class _Unchanged:
    """An enhancer that hands back its input, which is all score_pair needs."""
    cfg = SimpleNamespace(audio=SimpleNamespace(sr=SR))

    def enhance(self, noisy):
        return np.asarray(noisy, dtype=np.float32)


def test_pesq_stoi_and_sdr_get_the_files_as_loaded(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    clean = 0.1 * rng.standard_normal(2 * SR).astype(np.float32)
    noisy = clean[:2 * SR - 3200] + 0.01 * rng.standard_normal(2 * SR - 3200).astype(np.float32)
    kudio.save_wave(tmp_path / "clean.wav", clean, SR)
    kudio.save_wave(tmp_path / "noisy.wav", noisy, SR)
    handed = []

    def eval_metrics(sr, ref, deg, **kwargs):
        handed.append((len(ref), len(deg)))
        return (2.5, 0.9, 5.0)

    monkeypatch.setattr(kudio, "eval_metrics", eval_metrics)
    pair = SimpleNamespace(noisy=str(tmp_path / "noisy.wav"),
                           clean=str(tmp_path / "clean.wav"), noise="white", snr_db=0.0)
    row = score_pair(_Unchanged(), pair, optional=True)
    # noisy against clean, then the output, as long as the noisy file
    assert handed == [(2 * SR, 2 * SR - 3200)] * 2
    assert row["noisy_pesq"] == 2.5 and row["delta_pesq"] == 0.0
    assert np.isfinite(row["noisy_snr"])


def test_a_pair_kudio_refuses_keeps_its_numpy_scores(tmp_path, monkeypatch):
    """Past MAX_LENGTH_MISMATCH kudio refuses the pair for PESQ, STOI and
    SDR; the row keeps SNR, SI-SDR and segmental SNR on the overlap."""
    rng = np.random.default_rng(0)
    clean = 0.1 * rng.standard_normal(2 * SR).astype(np.float32)
    noisy = clean[:SR // 2] + 0.01 * rng.standard_normal(SR // 2).astype(np.float32)
    kudio.save_wave(tmp_path / "clean.wav", clean, SR)
    kudio.save_wave(tmp_path / "noisy.wav", noisy, SR)

    def eval_metrics(sr, ref, deg, **kwargs):
        raise ValueError("eval_metrics(): ref is 32000 samples and deg 8000: ...")

    monkeypatch.setattr(kudio, "eval_metrics", eval_metrics)
    pair = SimpleNamespace(noisy=str(tmp_path / "noisy.wav"),
                           clean=str(tmp_path / "clean.wav"), noise="white", snr_db=0.0)
    row = score_pair(_Unchanged(), pair, optional=True)
    assert row["noisy_pesq"] is None and row["delta_stoi"] is None
    assert np.isfinite(row["noisy_snr"]) and np.isfinite(row["delta_si_sdr"])
