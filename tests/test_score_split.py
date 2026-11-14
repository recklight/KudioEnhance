# -*- coding: utf-8 -*-
"""score_split: the enhanced files written out, the split scored by
kudio.score_dataset against the noisy files, and the table in every form
saying what its intervals resampled.

The enhancer here is a five-tap moving average. On a low tone in white noise
it takes out about four fifths of the noise power and leaves the tone, so
every score moves, and no model or torch is needed."""
from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace

import kudio
import numpy as np
import pytest
import soundfile

from kudioenhance.data import Pair
from kudioenhance.evaluate import (
    FULL_METRICS,
    VCTK_SPEAKER,
    item_ids,
    resolve_metrics,
    score_split,
)

SR = 16000


class _Smoothing:
    cfg = SimpleNamespace(audio=SimpleNamespace(sr=SR))

    def enhance(self, noisy):
        noisy = np.asarray(noisy, dtype=np.float32)
        return np.convolve(noisy, np.ones(5) / 5, mode="same").astype(np.float32)


def _split(root, ids, clean_sr=SR):
    """One pair per id: a tone under a Hann window, and the same in white
    noise, in ``clean/`` and ``noisy/``."""
    rng = np.random.default_rng(0)
    pairs = []
    for k, key in enumerate(ids):
        t = np.arange(SR) / SR
        clean = 0.3 * np.sin(2 * np.pi * (150 + 10 * k) * t) * np.hanning(SR)
        noisy = clean + 0.05 * rng.standard_normal(SR)
        clean_path = root / "clean" / f"{key}.wav"
        noisy_path = root / "noisy" / f"{key}.wav"
        for path in (clean_path, noisy_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        kudio.save_wave(clean_path, clean.astype(np.float32), SR, subtype="FLOAT",
                        d_sample=None if clean_sr == SR else clean_sr)
        kudio.save_wave(noisy_path, noisy.astype(np.float32), SR, subtype="FLOAT")
        pairs.append(Pair(noisy=str(noisy_path), clean=str(clean_path),
                          noise="white", snr_db=10))
    return pairs


def _vbd_ids(per_speaker=6, speakers=("p232", "p257")):
    return [f"{s}_{i:03d}" for s in speakers for i in range(1, per_speaker + 1)]


def test_ids_are_the_noisy_paths_below_their_shared_folder(tmp_path):
    pairs = [Pair(noisy=str(tmp_path / "noisy" / "a" / "p232_001.wav"), clean="c"),
             Pair(noisy=str(tmp_path / "noisy" / "b" / "p232_001.flac"), clean="c")]
    assert item_ids(pairs) == ["a/p232_001", "b/p232_001"]
    assert item_ids(pairs[:1]) == ["p232_001"]


def test_two_noisy_files_with_one_id_are_refused(tmp_path):
    pairs = [Pair(noisy=str(tmp_path / "p232_001.wav"), clean="c"),
             Pair(noisy=str(tmp_path / "p232_001.flac"), clean="c")]
    with pytest.raises(ValueError, match="come to one id"):
        item_ids(pairs)


def test_two_speakers_resample_sentences_and_every_form_says_so(tmp_path):
    pairs = _split(tmp_path, _vbd_ids())
    result = score_split(_Smoothing(), pairs, tmp_path / "enhanced")

    written = tmp_path / "enhanced" / "p232_001.wav"
    info = soundfile.info(str(written))
    assert (info.samplerate, info.subtype) == (SR, "FLOAT")
    assert len(list((tmp_path / "enhanced").glob("*.wav"))) == 12

    assert not result.grouped and result.resampled == "sentences"
    assert not result.score.grouped and result.score.has_baseline
    assert result.score.metrics == ("segsnr", "si_sdr", "snr")
    assert "2 speakers (p232, p257)" in result.note

    markdown = result.to_markdown().splitlines()
    blank = markdown.index("")
    assert all(line.startswith("|") for line in markdown[:blank])
    assert markdown[blank + 1] == result.note
    assert any(line.startswith("Signature") for line in markdown)

    latex = result.to_latex().splitlines()
    assert latex[latex.index(r"\end{tabular}") + 1] == f"% {result.note}"

    data = json.loads(result.to_json())
    assert data["resampled"] == "sentences"
    assert data["speakers"] == ["p232", "p257"]
    assert data["note"] == result.note
    assert data["baseline"] is True and data["items"] == 12
    assert "@misc{kudio" in result.cite()

    saved = result.save(tmp_path / "run")
    assert sorted(saved) == [".bib", ".json", ".md", ".tex"]
    assert saved[".md"].read_text(encoding="utf-8").startswith("| Metric")


def test_the_report_keeps_its_layout_and_takes_the_table_means(tmp_path):
    pairs = _split(tmp_path, _vbd_ids())
    rows, summary = score_split(_Smoothing(), pairs, tmp_path / "enhanced").report()

    assert [r["file"] for r in rows] == [f"{k}.wav" for k in _vbd_ids()]
    for name in ("seg_snr", "si_sdr", "snr"):
        assert summary[f"delta_{name}"] == pytest.approx(
            summary[f"enhanced_{name}"] - summary[f"noisy_{name}"], abs=1e-12)
        assert summary[f"delta_{name}"] > 3
        assert rows[0][f"delta_{name}"] == pytest.approx(
            rows[0][f"enhanced_{name}"] - rows[0][f"noisy_{name}"])


def test_the_bibtex_puts_kudioenhance_and_the_architecture_first(tmp_path):
    import kudioenhance

    pairs = _split(tmp_path, _vbd_ids(per_speaker=2))
    result = score_split(_Smoothing(), pairs, tmp_path / "enhanced",
                         metrics=["si_sdr"], model="crn")

    assert result.references[:3] == ["kudioenhance", "tan2018crn", "kudio"]
    assert len(result.references) == len(set(result.references))
    bib = result.cite()
    assert bib.startswith("@misc{kudioenhance,")
    assert f"note = {{Version {kudioenhance.__version__}}}" in bib
    assert "\n\n@inproceedings{tan2018crn," in bib and "\n\n@misc{kudio," in bib
    # every key the table names has its entry in the .bib, and the other way round
    entries = re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE)
    assert entries == result.references

    listed = ", ".join(result.references)
    assert f"References: {listed}" in result.to_markdown()
    assert f"% \\cite{{{','.join(result.references)}}}" in result.to_latex()
    assert json.loads(result.to_json())["references"] == result.references

    # without a model, KudioEnhance's entry still comes first
    result = score_split(_Smoothing(), pairs, tmp_path / "enhanced", metrics=["si_sdr"])
    assert result.references[:2] == ["kudioenhance", "kudio"]


def test_every_architecture_names_the_works_it_follows():
    from kudioenhance.models import REGISTRY, model_bibtex

    for name, spec in REGISTRY.items():
        assert spec.references, name
        for entry in spec.references:
            assert re.match(r"@(article|inproceedings)\{\w+,\n", entry), entry
            assert entry.endswith("}\n") and re.search(r"\n  doi = \{10\.", entry)
        keys = re.findall(r"^@\w+\{([^,]+),", model_bibtex(name), flags=re.MULTILINE)
        assert len(keys) == len(spec.references)


def test_ten_speakers_are_resampled_whole(tmp_path):
    speakers = tuple(f"p{230 + i}" for i in range(10))
    pairs = _split(tmp_path, _vbd_ids(per_speaker=2, speakers=speakers))
    result = score_split(_Smoothing(), pairs, tmp_path / "enhanced",
                         metrics=["si_sdr"])

    assert result.grouped and result.score.grouped
    assert result.note.startswith("Intervals and tests resample the 10 speakers")
    s = result.score.summary[0]
    assert s.mean.n_groups == 10 and s.difference.n_groups == 10


def test_names_with_no_speaker_in_them_resample_sentences(tmp_path):
    pairs = _split(tmp_path, ["utt0_white_5dB", "utt1_white_5dB", "utt2_hum_5dB"])
    result = score_split(_Smoothing(), pairs, tmp_path / "enhanced",
                         metrics=["si_sdr"])
    assert not result.grouped and result.speakers is None
    assert "no speaker could be read from the file names" in result.note

    result = score_split(_Smoothing(), pairs, tmp_path / "enhanced",
                         metrics=["si_sdr"], speakers=None)
    assert "no speakers were given" in result.note


def test_a_pattern_that_fits_some_names_stops_before_anything_is_enhanced(tmp_path):
    pairs = _split(tmp_path, ["p232_001", "p232_002", "utt0"])
    with pytest.raises(ValueError,
                       match="finds a speaker in 2 of 3 files but not in utt0"):
        score_split(_Smoothing(), pairs, tmp_path / "enhanced")
    assert not (tmp_path / "enhanced").exists()


def test_the_speaker_pattern_can_be_given_as_text(tmp_path):
    pairs = _split(tmp_path, ["a-001", "a-002", "b-001"])
    result = score_split(_Smoothing(), pairs, tmp_path / "enhanced",
                         metrics=["si_sdr"], speakers=r"^(\w)-")
    assert result.speakers == ["a", "a", "b"]
    with pytest.raises(ValueError, match="no group in parentheses"):
        score_split(_Smoothing(), pairs, tmp_path / "enhanced", speakers=r"^\w-")


def test_clean_files_at_another_rate_are_resampled_as_they_are_read(tmp_path):
    pairs = _split(tmp_path, _vbd_ids(per_speaker=2), clean_sr=32000)
    assert soundfile.info(pairs[0].clean).samplerate == 32000
    result = score_split(_Smoothing(), pairs, tmp_path / "enhanced",
                         metrics=["si_sdr"])
    assert result.score.sr == SR
    line = next(iter(result.score.signatures.values()))
    assert "resample" in kudio.parse_signature(line)

    # all at the model's rate, nothing is resampled and the signature says so
    pairs = _split(tmp_path / "same", _vbd_ids(per_speaker=2))
    result = score_split(_Smoothing(), pairs, tmp_path / "same" / "enhanced",
                         metrics=["si_sdr"])
    line = next(iter(result.score.signatures.values()))
    assert "resample" not in kudio.parse_signature(line)


def test_full_leaves_out_what_is_not_installed(monkeypatch, caplog):
    import importlib.util

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "pesq"
                        else find_spec(name, *a))
    with caplog.at_level(logging.WARNING, logger="kudioenhance.evaluate"):
        names = resolve_metrics(full=True)
    pesq_based = ("pesq", "csig", "cbak", "covl")
    assert names == [m for m in FULL_METRICS if m not in pesq_based]
    assert 'pesq, csig, cbak, covl left out' in caplog.text
    assert 'kudio[pesq]' in caplog.text

    # asked for by name, a metric is never dropped
    with pytest.raises(kudio.DependencyError, match="pesq"):
        resolve_metrics(["pesq", "stoi"])


def test_dnsmos_without_onnxruntime_is_refused(monkeypatch):
    import importlib.util

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "onnxruntime"
                        else find_spec(name, *a))
    with pytest.raises(kudio.DependencyError, match="onnxruntime"):
        resolve_metrics(dnsmos=True)


def test_metric_names_and_the_default_set(monkeypatch):
    import importlib.util

    assert resolve_metrics() == ["segsnr", "si_sdr", "snr"]
    assert resolve_metrics(["seg_snr", "si_sdr"]) == ["segsnr", "si_sdr"]
    with pytest.raises(ValueError, match="no metric to score"):
        resolve_metrics([])

    # whether onnxruntime is installed here or not
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name == "onnxruntime"
                        else find_spec(name, *a))
    assert resolve_metrics(["si_sdr"], dnsmos=True) == [
        "si_sdr", "dnsmos", "dnsmos_p808"]


def test_vctk_names_give_their_speaker():
    assert VCTK_SPEAKER.search("p232_001").group(1) == "p232"
    assert VCTK_SPEAKER.search("p257_434_white_5dB").group(1) == "p257"
    assert VCTK_SPEAKER.search("utt0_white_5dB") is None
    assert isinstance(VCTK_SPEAKER, re.Pattern)


def test_evaluate_passes_its_options_through(tmp_path, config, monkeypatch, capsys):
    from kudioenhance import pipeline
    from kudioenhance.cli import main

    seen = {}

    def evaluate(cfg, name, **kwargs):
        seen.update(kwargs)
        return [], {"noisy_si_sdr": 1.0, "enhanced_si_sdr": 3.5}

    monkeypatch.setattr(pipeline, "evaluate", evaluate)
    path = str(config.to_yaml(tmp_path / "config.yaml"))

    assert main(["evaluate", "-c", path, "-n", "exp1"]) == 0
    assert seen["speakers"] is VCTK_SPEAKER and seen["metrics"] is None
    assert seen["jobs"] == 1 and seen["dnsmos_dir"] is None and not seen["optional"]
    out = capsys.readouterr().out
    assert "si_sdr" in out and "(+2.500)" in out
    assert f"table  -> {config.results_path('exp1')}" in out

    assert main(["evaluate", "-c", path, "-n", "exp1", "--metrics", "pesq, stoi",
                 "--speakers", "", "--jobs", "2", "--dnsmos-dir", "models"]) == 0
    assert seen["metrics"] == ["pesq", "stoi"] and seen["speakers"] is None
    assert seen["jobs"] == 2 and seen["dnsmos_dir"] == "models"
