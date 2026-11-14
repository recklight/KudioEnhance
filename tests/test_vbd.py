# -*- coding: utf-8 -*-
"""VoiceBank-DEMAND preparation, on a corpus laid out as the record's zips.

The corpus here is a few short synthetic files at 48 kHz with VCTK names, in
zips with the record's names and layout, so every step of ``prepare`` runs as
it does on the real thing, in a second.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa", reason="resampling needs librosa")
sf = pytest.importorskip("soundfile")

from kudioenhance.vbd import (  # noqa: E402
    RESAMPLER,
    VALIDATION_SPEAKERS,
    prepare,
    read_log,
)

RATE = 48000
TRAIN = {"p226": 2, "p227": 3, "p287": 2}
TEST = {"p232": 2, "p257": 2}
NOISES = ["cafe", "bus", "living"]


def _clean(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.25 * RATE)) / RATE
    f0 = rng.uniform(110, 220)
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    y *= np.hanning(len(t))
    return np.round(9000 * y / np.max(np.abs(y))).astype(np.int16)


def _noise(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    return np.round(rng.normal(0, 2000, n)).astype(np.int16)


def _corpus(root: Path, *, zipped: bool = True, log: bool = True,
            drop_noisy=None, drop_log=None) -> dict:
    """The corpus under *root*, zipped or unpacked; returns the 48 kHz
    clean and noise samples by id, and each id's noise and SNR."""
    samples, conditions = {}, {}
    seed = 0
    for set_name, speakers, snrs, clean_stem, noisy_stem, log_name in (
            ("train", TRAIN, [0, 5, 10, 15], "clean_trainset_28spk_wav",
             "noisy_trainset_28spk_wav", "log_trainset_28spk.txt"),
            ("test", TEST, [2.5, 7.5, 12.5, 17.5], "clean_testset_wav",
             "noisy_testset_wav", "log_testset.txt")):
        files = {clean_stem: {}, noisy_stem: {}}
        lines = []
        for speaker, count in speakers.items():
            for k in range(1, count + 1):
                key = f"{speaker}_{k:03d}"
                clean = _clean(seed)
                noise = _noise(seed, len(clean))
                noisy = (clean.astype(np.int32) + noise).astype(np.int16)
                samples[key] = (clean, noisy)
                noise_name, snr = NOISES[seed % 3], snrs[seed % 4]
                conditions[key] = (noise_name, snr)
                files[clean_stem][key] = clean
                if key != drop_noisy:
                    files[noisy_stem][key] = noisy
                if key != drop_log:
                    # the record writes the SNR as 1.750000e+01
                    lines.append(f"{key} {noise_name} {float(snr):e}")
                seed += 1
        for stem, waves in files.items():
            if zipped:
                with zipfile.ZipFile(root / f"{stem}.zip", "w") as zf:
                    for key, y in waves.items():
                        data = _wav_bytes(y)
                        zf.writestr(f"{stem}/{key}.wav", data)
                        # what a zip made on a Mac carries beside each file
                        zf.writestr(f"__MACOSX/{stem}/._{key}.wav", b"\x00\x05")
            else:
                (root / stem).mkdir(parents=True)
                for key, y in waves.items():
                    sf.write(root / stem / f"{key}.wav", y, RATE, subtype="PCM_16")
        if log:
            text = "\n".join(lines) + "\n"
            if zipped:
                # as the record has it: the logs at the top level, and a
                # resource fork of each that is not text
                with zipfile.ZipFile(root / "logfiles.zip", "a") as zf:
                    zf.writestr(log_name, text)
                    zf.writestr(f"__MACOSX/._{log_name}", b"\x00\x05\xab\xcd")
            else:
                (root / "logfiles").mkdir(exist_ok=True)
                (root / "logfiles" / log_name).write_text(text, encoding="utf-8")
    return {"samples": samples, "conditions": conditions}


def _wav_bytes(y: np.ndarray) -> bytes:
    import io
    buffer = io.BytesIO()
    sf.write(buffer, y, RATE, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def _rows(path: Path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def zips(tmp_path):
    root = tmp_path / "zips"
    root.mkdir()
    return root, _corpus(root)


# ------------------------------------------------------------------ prepare

def test_validation_speakers_are_held_out_whole_and_test_stays_test(zips, tmp_path):
    root, _ = zips
    out = tmp_path / "16k"
    summary = prepare(root, out)

    rows = _rows(out / "utterances.csv")
    by_split = {split: {r["speaker"] for r in rows if r["split"] == split}
                for split in ("train", "val", "test")}
    assert by_split == {"train": {"p227"}, "val": set(VALIDATION_SPEAKERS),
                        "test": set(TEST)}
    assert summary["utterances"] == {"train": 3, "val": 4, "test": 4}
    assert summary["validation_speakers"] == list(VALIDATION_SPEAKERS)
    assert summary["test_speakers"] == sorted(TEST)


def test_every_file_is_at_the_new_rate_in_float_and_matches_its_sum(zips, tmp_path):
    root, corpus = zips
    out = tmp_path / "16k"
    prepare(root, out)

    rows = _rows(out / "utterances.csv")
    assert len(rows) == len(corpus["samples"])
    for row in rows:
        for side in ("clean", "noisy"):
            path = out / row[side]
            assert path.parent == out / row["split"] / side
            info = sf.info(str(path))
            assert (info.samplerate, info.subtype) == (16000, "FLOAT")
            assert info.frames == int(row["frames"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == row[f"{side}_sha256"]
        # 0.25 s at 48 kHz is 12000 samples, 4000 at 16 kHz
        assert int(row["frames"]) == 4000
        assert float(row["seconds"]) == pytest.approx(0.25)


def test_the_resampled_noisy_file_is_clean_plus_the_resampled_noise(zips, tmp_path):
    """The irm target takes noisy - clean as the noise. The resampler is
    linear, so that still holds after it, to float precision."""
    import kudio

    root, corpus = zips
    out = tmp_path / "16k"
    prepare(root, out)
    clean48, noisy48 = corpus["samples"]["p227_002"]
    noise48 = (noisy48.astype(np.float32) - clean48.astype(np.float32)) / 32768
    expected = kudio.resample(noise48, RATE, 16000, res_type=RESAMPLER)

    clean, _ = sf.read(out / "train/clean/p227_002.wav", dtype="float32")
    noisy, _ = sf.read(out / "train/noisy/p227_002.wav", dtype="float32")
    np.testing.assert_allclose(noisy - clean, expected, atol=1e-6)


def test_each_pair_carries_the_noise_and_snr_of_the_log(zips, tmp_path):
    from kudioenhance.data import load_prepared

    root, corpus = zips
    out = tmp_path / "16k"
    prepare(root, out)

    seen = 0
    for split in ("train", "val", "test"):
        for pair in load_prepared(out, split):
            key = Path(pair.noisy).stem
            noise, snr = corpus["conditions"][key]
            assert (pair.noise, pair.snr_db) == (noise, snr)
            # whole dB stay whole, as the Synthesizer writes them
            assert isinstance(pair.snr_db, int) == float(snr).is_integer()
            seen += 1
    assert seen == len(corpus["samples"])


def test_manifests_are_relative_and_load_from_wherever_the_folder_went(zips, tmp_path):
    from kudioenhance.data import load_prepared

    root, _ = zips
    prepare(root, tmp_path / "16k")
    moved = tmp_path / "elsewhere" / "vbd16k"
    moved.parent.mkdir()
    (tmp_path / "16k").rename(moved)

    raw = json.loads((moved / "test.json").read_text(encoding="utf-8"))
    assert raw[0]["noisy"].startswith("test/noisy/")
    pairs = load_prepared(moved, "test")
    assert all(pair.exists() for pair in pairs)
    assert Path(pairs[0].clean).parent == moved / "test" / "clean"


def test_speaker_table_counts_utterances_and_seconds(zips, tmp_path):
    root, _ = zips
    out = tmp_path / "16k"
    prepare(root, out)

    table = {r["speaker"]: r for r in _rows(out / "speakers.csv")}
    assert set(table) == set(TRAIN) | set(TEST)
    assert table["p227"]["split"] == "train"
    assert int(table["p227"]["utterances"]) == 3
    assert float(table["p227"]["seconds"]) == pytest.approx(0.75)
    assert table["p257"]["split"] == "test"


def test_prepare_json_records_the_source_resampler_and_checksums(zips, tmp_path):
    root, _ = zips
    out = tmp_path / "16k"
    prepare(root, out)

    record = json.loads((out / "prepare.json").read_text(encoding="utf-8"))
    assert record["source_rates"] == {"48000": 2 * 11}
    assert record["sr"] == 16000
    assert RESAMPLER in record["resampler"]
    assert {"kudio", "librosa", "soxr", "kudioenhance"} <= set(record["versions"])
    names = {entry["name"] for entry in record["source"]}
    assert "clean_testset_wav.zip" in names and "logfiles.zip" in names
    zipped = root / "clean_testset_wav.zip"
    entry = next(e for e in record["source"] if e["name"] == zipped.name)
    assert entry["sha256"] == hashlib.sha256(zipped.read_bytes()).hexdigest()
    for name, digest in record["sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert record["conditions"]["test"]["snr_db"] == {
        "2.5": 1, "7.5": 1, "12.5": 1, "17.5": 1}
    assert record["unequal_lengths"] == 0


def test_unpacked_folders_give_the_same_files_as_the_zips(tmp_path):
    zipped, unpacked = tmp_path / "zips", tmp_path / "folders"
    zipped.mkdir()
    unpacked.mkdir()
    _corpus(zipped)
    _corpus(unpacked, zipped=False)

    prepare(zipped, tmp_path / "a")
    prepare(unpacked, tmp_path / "b")
    assert (_rows(tmp_path / "a" / "utterances.csv")
            == _rows(tmp_path / "b" / "utterances.csv"))


def test_preparing_again_later_gives_the_same_bytes(zips, tmp_path):
    """libsndfile stamps the time into a float WAV's PEAK chunk; with it
    zeroed, a rebuild matches the checksums of the first build."""
    import time

    root, _ = zips
    prepare(root, tmp_path / "a")
    # the stamp is in whole seconds, so the second build has to start in
    # another one for the test to mean anything
    time.sleep(1.1)
    prepare(root, tmp_path / "b")
    assert (_rows(tmp_path / "a" / "utterances.csv")
            == _rows(tmp_path / "b" / "utterances.csv"))
    with open(tmp_path / "b" / "test" / "clean" / "p232_001.wav", "rb") as fh:
        data = fh.read()
    at = data.index(b"PEAK")
    assert data[at + 12:at + 16] == bytes(4)


def test_other_validation_speakers(zips, tmp_path):
    root, _ = zips
    out = tmp_path / "16k"
    summary = prepare(root, out, validation=["p227"])
    assert summary["utterances"] == {"train": 4, "val": 3, "test": 4}
    assert {r["speaker"] for r in _rows(out / "utterances.csv")
            if r["split"] == "val"} == {"p227"}


def test_a_validation_speaker_not_in_the_training_set_writes_nothing(zips, tmp_path):
    root, _ = zips
    out = tmp_path / "16k"
    with pytest.raises(ValueError, match="p232"):
        prepare(root, out, validation=["p226", "p232"])
    assert not out.exists()
    with pytest.raises(ValueError, match="every training speaker"):
        prepare(root, out, validation=list(TRAIN))


def test_an_existing_corpus_is_kept_unless_overwrite(zips, tmp_path):
    root, _ = zips
    out = tmp_path / "16k"
    prepare(root, out)
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare(root, out, validation=["p227"])

    prepare(root, out, validation=["p227"], overwrite=True)
    # the old validation speakers' files are not left lying in val/
    assert sorted(p.stem for p in (out / "val" / "clean").iterdir()) == [
        "p227_001", "p227_002", "p227_003"]


def test_a_noisy_file_missing_from_its_set_is_refused(tmp_path):
    root = tmp_path / "zips"
    root.mkdir()
    _corpus(root, drop_noisy="p227_002")
    with pytest.raises(ValueError, match="p227_002"):
        prepare(root, tmp_path / "16k")


def test_a_log_that_misses_an_utterance_is_refused(tmp_path):
    root = tmp_path / "zips"
    root.mkdir()
    _corpus(root, drop_log="p257_001")
    with pytest.raises(ValueError, match="log_testset.txt.*p257_001"):
        prepare(root, tmp_path / "16k")


def test_without_the_log_files_the_pairs_carry_no_condition(tmp_path, caplog):
    from kudioenhance.data import load_prepared

    root = tmp_path / "zips"
    root.mkdir()
    _corpus(root, log=False)
    prepare(root, tmp_path / "16k")
    assert "log_trainset_28spk.txt not found" in caplog.text
    pair = load_prepared(tmp_path / "16k", "train")[0]
    assert (pair.noise, pair.snr_db) == ("", None)


def test_a_missing_set_names_both_forms_it_could_take(tmp_path):
    root = tmp_path / "zips"
    root.mkdir()
    _corpus(root)
    (root / "noisy_testset_wav.zip").unlink()
    with pytest.raises(FileNotFoundError, match="noisy_testset_wav"):
        prepare(root, tmp_path / "16k")


# ----------------------------------------------------------------- the logs

def test_read_log():
    text = "p232_001 living 17.5\n\np232_002  bus   2.5\np226_001 cafe 15\n"
    assert read_log(text) == {"p232_001": ("living", 17.5),
                              "p232_002": ("bus", 2.5),
                              "p226_001": ("cafe", 15.0)}


@pytest.mark.parametrize("text, message", [
    ("p232_001 living\n", "line 1"),
    ("p232_001 living loud\n", "not a number"),
    ("p232_001 bus 5\np232_001 cafe 10\n", "twice"),
])
def test_read_log_refuses_what_it_cannot_read(text, message):
    with pytest.raises(ValueError, match=message):
        read_log(text)


# ------------------------------------------------------- the rest of the run

def test_a_prepared_corpus_has_nothing_to_synthesize(zips, tmp_path, config):
    from kudioenhance import pipeline

    root, _ = zips
    prepare(root, tmp_path / "16k")
    config.data.prepared = str(tmp_path / "16k")
    with pytest.raises(ValueError, match="nothing to synthesize"):
        pipeline.synthesize(config, "vbd")


def test_prepared_is_kept_through_the_config_file(tmp_path):
    from kudioenhance import Config

    assert Config().data.prepared is None
    cfg = Config.from_dict({"data": {"prepared": "data/vbd16k"}})
    cfg.to_yaml(tmp_path / "config.yaml")
    assert Config.from_yaml(tmp_path / "config.yaml").data.prepared == "data/vbd16k"


def test_train_and_evaluate_on_a_prepared_corpus(zips, tmp_path, config, torch):
    """Trained on train/, stopped on val/, scored on test/, with the test
    set's two speakers read from the file names."""
    from kudioenhance import pipeline

    root, _ = zips
    prepare(root, tmp_path / "16k")
    config.data.prepared = str(tmp_path / "16k")
    config.audio.sr = 16000
    pipeline.train(config, "vbd", verbose=0)
    rows, summary = pipeline.evaluate(config, "vbd")

    assert sorted(row["file"] for row in rows) == [
        "p232_001.wav", "p232_002.wav", "p257_001.wav", "p257_002.wav"]
    assert {row["noise"] for row in rows} <= set(NOISES)
    assert "noisy_si_sdr" in summary
    path = config.results_path("vbd", ".json")
    results = json.loads(path.read_text(encoding="utf-8"))
    assert results["speakers"] == sorted(TEST)
    # two speakers is too few to resample, so the table says it took sentences
    assert results["resampled"] == "sentences"


def test_cli_prepare_and_synth(zips, tmp_path, capsys):
    from kudioenhance import Config
    from kudioenhance.cli import main

    root, _ = zips
    out = tmp_path / "16k"
    assert main(["prepare", str(root), str(out), "--val-speakers", "p227"]) == 0
    printed = capsys.readouterr().out
    assert "4 train / 3 val / 4 test" in printed and "p227" in printed

    assert main(["prepare", str(root), str(out)]) == 1
    assert "overwrite" in capsys.readouterr().err

    config = tmp_path / "config.yaml"
    Config.from_dict({"data": {"prepared": str(out)}}).to_yaml(config)
    assert main(["synth", "-c", str(config), "-n", "vbd"]) == 1
    assert "nothing to synthesize" in capsys.readouterr().err
