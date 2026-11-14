# -*- coding: utf-8 -*-
"""VoiceBank-DEMAND, brought to the model's rate and split for training.

Valentini-Botinhao, Wang, Takaki and Yamagishi's corpus (SSW 2016; the
28-speaker training set, DOI 10.7488/ds/2117) comes already mixed, at
48 kHz, as the zips the DataShare record lists. :func:`prepare` reads them,
or the folders they unpack to, and writes:

``<out>/{train,val,test}/{clean,noisy}/<id>.wav``
    every file resampled to *sr*, in 32-bit float. Float, because the noisy
    files come close to full scale and a resampler can overshoot by a few
    per cent: written as 16-bit those samples would clip, and ``noisy -
    clean``, which the ``irm`` target takes as the noise, would stop being
    the noise there.
``<out>/{train,val,test}.json``
    the pairs as kudio's manifests, with each mixture's noise and SNR from
    the corpus's log files. The paths are relative to *out*, so the folder
    can be moved; :func:`kudioenhance.data.load_prepared` resolves them.
``<out>/utterances.csv``
    one row per utterance: speaker, split, noise, SNR, length, and both
    files with their SHA-256.
``<out>/speakers.csv``
    one row per speaker: split, utterances and seconds.
``<out>/prepare.json``
    the source files, the rates, the resampler and the library versions
    behind it, the validation speakers and the counts. It is written last,
    so a folder without it is a run that did not finish.

Nothing is downloaded here. The corpus is fetched by hand from the record,
which also gives each file's MD5.

**The validation split.** The corpus has a training set and a test set and
no validation set, and the checkpoint must not be chosen on the test set. Two
training speakers, :data:`VALIDATION_SPEAKERS`, are held out whole for it, so
validation, like the test set, is on voices the model has not heard. They are
the two SpeechBrain's VoiceBank recipe holds out, so the split is one that
already exists rather than one more of my own. Both are men, which leaves 12
men and 14 women to train on; the test set is a man and a woman.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from kudio import Pair, save_manifest

log = logging.getLogger(__name__)

__all__ = ["prepare", "read_log", "VALIDATION_SPEAKERS", "TEST_SPEAKERS",
           "SPLITS", "RESAMPLER", "Utterance"]

#: each set as the record names it: clean files, noisy files, log file
SOURCES = {
    "train": ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav",
              "log_trainset_28spk.txt"),
    "test": ("clean_testset_wav", "noisy_testset_wav", "log_testset.txt"),
}

#: where the log files are: ``logfiles.zip``, or the folder it unpacks to
LOGS = "logfiles"

#: training speakers held out for validation; see the module notes
VALIDATION_SPEAKERS = ("p226", "p287")

#: the two speakers of the test set
TEST_SPEAKERS = ("p232", "p257")

SPLITS = ("train", "val", "test")

#: librosa's name for the resampler, through :func:`kudio.resample`. The
#: soxr "HQ" preset, also what ``kudio.file_load(path, sr=...)`` uses, so a
#: file prepared here and one resampled as kudio reads it go through the same
#: filter.
RESAMPLER = "soxr_hq"

#: written into every file: see the module notes for why float
SUBTYPE = "FLOAT"

#: VCTK file names start with the speaker: ``p226_001``
SPEAKER = re.compile(r"^(p\d+)_")


@dataclass(frozen=True)
class Utterance:
    """One row of ``utterances.csv``."""

    id: str
    speaker: str
    split: str
    noise: str
    snr_db: Optional[float]
    frames: int
    clean: str
    clean_sha256: str
    noisy: str
    noisy_sha256: str


# ----------------------------------------------------------------- reading

class _Folder:
    """The wave files of one set, from its zip or from the folder it unpacks
    to, by id. The zip wins when both are there: it is what the record's MD5
    covers."""

    def __init__(self, root: Path, stem: str):
        self.name = stem
        zipped = root / f"{stem}.zip"
        unpacked = root / stem
        self._zip: Optional[zipfile.ZipFile] = None
        if zipped.is_file():
            self.origin = zipped
            self._zip = zipfile.ZipFile(zipped)
            members = [m for m in self._zip.namelist() if _is_wave(m)]
            self._members = _by_id(members, zipped)
        elif unpacked.is_dir():
            self.origin = unpacked
            members = [str(p) for p in unpacked.rglob("*.wav") if _is_wave(p.name)]
            self._members = _by_id(members, unpacked)
        else:
            raise FileNotFoundError(
                f"neither {zipped.name} nor a folder {stem}/ in {root}")
        if not self._members:
            raise ValueError(f"{self.origin} holds no .wav files")

    @property
    def ids(self) -> List[str]:
        return sorted(self._members)

    def read(self, key: str) -> Tuple[np.ndarray, int]:
        import soundfile as sf
        member = self._members[key]
        if self._zip is not None:
            source = io.BytesIO(self._zip.read(member))
        else:
            source = member
        y, rate = sf.read(source, dtype="float32", always_2d=False)
        if y.ndim > 1:
            raise ValueError(f"{self.name}/{key}: {y.shape[1]} channels; the "
                             f"corpus is mono")
        return y, int(rate)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def _is_wave(name: str) -> bool:
    # macOS leaves __MACOSX/ and ._name entries in zips made on it
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    return (name.lower().endswith(".wav") and not base.startswith("._")
            and "__MACOSX" not in name)


def _by_id(members: Sequence[str], origin: Path) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for member in members:
        key = Path(member.replace("\\", "/")).stem
        if key in found:
            raise ValueError(f"{origin} holds {key}.wav twice: {found[key]} "
                             f"and {member}")
        found[key] = member
    return found


def _log_text(root: Path, name: str) -> Optional[str]:
    """The log file *name*, from ``logfiles.zip``, a ``logfiles/`` folder, or
    *root* itself; None when it is nowhere."""
    zipped = root / f"{LOGS}.zip"
    if zipped.is_file():
        # the record's zip holds the logs at its top level, beside a __MACOSX/
        # folder of resource forks that are not text
        with zipfile.ZipFile(zipped) as zf:
            for member in zf.namelist():
                if ("__MACOSX" not in member
                        and member.replace("\\", "/").rsplit("/", 1)[-1] == name):
                    return zf.read(member).decode("utf-8")
    for path in (root / LOGS / name, root / name):
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def read_log(text: str, source: str = "log") -> Dict[str, Tuple[str, float]]:
    """``{id: (noise, snr_db)}`` from a VoiceBank-DEMAND log file, whose lines
    are ``p226_001 cafe 15``: the utterance, the noise it was mixed with, and
    the SNR in dB.

    :raises ValueError: for a line that is not three fields with a number
        last, or an utterance listed twice.
    """
    out: Dict[str, Tuple[str, float]] = {}
    for n, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"{source} line {n}: expected 'id noise snr', "
                             f"got {line.strip()!r}")
        key, noise, snr = fields
        try:
            value = float(snr)
        except ValueError:
            raise ValueError(f"{source} line {n}: the SNR {snr!r} is not a "
                             f"number") from None
        if key in out:
            raise ValueError(f"{source} lists {key} twice")
        out[key] = (noise, value)
    return out


def _speaker(key: str) -> str:
    found = SPEAKER.match(key)
    if found is None:
        raise ValueError(f"{key}: not a VCTK file name, so the speaker is unknown")
    return found.group(1)


# ----------------------------------------------------------------- writing

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _without_timestamp(data: bytes) -> bytes:
    """*data*, a WAV file, with the time in its PEAK chunk set to zero.

    libsndfile writes a PEAK chunk into every float WAV, and puts the time of
    writing in it, so the same samples written a second apart are two
    different files with two different SHA-256s. With the time zeroed,
    preparing the corpus again gives the same bytes, and the checksums in
    ``utterances.csv`` can check a rebuild as well as a copy. The chunk's
    peak values are left as they are.
    """
    out = bytearray(data)
    at = 12                                  # past "RIFF", size, "WAVE"
    while at + 8 <= len(out):
        chunk = bytes(out[at:at + 4])
        size = int.from_bytes(out[at + 4:at + 8], "little")
        if chunk == b"PEAK" and size >= 8:
            # version (4 bytes), then the time (4 bytes), then the peaks
            out[at + 12:at + 16] = bytes(4)
            break
        at += 8 + size + (size & 1)          # chunks are padded to even size
    return bytes(out)


def _write(path: Path, y: np.ndarray, sr: int) -> str:
    """Write *y* to *path* and return the SHA-256 of the bytes written."""
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, subtype=SUBTYPE, format="WAV")
    data = _without_timestamp(buffer.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _snr(value: Optional[float]):
    # whole numbers as ints, as kudio's Synthesizer writes them
    if value is None or not float(value).is_integer():
        return value
    return int(value)


def _versions() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for package in ("kudio", "librosa", "soxr", "soundfile", "numpy"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "not installed"
    from kudioenhance._version import __version__
    out["kudioenhance"] = __version__
    return out


def _clear(out: Path) -> None:
    """Remove what an earlier :func:`prepare` wrote into *out*, so a folder of
    the old split cannot be read beside the new one. Only called where
    ``prepare.json`` shows the folder is one of ours."""
    (out / "prepare.json").unlink()
    for split in SPLITS:
        if (out / split).is_dir():
            shutil.rmtree(out / split)
    for name in ("train.json", "val.json", "test.json", "utterances.csv",
                 "speakers.csv"):
        if (out / name).is_file():
            (out / name).unlink()


# ----------------------------------------------------------------- prepare

def _assign(speaker: str, source_split: str, validation: Sequence[str]) -> str:
    if source_split == "test":
        return "test"
    return "val" if speaker in validation else "train"


#: an utterance on its way through :func:`prepare`: id, split, the clean and
#: noisy sets it is read from, and its noise and SNR
_Item = Tuple[str, str, "_Folder", "_Folder", Tuple[str, Optional[float]]]


def _utterances(root: Path, source_split: str, validation: Sequence[str]
                ) -> Iterator[_Item]:
    clean_stem, noisy_stem, log_name = SOURCES[source_split]
    clean, noisy = _Folder(root, clean_stem), _Folder(root, noisy_stem)
    try:
        ids = clean.ids
        unpaired = sorted(set(ids) ^ set(noisy.ids))
        if unpaired:
            more = f" and {len(unpaired) - 5} more" if len(unpaired) > 5 else ""
            raise ValueError(f"{clean_stem} and {noisy_stem} do not hold the same "
                             f"utterances: {', '.join(unpaired[:5])}{more} are in "
                             f"one only")
        text = _log_text(root, log_name)
        if text is None:
            log.warning("%s not found, so the %s pairs carry no noise type or SNR",
                        log_name, source_split)
            conditions: Dict[str, Tuple[str, float]] = {}
        else:
            conditions = read_log(text, log_name)
            missing = [key for key in ids if key not in conditions]
            if missing:
                raise ValueError(f"{log_name} has no line for {len(missing)} of "
                                 f"the {len(ids)} {source_split} utterances, "
                                 f"e.g. {missing[0]}")
        for key in ids:
            speaker = _speaker(key)
            yield (key, _assign(speaker, source_split, validation), clean, noisy,
                   conditions.get(key, ("", None)))
    finally:
        clean.close()
        noisy.close()


def prepare(source, out, *, sr: int = 16000,
            validation: Sequence[str] = VALIDATION_SPEAKERS,
            overwrite: bool = False) -> Dict[str, object]:
    """Read VoiceBank-DEMAND from *source* and write it to *out* at *sr*.

    *source* is the folder holding the record's zips, or the folders they
    unpack to: ``clean_trainset_28spk_wav``, ``noisy_trainset_28spk_wav``,
    ``clean_testset_wav``, ``noisy_testset_wav`` and ``logfiles``. The
    transcripts are not needed. What lands in *out* is in the module notes.

    :param validation: the training speakers held out whole as the
        validation set; see the module notes.
    :param overwrite: replace an earlier run's output in *out*. Without it a
        folder that already holds ``prepare.json`` is refused.
    :returns: what ``prepare.json`` holds.
    :raises FileExistsError: for an *out* that already holds a prepared
        corpus, without *overwrite*.
    :raises FileNotFoundError: for a set that is neither zipped nor unpacked
        in *source*.
    :raises ValueError: before anything is written, for validation speakers
        that are not training speakers, and on the way, for a clean and noisy
        set that do not hold the same utterances, an utterance missing from a
        log file that is there, or a pair whose two files are at different
        rates.
    """
    import kudio

    root, out = Path(source), Path(out)
    validation = tuple(dict.fromkeys(validation))
    if sr <= 0:
        raise ValueError(f"sr must be a positive rate, got {sr!r}")
    if (out / "prepare.json").is_file():
        if not overwrite:
            raise FileExistsError(f"{out} already holds a prepared corpus "
                                  f"(prepare.json); pass overwrite to replace it")
        _clear(out)

    # the speakers are known from the file names before any audio is read, so
    # a mistyped validation speaker fails here rather than ten minutes in
    train_ids = _Folder(root, SOURCES["train"][0])
    try:
        speakers = {_speaker(key) for key in train_ids.ids}
    finally:
        train_ids.close()
    strangers = [s for s in validation if s not in speakers]
    if strangers:
        raise ValueError(f"validation speaker(s) {', '.join(strangers)} are not in "
                         f"the training set, whose speakers are "
                         f"{', '.join(sorted(speakers))}")
    if len(validation) == len(speakers):
        raise ValueError("every training speaker would be held out for validation")

    rows: List[Utterance] = []
    pairs: Dict[str, List[Pair]] = {split: [] for split in SPLITS}
    unequal = 0
    rates: Counter = Counter()
    for source_split in ("train", "test"):
        for key, split, clean, noisy, (noise, snr) in _utterances(
                root, source_split, validation):
            written = {}
            frames = {}
            for side, folder in (("clean", clean), ("noisy", noisy)):
                y, rate = folder.read(key)
                rates[rate] += 1
                y = kudio.resample(y, rate, sr, res_type=RESAMPLER)
                relative = f"{split}/{side}/{key}.wav"
                written[side] = (relative, _write(out / relative, y, sr))
                frames[side] = (rate, len(y))
            if frames["clean"][0] != frames["noisy"][0]:
                raise ValueError(f"{key}: the clean file is at {frames['clean'][0]} Hz "
                                 f"and the noisy one at {frames['noisy'][0]} Hz")
            unequal += frames["clean"][1] != frames["noisy"][1]
            rows.append(Utterance(
                id=key, speaker=_speaker(key), split=split, noise=noise,
                snr_db=snr, frames=frames["clean"][1],
                clean=written["clean"][0], clean_sha256=written["clean"][1],
                noisy=written["noisy"][0], noisy_sha256=written["noisy"][1]))
            pairs[split].append(Pair(noisy=written["noisy"][0],
                                     clean=written["clean"][0], noise=noise,
                                     snr_db=_snr(snr)))
            if len(rows) % 1000 == 0:
                log.info("%d utterances written", len(rows))
    if unequal:
        log.warning("%d pair(s) whose clean and noisy files differ in length",
                    unequal)

    for split in SPLITS:
        save_manifest(out / f"{split}.json", pairs[split])
    _write_utterances(out / "utterances.csv", rows, sr)
    speaker_rows = _speaker_table(rows, sr)
    _write_csv(out / "speakers.csv", ["speaker", "split", "utterances", "seconds"],
               speaker_rows)

    summary = {
        "corpus": "VoiceBank-DEMAND, 28-speaker training set",
        "doi": "10.7488/ds/2117",
        "source": _source_record(root),
        "source_rates": {str(rate): n for rate, n in sorted(rates.items())},
        "sr": sr,
        "resampler": f"kudio.resample, res_type={RESAMPLER}",
        "subtype": SUBTYPE,
        "versions": _versions(),
        "validation_speakers": list(validation),
        "test_speakers": sorted({r.speaker for r in rows if r.split == "test"}),
        "utterances": {split: len(pairs[split]) for split in SPLITS},
        "speakers": {split: len({r.speaker for r in rows if r.split == split})
                     for split in SPLITS},
        "seconds": {split: round(sum(r.frames for r in rows if r.split == split)
                                 / sr, 2) for split in SPLITS},
        "unequal_lengths": unequal,
        "conditions": {split: _conditions([r for r in rows if r.split == split])
                       for split in SPLITS},
        "files": {"manifests": [f"{split}.json" for split in SPLITS],
                  "utterances": "utterances.csv", "speakers": "speakers.csv"},
        "sha256": {name: _sha256_file(out / name)
                   for name in ("train.json", "val.json", "test.json",
                                "utterances.csv", "speakers.csv")},
    }
    with open(out / "prepare.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    log.info("prepared %s -> %s", ", ".join(f"{len(pairs[s])} {s}" for s in SPLITS),
             out)
    return summary


def _source_record(root: Path) -> List[Dict[str, object]]:
    """Each file or folder read from *root*, with the zip's size and SHA-256."""
    record = []
    names = [stem for stems in SOURCES.values() for stem in stems[:2]] + [LOGS]
    for stem in names:
        zipped, unpacked = root / f"{stem}.zip", root / stem
        if zipped.is_file():
            record.append({"name": zipped.name, "bytes": zipped.stat().st_size,
                           "sha256": _sha256_file(zipped)})
        elif unpacked.is_dir():
            record.append({"name": f"{stem}/", "unpacked": True})
    return record


def _conditions(rows: Sequence[Utterance]) -> Dict[str, Dict[str, int]]:
    noises = Counter(r.noise for r in rows if r.noise)
    snrs = Counter(_snr(r.snr_db) for r in rows if r.snr_db is not None)
    return {"noise": dict(sorted(noises.items())),
            "snr_db": {str(k): v for k, v in sorted(snrs.items())}}


def _speaker_table(rows: Sequence[Utterance], sr: int) -> List[List[object]]:
    table: Dict[Tuple[str, str], List[int]] = {}
    for r in rows:
        entry = table.setdefault((r.speaker, r.split), [0, 0])
        entry[0] += 1
        entry[1] += r.frames
    return [[speaker, split, n, f"{frames / sr:.2f}"]
            for (speaker, split), (n, frames) in sorted(table.items())]


def _write_utterances(path: Path, rows: Sequence[Utterance], sr: int) -> None:
    header = ["id", "speaker", "split", "noise", "snr_db", "seconds", "frames",
              "clean", "clean_sha256", "noisy", "noisy_sha256"]
    _write_csv(path, header, [
        [r.id, r.speaker, r.split, r.noise,
         "" if r.snr_db is None else _snr(r.snr_db), f"{r.frames / sr:.4f}",
         r.frames, r.clean, r.clean_sha256, r.noisy, r.noisy_sha256]
        for r in rows])


def _write_csv(path: Path, header: Sequence[str], rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
