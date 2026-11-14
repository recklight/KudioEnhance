# -*- coding: utf-8 -*-
"""Score enhancement against the clean reference.

Every metric is reported for the noisy input as well as the enhanced output.
An absolute SI-SDR means little on its own; the improvement over the mixture
is the number that says whether the model did anything.

:func:`score_split` is what ``evaluate`` runs. It writes every enhanced file
to disk and hands the three sides, enhanced, clean and noisy, to
:func:`kudio.score_dataset`, so the table that comes back is the one a paper
needs: each mean with its BCa interval, the difference from the noisy input
with its sign-flip test, Holm's adjustment over the rows, the signature that
says how the numbers were computed, and the BibTeX for the metrics. The files
stay on disk, so the same outputs can be scored again, or by someone else's
script, without running the model.

**What the intervals resample.** Sentences of one speaker are not
independent, so where the file names give the speaker (VCTK's do:
``p232_001.wav``) the intervals and tests resample whole speakers. That only
holds its level with enough speakers: kudio warns under
``kudio.core.stats.FEW_UNITS``, 10, and over 2 speakers a 95% interval held
the true mean half the time in its simulation. VoiceBank-DEMAND's test set
has 2, p232 and p257. Under 10 speakers the intervals resample sentences
instead, and a line under the table says so: they say how far the means
would move over other sentences from the same speakers, not how a third
speaker would fare.

:func:`score_pair` and :func:`evaluate_pairs` score in memory, one pair at a
time, with no interval; they stay for scripts written against them.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import kudio
from kudioenhance.config import Config
from kudioenhance.data import Pair
from kudioenhance.inference import Enhancer

log = logging.getLogger(__name__)

__all__ = ["score_pair", "evaluate_pairs", "write_report", "score_split",
           "SplitScore", "item_ids", "resolve_metrics", "VCTK_SPEAKER",
           "FULL_METRICS"]

#: metrics computed from numpy alone, always available
BASE_METRICS = ("snr", "si_sdr", "seg_snr")
#: metrics needing kudio[eval]
OPTIONAL_METRICS = ("pesq", "stoi", "sdr")

#: VCTK's file names start with the speaker, ``p232_001.wav``, and so do the
#: mixtures kudio's Synthesizer writes from them, ``p232_001_white_5dB.wav``
VCTK_SPEAKER = re.compile(r"^(p\d+)_")

#: what ``evaluate --full`` scores, in kudio's names and in the order of
#: VoiceBank-DEMAND tables: PESQ, the three composite ratings, segmental SNR,
#: STOI and SI-SDR, then SNR and SDR
FULL_METRICS = ("pesq", "csig", "cbak", "covl", "segsnr", "stoi", "si_sdr",
                "snr", "sdr")

#: what is scored without ``--full``, from numpy alone
_PLAIN = ("segsnr", "si_sdr", "snr")

#: the two DNSMOS models, scored when their folder is given
_DNSMOS = ("dnsmos", "dnsmos_p808")

#: the module each metric that is not numpy alone needs, and the extra of
#: kudio that installs it
_NEEDS = {"pesq": ("pesq", "pesq"), "csig": ("pesq", "pesq"),
          "cbak": ("pesq", "pesq"), "covl": ("pesq", "pesq"),
          "stoi": ("pystoi", "eval"), "estoi": ("pystoi", "eval"),
          "sdr": ("mir_eval", "eval"), "dnsmos": ("onnxruntime", "dnsmos"),
          "dnsmos_p808": ("onnxruntime", "dnsmos")}

#: the report has called segmental SNR seg_snr since 1.0; kudio calls it segsnr
_REPORT_NAME = {"segsnr": "seg_snr"}
_KUDIO_NAME = {v: k for k, v in _REPORT_NAME.items()}


def _align(*signals: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = min(len(s) for s in signals)
    return tuple(np.asarray(s[:n], dtype=np.float32) for s in signals)


def _base_scores(ref: np.ndarray, deg: np.ndarray) -> Dict[str, float]:
    return {
        "snr": float(kudio.snr(ref, deg)),
        "si_sdr": float(kudio.si_sdr(ref, deg)),
        "seg_snr": float(kudio.segmental_snr(ref, deg)),
    }


def score_pair(enhancer: Enhancer, pair: Pair, *,
               optional: bool = False, pesq_mode: str = "auto",
               pesq_scale: str = "lqo") -> Dict[str, object]:
    """Score one mixture: noisy vs clean, then enhanced vs clean.

    The numpy scores take the three cut to the shortest. PESQ, STOI and SDR
    get them as loaded: kudio cuts for STOI and SDR itself, with a warning,
    and PESQ finds the delay on its own and is scored on both whole. Where
    kudio refuses the pair, past ``kudio.core.evaluator.MAX_LENGTH_MISMATCH``,
    those three are left out of the row and the numpy scores stay.
    """
    sr = enhancer.cfg.audio.sr
    noisy, _ = kudio.file_load(pair.noisy, sr=sr)
    clean, _ = kudio.file_load(pair.clean, sr=sr)
    enhanced = enhancer.enhance(noisy)
    loaded = (clean, noisy, enhanced)
    clean, noisy, enhanced = _align(clean, noisy, enhanced)

    row: Dict[str, object] = {
        "file": Path(pair.noisy).name,
        "noise": pair.noise,
        "snr_db": pair.snr_db,
    }
    for name, value in _base_scores(clean, noisy).items():
        row[f"noisy_{name}"] = value
    for name, value in _base_scores(clean, enhanced).items():
        row[f"enhanced_{name}"] = value
    for name in BASE_METRICS:
        row[f"delta_{name}"] = row[f"enhanced_{name}"] - row[f"noisy_{name}"]

    if optional:
        clean, noisy, enhanced = loaded
        try:
            noisy_opt = kudio.eval_metrics(sr, clean, noisy, pesq_mode=pesq_mode,
                                           pesq_scale=pesq_scale)
            enhanced_opt = kudio.eval_metrics(sr, clean, enhanced,
                                              pesq_mode=pesq_mode,
                                              pesq_scale=pesq_scale)
        except ValueError as e:
            log.warning("%s: PESQ, STOI and SDR not scored: %s",
                        Path(pair.noisy).name, e)
            noisy_opt = enhanced_opt = (None, None, None)
        for name, n_val, e_val in zip(OPTIONAL_METRICS, noisy_opt, enhanced_opt):
            row[f"noisy_{name}"] = n_val
            row[f"enhanced_{name}"] = e_val
            row[f"delta_{name}"] = (None if n_val is None or e_val is None
                                    else e_val - n_val)
    return row


def evaluate_pairs(enhancer: Enhancer, pairs: Sequence[Pair], *,
                   optional: bool = False,
                   pesq_mode: str = "auto", pesq_scale: str = "lqo"
                   ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Score every pair and average the numeric columns.

    :param pesq_scale: ``'lqo'`` for MOS-LQO, or ``'raw'`` for the narrowband
        P.862 score before the P.862.1 mapping. The two are different scales,
        not different roundings: raw 2.03 is about 1.66 MOS-LQO.
    """
    if not pairs:
        raise ValueError("nothing to evaluate")

    rows = []
    for pair in pairs:
        try:
            rows.append(score_pair(enhancer, pair, optional=optional,
                                   pesq_mode=pesq_mode,
                                   pesq_scale=pesq_scale))
        except Exception as e:
            log.warning("skipping %s — %s: %s", pair.noisy, type(e).__name__, e)
    if not rows:
        raise RuntimeError("every pair failed to score")

    summary = {}
    for key in rows[0]:
        values = [r[key] for r in rows
                  if isinstance(r.get(key), (int, float)) and r[key] is not None]
        if values and key != "snr_db":
            summary[key] = float(np.mean(values))
    log.info("evaluated %d file(s): SI-SDR %.2f -> %.2f dB (%+.2f)",
             len(rows), summary.get("noisy_si_sdr", float("nan")),
             summary.get("enhanced_si_sdr", float("nan")),
             summary.get("delta_si_sdr", float("nan")))
    return rows, summary


def write_report(path, rows: Sequence[Dict[str, object]],
                 summary: Optional[Dict[str, float]] = None) -> Path:
    """Per-file scores as CSV, with a trailing MEAN row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        if summary:
            mean_row = {k: summary.get(k, "") for k in fieldnames}
            mean_row[fieldnames[0]] = "MEAN"
            writer.writerow(mean_row)
    log.info("report -> %s", path)
    return path


# ---------------------------------------------------------- the whole split

def item_ids(pairs: Sequence[Pair]) -> List[str]:
    """One id per pair, as kudio names the files of a folder: the noisy
    file's path below the folder every noisy file is in, without the
    extension, with forward slashes. ``noisy/p232_001.wav`` is ``p232_001``.

    :raises ValueError: for two pairs that come to one id, ``a.wav`` and
        ``a.flac`` or one file listed twice; the table would score one of
        them and say nothing of the other.
    """
    paths = [Path(os.path.abspath(pair.noisy)) for pair in pairs]
    root = Path(os.path.commonpath([str(path.parent) for path in paths]))
    ids = [path.relative_to(root).with_suffix("").as_posix() for path in paths]
    first: Dict[str, str] = {}
    twice = []
    for key, pair in zip(ids, pairs):
        if key in first:
            twice.append(f"{first[key]} and {pair.noisy}")
        first.setdefault(key, pair.noisy)
    if twice:
        more = f", and {len(twice) - 3} more" if len(twice) > 3 else ""
        raise ValueError(f"these noisy files come to one id, so the table would "
                         f"leave one of each out: {'; '.join(twice[:3])}{more}")
    return ids


def resolve_metrics(metrics: Optional[Sequence[str]] = None, *,
                    full: bool = False, dnsmos: bool = False) -> List[str]:
    """The metrics to score, in kudio's names.

    *metrics*, when given, is taken as it is, with ``seg_snr`` read as
    kudio's ``segsnr``. Otherwise it is segmental SNR, SI-SDR and SNR, which
    numpy computes alone, or :data:`FULL_METRICS` with *full*. *dnsmos* adds
    DNSMOS P.835 and P.808.

    ``--full`` has always scored what is installed, so a metric of
    :data:`FULL_METRICS` whose package is missing is left out, with a warning
    that says what to install. A metric named in *metrics*, or DNSMOS, is not
    left out that way: the table would lack a column that was asked for.

    :raises kudio.DependencyError: for a metric named in *metrics*, or
        DNSMOS, whose package is not installed.
    """
    if metrics is not None:
        names = [_KUDIO_NAME.get(m, m) for m in metrics]
        if not names:
            raise ValueError("no metric to score; leave metrics out for the "
                             "default ones")
    else:
        names = list(FULL_METRICS if full else _PLAIN)
    if dnsmos:
        names += [m for m in _DNSMOS if m not in names]
    missing = [m for m in names
               if m in _NEEDS and importlib.util.find_spec(_NEEDS[m][0]) is None]
    asked = [m for m in missing if metrics is not None or m in _DNSMOS]
    if asked:
        module, extra = _NEEDS[asked[0]]
        raise kudio.DependencyError(module, extra=extra)
    if missing:
        by_module: Dict[Tuple[str, str], List[str]] = {}
        for m in missing:
            by_module.setdefault(_NEEDS[m], []).append(m)
        for (module, extra), left in by_module.items():
            log.warning("%s left out of the table: %s is not installed "
                        "(pip install \"kudio[%s]\")", ", ".join(left), module, extra)
        names = [m for m in names if m not in missing]
    return names


def _speakers(ids: Sequence[str], pattern: Union[str, re.Pattern, None]
              ) -> Tuple[Optional[List[str]], bool, str]:
    """Each item's speaker, None where there are none; whether the intervals
    resample speakers; and the line that says which, for under the table."""
    from kudio.core.stats import FEW_UNITS

    alone = "so each sentence counts as independent of the others"
    if pattern is None or pattern == "":
        return None, False, (f"Intervals and tests resample sentences: no speakers "
                             f"were given, {alone}.")
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    if pattern.groups < 1:
        raise ValueError(f"the speaker pattern {pattern.pattern!r} has no group in "
                         f"parentheses to take the speaker from")
    found = [pattern.search(key) for key in ids]
    labels = [None if m is None else m.group(1) for m in found]
    missing = [key for key, label in zip(ids, labels) if not label]
    if len(missing) == len(ids):
        return None, False, (f"Intervals and tests resample sentences: no speaker "
                             f"could be read from the file names, {alone}.")
    if missing:
        named = ", ".join(missing[:3]) + (f" and {len(missing) - 3} more"
                                         if len(missing) > 3 else "")
        raise ValueError(
            f"the speaker pattern {pattern.pattern!r} finds a speaker in "
            f"{len(ids) - len(missing)} of {len(ids)} files but not in {named}; "
            f"give a pattern that fits every file name, or none to resample "
            f"sentences")
    names = sorted(set(labels))
    k = len(names)
    if k >= FEW_UNITS:
        return labels, True, (f"Intervals and tests resample the {k} speakers, "
                              f"each with all of their sentences.")
    some = "speaker" if k == 1 else "speakers"
    return labels, False, (
        f"Intervals and tests resample sentences, not speakers: the test set has "
        f"{k} {some} ({', '.join(names)}), and an interval over fewer than "
        f"{FEW_UNITS} falls well short of its level. They say how far the means "
        f"would move over other sentences from the same {some}, not how another "
        f"speaker would fare.")


def _own_bibtex() -> str:
    from kudioenhance._version import __author__, __version__
    return ("@misc{kudioenhance,\n"
            f"  author = {{{__author__}}},\n"
            "  title = {KudioEnhance},\n"
            f"  note = {{Version {__version__}}},\n"
            "  url = {https://github.com/recklight/KudioEnhance},\n"
            "}\n")


def _at_rate(paths: Sequence[Path], sr: int) -> bool:
    import soundfile
    return all(soundfile.info(str(path)).samplerate == sr for path in paths)


def score_split(enhancer: Enhancer, pairs: Sequence[Pair], out_dir, *,
                metrics: Optional[Sequence[str]] = None, full: bool = False,
                pesq_mode: str = "auto", pesq_scale: str = "lqo",
                speakers: Union[str, re.Pattern, None] = VCTK_SPEAKER,
                dnsmos_dir=None, jobs: int = 1, seed: int = 0,
                model: Optional[str] = None) -> "SplitScore":
    """Enhance every pair into *out_dir*, then score the split with
    :func:`kudio.score_dataset`: the enhanced files against the clean ones,
    with the noisy ones as the baseline.

    Each output is written as ``<out_dir>/<id>.wav``, with the id of
    :func:`item_ids`, in 32-bit float at the model's rate, so what is scored
    is what the model put out, not that rounded to 16 bits and clipped at
    full scale. When a clean or noisy file is at another rate, kudio
    resamples every file to the model's as it reads it, and the signature
    names the resampler.

    :param metrics, full: as :func:`resolve_metrics` takes them.
    :param speakers: a regular expression whose first group, searched in
        each id, is the speaker; None to give no speakers. Whether the
        intervals then resample speakers or sentences is in the module notes.
    :param dnsmos_dir: the folder of the DNSMOS models, as
        :func:`kudio.find_dnsmos_model` takes it; given, DNSMOS P.835 and
        P.808 are scored too.
    :param jobs: items kudio scores at once, each in a process of its own.
    :param seed: for every interval and test.
    :param model: the architecture that was trained, by its registry name;
        the works it follows go into the BibTeX after KudioEnhance's own
        entry and before kudio's.
    :raises ValueError: before anything is enhanced, for an empty split, two
        pairs that come to one id, and a speaker pattern that fits some file
        names and not others.
    """
    if not pairs:
        raise ValueError("nothing to evaluate")
    ids = item_ids(pairs)
    names = resolve_metrics(metrics, full=full, dnsmos=dnsmos_dir is not None)
    labels, grouped, note = _speakers(ids, speakers)

    sr = enhancer.cfg.audio.sr
    out_dir = Path(out_dir)
    enhanced: Dict[str, Path] = {}
    for key, pair in zip(ids, pairs):
        path = out_dir / f"{key}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        noisy, _ = kudio.file_load(pair.noisy, sr=sr)
        kudio.save_wave(path, enhancer.enhance(noisy), sr, subtype="FLOAT")
        enhanced[key] = path
    log.info("enhanced %d file(s) -> %s", len(enhanced), out_dir)

    reference = {key: Path(pair.clean) for key, pair in zip(ids, pairs)}
    baseline = {key: Path(pair.noisy) for key, pair in zip(ids, pairs)}
    same = _at_rate(list(reference.values()) + list(baseline.values()), sr)
    score = kudio.score_dataset(
        enhanced, reference=reference, baseline=baseline, metrics=names,
        sr=None if same else sr, pesq_mode=pesq_mode, pesq_scale=pesq_scale,
        groups=dict(zip(ids, labels)) if grouped else None, jobs=jobs,
        seed=seed, dnsmos_dir=dnsmos_dir)
    entries = [_own_bibtex()]
    if model is not None:
        from kudioenhance.models import model_bibtex
        entries.append(model_bibtex(model))
    # every entry ends in a newline, so one more between them leaves a blank line
    return SplitScore(score, pairs, labels, grouped, note,
                      "\n".join(entry for entry in entries if entry))


class SplitScore:
    """A test split as :func:`score_split` scored it: kudio's
    :class:`~kudio.DatasetScore`, and the line on what its intervals
    resampled, which every form of the table carries.

    :ivar score: what :func:`kudio.score_dataset` returned
    :ivar pairs: the pairs, in the order of ``score.rows``
    :ivar speakers: each item's speaker, or None when none was found
    :ivar grouped: whether the intervals and tests resampled speakers
    :ivar note: the line on that
    :ivar bibtex: the entries that go before kudio's: KudioEnhance's own,
        and those of the works the trained architecture follows
    """

    def __init__(self, score: "kudio.DatasetScore", pairs: Sequence[Pair],
                 speakers: Optional[List[str]], grouped: bool, note: str,
                 bibtex: str = ""):
        self.score = score
        self.pairs = list(pairs)
        self.speakers = speakers
        self.grouped = grouped
        self.note = note
        self.bibtex = bibtex

    @property
    def references(self) -> List[str]:
        """The BibTeX keys of :meth:`cite`, in its order."""
        keys = re.findall(r"^@\w+\{([^,\s]+),", self.bibtex, flags=re.MULTILINE)
        return keys + [key for key in self.score.references if key not in keys]

    def __repr__(self) -> str:
        return f"SplitScore({self.score!r}, resampled {self.resampled})"

    @property
    def resampled(self) -> str:
        """``'speakers'`` or ``'sentences'``."""
        return "speakers" if self.grouped else "sentences"

    def to_markdown(self) -> str:
        """kudio's Markdown table, with the line under it and the keys of
        :meth:`cite` at the end."""
        table, _, rest = self.score.to_markdown().partition("\n\n")
        listed = "References: " + ", ".join(self.references)
        rest = re.sub(r"^References: .*$", lambda _: listed, rest, flags=re.MULTILINE)
        return f"{table}\n\n{self.note}\n{rest}"

    def to_latex(self) -> str:
        r"""kudio's tabular, with the line as the first comment after it and
        the keys of :meth:`cite` in the closing ``\cite``."""
        head, end, rest = self.score.to_latex().partition("\\end{tabular}\n")
        listed = "% \\cite{" + ",".join(self.references) + "}"
        rest = re.sub(r"^% \\cite\{.*\}$", lambda _: listed, rest, flags=re.MULTILINE)
        return f"{head}{end}% {self.note}\n{rest}"

    def to_json(self) -> str:
        """kudio's JSON, with ``resampled``, the ``speakers`` found, the
        ``note``, and the keys of :meth:`cite` as ``references``."""
        data = json.loads(self.score.to_json())
        data["resampled"] = self.resampled
        data["speakers"] = sorted(set(self.speakers)) if self.speakers else []
        data["note"] = self.note
        data["references"] = self.references
        return json.dumps(data, indent=2)

    def cite(self) -> str:
        """The BibTeX for the table: :attr:`bibtex`, then kudio's entries for
        kudio and the metrics, as :meth:`kudio.DatasetScore.cite` gives them."""
        return "\n".join(text for text in (self.bibtex, self.score.cite()) if text)

    def report(self) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        """The rows and the means in the layout ``report.csv`` has had since
        1.0: ``noisy_``, ``enhanced_`` and ``delta_`` of each metric, per
        file and averaged. The means are the table's, over the files both
        sides scored, so each delta is the enhanced mean minus the noisy one;
        a score kudio could not give is None."""
        rows = []
        for pair, scored in zip(self.pairs, self.score.rows):
            row: Dict[str, object] = {"file": Path(pair.noisy).name,
                                      "noise": pair.noise, "snr_db": pair.snr_db}
            for column in self.score.columns:
                name = _REPORT_NAME.get(column, column)
                noisy, enhanced = scored.get(f"baseline_{column}"), scored.get(column)
                row[f"noisy_{name}"] = noisy
                row[f"enhanced_{name}"] = enhanced
                row[f"delta_{name}"] = (None if noisy is None or enhanced is None
                                        else enhanced - noisy)
            rows.append(row)
        summary: Dict[str, float] = {}
        for s in self.score.summary:
            name = _REPORT_NAME.get(s.column, s.column)
            for side, ci in (("noisy", s.baseline), ("enhanced", s.mean),
                             ("delta", s.difference)):
                if ci is not None and math.isfinite(ci.estimate):
                    summary[f"{side}_{name}"] = float(ci.estimate)
        return rows, summary

    def save(self, folder, stem: str = "results") -> Dict[str, Path]:
        """Write the table as ``<stem>.md``, ``<stem>.tex`` and
        ``<stem>.json``, and its BibTeX as ``<stem>.bib``, into *folder*.
        Returns the paths by suffix."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        written = {}
        for suffix, text in ((".md", self.to_markdown()), (".tex", self.to_latex()),
                             (".json", self.to_json()), (".bib", self.cite())):
            path = folder / f"{stem}{suffix}"
            # one line ending on every system, so the files compare across them
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text if text.endswith("\n") else text + "\n")
            written[suffix] = path
        log.info("table -> %s", written[".md"])
        return written
