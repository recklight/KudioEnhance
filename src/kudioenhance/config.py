# -*- coding: utf-8 -*-
"""Typed configuration.

One YAML file describes an entire experiment. Every field is a dataclass
attribute, so a typo fails at load time with the offending key named, rather
than silently taking a default three stages later.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = ["AudioConfig", "DataConfig", "ModelConfig", "TrainConfig",
           "LossConfig", "Config"]


def _build(cls, values: Optional[Dict[str, Any]], section: str):
    """Instantiate *cls* from a mapping, rejecting unknown keys."""
    values = values or {}
    if not isinstance(values, dict):
        raise TypeError(f"config section {section!r} must be a mapping, "
                        f"got {type(values).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in section {section!r}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}")
    return cls(**values)


def _is_number(value) -> bool:
    """A finite int or float, and not a bool, which YAML hands over for
    ``yes``."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _check_power(value, key: str) -> None:
    """*value* is ``None`` or an exponent in (0, 1], the range in which raising
    a magnitude to it compresses rather than expands."""
    if value is not None and not (_is_number(value) and 0 < value <= 1):
        raise ValueError(
            f"{key} must be an exponent in (0, 1], or null for none; "
            f"got {value!r}")


@dataclass
class AudioConfig:
    """STFT geometry — shared by feature extraction and reconstruction."""

    sr: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    #: ``None`` means "follow n_fft", and is left as ``None`` rather than being
    #: resolved at construction — resolving it froze the value, so setting
    #: ``cfg.audio.n_fft = 256`` afterwards left a stale 512 behind and the
    #: geometry stopped being buildable. Read :attr:`window_length` for the
    #: number.
    win_length: Optional[int] = None
    window: str = "hamming"
    #: The waveform path's models see the complex spectrum with its magnitude
    #: raised to this power and its phase kept, ``|X|^c e^{j angle X}``;
    #: ``None`` leaves it linear. 0.3 is the usual value: across a 40 dB
    #: range it leaves the quiet bins 10^-0.6 of the loud ones in magnitude
    #: instead of 10^-2. The log-power path takes the logarithm and does not
    #: use it. What the losses compress is ``loss.compress``.
    compress: Optional[float] = None
    #: What the waveform path's centred STFT puts in the ``n_fft // 2``
    #: samples it adds at each end: ``constant`` is zeros, as kudio pads, so
    #: its frames are the log-power path's; ``reflect`` mirrors the signal,
    #: as ``torch.stft`` does by default and GTCRN's code with it. The
    #: log-power path pads with zeros whatever this says.
    pad_mode: str = "constant"

    @property
    def window_length(self) -> int:
        """The window actually used: *win_length*, or *n_fft* when unset."""
        return self.n_fft if self.win_length is None else self.win_length

    @property
    def stft(self) -> "kudio.STFT":
        """This section as a :class:`kudio.STFT`.

        One value carries the geometry to feature extraction, to
        reconstruction, and into the run directory — nothing has to be kept in
        sync by hand.
        """
        import kudio
        return kudio.STFT(sr=self.sr, n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window)

    @property
    def n_bins(self) -> int:
        return self.stft.n_bins

    def __post_init__(self) -> None:
        if self.win_length is not None and self.win_length > self.n_fft:
            raise ValueError(f"win_length ({self.win_length}) cannot exceed "
                             f"n_fft ({self.n_fft})")
        if self.hop_length > self.window_length:
            raise ValueError(f"hop_length ({self.hop_length}) cannot exceed "
                             f"win_length ({self.window_length})")
        _check_power(self.compress, "audio.compress")
        if self.pad_mode not in ("constant", "reflect"):
            raise ValueError(f"audio.pad_mode must be 'constant' or "
                             f"'reflect', got {self.pad_mode!r}")


@dataclass
class DataConfig:
    """Where the audio lives and how the mixture is built."""

    clean_dir: str = "data/clean"
    noise_dir: str = "data/noise"
    mixed_dir: str = "runs/mixed"
    snr_db: List[int] = field(default_factory=lambda: [-5, 0, 5])
    #: Reverberation times, in seconds, to put the speech through before the
    #: noise is mixed in. Empty means no rooms, which is the default and what
    #: every run before this one did.
    #:
    #: The clean file stays the target, so a model trained on this is asked to
    #: undo the room as well as the noise. That is a standard setup and a
    #: choice worth making on purpose.
    rt60: List[float] = field(default_factory=list)
    #: Direct-to-reverberant ratio for those rooms, in dB -- distance in a
    #: readable unit. +10 and above is a close microphone, 0 is a metre or
    #: two, negative is the far end of a hall.
    drr_db: float = 0.0
    #: Train against the reverberant-but-clean signal instead of the dry
    #: clean file: remove the noise, leave the room. A different and easier
    #: problem than undoing both, and the honest one when the deployment
    #: keeps its room.
    #:
    #: It also makes the ``irm`` target correct in a reverberant dataset.
    #: The mask is built from ``noisy - reference``, and with a room on the
    #: speech the dry clean file is not what was mixed, so that subtraction
    #: hands the reverberation to the *noise* and the mask asks the model to
    #: remove the room's own tail as if it were noise.
    reverberant_target: bool = False
    #: A link for the finished mixture to travel down -- a kudio channel
    #: preset (``'telephone'``, ``'voip'``, ``'mu_law'``, ...) or ``None``.
    #: Constant for the dataset rather than an axis: a corpus is recorded
    #: over a phone line or it is not.
    channel: Optional[str] = None
    mode: str = "regular"          # 'regular' or 'inc' (see kudio.Synthesizer)
    seed: Optional[int] = 17
    val_split: float = 0.1
    test_split: float = 0.1
    max_files: Optional[int] = None
    #: A folder ``kudioenhance prepare`` wrote: a corpus that comes already
    #: mixed, with its splits fixed, VoiceBank-DEMAND's. Set, the splits are
    #: read from its ``train.json``, ``val.json`` and ``test.json``, there is
    #: nothing to synthesize, and the mixing settings above and the two
    #: split fractions are not used.
    prepared: Optional[str] = None
    #: The waveform path's training examples: this many seconds cut from
    #: each clean/noisy pair, at the same place in both and at a fresh random
    #: place every epoch. A pair shorter than that is padded with silence at
    #: its end. The log-power path trains on whole files and does not use it.
    crop_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not (_is_number(self.crop_seconds) and self.crop_seconds > 0):
            raise ValueError(
                f"crop_seconds must be a positive number of seconds, got "
                f"{self.crop_seconds!r}")
        if self.mode not in ("regular", "reg", "inc", "increment"):
            raise ValueError(f"unknown synthesis mode: {self.mode!r}")
        if not 0.0 <= self.val_split + self.test_split < 1.0:
            raise ValueError("val_split + test_split must be in [0, 1)")
        if any(v <= 0 for v in self.rt60):
            raise ValueError(
                f"rt60 values must be positive, got {self.rt60}")
        if self.reverberant_target and not self.rt60:
            raise ValueError(
                "reverberant_target needs rt60: with no room the clean file "
                "already is the reverberant-clean signal")
        if self.channel is not None:
            # fail here rather than an hour into a synthesis run
            import kudio
            kudio.channel_spec(self.channel)

    @property
    def reverberant(self) -> bool:
        """Is this dataset being built with rooms?"""
        return bool(self.rt60)


#: What the network is asked to produce.
#:
#: ``spectrum`` regresses the clean log-power spectrogram directly. Simple, and
#: it has to learn the *level* of the speech as well as its shape — including
#: for bins that are pure noise, where there is nothing to learn.
#:
#: ``irm`` predicts an ideal ratio mask instead: one number per bin saying how
#: much of the noisy signal to keep. The target is bounded in [0, 1], so a
#: sigmoid output cannot produce an impossible answer, and the level comes from
#: the input rather than having to be reconstructed. It is the usual reason a
#: mask beats direct regression.
TARGETS = ("spectrum", "irm")


@dataclass
class ModelConfig:
    """Architecture selection and its hyper-parameters."""

    name: str = "ddae"
    target: str = "spectrum"       # see TARGETS
    context: int = 2               # frame-wise models: +/- frames stacked in
    n_frames: int = 64             # sequence models: frames per training window
    #: Layer widths: a dense layer's units, an LSTM's, or a convolution's
    #: filters or channels, whichever the architecture has. ``None`` means
    #: the architecture's own default, which ``kudioenhance models`` lists.
    #: No single default serves them all: 1024 is a sensible width for a
    #: dense layer and an absurd one for a 2-D convolution, each of whose
    #: channels is a whole time-frequency map. Training writes the widths it
    #: built into the run's ``config.yaml``, so a run never depends on what
    #: the default happens to be when it is loaded.
    units: Optional[List[int]] = None
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.context < 0:
            raise ValueError("context must be >= 0")
        if self.n_frames < 1:
            raise ValueError("n_frames must be >= 1")
        if self.units is not None and (
                not isinstance(self.units, (list, tuple))
                or not all(isinstance(width, int) and width >= 1
                           for width in self.units)):
            raise ValueError(
                f"units must be a list of positive widths, or null for the "
                f"architecture's default; got {self.units!r}")
        if self.target not in TARGETS:
            raise ValueError(
                f"unknown target {self.target!r}. Choose from "
                f"{', '.join(TARGETS)}")

    @property
    def predicts_mask(self) -> bool:
        """Does the network output a mask rather than a spectrogram?

        It decides three things at once — the training target, the output
        activation, and what inference does with the prediction — so it is
        asked in one place instead of being re-derived in three.
        """
        return self.target == "irm"


#: Where training runs. ``auto`` takes an NVIDIA GPU when torch can see one
#: and the CPU otherwise, so one config file works on a laptop and on a
#: training box. Apple's GPUs (torch's ``mps``) are not used: on a Mac,
#: training runs on the CPU.
DEVICES = ("auto", "cpu", "cuda")


@dataclass
class TrainConfig:
    """Optimisation and output locations."""

    epochs: int = 50
    #: Examples per training step. The frame-wise ``ddae`` needs two or more,
    #: because its BatchNorm takes a variance over the batch and a single
    #: example has none; training refuses less for it. The sequence models
    #: can train on one window at a time: where they normalise, they take
    #: the statistics over the window's frames as well.
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 8
    out_dir: str = "runs"
    #: How many examples a streamed epoch holds back to mix with the files
    #: still to come -- frames for a frame-wise model, windows for a sequence
    #: model. Every batch is drawn at random from all the examples waiting,
    #: and at least this many are left waiting until the files run out.
    #:
    #: ``0`` holds nothing back beyond the file being read: the files still
    #: come in a fresh order every epoch and each file's examples are
    #: shuffled, but a batch draws only on that file and on what earlier
    #: draws left over, which is less than a batch. Held in memory, the
    #: training set is shuffled whole every epoch and this is not used.
    shuffle_buffer: int = 0
    device: str = "auto"           # see DEVICES; 'cuda:1' picks a card
    #: Seeds the weight initialisation, the dropout masks and the order the
    #: batches come in. ``None`` borrows ``data.seed``, so the one seed that
    #: already makes the mixture reproducible makes the training run
    #: reproducible too, unless the two are deliberately pulled apart.
    seed: Optional[int] = None
    #: Compute the training features from the audio a file at a time, every
    #: epoch, instead of holding them in memory: for a training set too big
    #: to hold, at the price of a pass over the audio per epoch. See
    #: :func:`kudioenhance.data.build_dataset`.
    streaming: bool = False
    #: Processes that read and crop the audio for the waveform path, beside
    #: the one that trains; 0 reads it in the training process. The batches
    #: are the same whatever the number, because where each crop is cut is
    #: drawn in the training process. See :mod:`kudioenhance.waveform`.
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        # a float or `true` would get as far as the first epoch before
        # failing somewhere inside the shuffle
        if (isinstance(self.batch_size, bool)
                or not isinstance(self.batch_size, int)
                or self.batch_size < 1):
            raise ValueError(
                f"batch_size must be a number of examples, 1 or more; "
                f"got {self.batch_size!r}")
        # YAML reads `streaming: "false"` as a string, and a non-empty string
        # is true: refused here rather than quietly streaming
        if not isinstance(self.streaming, bool):
            raise ValueError(
                f"streaming must be true or false, got {self.streaming!r}")
        if (isinstance(self.shuffle_buffer, bool)
                or not isinstance(self.shuffle_buffer, int)
                or self.shuffle_buffer < 0):
            raise ValueError(
                f"shuffle_buffer must be a number of examples, 0 or more; "
                f"got {self.shuffle_buffer!r}")
        if (isinstance(self.num_workers, bool)
                or not isinstance(self.num_workers, int)
                or self.num_workers < 0):
            raise ValueError(
                f"num_workers must be a number of processes, 0 or more; "
                f"got {self.num_workers!r}")
        base, _, index = self.device.partition(":")
        if base not in DEVICES or (
                index and not (base == "cuda" and index.isdigit())):
            raise ValueError(
                f"unknown device {self.device!r}. Choose from "
                f"{', '.join(DEVICES)}, or 'cuda:<index>'")


#: The terms of the waveform path's loss, in the order they are summed. Each
#: is a function of :mod:`kudioenhance.losses` and a weight in
#: :class:`LossConfig` by the same name.
LOSS_TERMS = ("complex_mse", "magnitude_mse", "time_l1", "neg_si_snr")


@dataclass
class LossConfig:
    """What the waveform path trains on: a weighted sum of losses.

    Each weight multiplies the function of :mod:`kudioenhance.losses` it is
    named after, and a weight of 0 leaves that term out and uncomputed. The
    log-power path trains on MSE and does not read this section.

    The defaults are GTCRN's loss as its repository trains it (``loss.py``
    in Xiaobin-Rong/gtcrn): 30 times the compressed complex MSE, 70 times the
    compressed magnitude MSE, both at an exponent of 0.3, and ``-log10`` of
    the SI-SNR ratio, taken without removing the mean. That last term is a
    tenth of the negative SI-SNR in dB, hence 0.1 here. The 30/70 split is
    Braun and Tashev's (2021) mix, 0.3 on the complex term and 0.7 on the
    magnitude, scaled by 100.

    The loss takes the STFT the ``audio`` section describes, so it is
    GTCRN's only with GTCRN's ``audio`` as well: ``n_fft: 512``,
    ``hop_length: 256``, ``window: sqrt_hann``, ``pad_mode: reflect``.
    ``sqrt_hann`` exists only on the waveform path; kudio's log-power
    features refuse it.
    """

    complex_mse: float = 30.0
    magnitude_mse: float = 70.0
    time_l1: float = 0.0
    neg_si_snr: float = 0.1
    #: The exponent the two spectral terms raise magnitudes to, phase kept;
    #: ``None`` compares the spectra linearly.
    compress: Optional[float] = 0.3
    #: Remove the mean of both waveforms before SI-SNR, as Conv-TasNet does.
    #: GTCRN's loss does not, so neither does the default; with it off, an
    #: offset in the output counts as error there too.
    zero_mean: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.zero_mean, bool):
            raise ValueError(
                f"loss.zero_mean must be true or false, got "
                f"{self.zero_mean!r}")
        for name in LOSS_TERMS:
            weight = getattr(self, name)
            if not (_is_number(weight) and weight >= 0):
                raise ValueError(
                    f"loss.{name} must be a weight of 0 or more, got "
                    f"{weight!r}")
        if not any(getattr(self, name) for name in LOSS_TERMS):
            raise ValueError(
                f"every loss weight is 0, which leaves nothing to train on; "
                f"give one of {', '.join(LOSS_TERMS)} a positive weight")
        _check_power(self.compress, "loss.compress")

    @property
    def weights(self) -> Dict[str, float]:
        """The terms with a weight above 0, by name, in summing order."""
        return {name: float(getattr(self, name)) for name in LOSS_TERMS
                if getattr(self, name)}


@dataclass
class Config:
    """The whole experiment."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    #: Read by the waveform path only. A config, or a run's ``config.yaml``,
    #: without this section loads with the defaults.
    loss: LossConfig = field(default_factory=LossConfig)

    # ----------------------------------------------------------------- yaml

    @classmethod
    def from_yaml(cls, path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        sections = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - sections
        if unknown:
            raise ValueError(
                f"unknown config section(s): {', '.join(sorted(unknown))}. "
                f"Valid sections: {', '.join(sorted(sections))}")
        return cls(
            audio=_build(AudioConfig, raw.get("audio"), "audio"),
            data=_build(DataConfig, raw.get("data"), "data"),
            model=_build(ModelConfig, raw.get("model"), "model"),
            train=_build(TrainConfig, raw.get("train"), "train"),
            loss=_build(LossConfig, raw.get("loss"), "loss"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False,
                           allow_unicode=True)
        return path

    # ---------------------------------------------------------------- paths

    def run_dir(self, name: str) -> Path:
        """Everything produced by experiment *name* lives here."""
        return Path(self.train.out_dir) / name

    def model_path(self, name: str) -> Path:
        """The trained weights. See :mod:`kudioenhance.checkpoint`."""
        return self.run_dir(name) / f"{self.model.name}.pt"

    def stats_path(self, name: str) -> Path:
        return self.run_dir(name) / "stats.npz"

    def manifest_path(self, name: str) -> Path:
        return self.run_dir(name) / "manifest.json"

    def report_path(self, name: str) -> Path:
        return self.run_dir(name) / "report.csv"

    def enhanced_dir(self, name: str) -> Path:
        """Where ``evaluate`` writes the enhanced test files it scores."""
        return self.run_dir(name) / "enhanced"

    def results_path(self, name: str, suffix: str = ".md") -> Path:
        """The results table, as ``.md``, ``.tex`` or ``.json``, and its
        BibTeX as ``.bib``."""
        return self.run_dir(name) / f"results{suffix}"

    def history_path(self, name: str) -> Path:
        """Where the training curves land. See :func:`kudioenhance.save_history`."""
        return self.run_dir(name) / "history.json"
