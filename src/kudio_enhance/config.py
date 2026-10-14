# -*- coding: utf-8 -*-
"""Typed configuration.

One YAML file describes an entire experiment. Every field is a dataclass
attribute, so a typo fails at load time with the offending key named, rather
than silently taking a default three stages later.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = ["AudioConfig", "DataConfig", "ModelConfig", "TrainConfig", "Config"]


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

    def __post_init__(self) -> None:
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
    #: the architecture's own default, which ``kudio-enhance models`` lists.
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
    #: :func:`kudio_enhance.data.build_dataset`.
    streaming: bool = False

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
        base, _, index = self.device.partition(":")
        if base not in DEVICES or (
                index and not (base == "cuda" and index.isdigit())):
            raise ValueError(
                f"unknown device {self.device!r}. Choose from "
                f"{', '.join(DEVICES)}, or 'cuda:<index>'")


@dataclass
class Config:
    """The whole experiment."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

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
        """The trained weights. See :mod:`kudio_enhance.checkpoint`."""
        return self.run_dir(name) / f"{self.model.name}.pt"

    def stats_path(self, name: str) -> Path:
        return self.run_dir(name) / "stats.npz"

    def manifest_path(self, name: str) -> Path:
        return self.run_dir(name) / "manifest.json"

    def report_path(self, name: str) -> Path:
        return self.run_dir(name) / "report.csv"

    def history_path(self, name: str) -> Path:
        """Where the training curves land. See :func:`kudio_enhance.save_history`."""
        return self.run_dir(name) / "history.json"
