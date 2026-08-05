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
    #: defaults to n_fft, so changing n_fft alone stays valid
    win_length: Optional[int] = None
    window: str = "hamming"

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
        if self.win_length is None:
            self.win_length = self.n_fft
        if self.win_length > self.n_fft:
            raise ValueError(f"win_length ({self.win_length}) cannot exceed "
                             f"n_fft ({self.n_fft})")
        if self.hop_length > self.win_length:
            raise ValueError(f"hop_length ({self.hop_length}) cannot exceed "
                             f"win_length ({self.win_length})")


@dataclass
class DataConfig:
    """Where the audio lives and how the mixture is built."""

    clean_dir: str = "data/clean"
    noise_dir: str = "data/noise"
    mixed_dir: str = "runs/mixed"
    snr_db: List[int] = field(default_factory=lambda: [-5, 0, 5])
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


@dataclass
class ModelConfig:
    """Architecture selection and its hyper-parameters."""

    name: str = "ddae"
    context: int = 2               # frame-wise models: +/- frames stacked in
    n_frames: int = 64             # sequence models: frames per training window
    units: List[int] = field(default_factory=lambda: [1024, 1024, 1024])
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.context < 0:
            raise ValueError("context must be >= 0")
        if self.n_frames < 1:
            raise ValueError("n_frames must be >= 1")


@dataclass
class TrainConfig:
    """Optimisation and output locations."""

    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 8
    out_dir: str = "runs"
    shuffle_buffer: int = 0        # 0 = shuffle the whole training set

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")


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
        return self.run_dir(name) / f"{self.model.name}.keras"

    def stats_path(self, name: str) -> Path:
        return self.run_dir(name) / "stats.npz"

    def manifest_path(self, name: str) -> Path:
        return self.run_dir(name) / "manifest.json"

    def report_path(self, name: str) -> Path:
        return self.run_dir(name) / "report.csv"
