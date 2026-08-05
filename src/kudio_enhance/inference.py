# -*- coding: utf-8 -*-
"""Run a trained model over audio.

The model, its normalisation statistics and the STFT geometry that produced
them travel together — loading a run directory gives you all three, so
inference can never silently disagree with training.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

import kudio
from kudio_enhance.config import Config
from kudio_enhance.features import (
    Standardizer,
    spectrogram,
    stack_context,
    to_waveform,
)
from kudio_enhance.models import is_sequence

log = logging.getLogger(__name__)

__all__ = ["Enhancer"]


class Enhancer:
    """Enhance a waveform or a file with a trained model.

    >>> enh = Enhancer.load('runs/exp1')
    >>> clean = enh.enhance_file('noisy.wav', 'clean.wav')
    """

    def __init__(self, model, standardizer: Standardizer, cfg: Config):
        self.model = model
        self.standardizer = standardizer
        self.cfg = cfg
        self.sequence = is_sequence(cfg.model.name)

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, run_dir, config_name: str = "config.yaml") -> "Enhancer":
        """Load model + stats + config from a run directory."""
        import keras
        run_dir = Path(run_dir)
        cfg = Config.from_yaml(run_dir / config_name)
        model_path = run_dir / f"{cfg.model.name}.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"no model at {model_path}")
        return cls(keras.models.load_model(str(model_path)),
                   Standardizer.load(run_dir / "stats.npz"),
                   cfg)

    # ------------------------------------------------------------- enhance

    def enhance(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Denoise a waveform, returning one the same length as the input."""
        if sr is not None and sr != self.cfg.audio.sr:
            raise ValueError(
                f"expected {self.cfg.audio.sr} Hz (the rate this model was "
                f"trained at), got {sr}; resample first")
        y = np.asarray(y, dtype=np.float32)
        if len(y) == 0:
            return y

        spec = spectrogram(y, self.cfg.audio)
        normalised = self.standardizer.transform(spec)

        if self.sequence:
            # trained on fixed windows, applied to the whole utterance at once
            predicted = self.model.predict(normalised[None, ...], verbose=0)[0]
        else:
            stacked = stack_context(normalised, self.cfg.model.context)
            predicted = self.model.predict(stacked, verbose=0)

        enhanced_spec = self.standardizer.inverse(predicted)[:len(spec)]
        return to_waveform(y, enhanced_spec, self.cfg.audio)

    def enhance_file(self, src, dst=None, subtype: str = "PCM_16") -> np.ndarray:
        """Enhance *src*; write to *dst* when given. Returns the waveform."""
        y, sr = kudio.file_load(src, sr=self.cfg.audio.sr)
        enhanced = self.enhance(y)
        if dst is not None:
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            kudio.save_wave(dst, enhanced, sr, subtype=subtype)
            log.info("enhanced %s -> %s", Path(src).name, dst)
        return enhanced
