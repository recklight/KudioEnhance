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
    apply_mask,
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
        self.mask = cfg.model.predicts_mask

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
        """Denoise a waveform, returning one the same length as the input.

        What the prediction *means* comes from the run directory's config, not
        from a guess about its shape: a mask model's output is multiplied into
        the noisy spectrum, a spectrum model's is un-normalised. Getting that
        backwards produces audio, just not the right audio — which is exactly
        why the config travels with the weights.
        """
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

        predicted = predicted[:len(spec)]
        if self.mask:
            enhanced_spec = apply_mask(spec[:len(predicted)], predicted)
        else:
            enhanced_spec = self.standardizer.inverse(predicted)
        return to_waveform(y, enhanced_spec, self.cfg.audio)

    def enhance_file(self, src, dst=None, subtype: str = "PCM_16") -> np.ndarray:
        """Enhance *src*; write to *dst* when given. Returns the waveform.

        The file is read at the model's rate, so a corpus at another rate is
        resampled on the way in rather than being fed to a model that was never
        trained for it.
        """
        y, sr = kudio.file_load(src, sr=self.cfg.audio.sr)
        enhanced = self.enhance(y)
        if dst is not None:
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            kudio.save_wave(dst, enhanced, sr, subtype=subtype)
            log.info("enhanced %s -> %s", Path(src).name, dst)
        return enhanced

    def enhance_folder(self, src, dst, *, subtype: str = "PCM_16",
                       overwrite: bool = False, progress=None,
                       on_error: str = "collect", report: bool = False):
        """Enhance every audio file under *src* into *dst*.

        >>> result = Enhancer.load('runs/exp1').enhance_folder('noisy/', 'clean/')
        >>> print(result)
        412/412 written, 0 failed

        The model is loaded once and reused, which is the whole point — the
        obvious loop over :meth:`enhance_file` is correct and reloads nothing,
        but a caller writing it themselves usually reloads per file.

        Output mirrors the input's directory structure, so two files with the
        same name in different subfolders do not collide. Returns a
        :class:`kudio.EnhanceFolderResult`; with ``report=True`` it also carries
        each file's noise floor before and after.

        :param on_error: ``'collect'`` records the failure and carries on;
            ``'raise'`` stops at the first one.
        """
        return kudio.enhance_folder(
            src, dst,
            # kudio dispatches on the method name; here the "method" is this
            # model, so it goes through the custom-enhancer door
            method=kudio.CustomEnhancer(name=self.cfg.model.name,
                                        fn=lambda y, sr: self.enhance(y)),
            sr=self.cfg.audio.sr, subtype=subtype, overwrite=overwrite,
            progress=progress, on_error=on_error, report=report)
