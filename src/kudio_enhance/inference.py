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
from kudio_enhance.models import is_sequence, resolve_units

log = logging.getLogger(__name__)

__all__ = ["Enhancer"]

#: Frame-wise models see one row per frame, and an hour of audio is over a
#: million of them; this many at a time keeps memory flat whatever the length.
FRAME_BATCH = 4096


class Enhancer:
    """Enhance a waveform or a file with a trained model.

    >>> enh = Enhancer.load('runs/exp1')
    >>> clean = enh.enhance_file('noisy.wav', 'clean.wav')
    """

    def __init__(self, model, standardizer: Standardizer, cfg: Config):
        # inference never wants dropout, or BatchNorm normalising one
        # utterance by its own statistics instead of the training set's
        self.model = model.eval()
        self.standardizer = standardizer
        self.cfg = cfg
        self.sequence = is_sequence(cfg.model.name)
        self.mask = cfg.model.predicts_mask

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, run_dir, config_name: str = "config.yaml", *,
             device: str = "auto") -> "Enhancer":
        """Load model + stats + config from a run directory.

        The checkpoint holds weights only. The architecture is rebuilt from
        the run's own config and the weights are poured into it, so the
        network that comes back is the one the config describes — and the
        config is checked against what the checkpoint says it was trained as.

        :param device: where to run — ``'auto'``, whatever ``train.device``
            the run was trained with, because a model trained on a GPU box is
            routinely used on a laptop.
        """
        run_dir = Path(run_dir)
        # a run trained here always names its widths; one whose config was
        # written by hand may leave them to the default, and `cfg` should
        # say what was built either way
        cfg = resolve_units(Config.from_yaml(run_dir / config_name))
        model_path = run_dir / f"{cfg.model.name}.pt"
        if not model_path.is_file():
            legacy = model_path.with_suffix(".keras")
            if legacy.is_file():
                raise FileNotFoundError(
                    f"{run_dir} was trained by kudio-enhance 0.1 on "
                    f"TensorFlow ({legacy.name}); kudio-enhance 0.2 runs on "
                    f"PyTorch and cannot read Keras weights. Retrain it with "
                    f"0.2 — the config and the splits in the run directory "
                    f"can be reused as they are")
            raise FileNotFoundError(f"no model at {model_path}")

        from kudio_enhance.checkpoint import load_checkpoint
        from kudio_enhance.models import build_model
        from kudio_enhance.train import resolve_device

        run_on = resolve_device(device)
        checkpoint = load_checkpoint(model_path, map_location=run_on)
        trained_as = (checkpoint["model"], checkpoint["target"])
        if trained_as != (cfg.model.name, cfg.model.target):
            raise ValueError(
                f"{model_path} holds a {trained_as[0]} trained for the "
                f"{trained_as[1]!r} target, but {config_name} describes a "
                f"{cfg.model.name} for {cfg.model.target!r}")
        model = build_model(cfg)
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model.to(run_on),
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
            predicted = self._predict(normalised[None, ...])[0]
        else:
            stacked = stack_context(normalised, self.cfg.model.context)
            predicted = np.concatenate(
                [self._predict(stacked[start:start + FRAME_BATCH])
                 for start in range(0, len(stacked), FRAME_BATCH)])

        predicted = predicted[:len(spec)]
        if self.mask:
            enhanced_spec = apply_mask(spec[:len(predicted)], predicted)
        else:
            enhanced_spec = self.standardizer.inverse(predicted)
        return to_waveform(y, enhanced_spec, self.cfg.audio)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Run the network on one float32 batch, without building a graph."""
        import torch
        device = next(self.model.parameters()).device
        with torch.no_grad():
            out = self.model(torch.from_numpy(
                np.ascontiguousarray(x, dtype=np.float32)).to(device))
        return out.cpu().numpy()

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
