# -*- coding: utf-8 -*-
"""Run a trained model over audio.

The model, its normalisation statistics and the STFT geometry that produced
them travel together — loading a run directory gives you all three, so
inference can never silently disagree with training.

A model of the waveform path has no statistics, and runs on the complex
spectrum. A long file is cut into chunks of fixed length that overlap, each
enhanced on its own and cross-faded into the next, so the memory a file
takes does not grow with its length beyond the output itself.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import kudio
from kudioenhance.config import Config
from kudioenhance.features import (
    Standardizer,
    apply_mask,
    spectrogram,
    stack_context,
    to_waveform,
)
from kudioenhance.models import is_sequence, is_waveform, resolve_units

log = logging.getLogger(__name__)

__all__ = ["Enhancer"]

#: Frame-wise models see one row per frame, and an hour of audio is over a
#: million of them; this many at a time keeps memory flat whatever the length.
FRAME_BATCH = 4096

#: How much audio a waveform-path model is given at once, in seconds, and how
#: much of it is shared with the next chunk and cross-faded. 30 s takes every
#: file of VoiceBank-DEMAND (the longest is 15 s) and of the DNS Challenge's
#: test sets (10 s) in one piece, so their scores are those of the whole
#: file; 1 s of overlap is 3.4% more work, and keeps the frames a chunk
#: starts or ends with, which see its zero padding instead of the audio
#: around it, at a weight below 0.003 for a 512-point frame at 16 kHz.
CHUNK_SECONDS = 30.0
OVERLAP_SECONDS = 1.0


class Enhancer:
    """Enhance a waveform or a file with a trained model.

    >>> enh = Enhancer.load('runs/exp1')
    >>> clean = enh.enhance_file('noisy.wav', 'clean.wav')
    """

    def __init__(self, model, standardizer: Optional[Standardizer],
                 cfg: Config, *,
                 chunk_seconds: Optional[float] = CHUNK_SECONDS,
                 overlap_seconds: float = OVERLAP_SECONDS):
        """*standardizer* is ``None`` for a model of the waveform path, which
        needs none. *chunk_seconds* and *overlap_seconds* say how that path
        cuts a long file (:meth:`chunking`); ``chunk_seconds=None`` takes
        every file whole."""
        # inference never wants dropout, or BatchNorm normalising one
        # utterance by its own statistics instead of the training set's
        self.model = model.eval()
        self.standardizer = standardizer
        self.cfg = cfg
        self.sequence = is_sequence(cfg.model.name)
        self.mask = cfg.model.predicts_mask
        self.waveform = is_waveform(cfg.model.name)
        if self.waveform:
            from kudioenhance.spectral import ComplexSTFT
            self.stft = ComplexSTFT.from_config(cfg.audio)
            self.chunk, self.overlap = self.chunking(chunk_seconds,
                                                     overlap_seconds)
        elif standardizer is None:
            raise ValueError(
                f"{cfg.model.name} works on the standardised log-power "
                f"spectrogram, and needs the run's statistics")

    def chunking(self, chunk_seconds: Optional[float],
                 overlap_seconds: float) -> Tuple[Optional[int], int]:
        """``(chunk, overlap)`` in samples for the waveform path, both
        rounded to whole hops; ``(None, 0)`` takes files whole.

        Whole hops, so a chunk's frames fall where the whole file's do, and
        a model that sees a few frames either side gives, away from the
        chunk's ends, what it gives on the whole file.

        :raises ValueError: for an overlap of more than half a chunk, or a
            chunk shorter than ``n_fft``.
        """
        if chunk_seconds is None:
            return None, 0
        sr, hop = self.cfg.audio.sr, self.cfg.audio.hop_length
        chunk = int(round(chunk_seconds * sr / hop)) * hop
        overlap = int(round(overlap_seconds * sr / hop)) * hop
        if chunk < self.cfg.audio.n_fft:
            raise ValueError(
                f"a chunk of {chunk_seconds} s is {chunk} samples, shorter "
                f"than one frame of {self.cfg.audio.n_fft}")
        if not 0 <= overlap <= chunk // 2:
            raise ValueError(
                f"the overlap ({overlap_seconds} s) must be between 0 and "
                f"half a chunk ({chunk_seconds} s)")
        return chunk, overlap

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, run_dir, config_name: str = "config.yaml", *,
             device: str = "auto",
             chunk_seconds: Optional[float] = CHUNK_SECONDS,
             overlap_seconds: float = OVERLAP_SECONDS) -> "Enhancer":
        """Load model + stats + config from a run directory.

        The checkpoint holds weights only. The architecture is rebuilt from
        the run's own config and the weights are poured into it, so the
        network that comes back is the one the config describes — and the
        config is checked against what the checkpoint says it was trained as.

        A run of the waveform path has no ``stats.npz``, and loads while it
        is still training: its best epoch so far.

        :param device: where to run — ``'auto'``, whatever ``train.device``
            the run was trained with, because a model trained on a GPU box is
            routinely used on a laptop.
        :param chunk_seconds, overlap_seconds: as the constructor takes them.
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
                    f"{run_dir} was trained by version 0.1 on TensorFlow "
                    f"({legacy.name}); since 0.2 training and inference run "
                    f"on PyTorch, which cannot read Keras weights. Retrain "
                    f"it — the config and the splits in the run directory "
                    f"can be reused as they are")
            raise FileNotFoundError(f"no model at {model_path}")

        from kudioenhance.checkpoint import load_checkpoint
        from kudioenhance.models import build_model
        from kudioenhance.train import resolve_device

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
        standardizer = (None if is_waveform(cfg.model.name)
                        else Standardizer.load(run_dir / "stats.npz"))
        return cls(model.to(run_on), standardizer, cfg,
                   chunk_seconds=chunk_seconds,
                   overlap_seconds=overlap_seconds)

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
        if self.waveform:
            return self._enhance_chunks(y)

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

    def _enhance_chunks(self, y: np.ndarray) -> np.ndarray:
        """The waveform path: *y* whole when it fits in a chunk, otherwise
        chunk by chunk.

        Chunk k starts at ``k (chunk - overlap)``, and the last one, which
        reaches the end, is padded with zeros to the full length, so the
        model always sees the same shape. Across each overlap the earlier
        chunk fades out as ``cos^2`` and the later one fades in as
        ``sin^2``, which add to 1, so where neighbouring chunks agree the
        result is what each says. They agree away from their ends for a
        model that sees a few frames either side; a recurrent one starts
        every chunk from its initial state, and the overlap is where it
        settles.

        Each overlap is blended in a buffer of its own, so the output is the
        only array as long as the file.
        """
        n = len(y)
        if self.chunk is None or n <= self.chunk:
            return self._run(y)
        step, overlap = self.chunk - self.overlap, self.overlap
        # sampled between the samples, so neither end of a fade is exactly 0
        fade_in = np.sin(0.5 * np.pi * (np.arange(overlap) + 0.5)
                         / max(overlap, 1)) ** 2
        out = np.empty(n, dtype=np.float32)
        tail = None
        start = 0
        while True:
            last = start + self.chunk >= n
            piece = y[start:start + self.chunk]
            if len(piece) < self.chunk:
                piece = np.pad(piece, (0, self.chunk - len(piece)))
            enhanced = self._run(piece).astype(np.float64)
            if tail is not None and overlap:
                enhanced[:overlap] = ((1.0 - fade_in) * tail
                                      + fade_in * enhanced[:overlap])
            if last:
                out[start:] = enhanced[:n - start]
                return out
            # up to where the next chunk takes over; this one's own end is
            # kept to fade out under it
            out[start:start + step] = enhanced[:step]
            tail = enhanced[step:]
            start += step

    def _run(self, y: np.ndarray) -> np.ndarray:
        """One waveform through the STFT, the model and back, as long as it
        came in."""
        import torch

        from kudioenhance.waveform_train import estimate_spectrum
        n = len(y)
        if n <= self.cfg.audio.n_fft // 2:
            # reflect padding needs more samples than it adds
            y = np.pad(y, (0, self.cfg.audio.n_fft - n))
        weight = next(self.model.parameters(), None)
        device = "cpu" if weight is None else weight.device
        stft = self.stft.to(device)
        wave = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        with torch.no_grad():
            spec = estimate_spectrum(self.model, stft, wave.to(device)[None])
            out = stft.waveform(spec, length=len(y))[0]
        return out.cpu().numpy()[:n]

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
