# -*- coding: utf-8 -*-
"""Model registry.

Adding an architecture is one entry here plus a builder function — no
``if/elif`` chain to extend in three separate places.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from kudioenhance.config import Config

log = logging.getLogger(__name__)

__all__ = ["ModelSpec", "REGISTRY", "list_models", "is_sequence",
           "is_waveform", "default_units", "resolve_units", "build_model",
           "count_params", "count_macs", "model_bibtex"]


@dataclass(frozen=True)
class ModelSpec:
    """How to build an architecture, and what shape of data it wants.

    ``sequence`` decides the whole data path: sequence models are trained on
    ``(N, n_frames, bins)`` windows, frame-wise models on context-stacked
    ``(N, bins * (2 * context + 1))`` rows.

    ``default_units`` is what the builder is handed when ``model.units`` is
    left unset. A width means something different to each architecture — a
    dense layer's units, a 2-D convolution's channels, each of which is a
    whole time-frequency map — so the default belongs to the architecture
    rather than to the config. It falls back to the 1024-wide stack every
    architecture shared before it had one, so an entry written without it
    builds what it always did.

    ``references`` holds the BibTeX of the works the architecture follows,
    which ``evaluate`` puts into ``results.bib`` beside kudio's entries for
    the metrics.

    ``waveform`` marks an architecture of the waveform path: it is trained
    on waveform crops through the complex STFT by
    :func:`kudioenhance.waveform_train.fit_waveform`, and takes and gives
    a complex spectrum, as
    :func:`~kudioenhance.waveform_train.estimate_spectrum` says.
    ``sequence`` means nothing to it.
    """

    builder: Callable
    sequence: bool
    description: str
    default_units: Tuple[int, ...] = (1024, 1024, 1024)
    references: Tuple[str, ...] = ()
    waveform: bool = False


def _ddae(cfg: Config):
    from kudioenhance.models.dense import build_ddae
    bins = cfg.audio.n_bins
    return build_ddae(input_dim=bins * (2 * cfg.model.context + 1),
                      output_dim=bins,
                      units=cfg.model.units,
                      dropout=cfg.model.dropout,
                      mask=cfg.model.predicts_mask)


def _blstm(cfg: Config):
    from kudioenhance.models.recurrent import build_blstm
    return build_blstm(n_bins=cfg.audio.n_bins,
                       units=cfg.model.units,
                       dropout=cfg.model.dropout,
                       mask=cfg.model.predicts_mask)


def _conv_ae(cfg: Config):
    from kudioenhance.models.convolutional import build_conv_ae
    return build_conv_ae(n_bins=cfg.audio.n_bins,
                         filters=cfg.model.units,
                         dropout=cfg.model.dropout,
                         mask=cfg.model.predicts_mask)


def _crn(cfg: Config):
    from kudioenhance.models.crn import build_crn
    return build_crn(n_bins=cfg.audio.n_bins,
                     channels=cfg.model.units,
                     dropout=cfg.model.dropout,
                     mask=cfg.model.predicts_mask)


def _unet(cfg: Config):
    from kudioenhance.models.unet import build_unet
    return build_unet(channels=cfg.model.units,
                      dropout=cfg.model.dropout,
                      mask=cfg.model.predicts_mask)


def _gtcrn(cfg: Config):
    from kudioenhance.models.gtcrn import build_gtcrn
    # n_fft // 2 + 1 rather than cfg.audio.n_bins, which asks kudio's STFT,
    # and kudio has no sqrt_hann window
    return build_gtcrn(n_bins=cfg.audio.n_fft // 2 + 1, sr=cfg.audio.sr,
                       units=cfg.model.units)


# The works each architecture follows, checked against the publisher's or
# ISCA's record. ddae maps a context window to one frame as Xu et al. do, and
# takes its name from Lu et al., whose autoencoder maps a patch of frames to a
# patch. blstm is Erdogan et al.'s stacked BLSTM, whose output is a mask; the
# spectrum head is the same network with a linear output. conv_ae keeps the
# part of Tan, Chen & Wang's gated residual network that convolves along time
# over STFT magnitudes with a doubling dilation, after Yu & Koltun, and drops
# its 2-D front end, the gates and the repeated schedule. unet is
# Ronneberger et al.'s layout, put on a spectrogram with a mask out as Jansson
# et al. do; their U-Net strides instead of pooling. gtcrn is Rong et al.'s
# network, with what the paper leaves open taken from the authors' code.
_LU_2013 = """@inproceedings{lu2013ddae,
  author = {X. Lu and Y. Tsao and S. Matsuda and C. Hori},
  title = {Speech enhancement based on deep denoising autoencoder},
  booktitle = {Proc. Interspeech},
  pages = {436--440},
  year = {2013},
  doi = {10.21437/Interspeech.2013-130},
}
"""
_XU_2015 = """@article{xu2015regression,
  author = {Y. Xu and J. Du and L.-R. Dai and C.-H. Lee},
  title = {A regression approach to speech enhancement based on deep neural networks},
  journal = {IEEE/ACM Trans. Audio, Speech, Lang. Process.},
  volume = {23},
  number = {1},
  pages = {7--19},
  year = {2015},
  doi = {10.1109/TASLP.2014.2364452},
}
"""
_ERDOGAN_2015 = """@inproceedings{erdogan2015phase,
  author = {H. Erdogan and J. R. Hershey and S. Watanabe and J. {Le Roux}},
  title = {Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks},
  booktitle = {Proc. ICASSP},
  pages = {708--712},
  year = {2015},
  doi = {10.1109/ICASSP.2015.7178061},
}
"""
_TAN_2018 = """@inproceedings{tan2018crn,
  author = {K. Tan and D. Wang},
  title = {A convolutional recurrent neural network for real-time speech enhancement},
  booktitle = {Proc. Interspeech},
  pages = {3229--3233},
  year = {2018},
  doi = {10.21437/Interspeech.2018-1405},
}
"""
_TAN_2019 = """@article{tan2019grn,
  author = {K. Tan and J. Chen and D. Wang},
  title = {Gated residual networks with dilated convolutions for monaural speech enhancement},
  journal = {IEEE/ACM Trans. Audio, Speech, Lang. Process.},
  volume = {27},
  number = {1},
  pages = {189--198},
  year = {2019},
  doi = {10.1109/TASLP.2018.2876171},
}
"""
_YU_2016 = """@inproceedings{yu2016dilated,
  author = {F. Yu and V. Koltun},
  title = {Multi-scale context aggregation by dilated convolutions},
  booktitle = {Proc. ICLR},
  year = {2016},
  doi = {10.48550/arXiv.1511.07122},
  eprint = {1511.07122},
  archiveprefix = {arXiv},
}
"""
_RONNEBERGER_2015 = """@inproceedings{ronneberger2015unet,
  author = {O. Ronneberger and P. Fischer and T. Brox},
  title = {{U-Net}: Convolutional networks for biomedical image segmentation},
  booktitle = {Proc. MICCAI},
  series = {Lecture Notes in Computer Science},
  volume = {9351},
  pages = {234--241},
  year = {2015},
  doi = {10.1007/978-3-319-24574-4_28},
}
"""
_JANSSON_2017 = """@inproceedings{jansson2017unet,
  author = {A. Jansson and E. Humphrey and N. Montecchio and R. Bittner and A. Kumar and T. Weyde},
  title = {Singing voice separation with deep {U-Net} convolutional networks},
  booktitle = {Proc. ISMIR},
  pages = {745--751},
  year = {2017},
  doi = {10.5281/zenodo.1414934},
}
"""

_RONG_2024 = """@inproceedings{rong2024gtcrn,
  author = {X. Rong and T. Sun and X. Zhang and Y. Hu and C. Zhu and J. Lu},
  title = {{GTCRN}: A speech enhancement model requiring ultralow
           computational resources},
  booktitle = {Proc. ICASSP},
  pages = {971--975},
  year = {2024},
  doi = {10.1109/ICASSP48485.2024.10448310},
}
"""

REGISTRY: Dict[str, ModelSpec] = {
    "ddae": ModelSpec(_ddae, sequence=False,
                      description="dense denoising autoencoder, frame-wise "
                                  "(Xu et al. 2015)",
                      references=(_XU_2015, _LU_2013)),
    "blstm": ModelSpec(_blstm, sequence=True,
                       description="stacked bidirectional LSTM "
                                   "(Erdogan et al. 2015)",
                       references=(_ERDOGAN_2015,)),
    "conv_ae": ModelSpec(_conv_ae, sequence=True,
                         description="dilated Conv1D autoencoder, residual "
                                     "(after Tan et al. 2019)",
                         references=(_TAN_2019, _YU_2016)),
    # The CRN's LSTMs read every channel at every bin left after five
    # halvings, so their width is the last channel count times
    # ceil(bins / 32): 64 x 9 = 576 at 257 bins, 64 x 5 = 320 at 129, for
    # about 5.5M and 1.8M parameters, most of them in the LSTMs. The paper
    # doubles on to 256, which its unpadded convolutions can afford by
    # leaving 4 of 161 bins; here 256 x 9 would be an LSTM of some 85M.
    "crn": ModelSpec(_crn, sequence=True, default_units=(16, 32, 64, 64, 64),
                     description="causal conv-recurrent network "
                                 "(Tan & Wang 2018)",
                     references=(_TAN_2018,)),
    # Nothing in a U-Net depends on the number of bins, so this is about
    # half a million parameters at any n_fft. Three halvings take the default
    # 64-frame training window down to 8 frames at the bottom, where each
    # position already sees 68 frames of input -- wider than the window, so a
    # fourth level would add parameters and no reach that training could use.
    "unet": ModelSpec(_unet, sequence=True, default_units=(16, 32, 64, 128),
                      description="2-D U-Net over time and frequency, "
                                  "non-causal (Ronneberger et al. 2015)",
                      references=(_RONNEBERGER_2015, _JANSSON_2017)),
    # The paper's width; at 257 bins that is 23,669 weights and 26.3M
    # multiply-accumulates a second (tests/test_gtcrn.py counts both layer by
    # layer).
    "gtcrn": ModelSpec(_gtcrn, sequence=True, waveform=True,
                       default_units=(16,),
                       description="grouped temporal conv-recurrent network, "
                                   "causal, complex mask (Rong et al. 2024)",
                       references=(_RONG_2024,)),
}


def list_models() -> Dict[str, str]:
    """``{name: description}`` for the CLI and the docs."""
    return {name: spec.description for name, spec in REGISTRY.items()}


def _spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}. Available: "
            f"{', '.join(sorted(REGISTRY))}") from None


def is_sequence(name: str) -> bool:
    return _spec(name).sequence


def is_waveform(name: str) -> bool:
    """Does architecture *name* train on the waveform path?"""
    return _spec(name).waveform


def default_units(name: str) -> List[int]:
    """The widths architecture *name* is built with when ``model.units`` is
    unset."""
    return list(_spec(name).default_units)


def resolve_units(cfg: Config) -> Config:
    """*cfg* with ``model.units`` spelled out.

    ``None`` means "the architecture's default", which lets one starter
    config switch between a 1024-wide dense model and a 16-channel
    convolution by changing the name alone. A run directory has to say what
    was actually built, though: a default is free to change between
    releases, and a ``config.yaml`` that still said ``None`` would then
    rebuild a network its own weights no longer fit.

    Returns *cfg* itself when the widths are already given, and otherwise a
    copy, so the caller's config is never changed underneath it.
    """
    if cfg.model.units is not None:
        return cfg
    resolved = copy.deepcopy(cfg)
    resolved.model.units = default_units(cfg.model.name)
    return resolved


def build_model(cfg: Config):
    """Build the configured architecture as a freshly initialised
    :class:`torch.nn.Module`.

    The optimiser and the loop belong to :mod:`kudioenhance.train`, so a
    model can be built and inspected without committing to either — and
    :class:`~kudioenhance.inference.Enhancer` rebuilds a trained one the same
    way before pouring its weights in.

    ``model.units`` left unset builds the architecture's default widths; see
    :func:`resolve_units`.
    """
    spec = _spec(cfg.model.name)
    cfg = resolve_units(cfg)
    model = spec.builder(cfg)
    log.info("built %s %s (%s target): %d parameters",
             cfg.model.name, cfg.model.units, cfg.model.target,
             count_params(model))
    return model


def model_bibtex(name: str) -> str:
    """The BibTeX of the works architecture *name* follows, one entry after
    another; empty when it has none."""
    return "\n".join(_spec(name).references)


def count_params(model) -> int:
    """How many numbers the optimiser can move.

    BatchNorm's running mean and variance are buffers in torch rather than
    weights, so they are not counted; Keras counted them as non-trainable
    weights, and reported a slightly larger number for the same network.
    """
    return sum(param.numel() for param in model.parameters())


def count_macs(model, *inputs) -> int:
    """Multiply-accumulates of one forward pass of *model* on *inputs*.

    What torch's ``FlopCounterMode`` counts, at two FLOPs a
    multiply-accumulate: the matrix products and convolutions, and whatever
    multiplies by a buffer, such as GTCRN's ERB filters. Biases,
    normalisation, activations and other elementwise arithmetic are left
    out, so ptflops, which counts those too, reports more for the same
    network: 33.0M for GTCRN on one second, where this counts 26.5M.

    The count runs on a copy of *model*, on the CPU, in eval mode, without a
    graph and with oneDNN off. oneDNN runs an LSTM as one fused operation
    the counter has no formula for, and so does cuDNN for any recurrent
    layer; on the CPU without oneDNN, every step of a GRU or an LSTM is a
    pair of matrix products and is counted. *model* itself is left as it was,
    modes and device included.
    """
    import copy

    import torch
    from torch.utils.flop_counter import FlopCounterMode

    probe = copy.deepcopy(model).cpu().eval()
    inputs = [x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs]
    # the property rather than mkldnn.flags(), which also sets TF32 and
    # warns about Intel GPUs on every call
    onednn = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            probe(*inputs)
    finally:
        torch.backends.mkldnn.enabled = onednn
    return counter.get_total_flops() // 2
