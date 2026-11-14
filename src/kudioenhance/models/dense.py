# -*- coding: utf-8 -*-
"""Deep denoising autoencoder — the frame-wise baseline.

Maps one (context-stacked) noisy frame to one clean frame, as the regression
network of Y. Xu, J. Du, L.-R. Dai and C.-H. Lee, "A regression approach to
speech enhancement based on deep neural networks", IEEE/ACM Trans. Audio,
Speech, Lang. Process. 23(1), 7-19, 2015, does; their paper also trains it
on the ideal ratio mask, the ``irm`` target here. The name comes from X. Lu,
Y. Tsao, S. Matsuda and C. Hori, "Speech enhancement based on deep denoising
autoencoder", Interspeech 2013, whose autoencoder maps a patch of frames to
a patch rather than to the frame at its centre.

No temporal modelling beyond the stacked context, which makes it fast to
train and a sensible reference point for the sequence models.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

from torch import nn

from kudioenhance.models._defaults import batch_norm, init_like_keras

__all__ = ["build_ddae"]


def build_ddae(input_dim: int, output_dim: int,
               units: Sequence[int] = (1024, 1024, 1024),
               dropout: float = 0.1,
               mask: bool = False) -> nn.Sequential:
    """Dense encoder/decoder, ``(N, input_dim) -> (N, output_dim)``.

    Nothing in it looks across rows, so a plain stack of layers is the whole
    model; the layer names are the ones the Keras version used.

    :param mask: end in a sigmoid instead of a linear layer. A mask lives in
        ``[0, 1]``, so a bounded output cannot produce an impossible answer —
        a linear head would spend part of its capacity learning not to.
    """
    layers: "OrderedDict[str, nn.Module]" = OrderedDict()
    width_in = input_dim
    for i, width in enumerate(units):
        layers[f"dense_{i}"] = nn.Linear(width_in, width)
        layers[f"bn_{i}"] = batch_norm(width)
        layers[f"relu_{i}"] = nn.ReLU()
        if dropout:
            layers[f"drop_{i}"] = nn.Dropout(dropout)
        width_in = width
    layers["head"] = nn.Linear(width_in, output_dim)
    if mask:
        layers["sigmoid"] = nn.Sigmoid()
    return init_like_keras(nn.Sequential(layers))
