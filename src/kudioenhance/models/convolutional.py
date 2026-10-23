# -*- coding: utf-8 -*-
"""Dilated 1-D convolutional autoencoder.

Dilation widens the receptive field without striding, so the time axis keeps
its length exactly and variable-length input needs no cropping or padding
bookkeeping. The output is a residual correction to the noisy input, which
starts training from "pass the signal through" rather than from noise.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn

from kudioenhance.models._defaults import batch_norm, init_like_keras

__all__ = ["ConvAE", "build_conv_ae"]


class ConvAE(nn.Module):
    """Stacked dilated Conv1D, ``(B, T, bins) -> (B, T, bins)``.

    Layer *i* is dilated by ``2 ** i`` and padded to keep its length, so the
    receptive field doubles per layer while every layer still emits one frame
    per input frame.
    """

    def __init__(self, n_bins: int, filters: Sequence[int], kernel_size: int,
                 dropout: float, mask: bool):
        super().__init__()
        self.mask = mask
        layers: "OrderedDict[str, nn.Module]" = OrderedDict()
        width_in = n_bins
        for i, width in enumerate(filters):
            layers[f"conv_{i}"] = nn.Conv1d(width_in, width, kernel_size,
                                            padding="same", dilation=2 ** i)
            layers[f"bn_{i}"] = batch_norm(width)
            layers[f"relu_{i}"] = nn.ReLU()
            if dropout:
                layers[f"drop_{i}"] = nn.Dropout(dropout)
            width_in = width
        self.encoder = nn.Sequential(layers)
        self.head = nn.Conv1d(width_in, n_bins, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the data path is (batch, time, bins); Conv1d wants the channels
        # (here the bins) ahead of the time axis
        out = self.head(self.encoder(x.transpose(1, 2))).transpose(1, 2)
        if self.mask:
            return torch.sigmoid(out)
        return x + out


def build_conv_ae(n_bins: int, filters: Sequence[int] = (256, 256, 256),
                  kernel_size: int = 5, dropout: float = 0.1,
                  mask: bool = False) -> ConvAE:
    """Stacked dilated Conv1D.

    :param mask: predict a mask through a sigmoid instead of a residual
        correction. The residual shortcut is dropped in that case — "add this
        to the input" and "keep this fraction of the input" are different
        parameterisations, and combining them would make the sigmoid an
        offset rather than a proportion.
    """
    return init_like_keras(ConvAE(n_bins, filters, kernel_size, dropout, mask))
