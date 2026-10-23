# -*- coding: utf-8 -*-
"""Bidirectional LSTM — temporal context in both directions.

Trained on fixed-length windows, but nothing in it fixes the length of the
time axis, so the same weights enhance a whole utterance in one pass at
inference time.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from kudioenhance.models._defaults import init_like_keras

__all__ = ["BLSTM", "build_blstm"]


class BLSTM(nn.Module):
    """Stacked BLSTM with a per-frame head, ``(B, T, bins) -> (B, T, bins)``.

    One :class:`torch.nn.LSTM` per layer rather than one with ``num_layers``,
    because the layers may differ in width and torch's stacked LSTM cannot.
    """

    def __init__(self, n_bins: int, units: Sequence[int], dropout: float,
                 mask: bool):
        super().__init__()
        self.mask = mask
        self.blstm = nn.ModuleList()
        width_in = n_bins
        for width in units:
            self.blstm.append(nn.LSTM(width_in, width, batch_first=True,
                                      bidirectional=True))
            width_in = 2 * width        # both directions, concatenated
        self.drop = nn.Dropout(dropout) if dropout else None
        # a Linear acts on the last axis, so it is already applied per frame
        self.head = nn.Linear(width_in, n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lstm in self.blstm:
            x, _ = lstm(x)
            if self.drop is not None:
                x = self.drop(x)
        x = self.head(x)
        return torch.sigmoid(x) if self.mask else x


def build_blstm(n_bins: int, units: Sequence[int] = (256, 256),
                dropout: float = 0.1, mask: bool = False) -> BLSTM:
    """Stacked BLSTM with a time-distributed head.

    :param mask: sigmoid output for a bounded mask; linear otherwise.
    """
    return init_like_keras(BLSTM(n_bins, units, dropout, mask))
