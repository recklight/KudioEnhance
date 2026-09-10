# -*- coding: utf-8 -*-
"""Bidirectional LSTM — temporal context in both directions.

Trained on fixed-length windows but built with an undefined time axis, so the
same weights enhance a whole utterance in one pass at inference time.
"""
from __future__ import annotations

from typing import Sequence

import keras
from keras import layers

__all__ = ["build_blstm"]


def build_blstm(n_bins: int, units: Sequence[int] = (256, 256),
                dropout: float = 0.1, mask: bool = False) -> keras.Model:
    """Stacked BLSTM with a time-distributed head.

    :param mask: sigmoid output for a bounded mask; linear otherwise.
    """
    inputs = keras.Input(shape=(None, n_bins), name="noisy_spec")
    x = inputs
    for i, width in enumerate(units):
        x = layers.Bidirectional(
            layers.LSTM(width, return_sequences=True), name=f"blstm_{i}")(x)
        if dropout:
            x = layers.Dropout(dropout, name=f"drop_{i}")(x)
    outputs = layers.TimeDistributed(
        layers.Dense(n_bins, activation="sigmoid" if mask else None),
        name="mask_spec" if mask else "clean_spec")(x)
    return keras.Model(inputs, outputs, name="blstm")
