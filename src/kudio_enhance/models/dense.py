# -*- coding: utf-8 -*-
"""Deep denoising autoencoder — the frame-wise baseline.

Maps one (context-stacked) noisy frame to one clean frame. No temporal
modelling beyond the stacked context, which makes it fast to train and a
sensible reference point for the sequence models.
"""
from __future__ import annotations

from typing import Sequence

import keras
from keras import layers

__all__ = ["build_ddae"]


def build_ddae(input_dim: int, output_dim: int,
               units: Sequence[int] = (1024, 1024, 1024),
               dropout: float = 0.1,
               mask: bool = False) -> keras.Model:
    """Dense encoder/decoder.

    :param mask: end in a sigmoid instead of a linear layer. A mask lives in
        ``[0, 1]``, so a bounded output cannot produce an impossible answer —
        a linear head would spend part of its capacity learning not to.
    """
    inputs = keras.Input(shape=(input_dim,), name="noisy_frame")
    x = inputs
    for i, width in enumerate(units):
        x = layers.Dense(width, name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        if dropout:
            x = layers.Dropout(dropout, name=f"drop_{i}")(x)
    outputs = layers.Dense(output_dim,
                           activation="sigmoid" if mask else None,
                           name="mask_frame" if mask else "clean_frame")(x)
    return keras.Model(inputs, outputs, name="ddae")
