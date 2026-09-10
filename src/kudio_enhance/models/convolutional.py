# -*- coding: utf-8 -*-
"""Dilated 1-D convolutional autoencoder.

Dilation widens the receptive field without striding, so the time axis keeps
its length exactly and variable-length input needs no cropping or padding
bookkeeping. The output is a residual correction to the noisy input, which
starts training from "pass the signal through" rather than from noise.
"""
from __future__ import annotations

from typing import Sequence

import keras
from keras import layers

__all__ = ["build_conv_ae"]


def build_conv_ae(n_bins: int, filters: Sequence[int] = (256, 256, 256),
                  kernel_size: int = 5, dropout: float = 0.1,
                  mask: bool = False) -> keras.Model:
    """Stacked dilated Conv1D.

    :param mask: predict a mask through a sigmoid instead of a residual
        correction. The residual shortcut is dropped in that case — "add this
        to the input" and "keep this fraction of the input" are different
        parameterisations, and combining them would make the sigmoid an
        offset rather than a proportion.
    """
    inputs = keras.Input(shape=(None, n_bins), name="noisy_spec")
    x = inputs
    for i, width in enumerate(filters):
        x = layers.Conv1D(width, kernel_size, padding="same",
                          dilation_rate=2 ** i, name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        if dropout:
            x = layers.Dropout(dropout, name=f"drop_{i}")(x)
    if mask:
        outputs = layers.Conv1D(n_bins, 1, padding="same",
                                activation="sigmoid", name="mask_spec")(x)
    else:
        residual = layers.Conv1D(n_bins, 1, padding="same", name="residual")(x)
        outputs = layers.Add(name="clean_spec")([inputs, residual])
    return keras.Model(inputs, outputs, name="conv_ae")
