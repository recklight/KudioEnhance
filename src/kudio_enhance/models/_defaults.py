# -*- coding: utf-8 -*-
"""Keras's defaults, reproduced in torch.

These architectures were tuned under Keras, and the configs people already
have — the learning rate, the patience, the dropout — were chosen against
Keras's initialisation and batch normalisation. torch's defaults differ in
both places, and in ways that change training rather than the last digit of a
score: torch's BatchNorm averages its running statistics ten times faster, so
the statistics inference uses are much noisier, and an LSTM whose forget gate
starts half-closed throws its context away before it has learned what is worth
keeping. Matching them keeps an existing config meaning what it meant.

The 2-D architectures were never Keras models, and use the same defaults
anyway: one config file trains every architecture with one learning rate and
one patience, and that only works if they all start from the same kind of
weights and normalise the same way.
"""
from __future__ import annotations

from torch import nn

__all__ = ["batch_norm", "batch_norm_2d", "init_like_keras"]


def batch_norm(features: int) -> nn.BatchNorm1d:
    """BatchNorm with Keras's running average and epsilon.

    The two libraries count momentum from opposite ends: Keras's 0.99 keeps
    99% of the running value on each batch, which is torch's ``momentum=0.01``.
    """
    return nn.BatchNorm1d(features, momentum=0.01, eps=1e-3)


def batch_norm_2d(channels: int) -> nn.BatchNorm2d:
    """:func:`batch_norm` for feature maps: one mean and variance per channel,
    taken over the batch, time and frequency together."""
    return nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)


def init_like_keras(model: nn.Module) -> nn.Module:
    """Re-initialise *model* the way Keras would have built it, in place.

    Glorot-uniform kernels and zero biases for dense and convolutional
    layers, transposed convolutions included; for an LSTM, Glorot input
    weights, orthogonal recurrent weights, and a forget-gate bias of one, so
    every cell starts out remembering.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
                               nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            hidden = module.hidden_size
            for name, param in module.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(param)
                elif name.startswith("weight_hh"):
                    nn.init.orthogonal_(param)
                else:
                    # torch keeps two biases that are summed; one carries
                    # the forget gate's offset. Gates are ordered i, f, g, o
                    nn.init.zeros_(param)
                    if name.startswith("bias_ih"):
                        nn.init.ones_(param[hidden:2 * hidden])
    return model
