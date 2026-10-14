# -*- coding: utf-8 -*-
"""Convolutional recurrent network — causal, after Tan & Wang (2018).

K. Tan and D. Wang, "A convolutional recurrent neural network for real-time
speech enhancement", Interspeech 2018. An encoder of strided 2-D convolutions
squeezes the frequency axis, an LSTM carries context forward in time over
what is left, and a mirrored decoder of transposed convolutions rebuilds the
spectrum. Each decoder layer takes the output of the encoder layer it mirrors
alongside its own input, so the detail the striding threw away comes back.

Everything in it looks only at the present and the past. The convolutions are
two frames long in time and padded on the past side only, the LSTM runs
forwards, and each transposed convolution drops the frame it adds on the
future side. The output for frame *t* is therefore settled as soon as frame
*t* arrives, which is what makes the architecture usable in real time.
BatchNorm keeps that at inference, where it scales by its running statistics;
in training it takes the statistics of the whole batch, which does no harm
because training sees whole windows anyway.

Two departures from the paper, both so that any spectrum fits. The paper's
convolutions are unpadded in frequency, which suits its 161 bins but runs a
short spectrum out of bins after a few layers; here every layer pads one bin
either side, so it halves the bins rounding up and never reaches zero. And
the output is linear or a sigmoid rather than the paper's softplus, because
what is predicted here is a standardised log spectrum or a mask, not a
magnitude.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Sequence

import torch
from torch import nn

from kudio_enhance.models._defaults import batch_norm_2d, init_like_keras

__all__ = ["CRN", "build_crn"]

#: (time, frequency): two frames — this one and the last — by three bins.
KERNEL = (2, 3)
#: Halve the bins at every layer; never skip a frame.
STRIDE = (1, 2)
#: The paper's bottleneck: two LSTMs, each as wide as the features it reads.
LSTM_LAYERS = 2


class _DropFutureFrame(nn.Module):
    """Trim the frame a transposed convolution adds on the future side.

    A kernel two frames long, run backwards with stride one, turns *T* frames
    into *T* + 1, and the last of them is the only one that a future input
    frame would reach. Dropping it keeps both the length and the causality.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-1]


def _encoder_layer(width_in: int, width_out: int) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        # (bins below, bins above, frames before, frames after): one frame
        # of the past and nothing of the future is what makes it causal
        ("pad", nn.ZeroPad2d((1, 1, 1, 0))),
        ("conv", nn.Conv2d(width_in, width_out, KERNEL, stride=STRIDE)),
        ("bn", batch_norm_2d(width_out)),
        ("elu", nn.ELU()),
    ]))


def _decoder_layer(width_in: int, width_out: int, bins_out: int,
                   last: bool) -> nn.Sequential:
    """Undo one encoder layer: back to *width_out* channels and *bins_out*
    bins, which are that layer's input channels and bins.

    Halving rounded up maps both 2n - 1 and 2n bins to n, so the way back has
    to be told which it was; ``output_padding`` adds the one extra bin when
    the encoder's input was even.
    """
    layers: "OrderedDict[str, nn.Module]" = OrderedDict()
    layers["deconv"] = nn.ConvTranspose2d(
        width_in, width_out, KERNEL, stride=STRIDE, padding=(0, 1),
        output_padding=(0, 1 - bins_out % 2))
    layers["trim"] = _DropFutureFrame()
    if not last:
        # the paper's last layer is the output itself: no normalisation or
        # nonlinearity between it and the head
        layers["bn"] = batch_norm_2d(width_out)
        layers["elu"] = nn.ELU()
    return nn.Sequential(layers)


class CRN(nn.Module):
    """Causal convolutional recurrent network, ``(B, T, bins) -> (B, T, bins)``.

    *channels* are the encoder's widths, one per layer; the decoder mirrors
    them. The LSTMs read every remaining channel at every remaining bin as one
    vector per frame, so their width is ``channels[-1]`` times the bins left
    after ``len(channels)`` halvings — and their size, which is most of the
    network's, grows with the square of it.
    """

    def __init__(self, n_bins: int, channels: Sequence[int], dropout: float,
                 mask: bool):
        super().__init__()
        if not channels:
            raise ValueError("a CRN needs at least one encoder layer")
        self.mask = mask
        widths: List[int] = [1, *channels]
        bins: List[int] = [n_bins]
        for _ in channels:
            bins.append((bins[-1] + 1) // 2)      # halved, rounding up

        self.encoder = nn.ModuleList(
            _encoder_layer(widths[i], widths[i + 1])
            for i in range(len(channels)))
        features = channels[-1] * bins[-1]
        self.lstm = nn.LSTM(features, features, num_layers=LSTM_LAYERS,
                            batch_first=True, dropout=dropout)
        # nn.LSTM drops out between its layers only; this covers its output
        self.drop = nn.Dropout(dropout) if dropout else None
        # decoder[i] mirrors encoder[i], and reads its own input stacked on
        # that encoder layer's output: twice the channels the encoder made
        self.decoder = nn.ModuleList(
            _decoder_layer(2 * widths[i + 1], widths[i], bins[i], last=i == 0)
            for i in range(len(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.unsqueeze(1)              # (B, 1, T, bins): one input channel
        skips = []
        for layer in self.encoder:
            out = layer(out)
            skips.append(out)

        batch, width, frames, bins = out.shape
        sequence = out.permute(0, 2, 1, 3).reshape(batch, frames, width * bins)
        sequence, _ = self.lstm(sequence)
        if self.drop is not None:
            sequence = self.drop(sequence)
        out = sequence.reshape(batch, frames, width, bins).permute(0, 2, 1, 3)

        for layer, skip in zip(reversed(self.decoder), reversed(skips)):
            out = layer(torch.cat([out, skip], dim=1))
        out = out.squeeze(1)
        return torch.sigmoid(out) if self.mask else out


def build_crn(n_bins: int, channels: Sequence[int] = (16, 32, 64, 64, 64),
              dropout: float = 0.1, mask: bool = False) -> CRN:
    """Causal CRN for *n_bins*-bin spectra of any length.

    The default widths are the ones :data:`~kudio_enhance.models.REGISTRY`
    builds when ``model.units`` is unset; the reasoning is beside the entry.

    :param mask: sigmoid output for a bounded mask; linear otherwise.
    """
    return init_like_keras(CRN(n_bins, channels, dropout, mask))
