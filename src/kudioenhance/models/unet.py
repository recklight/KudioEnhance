# -*- coding: utf-8 -*-
"""2-D U-Net over the spectrogram — time and frequency treated alike.

After Ronneberger, Fischer & Brox (2015), "U-Net: convolutional networks for
biomedical image segmentation", applied to a spectrogram as if it were an
image. Each level of the contracting path is two 3x3 convolutions and then a
halving of both axes; the expanding path doubles them back, and every level
on the way up is handed the feature maps from the same level on the way down,
so the fine time-frequency detail the pooling discarded is there to be
reused. Speech is local structure at several scales at once — a harmonic, a
formant, a syllable — which is the case U-Nets were built for.

It is **not causal**. A 3x3 convolution reaches one frame into the future,
and every halving doubles that reach, so the output for a frame depends on
frames well after it — fifty of them with the default widths. That is the
price of looking at the whole window, and the reason to choose the CRN
instead when the output must not wait.

The time and frequency axes are padded up to a multiple of ``2 ** depth`` and
cropped back afterwards, so every halving is exact, every skip connection
lines up with the level it rejoins, and any spectrogram fits whatever its
length or its number of bins.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional

from kudioenhance.models._defaults import batch_norm_2d, init_like_keras

__all__ = ["UNet", "build_unet"]


def _block(width_in: int, width_out: int) -> nn.Sequential:
    """Two 3x3 convolutions, each normalised and rectified; padded, so the
    feature map keeps its size and only the pooling changes it."""
    return nn.Sequential(OrderedDict([
        ("conv_0", nn.Conv2d(width_in, width_out, 3, padding=1)),
        ("bn_0", batch_norm_2d(width_out)),
        ("relu_0", nn.ReLU()),
        ("conv_1", nn.Conv2d(width_out, width_out, 3, padding=1)),
        ("bn_1", batch_norm_2d(width_out)),
        ("relu_1", nn.ReLU()),
    ]))


class UNet(nn.Module):
    """2-D U-Net, ``(B, T, bins) -> (B, T, bins)`` for any *T* and *bins*.

    *channels* gives one width per level, from the full-resolution level down
    to the bottom one, so ``len(channels) - 1`` halvings. Every layer is a
    convolution, so the number of parameters does not depend on the number of
    bins, and the same weights run on any spectrogram.
    """

    def __init__(self, channels: Sequence[int], dropout: float, mask: bool):
        super().__init__()
        if not channels:
            raise ValueError("a U-Net needs at least one level")
        self.mask = mask
        self.depth = len(channels) - 1
        widths = [1, *channels]
        self.down = nn.ModuleList(_block(widths[i], widths[i + 1])
                                  for i in range(len(channels)))
        self.pool = nn.MaxPool2d(2)
        # where Ronneberger et al. put it: at the end of the contracting
        # path, on the most abstract features and the fewest positions
        self.drop = nn.Dropout(dropout) if dropout else None
        # up[i] brings level i + 1 back to level i's size and width, and
        # merge[i] reads that stacked on level i's own output
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2)
            for i in range(self.depth))
        self.merge = nn.ModuleList(_block(2 * channels[i], channels[i])
                                   for i in range(self.depth))
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames, bins = x.shape[-2:]
        step = 2 ** self.depth
        # pad the end of time and the top of the spectrum. Zero is the mean
        # of the standardised features, so the padding looks like an average
        # frame rather than an edge
        out = functional.pad(x, (0, -bins % step, 0, -frames % step))
        out = out.unsqueeze(1)            # (B, 1, T', bins'): one channel

        skips = []
        for level, block in enumerate(self.down):
            if level:
                out = self.pool(out)
            out = block(out)
            skips.append(out)
        skips.pop()                       # the bottom level rejoins nothing
        if self.drop is not None:
            out = self.drop(out)

        for up, merge, skip in zip(reversed(self.up), reversed(self.merge),
                                   reversed(skips)):
            out = merge(torch.cat([up(out), skip], dim=1))
        out = self.head(out)[:, 0, :frames, :bins]
        return torch.sigmoid(out) if self.mask else out


def build_unet(channels: Sequence[int] = (16, 32, 64, 128),
               dropout: float = 0.1, mask: bool = False) -> UNet:
    """2-D U-Net with one level per entry of *channels*.

    It takes no ``n_bins``: nothing in it depends on the number of bins. The
    default widths are the ones :data:`~kudioenhance.models.REGISTRY` builds
    when ``model.units`` is unset; the reasoning is beside the entry.

    :param mask: sigmoid output for a bounded mask; linear otherwise.
    """
    return init_like_keras(UNet(channels, dropout, mask))
