# -*- coding: utf-8 -*-
"""GTCRN, the grouped temporal convolutional recurrent network of Rong et
al. (2024): causal, 23.7K trained weights, a complex ratio mask out.

X. Rong, T. Sun, X. Zhang, Y. Hu, C. Zhu and J. Lu, "GTCRN: A speech
enhancement model requiring ultralow computational resources", Proc. ICASSP
2024, pp. 971-975. At 16 kHz and 512 points the way through is:

* The noisy spectrum's magnitude, real and imaginary part are three input
  channels. The lowest 65 bins, up to about 2 kHz, are kept as they are;
  the other 192 are folded into 64 bands of equal width on the ERB-rate
  scale by fixed triangular filters, 129 bins in all. Each bin is then
  stacked with its two neighbours, the paper's subband feature extraction,
  for nine channels.
* Two strided convolutions, 1 x 5 in frequency, take the bins to 65 and 33
  at 16 channels, and three grouped temporal convolution blocks follow,
  dilated 1, 2 and 5 frames in time.
* Two grouped dual-path GRNNs: one GRU across the bins of each frame, both
  ways, then one along the frames, forwards only.
* A decoder mirrors the encoder, each layer reading the sum of its input and
  the output of the encoder layer it mirrors, and ends in two channels under
  a tanh: the real and imaginary part of the mask. The transposed filters
  spread it back over the 257 bins, and it multiplies the noisy spectrum.

A grouped temporal convolution block passes half its channels through
untouched. The other half is stacked with its neighbours in frequency and
goes through a pointwise convolution, a depthwise 3 x 3 convolution dilated
in time and padded on the past side only, and a second pointwise
convolution; then the temporal recurrent attention scales each channel by a
sigmoid of what a GRU makes of the channel's energy, frame by frame. The two
halves are interleaved on the way out, so the next block works on channels
this one passed by.

Nothing looks ahead. Every convolution is one frame long in time or padded
on the past side only, the GRUs along time run forwards, and the GRU and the
layer norm across frequency stay inside one frame; BatchNorm scales by its
running statistics at inference. The output for frame *t* is settled once
frame *t* is in.

The paper does not spell out every layer, and what it leaves open is taken
from the authors' ``gtcrn.py`` (MIT licence, its notice in
THIRD_PARTY_NOTICES.txt): the ERB filters, the channel interleave, the
widths of the GRUs and the padding. Two choices here are my own, and neither
changes what the network computes, how many weights it has or the range they
start from. The depthwise convolutions of the decoder's blocks are
transposed convolutions of stride 1 there, which are convolutions with the
kernel flipped, so they are ordinary convolutions here; their pointwise
convolutions stay transposed, because torch starts a transposed 1 x 1
convolution from a different range than an ordinary one. And the ERB filters
are a buffer rather than two frozen linear layers. The repository's count of
48.2K parameters includes those two layers, 2 x 64 x 192 coefficients; the
paper's 23.7K, and :func:`~kudioenhance.models.count_params` here, are the
weights training moves, 23,669 at the default width.

Unlike the other architectures in the package, this one keeps torch's own
initialisation and BatchNorm, since GTCRN was trained with them and its
published scores are what it is compared with. It has no dropout, so
``model.dropout`` is not used, and it always puts out a mask, so
``model.target`` is not read either.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
from torch import nn

__all__ = ["GTCRN", "build_gtcrn", "erb_filters"]

#: Bins kept as they are, and ERB bands the rest are folded into: the
#: paper's 65 and 64 at 512 points.
LOW_BINS = 65
ERB_BANDS = 64
#: Channels through the encoder, the dual-path GRNNs and the decoder.
WIDTH = 16
#: Dilations in time of the three grouped temporal convolution blocks.
DILATIONS = (1, 2, 5)
#: Under the square root of the magnitude, so that a silent bin has a
#: finite gradient; the authors' value.
MAGNITUDE_EPS = 1e-12
#: The dual-path GRNN's layer norm epsilon, the authors' value.
LAYER_NORM_EPS = 1e-8


# ---------------------------------------------------------------- ERB bands

def _hz_to_erb(hz):
    """ERB-rate, after Glasberg & Moore (1990)."""
    return 21.4 * math.log10(1.0 + 0.00437 * hz)


def _erb_to_hz(erb: float) -> float:
    return (10.0 ** (erb / 21.4) - 1.0) / 0.00437


def erb_filters(n_bins: int, sr: int, low_bins: int = LOW_BINS,
                bands: int = ERB_BANDS) -> torch.Tensor:
    """The triangular filters that fold bins ``low_bins`` and up into
    *bands*, ``(bands, n_bins - low_bins)``.

    The band centres lie evenly on the ERB-rate scale, from the first bin
    folded to the last bin, at ``sr / 2``, and are rounded to bins; each
    filter rises from the centre below and falls to the centre above, so the
    filters of neighbouring bands add up to one between their centres. The
    first band has no rising side and the last no falling one. A band sums
    its bins rather than averaging them, as the authors' filters do, and the
    same matrix transposed spreads the bands back over the bins.

    :raises ValueError: when the bins above ``low_bins`` are too few for
        *bands* distinct centres.
    """
    n_fft = 2 * (n_bins - 1)
    if not 0 < low_bins < n_bins - 1 or bands < 2:
        raise ValueError(
            f"cannot fold bins {low_bins} to {n_bins - 1} into {bands} "
            f"bands")
    low, high = _hz_to_erb(low_bins * sr / n_fft), _hz_to_erb(sr / 2)
    centres = [int(round(_erb_to_hz(low + (high - low) * i / (bands - 1))
                         * n_fft / sr)) for i in range(bands)]
    if any(b <= a for a, b in zip(centres, centres[1:])):
        raise ValueError(
            f"{n_bins - low_bins} bins above bin {low_bins} are too few for "
            f"{bands} ERB bands with distinct centres; use fewer bands or a "
            f"longer FFT")
    filters = torch.zeros(bands, n_bins, dtype=torch.float64)
    for band, centre in enumerate(centres):
        if band > 0:
            below = centres[band - 1]
            span = torch.arange(below, centre, dtype=torch.float64)
            filters[band, below:centre] = (span - below) / (centre - below)
        if band < bands - 1:
            above = centres[band + 1]
            span = torch.arange(centre, above, dtype=torch.float64)
            filters[band, centre:above] = (above - span) / (above - centre)
        else:
            filters[band, centre] = 1.0
    return filters[:, low_bins:].float()


class _ErbBands(nn.Module):
    """Fold the high bins into ERB bands and spread them back.

    ``fold`` takes ``(..., n_bins)`` to ``(..., low_bins + bands)``, and
    ``spread`` goes the other way with the same filters transposed; the low
    bins pass through both untouched.
    """

    def __init__(self, n_bins: int, sr: int, low_bins: int, bands: int):
        super().__init__()
        self.low_bins = low_bins
        self.register_buffer("filters",
                             erb_filters(n_bins, sr, low_bins, bands))

    def fold(self, x: torch.Tensor) -> torch.Tensor:
        low, high = x[..., :self.low_bins], x[..., self.low_bins:]
        return torch.cat([low, high @ self.filters.T], dim=-1)

    def spread(self, x: torch.Tensor) -> torch.Tensor:
        low, high = x[..., :self.low_bins], x[..., self.low_bins:]
        return torch.cat([low, high @ self.filters], dim=-1)


def _with_neighbours(x: torch.Tensor) -> torch.Tensor:
    """``(B, C, T, F) -> (B, 3C, T, F)``: each bin beside the bins on either
    side of it, zero beyond the edges; channel ``3c + k`` is channel *c*
    shifted by ``k - 1`` bins. The paper's subband feature extraction."""
    padded = nn.functional.pad(x, (1, 1))
    bins = x.shape[-1]
    stacked = torch.stack([padded[..., k:k + bins] for k in range(3)], dim=2)
    return stacked.flatten(1, 2)


# ---------------------------------------------------------- convolutions

class _FrequencyConv(nn.Module):
    """A 1 x 5 convolution that halves the bins (rounding up), or the
    transposed one that doubles them back to *bins_out*; then BatchNorm and
    a PReLU, or a tanh on the last layer."""

    def __init__(self, width_in: int, width_out: int, *, groups: int = 1,
                 up: bool = False, bins_out: int = 0, last: bool = False):
        super().__init__()
        if up:
            # halving rounds up, so 2n - 1 and 2n bins both went down to n;
            # the extra bin comes back for an even count
            self.conv = nn.ConvTranspose2d(
                width_in, width_out, (1, 5), stride=(1, 2), padding=(0, 2),
                output_padding=(0, 1 - bins_out % 2), groups=groups)
        else:
            self.conv = nn.Conv2d(width_in, width_out, (1, 5), stride=(1, 2),
                                  padding=(0, 2), groups=groups)
        self.norm = nn.BatchNorm2d(width_out)
        self.act = nn.Tanh() if last else nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _TemporalAttention(nn.Module):
    """Scale each channel, frame by frame, by a sigmoid of a GRU's reading of
    its mean energy over frequency: the paper's temporal recurrent
    attention. The GRU is twice as wide as the channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.gru = nn.GRU(channels, 2 * channels, batch_first=True)
        self.out = nn.Linear(2 * channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.square().mean(dim=-1).transpose(1, 2)       # (B, T, C)
        weight = torch.sigmoid(self.out(self.gru(energy)[0]))
        return x * weight.transpose(1, 2).unsqueeze(-1)


class _GroupedTemporalConv(nn.Module):
    """The grouped temporal convolution block; see the module docstring.

    ``(B, C, T, F) -> (B, C, T, F)``. The depthwise convolution is padded
    with ``2 * dilation`` frames of the past, the reach of a three-frame
    kernel, and none of the future. *transposed* makes the two pointwise
    convolutions transposed ones, as in the decoder.
    """

    def __init__(self, channels: int, hidden: int, dilation: int,
                 transposed: bool = False):
        super().__init__()
        half = channels // 2
        pointwise = nn.ConvTranspose2d if transposed else nn.Conv2d
        self.past = 2 * dilation
        self.expand = pointwise(3 * half, hidden, 1)
        self.expand_norm = nn.BatchNorm2d(hidden)
        self.expand_act = nn.PReLU()
        self.depthwise = nn.Conv2d(hidden, hidden, (3, 3),
                                   padding=(0, 1), dilation=(dilation, 1),
                                   groups=hidden)
        self.depthwise_norm = nn.BatchNorm2d(hidden)
        self.depthwise_act = nn.PReLU()
        self.project = pointwise(hidden, half, 1)
        self.project_norm = nn.BatchNorm2d(half)
        self.attention = _TemporalAttention(half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        worked, passed = x.chunk(2, dim=1)
        h = self.expand_act(self.expand_norm(self.expand(
            _with_neighbours(worked))))
        h = nn.functional.pad(h, (0, 0, self.past, 0))
        h = self.depthwise_act(self.depthwise_norm(self.depthwise(h)))
        h = self.attention(self.project_norm(self.project(h)))
        # channel 2i is worked channel i and 2i + 1 the passed one, so the
        # next block's first half is half of each
        return torch.stack([h, passed], dim=2).flatten(1, 2)


# --------------------------------------------------------------- the GRNNs

class _GroupedGRU(nn.Module):
    """Two GRUs side by side, each reading and writing half the features;
    *hidden* is the width of both together, per direction."""

    def __init__(self, features: int, hidden: int, bidirectional: bool):
        super().__init__()
        self.groups = nn.ModuleList(
            nn.GRU(features // 2, hidden // 2, batch_first=True,
                   bidirectional=bidirectional) for _ in range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([gru(part)[0] for gru, part in
                          zip(self.groups, x.chunk(2, dim=-1))], dim=-1)


class _DualPathGRNN(nn.Module):
    """``(B, C, T, F) -> (B, C, T, F)``: a grouped GRU across the bins of each
    frame, both ways, then one along the frames, forwards; each followed by a
    linear layer and a layer norm over the frame's bins and channels, and
    added to what it read."""

    def __init__(self, channels: int, bins: int):
        super().__init__()
        self.across = _GroupedGRU(channels, channels // 2, bidirectional=True)
        self.across_out = nn.Linear(channels, channels)
        self.across_norm = nn.LayerNorm((bins, channels), eps=LAYER_NORM_EPS)
        self.along = _GroupedGRU(channels, channels, bidirectional=False)
        self.along_out = nn.Linear(channels, channels)
        self.along_norm = nn.LayerNorm((bins, channels), eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, bins = x.shape
        x = x.permute(0, 2, 3, 1)                              # (B, T, F, C)
        across = self.across(x.reshape(batch * frames, bins, channels))
        x = x + self.across_norm(self.across_out(across).reshape(x.shape))

        along = x.transpose(1, 2).reshape(batch * bins, frames, channels)
        along = self.along_out(self.along(along))
        along = along.reshape(batch, bins, frames, channels).transpose(1, 2)
        x = x + self.along_norm(along)
        return x.permute(0, 3, 1, 2)


# ------------------------------------------------------------- the network

class GTCRN(nn.Module):
    """GTCRN on a complex spectrum, ``(B, bins, T) -> (B, bins, T)``.

    *n_bins* and *sr* place the ERB bands; the paper's network is 257 bins
    at 16 kHz. *width* is the channel count throughout, 16 in the paper, and
    has to be a multiple of 4: the blocks halve it, and the GRU across
    frequency halves it again for each direction of each of its two groups.
    """

    def __init__(self, n_bins: int, sr: int, width: int = WIDTH,
                 low_bins: int = LOW_BINS, bands: int = ERB_BANDS):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError(
                f"GTCRN's width has to be a positive multiple of 4, got "
                f"{width}")
        self.bands = _ErbBands(n_bins, sr, low_bins, bands)
        folded = low_bins + bands
        half, quarter = (folded + 1) // 2, (folded + 3) // 4
        blocks = [_GroupedTemporalConv(width, width, d) for d in DILATIONS]
        self.encoder = nn.ModuleList([
            _FrequencyConv(9, width),
            _FrequencyConv(width, width, groups=2),
            *blocks])
        self.dual_path = nn.ModuleList(
            _DualPathGRNN(width, quarter) for _ in range(2))
        self.decoder = nn.ModuleList([
            *(_GroupedTemporalConv(width, width, d, transposed=True)
              for d in reversed(DILATIONS)),
            _FrequencyConv(width, width, groups=2, up=True, bins_out=half),
            _FrequencyConv(width, 2, up=True, bins_out=folded, last=True)])

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return spec * self.mask(spec)

    def mask(self, spec: torch.Tensor) -> torch.Tensor:
        """The complex ratio mask for *spec*, ``(B, bins, T)``; its real
        and imaginary part each lie in [-1, 1]."""
        real, imag = spec.real, spec.imag
        magnitude = torch.sqrt(real.square() + imag.square() + MAGNITUDE_EPS)
        x = torch.stack([magnitude, real, imag], dim=1).transpose(2, 3)
        x = _with_neighbours(self.bands.fold(x))           # (B, 9, T, F')

        skips: List[torch.Tensor] = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        for rnn in self.dual_path:
            x = rnn(x)
        for layer, skip in zip(self.decoder, reversed(skips)):
            x = layer(x + skip)

        mask = self.bands.spread(x).transpose(2, 3)          # (B, 2, F, T)
        return torch.complex(mask[:, 0], mask[:, 1])


def build_gtcrn(n_bins: int, sr: int, units: Sequence[int] = (WIDTH,)
                ) -> GTCRN:
    """GTCRN for *n_bins*-bin spectra at *sr*; ``units`` is ``[width]``.

    The default is what :data:`~kudioenhance.models.REGISTRY` builds when
    ``model.units`` is unset.
    """
    if len(units) != 1:
        raise ValueError(
            f"GTCRN takes one width in model.units, its channel count, got "
            f"{list(units)}")
    return GTCRN(n_bins, sr, width=int(units[0]))
