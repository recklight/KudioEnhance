# KudioEnhance

**English** | [繁體中文](https://github.com/recklight/KudioEnhance/blob/master/README.zh-TW.md)

[![PyPI](https://img.shields.io/pypi/v/kudioenhance.svg)](https://pypi.org/project/kudioenhance/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/recklight/KudioEnhance/blob/master/LICENSE.txt)
[![kudio](https://img.shields.io/pypi/v/kudio.svg?label=kudio)](https://pypi.org/project/kudio/)

**Train and run learned speech-enhancement (denoising) models**, built on the
[`kudio`](https://pypi.org/project/kudio/) audio toolkit, or denoise with the
weights trained here.

Three stages, one config file:

[![KudioEnhance workflow](https://raw.githubusercontent.com/recklight/KudioEnhance/master/docs/images/workflow.png)](https://raw.githack.com/recklight/KudioEnhance/master/docs/workflow.html?theme=dark)

The figure is a link to an interactive version — [open it online](https://raw.githack.com/recklight/KudioEnhance/master/docs/workflow.html?theme=dark), or open `docs/workflow.html` from a checkout. You can zoom, search for a node, or trace the path between two of them, and most boxes carry a link to the lines of source they were drawn from, at the commit they were checked against.

Each stage reads what the one before it wrote, so you can re-train without
re-mixing and re-evaluate without re-training. `run` does all three in order.

There are two paths through them. Five models work in the **log-power
spectrogram domain** and reuse the noisy phase — the standard magnitude-only
enhancement setup. Feature extraction and reconstruction both go through
`kudio`, so what the model sees is exactly what any other kudio-based tool
produces. The sixth, `gtcrn`, reads the **complex spectrum**, puts out a
complex mask and learns from waveforms. It has 23,669 weights, few enough to
train on a CPU, and its weights trained on VoiceBank-DEMAND are published
([Pretrained weights](#pretrained-weights)). [The two paths](#the-two-paths)
sets them side by side, and
[Results on VoiceBank-DEMAND](#results-on-voicebank-demand) scores both.

This page covers installing, the published weights, running the three
stages and the results. The [guide](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md) has the rest: what
`evaluate` writes, the models and the two paths in detail, rooms and
channels, streaming, the run directory and the layout.

## Install

```bash
pip install kudioenhance
```

The command is `kudioenhance` and the package `import kudioenhance`.

Needs **Python 3.9+** and **PyTorch ≥ 2.4.1**. The models are plain
`torch.nn` modules, trained by a loop in this package — no Keras, no
Lightning. Older torch wheels, and 2.4.0's on Windows, were built against
NumPy 1, which nothing stops pip from installing beside NumPy 2, and there
they fail at the first `torch.from_numpy`.

On Linux the default `torch` wheel bundles CUDA and runs to several
gigabytes. Without an NVIDIA GPU, install the CPU-only build **first** and pip
will keep it when it installs this package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install kudioenhance
```

The Windows and macOS wheels on PyPI carry no CUDA, so they stay small.
Training takes an NVIDIA GPU whenever torch can see one (`device: auto` in the
`train` section), so the same config works on both kinds of machine. Apple's
GPUs are not used: torch can drive them through `mps`, but this package does
not ask it to, and on a Mac training runs on the CPU.

STOI and SDR are optional: `pip install "kudioenhance[eval]"`. **PESQ is a
separate extra**, `pip install "kudio[pesq]"`, because it builds from source
— a C compiler, and the MSVC Build Tools on Windows — so it is deliberately
left out of `[eval]` and `[all]`, which would otherwise stop installing.

PESQ comes on two scales. `evaluate` reports MOS-LQO by default; pass
`--pesq-scale raw` for the narrowband P.862 score before the P.862.1 mapping.
They are different scales, not different roundings: raw 2.03 is about
1.66 MOS-LQO. Papers quote both, so check which one a paper used before
setting your numbers beside it.

## Pretrained weights

To denoise without training anything:

```bash
kudioenhance pretrained                          # what is published, what is cached
kudioenhance denoise gtcrn-vbd16k noisy.wav enhanced.wav
kudioenhance denoise gtcrn-vbd16k noisy/ clean/ --recursive
```

| Name | Model | Trained on | WB-PESQ | Licence | |
|---|---|---|---|---|---|
| `gtcrn-vbd16k` | `gtcrn`, 23,669 weights, 16 kHz, causal | VoiceBank-DEMAND, 26 speakers, on the CPU | 2.77 on its test set | CC BY 4.0 | [model card](https://github.com/recklight/KudioEnhance/blob/master/docs/models/gtcrn-vbd16k.md) |

The first use downloads the file into the cache, `~/.cache/kudioenhance`
(`$KUDIOENHANCE_CACHE` or `$XDG_CACHE_HOME` to move it, or `--cache DIR` on
both commands), and `kudioenhance pretrained gtcrn-vbd16k` downloads it
ahead of time, for a machine that will be offline. The file is the `.pt` its training run wrote,
unchanged, and its SHA-256 is part of the release: a download is checked
before it is kept, and the cached file again every time it is loaded, so a
file that changed on disk is fetched anew rather than used. Wherever a run
directory goes, in `denoise` and in `Enhancer.load`, the name goes too; a
folder of that name wins. In Python:

```python
from kudioenhance import Enhancer

enh = Enhancer.pretrained("gtcrn-vbd16k")
enh.enhance_file("noisy.wav", "enhanced.wav")
```

Only waveform-path weights are published. A log-power model needs its run's
normalisation statistics too, and an entry is one file. The weights are
under CC BY 4.0, like the data they were trained on, which asks to be
credited; the model card has the training run, the scores with kudio's
signature, and what to cite.

## Quick start

```bash
kudioenhance init config.yaml        # starter config
kudioenhance run -c config.yaml -n exp1
```

That mixes the dataset, trains, and prints the before/after scores:

```
si_sdr      -4.812 ->    6.933  (+11.745)
snr         -4.998 ->    5.221  (+10.219)
seg_snr     -6.114 ->    1.870  (+7.984)
report -> runs/exp1/report.csv
table  -> runs/exp1/results.md (.tex, .json, .bib beside it)
```

Then denoise anything with the trained run:

```bash
kudioenhance denoise runs/exp1 noisy.wav enhanced.wav
kudioenhance denoise runs/exp1 noisy/ clean/ --recursive --report
```

The folder form loads the model **once** for the whole tree, mirrors the input's
directory structure, and with `--report` says how far the noise floor moved.

And afterwards, whether it actually learned anything:

```bash
kudioenhance curves -c config.yaml -n exp1
```

```
3 epoch(s), monitoring val_loss
  best   0.12172 at epoch 3
  first  0.28104   last 0.12172
```

A training loop hands back its curves and almost every script drops them,
which makes "did it converge, or stop early because validation was already
climbing?" unanswerable the moment the terminal closes. They are written to
`runs/exp1/history.json` (and `history.png` when matplotlib is around). If the
best epoch was the *first* one the command says so and exits non-zero — that is
what "this did not learn" looks like.

### Stage by stage

```bash
kudioenhance synth    -c config.yaml -n exp1   # mix + freeze the splits
kudioenhance train    -c config.yaml -n exp1   # fit (--epochs 1 to smoke-test)
kudioenhance evaluate -c config.yaml -n exp1   # score (--full adds PESQ, STOI, ...)
```

Each stage reads what the previous one wrote, so you can re-train without
re-mixing, or re-evaluate without re-training.

`evaluate` writes the enhanced test split to `runs/exp1/enhanced/`, the
per-file scores to `report.csv`, and the table, with 95% intervals,
sign-flip tests and kudio's signature, to `results.md`, `.tex`, `.json`
and `.bib`. `--full` adds PESQ, CSIG/CBAK/COVL, STOI and SDR, and
`--dnsmos-dir` DNSMOS; the guide's
[What `evaluate` writes](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#what-evaluate-writes) has the
options and how the intervals resample.

## Results on VoiceBank-DEMAND

The 824 test utterances at 16 kHz, prepared by `kudioenhance prepare` and
scored by `kudioenhance evaluate --full --dnsmos-dir ...` through kudio
4.0.0. Each checkpoint was chosen on two training speakers held out for
validation, never on the test set, and everything was trained on one CPU, an
Intel Core i5-13600K, without a GPU.

| | Weights | WB-PESQ | CSIG | CBAK | COVL | SSNR | STOI | SI-SDR | DNSMOS OVRL |
|---|---|---|---|---|---|---|---|---|---|
| Noisy | – | 1.97 | 3.33 | 2.44 | 2.62 | 1.68 | 0.921 | 8.45 | 2.70 |
| `unet`, `irm` | 483K | 2.65 [2.61, 2.70] | 4.03 | 3.26 | 3.35 | 8.19 | 0.940 | 17.23 | 3.07 |
| `crn`, `irm` | 5.51M | 2.70 [2.66, 2.75] | 4.14 | 3.28 | 3.44 | 7.98 | 0.941 | 16.69 | 3.06 |
| `gtcrn` (`gtcrn-vbd16k`) | 23,669 | **2.77** [2.73, 2.81] | 4.09 | 3.41 | 3.43 | 9.85 | 0.939 | 18.69 | 3.07 |

- The brackets are 95% BCa intervals over sentences; the run's own
  `results.md` has one for every cell, and the
  [model card](https://github.com/recklight/KudioEnhance/blob/master/docs/models/gtcrn-vbd16k.md)
  prints `gtcrn`'s whole table with kudio's signature. In those tables
  every difference from the noisy input is significant after Holm's
  adjustment, except `gtcrn`'s DNSMOS SIG.
- SSNR is the segmental SNR of Loizou's `composite.m`,
  `kudio.composite(...).segsnr`, the column papers on this test set print.
  `evaluate`'s segSNR row is kudio's `segsnr`, which frames differently and
  gives 1.56 for the noisy files.
- The noisy row is the published one, 1.97, 2.44, 1.68, 0.921 and 8.45,
  with CSIG and COVL about 0.01 lower. The 16 kHz files here are 32-bit
  float; CMGAN's per-file logs of that row match 16-bit files with dither,
  whose noise floor moves CSIG and COVL in near-silent frames.
- `unet` and `crn` are on the log-power path with the `irm` target, at
  `n_fft: 512`, `hop_length: 256`, 64-frame windows, batch 16 and up to 40
  epochs with patience 8, streamed. They trained side by side on 4 threads
  each: `crn` stopped after 24 epochs (best 16) in 10.5 h, `unet` ran all
  40 (best 39, still improving) in 16.3 h.
- `gtcrn` trained with the settings in
  [The waveform path](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#the-waveform-path-gtcrn) on 2 threads: it stopped
  after 61 epochs (best 51) in 14 h 45 min.

Published results on the same test set for light models, as the GTCRN paper
and its repository print them:

| | Weights | WB-PESQ | STOI | SI-SNR |
|---|---|---|---|---|
| RNNoise | 0.06M | 2.29 | | |
| PercepNet | 8.00M | 2.73 | | |
| DeepFilterNet | 1.80M | 2.81 | | |
| S-DCCRN | 2.34M | 2.84 | | |
| GTCRN | 23.7K | 2.87 | 0.940 | 18.83 |

The trained `gtcrn` comes within 0.001 of the paper's STOI and 0.14 dB of
its SI-SNR, but 0.10 short of its WB-PESQ, which is outside the interval.
The paper gives no crop length or epoch count for this dataset; these
weights saw 2 s crops and stopped at epoch 61, and the GTCRN author has
suggested crops of about 8 s and 200 epochs elsewhere. Whether that closes
the gap has not been tried. The GTCRN repository lists 48.2K weights; that
count includes the two fixed ERB matrices, which here are buffers. Larger
models go further: CMGAN reaches 3.41, and MP-SENet, with 2.05M weights,
3.50.

To repeat a row, prepare the corpus as in [VoiceBank-DEMAND](#voicebank-demand),
then train and score:

```bash
kudioenhance train    -c gtcrn.yaml -n vbd-gtcrn
kudioenhance evaluate -c gtcrn.yaml -n vbd-gtcrn --full --jobs 8 --dnsmos-dir models/dnsmos
```

## Configuration

```yaml
audio:                      # STFT geometry, shared by features and reconstruction
  sr: 16000
  n_fft: 512
  hop_length: 256
  win_length: 512

data:
  clean_dir: data/clean
  noise_dir: data/noise
  mixed_dir: runs/mixed
  snr_db: [-5, 0, 5]
  rt60: []                  # reverberation times in seconds; [] = no rooms
  drr_db: 0.0               # how close the microphone is, in dB
  reverberant_target: false # train to remove the noise and leave the room
  channel: null             # telephone / voip / mu_law / a_law, or null
  mode: regular             # 'regular' = one mix per clean file; 'inc' = every combination
  seed: 17                  # makes the mixture and the splits reproducible
  val_split: 0.1
  test_split: 0.1

model:
  name: ddae                # ddae | blstm | conv_ae | crn | unet | gtcrn
  target: spectrum          # spectrum | irm  -- see the guide's log-power path
  context: 2                # frame-wise models: ±frames stacked into the input
  n_frames: 64              # sequence models: frames per training window
  units: null               # layer widths; null = the architecture's default
  dropout: 0.1

train:
  epochs: 50
  batch_size: 256           # ddae needs 2 or more; the sequence models train on 1
  learning_rate: 0.001
  patience: 8
  out_dir: runs
  device: auto              # auto | cpu | cuda | cuda:1
  seed: null                # weights, dropout and batch order; null = data.seed
  streaming: false          # true: read the features from disk every epoch
  shuffle_buffer: 0         # streaming only: examples held back for mixing
```

Every key is a typed dataclass field. A typo fails at load time and names the
offending key — it does not silently fall back to a default three stages later.
`gtcrn` has a few keys of its own and a `loss` section, listed under
[The waveform path](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#the-waveform-path-gtcrn).

## Models

```bash
kudioenhance models
```

```
blstm       stacked bidirectional LSTM (Erdogan et al. 2015)                                  units: [1024, 1024, 1024]
conv_ae     dilated Conv1D autoencoder, residual (after Tan et al. 2019)                      units: [1024, 1024, 1024]
crn         causal conv-recurrent network (Tan & Wang 2018)                                   units: [16, 32, 64, 64, 64]
ddae        dense denoising autoencoder, frame-wise (Xu et al. 2015)                          units: [1024, 1024, 1024]
gtcrn       grouped temporal conv-recurrent network, causal, complex mask (Rong et al. 2024)  units: [16]
unet        2-D U-Net over time and frequency, non-causal (Ronneberger et al. 2015)           units: [16, 32, 64, 128]
```

| Name | Input layout | Causal | What it is, and when to choose it |
|---|---|---|---|
| `ddae` | `(N, bins × (2·context+1))` | no — sees ±`context` frames | Dense network from a window of frames to the one in the middle, as in Xu et al., *A regression approach to speech enhancement based on deep neural networks* (IEEE/ACM TASLP 2015). The name is from Lu et al., *Speech enhancement based on deep denoising autoencoder* (Interspeech 2013). Frame-wise, fast, a solid baseline. |
| `blstm` | `(N, n_frames, bins)` | no | Stacked bidirectional LSTM, as in Erdogan et al., *Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks* (ICASSP 2015). Temporal context in both directions. |
| `conv_ae` | `(N, n_frames, bins)` | no | Dilated Conv1D autoencoder with a residual connection — starts from "pass the signal through" rather than from noise. The convolutions along time are a cut-down form of Tan, Chen & Wang, *Gated residual networks with dilated convolutions for monaural speech enhancement* (IEEE/ACM TASLP 2019), without its 2-D front end or its gates; the doubling dilation is Yu & Koltun's (ICLR 2016). |
| `crn` | `(N, n_frames, bins)` | **yes** | Convolutional recurrent network after Tan & Wang, *A convolutional recurrent neural network for real-time speech enhancement* (Interspeech 2018): a strided 2-D convolutional encoder over time and frequency, an LSTM over time, and a mirrored decoder that takes each encoder layer's output back in. Choose it when the output must not wait for what comes next. |
| `unet` | `(N, n_frames, bins)` | no — about 50 frames ahead | 2-D U-Net over time and frequency, after Ronneberger et al., *U-Net: Convolutional networks for biomedical image segmentation* (MICCAI 2015), on a spectrogram with a mask out as in Jansson et al., *Singing voice separation with deep U-Net convolutional networks* (ISMIR 2017): convolution blocks, halving both axes on the way down, skip connections on the way back up. Choose it for offline enhancement of whole files; at about half a million parameters it is the smallest log-power model here at the default widths. |
| `gtcrn` | complex spectrum `(N, bins, frames)` | **yes** | GTCRN after Rong et al., *GTCRN: A speech enhancement model requiring ultralow computational resources* (ICASSP 2024): low bins kept and high ones folded into ERB bands, grouped temporal convolutions with a recurrent attention, a grouped dual-path GRU network, and a complex ratio mask. 23,669 weights and 26.3M multiply-accumulates per second of 16 kHz audio in its matrix products and convolutions (33.0M the way GTCRN's repository counts, with biases, norms, activations and the GRUs' gate arithmetic). Choose it when compute is scarce, or when the phase has to be estimated too. |

The sequence models train on fixed windows and enhance a whole utterance
in one pass, `crn` and `gtcrn` never look ahead, and `model.units` sets
the layer widths. The guide's
[Models in detail](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#models-in-detail) has the rest,
`crn`'s size against `n_fft` and how to add an architecture.

## The two paths

| | Log-power path | Waveform path |
|---|---|---|
| Models | `ddae`, `blstm`, `conv_ae`, `crn`, `unet` | `gtcrn` |
| The network sees | the log-power spectrogram from kudio's STFT, standardised | the complex spectrum from `torch.stft` |
| It puts out | a log-power spectrum, or a mask in [0, 1] (`model.target`) | a complex ratio mask |
| Phase | the noisy input's | its own |
| Trained on | every frame of every file, in memory or streamed | a fresh crop of every pair each epoch (`data.crop_seconds`) |
| Loss | MSE on the target | the `loss` section: compressed complex and magnitude MSE, SI-SNR |
| Statistics | `stats.npz`, fitted on the training set | none |
| A stopped run | trains again from the start | `--resume`, to the last bit on the CPU |
| A long file | in one pass | in 30 s chunks overlapping by 1 s |
| Published weights | – | `gtcrn-vbd16k` |

Both read the same `audio` section, framed the same way: by default the
complex values `torch.stft` gives on the waveform path agree with kudio's to
a millionth of the peak. `train`, `evaluate`, `denoise` and `Enhancer` take
either; the registry entry of the architecture says which path it is on.

The guide goes through each path: the `spectrum` and `irm` targets of
[the log-power path](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#the-log-power-path), and
[the waveform path](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#the-waveform-path-gtcrn)'s settings,
its loss, `--resume`, `progress.json` and chunked inference.

## Datasets

Point `clean_dir` and `noise_dir` at any two folders. Public sets that work
well out of the box:

- **[VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791)** — the standard speech-enhancement benchmark. It comes mixed already, so it skips the synth stage; see below.
- **[LibriSpeech](https://www.openslr.org/12/)** + **[DEMAND](https://zenodo.org/record/1227121)** or **[MUSAN](https://www.openslr.org/17/)** — mix them yourself at the SNRs you care about.
- **[ESC-50](https://github.com/karolpiczak/ESC-50)** — environmental noise, handy for non-speech targets.

Set a `seed` and the mixture is reproducible: the same noise offsets, the same
pairings, the same splits.

`synth` can also put the speech through a room (`data.rt60`,
`data.drr_db`) and the mixture through a telephone or VoIP link
(`data.channel`). The guide's [Rooms](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#rooms) and
[The channel](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#the-channel) say what each does to the
target. A training set too large for memory can be
[streamed](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#training-data-in-memory-or-streamed).

### VoiceBank-DEMAND

Download the 28-speaker set from the DataShare record by hand (the clean and
noisy zips for the training and test sets, and `logfiles.zip`), then:

```bash
kudioenhance prepare path/to/zips data/vbd16k
```

`prepare` reads the zips, or the folders they unpack to, and never downloads
anything. It resamples the 48 kHz files to 16 kHz (`--sr` for another rate)
with soxr's HQ filter and writes them as 32-bit float: the noisy files come
close to full scale, and in 16 bits the resampler's overshoot would clip, so
`noisy - clean` would stop being the noise the `irm` target is built from.
The corpus has no validation set, and picking a checkpoint on the test set
would make the test score meaningless, so two training speakers, p226 and
p287, are held out whole for validation (`--val-speakers` to choose others).
They are the two SpeechBrain's VoiceBank recipe holds out.

Into `data/vbd16k` go `train/`, `val/` and `test/`, the three manifests with
each pair's noise and SNR from the corpus's log files, `utterances.csv` with
the SHA-256 of every file, `speakers.csv`, and `prepare.json`, which records
the source zips with their SHA-256, the resampler and the library versions.
Then point the config at the folder and skip `synth`:

```yaml
data:
  prepared: data/vbd16k
```

## Python API

```python
from kudioenhance import Config, Enhancer, pipeline

cfg = Config.from_yaml("config.yaml")
summary = pipeline.run(cfg, "exp1")          # synth + train + evaluate

enh = Enhancer.load("runs/exp1")             # model + stats + config together
clean = enh.enhance_file("noisy.wav", "enhanced.wav")
result = enh.enhance_folder("noisy/", "clean/", report=True)
print(result, result.mean_reduction_db())

enh = Enhancer.pretrained("gtcrn-vbd16k")    # published weights, from the cache
```

The model is loaded once for the whole tree, and the result is a
`kudio.EnhanceFolderResult` — the same shape `kudio.convert_folder` returns, so
"I processed a folder" has one answer whatever did the processing.

A trained run also drops straight into kudio's comparison table, beside the
statistical methods:

```python
import kudio
kudio.compare_enhancers(y, sr, ['logmmse', ('exp1', enh.enhance)],
                        reference=clean)
```

`Enhancer.load` reads the model, its normalisation statistics **and** the STFT
geometry that produced them from the same run directory, so inference can never
silently disagree with training.

The guide lists [what a run directory holds](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#what-a-run-directory-holds),
and why a `.pt` file somebody sent you is safe to open.

## Tests

From a checkout, with the development extras:

```bash
git clone https://github.com/recklight/KudioEnhance.git
cd KudioEnhance
pip install -e ".[dev]"
pytest -q
```

509 tests. The suite builds a tiny synthetic corpus in a temp directory and runs
the whole pipeline — mix, train one epoch, enhance, score — in seconds. Tests
that need PyTorch ask for it by name and skip cleanly when it is not
installed; the other 207 still run.

The guide's [What the tests check](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#what-the-tests-check)
says what they hold the code to.

## Known limits

- A streamed epoch of the log-power path reads and transforms its files in
  the training process, one at a time, between batches. It has no loader
  worker processes, so a GPU can end up waiting on the CPU; an in-memory run
  pays that cost once. The waveform path reads its crops in
  `train.num_workers` processes.
- The log-power models enhance the magnitude only, and the phase is the noisy
  input's. `gtcrn` estimates a complex mask; time-domain models are not here.
- The log-power sequence models enhance a whole file in one pass. For `crn`
  and `unet` that means a feature map per channel per bin for every frame:
  about half a gigabyte per minute of 16 kHz audio at the default widths, so
  split long recordings before enhancing them.
- `gtcrn`'s GRUs start every 30 s chunk from rest, and 1 s of overlap does
  not hide that. With the published weights the chunked output differs from
  the whole file's by about 3% of the file's RMS for two seconds after a
  seam, about 1% from two to five seconds on, and 0.3–0.6% from six to ten.
  The scores hardly move: on four files of 96–98 s joined from test
  sentences, the chunked output's WB-PESQ was within 0.005 of the whole
  file's and its SI-SDR within 0.05 dB, higher on some files and lower on
  others. A longer overlap only moves the same curve later.
- The `irm` target needs the noise, and takes it as `noisy - reference` in the
  time domain. That is exact for anything `kudio.Synthesizer` mixed, and wrong
  for a corpus whose noisy and clean files are separate recordings rather than
  a sum. In a reverberant dataset the reference has to be the reverberant
  target, or the room's tail is handed to the noise -- see
  `reverberant_target` in the guide's [Rooms](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md#rooms).

## Changes

[CHANGELOG.md](https://github.com/recklight/KudioEnhance/blob/master/CHANGELOG.md) lists what each
release changed, and what a config, a run directory or a script from an
earlier release needs.

## Citing KudioEnhance

If you use KudioEnhance in your research, please cite it. `CITATION.cff` holds
what GitHub shows under "Cite this repository". `evaluate` writes
`results.bib` beside the table: KudioEnhance's entry, the works behind the
architecture you trained, then kudio and every metric in the table, each
once, under the keys the table lists. `kudioenhance.models.model_bibtex("crn")`
gives an architecture's entries on their own.

Work that uses the published weights should also credit the data they were
trained on, which its CC BY 4.0 licence asks for; each model card has the
BibTeX for the architecture and the dataset.

## Related

- **[kudio](https://pypi.org/project/kudio/)** — the audio toolkit underneath.
- **[KudioStudio](https://pypi.org/project/kudiostudio/)** — desktop workbench
  for the same toolkit. With this package installed alongside it
  (`pip install "kudiostudio[enhance]"`), its Enhance panel loads the run
  directories produced here, so you can listen to a model's output and score
  it interactively instead of only reading the CSV.

## Licence

MIT — see [LICENSE.txt](https://github.com/recklight/KudioEnhance/blob/master/LICENSE.txt).

`gtcrn` and its default loss follow the GTCRN authors' MIT-licensed code,
and their notice is in
[THIRD_PARTY_NOTICES.txt](https://github.com/recklight/KudioEnhance/blob/master/THIRD_PARTY_NOTICES.txt).
