# KudioEnhance guide

**English** | [繁體中文](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.zh-TW.md)

What the [README](https://github.com/recklight/KudioEnhance/blob/master/README.md) leaves out: what
`evaluate` writes, the models and the two paths in detail, training data in
memory or streamed, the degradations `synth` can add, the run directory,
the layout and what the tests check.

## What `evaluate` writes

`evaluate` enhances every file of the test split into `runs/exp1/enhanced/`,
in 32-bit float so that what gets scored is what the model put out, and hands
the three sides to `kudio.score_dataset`: the enhanced files against the clean
ones, with the noisy files as the baseline. Beside `report.csv` the run gets
the table as `results.md`, `results.tex` and `results.json`, and
`results.bib` with the BibTeX for kudio and every metric in it. A row of the
table is the noisy input's mean, the enhanced mean, and the difference, each
with its 95% BCa interval, then the two-sided sign-flip test of the
difference and its p-value after Holm's adjustment over the rows. The
signature under the table records kudio's version, the rate, each metric's
settings and the versions of the packages behind them, so a number can be
traced back to how it was computed.

Without `--full` it scores segmental SNR, SI-SDR and SNR, which need nothing
but numpy. `--full` adds PESQ, CSIG/CBAK/COVL, STOI and SDR, and leaves out,
with a warning, whatever is not installed. `--metrics pesq,stoi,si_sdr` takes
kudio's names and scores exactly those. `--dnsmos-dir` adds DNSMOS P.835 and
P.808; that needs `pip install "kudio[dnsmos]"` and the ONNX models from
[microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge), which
kudio does not ship. `--jobs 4` scores four files at a time.

The sentences of one speaker are not independent, so where the file names
give the speaker, as VCTK's `p232_001.wav` does, the intervals resample whole
speakers. That needs enough speakers: under 10 an interval falls well short
of 95%, and over 2 it held the true mean half the time in kudio's
simulation. VoiceBank-DEMAND's test set has 2, p232 and p257, so there the
intervals resample sentences, and the line under the table says so: they
cover other sentences from those two speakers, not a third one. `--speakers`
takes a regular expression whose first group is the speaker, or `''` for
none.

## Models in detail

Sequence models are trained on fixed-length windows, but nothing in them
fixes the length of the time axis, so the same weights enhance a whole
utterance in one pass. The two 2-D models also take any `n_fft`: they pad
time and frequency internally as far as their halvings need, and crop back.

**Causal** means the output for a frame depends on that frame and earlier
ones only — the convolutions in `crn` and `gtcrn` are padded on the past
side, their recurrent layers along time run forwards, and tests hold both to
that. It is what a real-time port needs, but this package does not stream:
`Enhancer` enhances a whole file at once, or a long one 30 s at a time on
the waveform path, and the STFT adds a window of latency of its own.

**`model.units`** is the list of layer widths: a dense layer's units, an
LSTM's, a convolution's filters or channels. Left as `null` it takes the
architecture's default, which `kudioenhance models` lists. A width of 1024
is sensible for a dense layer and absurd for a 2-D convolution, each of
whose channels is a whole time-frequency map, so the default belongs to the
architecture. Set it to a list to choose the widths yourself. Training
writes the widths it actually built into the run's `config.yaml`, so a run
keeps loading even if a later release changes a default.

`crn`'s LSTM reads every channel at every bin left after halving the bins
once per layer, so its size grows with `n_fft`: about 1.8M parameters at
`n_fft: 256`, 5.5M at 512 and 19M at 1024. For a large `n_fft`, add a layer
to halve once more — `units: [16, 32, 64, 64, 64, 64]` brings 1024 back to
5.6M. Nothing in `unet` depends on the number of bins.

Adding an architecture is a builder function plus one `REGISTRY` entry —
there is no `if/elif` chain to extend in three places. The entry's
`default_units` is what `units: null` builds; leave it out and the
architecture gets the 1024-wide stack the first three share.

## The log-power path

`ddae`, `blstm`, `conv_ae`, `crn` and `unet` see the log-power spectrogram
kudio computes, standardised with statistics fitted on the training set and
kept as `stats.npz` beside the weights. They learn the clean spectrum or a
mask, and the enhanced magnitude goes back out with the noisy phase. Their
training set is held in memory or streamed
([Training data](#training-data-in-memory-or-streamed)).

### What the network is asked to produce

`model.target` is a separate choice from `model.name`, and every architecture
of this path supports both:

| Target | The network outputs | |
|---|---|---|
| `spectrum` | the clean log-power spectrogram, directly | unbounded output; it has to learn the *level* as well as the shape |
| `irm` | an **ideal ratio mask** in `[0, 1]` | bounded, so a sigmoid cannot produce an impossible answer, and the level comes from the input |

```yaml
model:
  name: ddae
  target: irm
```

The mask is `sqrt(|S|² / (|S|² + |N|²))` — how much of each time-frequency bin
is speech — and applying it is an addition in the log domain, so the network
only ever decides *how much to keep*.

**Measured on this repo's own smoke corpus, three epochs of a small `ddae`:**

| target | SI-SDR |
|---|---|
| `spectrum` | +0.05 → **−7.59 dB** |
| `irm` | +0.05 → **+2.61 dB** |
| an oracle mask | −3.02 → **+14.16 dB** |

An undertrained spectrum model has not learned the level yet, so it outputs
something loud and wrong; an undertrained mask model is already useful, because
the worst it can do is keep or drop what was already there. The oracle row is
the ceiling this target aims at.

That is a tiny model on a tiny corpus, not a benchmark — but it is the reason
mask targets are the usual choice, made visible in about twenty seconds.

`gtcrn` always puts out a complex mask and does not read `model.target`.

## The waveform path: `gtcrn`

`gtcrn` trains differently from the other five. It reads the noisy complex
spectrum and puts out a complex ratio mask, so the phase it gives back is its
own; it learns from waveforms, a fresh crop of every pair each epoch; and its
loss is taken on the spectrum it puts out and on that spectrum's waveform, so
no normalisation statistics are fitted. GTCRN's own STFT and the settings its
paper gives for VoiceBank-DEMAND look like this:

```yaml
audio:
  sr: 16000
  n_fft: 512
  hop_length: 256
  win_length: 512
  window: sqrt_hann         # the square root of Hann, for analysis and synthesis
  pad_mode: reflect         # how torch.stft pads the two ends, GTCRN's choice
  compress: null            # the model sees the spectrum on the linear scale
data:
  prepared: data/vbd16k
  crop_seconds: 2.0         # cut from every pair at a fresh place each epoch
model:
  name: gtcrn               # units: [16], the paper's channel count
train:
  batch_size: 4
  learning_rate: 0.001
  patience: 10              # the rate halves after 5 epochs without progress
  grad_clip: 3.0            # largest gradient norm; null does not clip
  num_workers: 2            # processes that read and crop the audio
```

Without a `loss` section the loss is the one in GTCRN's repository: 30 times
the MSE of the spectrum compressed to `|S|^0.3` with its phase, 70 times the
MSE of the compressed magnitude, and 0.1 times the negative SI-SNR in dB,
taken without removing the mean. The section's keys, `complex_mse`,
`magnitude_mse`, `time_l1`, `neg_si_snr`, `compress` and `zero_mean`, change
the weights, the exponent and the mean.

A run of this path writes `config.yaml` before its first epoch and, after
every epoch, the best weights so far, `resume.pt`, `history.json` and
`progress.json`. If it is stopped, the same command with `--resume` carries
on from the last epoch that finished, and on the CPU the resumed run is the
uninterrupted one to the last bit, as long as torch runs with the same
number of threads. `--resume --epochs N` extends a run that finished. Only
`train.epochs`, `train.patience`, `train.num_workers`, `train.device` and
`train.out_dir` may change on resume; anything else is refused with the keys
named, since it would be a different run. `progress.json` says where a run
stands: the epoch, the loss and each of its terms, the best epoch so far, the
learning rate, the seconds per epoch and the time left. A run that is still
training can already be evaluated or used to denoise; what it gives is its
best epoch so far.

`Enhancer` enhances a file of up to 30 s in one pass. A longer one is cut
into chunks of 30 s that overlap by 1 s, cross-faded where they meet, so
memory stays bounded whatever the length. The last chunk ends where the
file does, so the file's end goes through the STFT exactly as it would
whole.

On VoiceBank-DEMAND with these settings, 2 CPU threads trained it in 14 h
45 min: 61 epochs, the best at 51, about 14.5 min each. Those are the
published weights, and
[Results on VoiceBank-DEMAND](https://github.com/recklight/KudioEnhance/blob/master/README.md#results-on-voicebank-demand) has their
scores.

## Training data: in memory or streamed

By default training holds its features in memory: they are computed once,
and every epoch is a full shuffle of the whole set. That is the right choice
for the tens of hours these models are normally trained on.

A frame-wise model (`ddae`) sees every frame with its ±`context` neighbours,
and stacking those for the whole set up front used to multiply the memory by
`2·context + 1` — eleven times over at `context: 5`. The training set now
keeps each frame once, in a `kudio.ContextFrames`, and stacks the neighbours
of the rows a batch draws as it draws them. The rows are the same, and so is
the run: a test holds the set to the stacked matrix row for row, and fitting
on either gives the same weights to the last bit. Sequence models' windows
were never larger than their features.

What is left is the features themselves, input and target: at 16 kHz with
the default STFT, about 0.46 GB per hour of training audio. When that does
not fit, stream:

```yaml
train:
  streaming: true
  shuffle_buffer: 25000     # a hundred 4-second files of ddae frames
```

Nothing is held then. Every epoch reads the training files again, in a fresh
order, and computes each one's examples as it goes — the same examples, row
for row, as the in-memory set. The price is that pass over the audio, every
epoch: reading, resampling and the STFT, done by the training process between
batches.

- **The normalisation** is fitted by one extra pass before the first epoch, a
  file at a time. It is kudio's formula — the mean, the population standard
  deviation and the same `1e-8` added to it — summed in float64, and an
  in-memory run fits it the same way, a file's spectrogram at a time, so the
  two modes normalise with the same statistics to the bit. (kudio's own
  `Standardizer.fit` sums in float32, and on an hour of frames drifts from
  the true statistics by a few parts in 10⁵; this stays within a part in
  10⁷.)
- **Validation** is streamed too, in manifest order, and its batches fall
  exactly where the in-memory set's would, so it scores what the in-memory
  set would score.
- **The shuffle buffer.** Each file's examples join a pool, and every batch is
  drawn at random from everything in it, for as long as at least
  `shuffle_buffer` examples would be left waiting; when the files run out the
  rest goes out in random order. An example is a frame for `ddae` and an
  `n_frames` window for the sequence models. To mix well, let the buffer span
  a hundred files or so: a hundred times the frames in a typical file, or
  the windows in one. It costs that many inputs and targets of memory: 6 KB
  an example for `ddae` at 257 bins and `context: 2`, 132 KB for a 64-frame
  window.
- **`shuffle_buffer: 0` holds nothing back.** The files still come in a fresh
  order and each file's examples are shuffled, but every batch is drawn from
  the file being read and what earlier draws left over, less than a batch —
  neighbouring frames of one recording, which trains worse. Training warns
  when it is asked for. In memory the setting is not used.
- **It repeats.** The order of the files and every draw from the pool come
  from `train.seed`, so a streamed run repeats to the last digit on the CPU,
  as an in-memory one does. It is a different run from the in-memory one,
  though: the same examples, in a different order.

`kudioenhance.data.build_dataset` is what both modes go through, and
`train.fit` takes what it returns in place of arrays:

```python
from kudioenhance.data import build_dataset
from kudioenhance.train import fit

train_set, std = build_dataset(train_pairs, cfg, sequence=False, fit=True)
history = fit(model, train_set, None, cfg, "runs/exp1/ddae.pt")
```

`build_arrays` still returns the plain stacked matrices, for code that wants
them.

## Rooms

Noise is additive and a room is convolutive, and a real recording has both.
`data.rt60` puts the speech through a room before the noise is mixed in:

```yaml
data:
  snr_db: [-5, 0, 5]
  rt60: [0.3, 0.6, 0.9]     # a fourth axis -- in `inc` mode, x3 the files
  drr_db: 0.0               # +10 is a close mic, 0 is a metre or two
```

The SNR is then measured against the **reverberant** speech, because that is
what is competing with the noise at the microphone.

**By default the clean file stays the target.** A model trained on that is
asked to undo the room as well as the noise, which is a standard setup and a
choice worth making on purpose rather than discovering.

`reverberant_target: true` changes it. `kudio.Synthesizer` then writes the
reverberant-but-clean signal to `TARGETS/` beside each mixture — the same
speech with the room left on — and training asks the model to remove the
noise and **leave** the room, which is a different and easier problem, and
the honest one when the deployment keeps its room.

It also makes the `irm` target correct. The mask is built from
`noisy - reference`, and with a room on the speech the dry clean file is not
what was mixed, so that subtraction hands the reverberation to the *noise*
and the mask asks the model to strip the room's own tail as if it were noise.
Against the reverberant target the subtraction is the noise again.

Each mixture records the room it was built in — `Pair.rt60`, and the file name
carries it too (`speech_white_0dB_rt600ms.wav`), so a manifest says what every
example was asked to survive.

## The channel

The third kind of degradation, and the one that is neither additive nor
convolutive: a codec quantises, a link drops packets, a converter throws bits
away.

```yaml
data:
  channel: telephone        # 300-3400 Hz at 8 kHz through G.711
```

`telephone`, `telephone_a`, `voip` (G.711 with 5% of packets lost in runs of
four, held), `mu_law`, `a_law`. It is applied to the **mixture**, noise and
all, because a wire carries whatever reached the microphone.

**It is not an axis.** Unlike `snr_db` and `rt60` it is constant for the
dataset and does not appear in the file names: a corpus is recorded over a
phone line or it is not. `Pair.channel` records which link, and the target
does not go down it — asking a model to reproduce G.711 is not denoising.

## What a run directory holds

```
runs/exp1/
├── config.yaml      # the experiment, as it was trained
├── ddae.pt          # the weights: <model.name>.pt
├── stats.npz        # the Standardizer the features went through (not for gtcrn)
├── history.json     # the curves (and history.png with matplotlib)
├── resume.pt  progress.json   # gtcrn: the latest epoch's state, and where the run stands
├── train.json  val.json  test.json  manifest.json
├── enhanced/        # after evaluate: the test split, as the model put it out
├── report.csv       # per-file scores
└── results.md  results.tex  results.json  results.bib   # the table
```

The `.pt` file is a plain dictionary — a format tag, the architecture and
target it was trained as, and a `state_dict` — read back with
`torch.load(..., weights_only=True)`. Nothing in it is a pickled class, so
opening a run somebody sent you cannot run their code, and `Enhancer.load`
rebuilds the network from `config.yaml` before pouring the weights in. It is
also the best epoch, always: training checkpoints whenever the monitored loss
improves and puts those weights back into the model when it stops.

Set `train.seed` (or just `data.seed`) and the initial weights, the dropout
masks and the order of the batches all come from it, so on the CPU a run
repeats to the last digit. Some GPU kernels are not deterministic, so there
it repeats closely rather than exactly.

## Layout

```
docs/
├── images/workflow.png  # the README figure
├── workflow.html        # the same diagram, interactive
├── guide.md             # this guide (guide.zh-TW.md in Chinese)
└── models/              # a model card for each published set of weights
src/kudioenhance/
├── config.py        # typed dataclasses, YAML in/out, run paths
├── data.py          # mixing, manifests, splits, training sets
├── vbd.py           # VoiceBank-DEMAND: resampling, the validation split
├── datasets.py      # the training set as torch sees it: held or streamed
├── features.py      # spectrogram / context stacking / windows / Standardizer
├── waveform.py      # random crops of clean/noisy pairs, a seeded DataLoader
├── spectral.py      # complex STFT in torch, power-law compression
├── losses.py        # complex, magnitude, L1 and SI-SNR losses, and their mix
├── models/
│   ├── registry.py      # name -> builder + data layout
│   ├── dense.py         # ddae
│   ├── recurrent.py     # blstm
│   ├── convolutional.py # conv_ae
│   ├── crn.py           # crn
│   ├── unet.py          # unet
│   └── gtcrn.py         # gtcrn
├── train.py         # the training loop: Adam, checkpoints, early stopping
├── waveform_train.py # the waveform path's loop, which resumes where it stopped
├── progress.py      # progress.json and history.json, rewritten every epoch
├── checkpoint.py    # the .pt formats: weights, and a run's training state
├── inference.py     # Enhancer
├── pretrained.py    # published weights: the entries, the cache, the SHA-256 check
├── evaluate.py      # the test split through kudio.score_dataset, the table
├── pipeline.py      # the three stages
└── cli.py           # kudioenhance
```

Model builders return plain `torch.nn.Module`s; `train.py` owns the
optimiser, the loss and the loop. That keeps architectures inspectable without
committing to an optimisation setup.

## What the tests check

The mask tests check the *formula*, not just that it runs: all-speech gives
1, all-noise gives 0, equal power gives 1/√2, and an oracle mask has to
actually improve SI-SDR. Those hold or the target is wrong, whatever the
loss curve says. The architecture tests work the same way: change any frame
after *t* and `crn`'s output up to *t* has to stay the same bit for bit,
while `unet`'s has to move, because its description says it is non-causal.
The same holds `gtcrn` to no look-ahead, and its weights are counted layer
by layer against the paper's 23,669, its multiply-accumulates frame by
frame. And the training sets: the in-memory one has to hand back the stacked
matrix's rows bit for bit, so does a streamed pass in manifest order, and a
shuffled epoch has to be every example exactly once, in the same order again
from the same seed. The two modes have to normalise with the same
statistics, to the bit.
