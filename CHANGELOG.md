# Changelog

Notable changes to **KudioEnhance** from 1.0 on. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). What changed
before 1.0 is under *Before 1.0* at the end.

## [Unreleased]

### Added

- `THIRD_PARTY_NOTICES.txt`, in the wheel and the sdist beside
  `LICENSE.txt`: the MIT notice of the GTCRN authors' repository, which
  `models/gtcrn.py` and the default loss follow.

### Changed

- The README keeps installing, the published weights, the three stages,
  the results, the models and the two paths; the rest is in
  `docs/guide.md` and `docs/guide.zh-TW.md`. The notes on what changed
  before 1.0 are at the end of this file.

## [2.0.0] - 2026-11-16

A second path for models that work on the complex spectrum and learn from
waveforms, GTCRN on it, weights for it trained on VoiceBank-DEMAND on the
CPU, and a test set scored through kudio 4.0 with intervals, tests and a
signature. Configs and run directories from 1.0 load as they are, and the
log-power path trains as it did.

### Added

- `gtcrn`, after Rong et al. (ICASSP 2024): 65 low bins kept and the rest
  folded into 64 ERB bands, grouped temporal convolutions with recurrent
  attention, two grouped dual-path GRNNs and a complex ratio mask; 23,669
  trained weights, no look-ahead. Tests hold its mask to no dependence on a
  later frame and count its weights and multiply-accumulates layer by
  layer. `models.count_macs` counts the multiply-accumulates of a model's
  matrix products and convolutions: 26.3M a second for `gtcrn` at 16 kHz.
- The waveform path. `ModelSpec(waveform=True)` marks an architecture that
  takes and gives a complex spectrum, and `train`, `evaluate`, `denoise`
  and `Enhancer` follow it there.
  - `waveform.py`: a fresh crop of `data.crop_seconds` from every
    clean/noisy pair each epoch, drawn in the main process so the batches
    do not depend on `train.num_workers`.
  - `spectral.py`: the STFT and its inverse in torch, framed as kudio frames
    the log-power path; `audio.pad_mode`, `audio.compress` and the
    `sqrt_hann` window.
  - `losses.py` and the `loss` section: compressed complex and magnitude
    MSE, L1 and SI-SNR, weighted by default as GTCRN's repository weights
    them.
  - `waveform_train.fit_waveform`: validation every epoch, the best epoch
    kept, the learning rate halved on a plateau, early stopping, and
    `train.grad_clip`.
  - `kudioenhance train --resume`: `resume.pt` after every epoch holds the
    weights, the optimiser, the schedule and every random generator, and a
    resumed run on the CPU is the uninterrupted one to the last bit. Only
    `train.epochs`, `patience`, `num_workers`, `device` and `out_dir` may
    change on a resume.
  - `progress.json`, rewritten after every epoch: the losses and their
    terms, the best epoch, the learning rate, the seconds per epoch and
    the time left.
  - Chunked inference: a file longer than 30 s is enhanced in 30 s chunks
    that overlap by 1 s and are cross-faded, so memory stays bounded. The
    last chunk ends where the file does, and the file's end comes out as
    it would whole.
- Published weights. `kudioenhance pretrained` lists them and downloads
  them into the cache, checking each file's SHA-256 before it is kept and
  every time it is loaded. The cache is `$KUDIOENHANCE_CACHE` if set, else
  `kudioenhance` under `$XDG_CACHE_HOME` or `~/.cache`, or `--cache DIR`.
  `Enhancer.pretrained(name)` loads them, and `Enhancer.load` and `denoise`
  take the name where they take a run directory.
  `kudioenhance.pretrained.Pretrained.from_run` makes the entry for a run's
  weights.
- `gtcrn-vbd16k`: GTCRN trained on VoiceBank-DEMAND at 16 kHz, on the CPU
  in 14 h 45 min; WB-PESQ 2.77 [2.73, 2.81], STOI 0.939 and SI-SDR
  18.69 dB on the test set. Its model card is
  `docs/models/gtcrn-vbd16k.md`.
- `kudioenhance prepare` brings VoiceBank-DEMAND from the DataShare zips to
  16 kHz with soxr's HQ filter, as 32-bit float, holds out p226 and p287
  for validation, and writes manifests with each pair's noise and SNR,
  the SHA-256 of every file and `prepare.json`. `data.prepared` points a
  config at it. Preparing twice gives the same bytes.
- `evaluate` writes the enhanced test split to `runs/<name>/enhanced/` and
  scores it with `kudio.score_dataset`: `results.md`, `.tex`, `.json` and
  `.bib` beside `report.csv`, each metric's mean, the noisy mean and the
  difference with 95% BCa intervals, a sign-flip test with Holm's
  adjustment, and kudio's signature. The intervals resample speakers when
  there are 10 or more and sentences otherwise. `--metrics`, `--speakers`,
  `--dnsmos-dir` and `--jobs` are new, and `--full` adds CSIG, CBAK and
  COVL.
- BibTeX for every architecture, in `results.bib` and from
  `models.model_bibtex`, and `CITATION.cff`.
- The README's results on VoiceBank-DEMAND for `unet`, `crn` and `gtcrn`
  beside the noisy input and published light models, and a section on the
  two paths.

### Changed

- kudio 4.0.0 is the floor. Its PESQ takes a pair of two lengths whole, so
  such a pair can score differently from 1.0.
- `report.csv` keeps its columns and takes its means from the table.
- A checkpoint is written beside its destination, synced to disk and moved
  into place, retried while another process holds the file, so loading a
  run that is still training cannot stop it.
- `denoise` given a path that is neither a run nor a published name prints
  why and exits 1 instead of a traceback.
- `train.grad_clip` is refused on the log-power path, in `fit` and before
  `run` mixes anything, rather than ignored.
- The `--pesq-scale` help describes `raw` as the narrowband P.862 score
  before the P.862.1 mapping: raw 2.03 is about 1.66 MOS-LQO.

## Before 1.0

### 0.2.0

The models are **native PyTorch**; 0.1 was TensorFlow and Keras 3. The configs
are unchanged — `train.device` and `train.seed` are new and optional — and so
are the architectures, down to Keras's initialisation, BatchNorm momentum and
LSTM forget-gate bias, so a config tuned on 0.1 keeps its meaning on 0.2 —
though not its exact numbers, since the two frameworks' kernels differ.

The weights are not. A run directory trained by 0.1 holds `ddae.keras` and no
`ddae.pt`, and `Enhancer.load` says so and asks for a retrain rather than
reporting a missing file. The config and the frozen splits in it can be
reused as they are:

```bash
kudioenhance train -c runs/exp1/config.yaml -n exp1
```

`fit` returns the curves as a plain dict instead of a Keras `History`, and
`compile_model` is a deprecated no-op kept so 0.1-era scripts that call
`fit(compile_model(...))` still run. `build_callbacks` is gone, and importing
it fails: `fit` does the checkpointing, the early stopping and the
learning-rate halving itself, so code that built Keras callbacks for
`model.fit` calls `kudioenhance.train.fit` instead.

`train.batch_size` has to be 2 or more for `ddae`, and training says so before
it reads any audio; a `ddae` training set of a single frame is refused too.
Its BatchNorm takes a variance over the batch, and one example has none. 0.1
accepted a batch size of 1, but Keras normalised each unit to a constant
there, so nothing below the last BatchNorm ever trained. The sequence models
still train on a batch of one window.

The normalisation statistics are now summed in float64 in memory as well as
streamed, so the same training set gives a `stats.npz` a little different
from 0.1's: a few parts in 10⁵ on an hour of audio, less on a smaller set.

Training goes through `build_dataset` instead of `build_arrays`, so a
frame-wise model's context is no longer stacked for the whole set in memory;
the run is the same to the last digit. `fit` takes the dataset in place of
`x` with `y=None`, and still takes arrays. `train.streaming` is new, and
`train.shuffle_buffer`, which was in the config but read by nothing, now
sizes the streamed shuffle — and only that.

Two architectures are new, `crn` and `unet`, and `model.units` now defaults
to `null`, meaning the architecture's own widths. A config that sets `units`
builds what it did, and so does one that left it out: `ddae`, `blstm` and
`conv_ae` still default to `[1024, 1024, 1024]`. Code that reads
`cfg.model.units` directly can now find `None` there;
`kudioenhance.models.resolve_units(cfg)` returns a copy with the widths
spelled out.

### On kudio 3.5.0

`Pair`, `save_manifest`, `load_manifest` and `split_pairs` now come from
**kudio** — nothing in them was specific to denoising. The JSON on disk is
unchanged and the names are still importable from `kudioenhance.data`, but
kudio's `split_pairs` is stricter: `val_split + test_split >= 1.0` raises
instead of quietly handing everything to training.
