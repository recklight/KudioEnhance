# Changelog

Notable changes to **KudioEnhance** from 1.0 on. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). What changed
before 1.0 is in the README's *Since 0.2.0* and *Since 3.5.0* notes.

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
- The `--pesq-scale` help describes `raw` as the narrowband P.862 score
  before the P.862.1 mapping: raw 2.03 is about 1.66 MOS-LQO.
