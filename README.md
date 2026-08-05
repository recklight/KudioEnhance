# kudio-enhance

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-3-d00000.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![kudio](https://img.shields.io/pypi/v/kudio.svg?label=kudio)](https://pypi.org/project/kudio/)

**Train and run learned speech-enhancement (denoising) models**, built on the
[`kudio`](https://pypi.org/project/kudio/) audio toolkit.

Three stages, one config file:

```
synthesize  ──►  train  ──►  evaluate
 clean × noise    noisy spec      SI-SDR / PESQ / STOI,
 at your SNRs     → clean spec    before vs. after
```

Models work in the **log-power spectrogram domain** and reuse the noisy phase —
the standard magnitude-only enhancement setup. Feature extraction and
reconstruction both go through `kudio`, so what the model sees is exactly what
any other kudio-based tool produces.

## Install

```bash
pip install kudio-enhance
```

From a checkout:

```bash
pip install -e ".[dev]"
```

Needs **Python 3.9+** and **TensorFlow ≥ 2.16** (Keras 3 — no legacy shim).
PESQ/STOI/SDR metrics are optional: `pip install "kudio-enhance[eval]"`.

## Quick start

```bash
kudio-enhance init config.yaml        # starter config
kudio-enhance run -c config.yaml -n exp1
```

That mixes the dataset, trains, and prints the before/after scores:

```
si_sdr      -4.812 ->    6.933  (+11.745)
snr         -4.998 ->    5.221  (+10.219)
seg_snr     -6.114 ->    1.870  (+7.984)
report -> runs/exp1/report.csv
```

Then denoise anything with the trained run:

```bash
kudio-enhance denoise runs/exp1 noisy.wav enhanced.wav
```

### Stage by stage

```bash
kudio-enhance synth    -c config.yaml -n exp1   # mix + freeze the splits
kudio-enhance train    -c config.yaml -n exp1   # fit (--epochs 1 to smoke-test)
kudio-enhance evaluate -c config.yaml -n exp1   # score (--full adds PESQ/STOI)
```

Each stage reads what the previous one wrote, so you can re-train without
re-mixing, or re-evaluate without re-training.

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
  mode: regular             # 'regular' = one mix per clean file; 'inc' = every combination
  seed: 17                  # makes the mixture and the splits reproducible
  val_split: 0.1
  test_split: 0.1

model:
  name: ddae                # ddae | blstm | conv_ae
  context: 2                # frame-wise models: ±frames stacked into the input
  n_frames: 64              # sequence models: frames per training window
  units: [1024, 1024, 1024]
  dropout: 0.1

train:
  epochs: 50
  batch_size: 256
  learning_rate: 0.001
  patience: 8
  out_dir: runs
```

Every key is a typed dataclass field. A typo fails at load time and names the
offending key — it does not silently fall back to a default three stages later.

## Models

```bash
kudio-enhance models
```

| Name | Input layout | What it is |
|---|---|---|
| `ddae` | `(N, bins × (2·context+1))` | Dense denoising autoencoder. Frame-wise, fast, a solid baseline. |
| `blstm` | `(N, n_frames, bins)` | Stacked bidirectional LSTM. Temporal context in both directions. |
| `conv_ae` | `(N, n_frames, bins)` | Dilated Conv1D autoencoder with a residual connection — starts from "pass the signal through" rather than from noise. |

Sequence models are trained on fixed-length windows but built with an
undefined time axis, so the same weights enhance a whole utterance in one pass.

Adding an architecture is a builder function plus one `REGISTRY` entry —
there is no `if/elif` chain to extend in three places.

## Datasets

Point `clean_dir` and `noise_dir` at any two folders. Public sets that work
well out of the box:

- **[VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791)** — the standard speech-enhancement benchmark (comes pre-mixed; use its clean/noisy folders directly and skip the synth stage).
- **[LibriSpeech](https://www.openslr.org/12/)** + **[DEMAND](https://zenodo.org/record/1227121)** or **[MUSAN](https://www.openslr.org/17/)** — mix them yourself at the SNRs you care about.
- **[ESC-50](https://github.com/karolpiczak/ESC-50)** — environmental noise, handy for non-speech targets.

Set a `seed` and the mixture is reproducible: the same noise offsets, the same
pairings, the same splits.

## Python API

```python
from kudio_enhance import Config, pipeline
from kudio_enhance.inference import Enhancer

cfg = Config.from_yaml("config.yaml")
summary = pipeline.run(cfg, "exp1")          # synth + train + evaluate

enh = Enhancer.load("runs/exp1")             # model + stats + config together
clean = enh.enhance_file("noisy.wav", "enhanced.wav")
```

`Enhancer.load` reads the model, its normalisation statistics **and** the STFT
geometry that produced them from the same run directory, so inference can never
silently disagree with training.

## Layout

```
src/kudio_enhance/
├── config.py        # typed dataclasses, YAML in/out, run paths
├── data.py          # mixing, manifests, splits, feature matrices
├── features.py      # spectrogram / context stacking / windows / Standardizer
├── models/
│   ├── registry.py      # name -> builder + data layout
│   ├── dense.py         # ddae
│   ├── recurrent.py     # blstm
│   └── convolutional.py # conv_ae
├── train.py         # compile, callbacks, fit
├── inference.py     # Enhancer
├── evaluate.py      # per-file scoring + CSV report
├── pipeline.py      # the three stages
└── cli.py           # kudio-enhance
```

Model builders return **uncompiled** models; `train.py` owns the optimiser,
loss and callbacks. That keeps architectures inspectable without committing to
an optimisation setup.

## Tests

```bash
pytest -q
```

The suite builds a tiny synthetic corpus in a temp directory and runs the whole
pipeline — mix, train one epoch, enhance, score — in seconds. Tests that need
TensorFlow skip cleanly when it is not installed.

## Known limits

- `build_arrays` holds the training set in memory. Fine for the tens of hours
  these models are normally trained on; a `tf.data` pipeline is the answer for
  more.
- Magnitude-only enhancement: the phase is the noisy input's. Phase-aware or
  time-domain models are a different architecture family.

## Related

- **[kudio](https://pypi.org/project/kudio/)** — the audio toolkit underneath.
- **[KudioStudio](https://github.com/recklight/KudioStudio)** — desktop workbench
  for the same toolkit. Its Enhance panel loads the run directories this package
  produces (`pip install "kudiostudio[enhance]"`), so you can listen to a model's
  output and score it interactively instead of only reading the CSV.

## Licence

MIT — see [LICENSE.txt](LICENSE.txt).
