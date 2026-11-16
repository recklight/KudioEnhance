# Model card: `gtcrn-vbd16k`

GTCRN trained on VoiceBank-DEMAND at 16 kHz, on the CPU.

| | |
|---|---|
| Architecture | `gtcrn`, after Rong et al., *GTCRN: A speech enhancement model requiring ultralow computational resources* (ICASSP 2024) |
| Weights | 23,669 trained; the file also holds the ERB filters and the BatchNorm statistics |
| Compute | 26.3M weight multiply-accumulates per second of audio (33.0M counted the way the GTCRN repository counts) |
| Input and output | 16 kHz mono; the complex spectrum of a 512-point sqrt-Hann STFT with a 256-sample hop, enhanced by a complex ratio mask |
| Causal | yes: no frame looks ahead; the STFT adds 32 ms of its own |
| File | `gtcrn-vbd16k.pt`, 226,275 bytes |
| SHA-256 | `357f4c9da4fd09895bd28e3933aed729dc84869104343b335d2ec990448bee99` |
| Download | {URL} |
| Licence | the weights CC BY 4.0, the code MIT |

## Using it

```bash
kudioenhance pretrained gtcrn-vbd16k            # download, check the SHA-256
kudioenhance denoise gtcrn-vbd16k noisy.wav enhanced.wav
```

```python
from kudioenhance import Enhancer

enh = Enhancer.pretrained("gtcrn-vbd16k")
enhanced = enh.enhance_file("noisy.wav", "enhanced.wav")
```

The file is the `gtcrn.pt` the training run wrote, unchanged: a plain
dictionary of tensors read with `torch.load(..., weights_only=True)`, so
loading it runs no code. The download is checked against the SHA-256 above
before it is used, and again every time it is loaded from the cache. Files
at another rate are resampled to 16 kHz on the way in.

On one CPU thread a 10 s file takes 0.14 s (real-time factor 0.014). A
file longer than 30 s is enhanced in 30 s chunks that overlap by 1 s; see
*Limits*.

## Training data

**VoiceBank-DEMAND**, the 28-speaker set: Valentini-Botinhao, *Noisy speech
database for training speech enhancement algorithms and TTS models*,
University of Edinburgh, CSTR, 2017,
[doi:10.7488/ds/2117](https://doi.org/10.7488/ds/2117), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The clean speech
is from the CSTR VCTK Corpus; the noise is from the DEMAND database, plus
speech-shaped noise and babble.

`kudioenhance prepare` resampled the 48 kHz files to 16 kHz with soxr's HQ
filter and kept them as 32-bit float. Two training speakers, p226 and p287,
were held out whole for validation, as in SpeechBrain's VoiceBank recipe:

| Split | Speakers | Utterances | Hours | Noise |
|---|---|---|---|---|
| train | 26 | 10,802 | 8.76 | 10 types at 0, 5, 10 and 15 dB |
| validation | p226, p287 | 770 | 0.63 | as train |
| test | p232, p257 | 824 | 0.58 | 5 other types at 2.5, 7.5, 12.5 and 17.5 dB |

The test set was used for the scores below and for nothing else: the
checkpoint was chosen on the validation loss.

## Training

`kudioenhance train` on the waveform path, with this config:

```yaml
audio:
  sr: 16000
  n_fft: 512
  hop_length: 256
  win_length: 512
  window: sqrt_hann
  pad_mode: reflect
  compress: null
data:
  prepared: data/vbd16k
  seed: 17
  crop_seconds: 2.0
model:
  name: gtcrn
  units: null               # [16]
  dropout: 0.0
train:
  epochs: 200
  batch_size: 4
  learning_rate: 0.001
  patience: 10
  grad_clip: 3.0
  device: cpu
  seed: 17
  num_workers: 2
```

- The loss is the `loss` section's default, the one in GTCRN's repository:
  30 times the MSE of the spectrum compressed to `|S|^0.3` with its phase,
  70 times the MSE of the compressed magnitude, and 0.1 times the negative
  SI-SNR in dB.
- Adam at 1e-3, halved whenever the validation loss had not improved for 5
  epochs, as the paper does; training stopped after 10 epochs without
  improvement.
- Each epoch takes a fresh 2 s crop of every training pair; validation takes
  the middle 2 s of each pair, the same every epoch.
- **CPU only**: an Intel Core i5-13600K with 2 PyTorch threads and 2
  processes reading the audio; no GPU. PyTorch 2.15.0, Python 3.13.
- **Time**: 14 h 45 min for 61 epochs, about 14.5 min an epoch. The
  validation loss was lowest at epoch 51 (2.42398), and those are the
  weights; the learning rate had been halved three times by the end.

## Results

The VoiceBank-DEMAND test set, 824 utterances, scored by `kudioenhance
evaluate --full` through `kudio.score_dataset` (kudio 4.0.0). The brackets
are 95% BCa intervals. Every difference from the noisy input is significant
after Holm's adjustment, except DNSMOS SIG.

| Metric | Noisy | `gtcrn-vbd16k` | GTCRN paper |
|---|---|---|---|
| WB-PESQ | 1.97 [1.92, 2.02] | **2.77** [2.73, 2.81] | 2.87 |
| CSIG | 3.33 [3.28, 3.39] | 4.09 [4.04, 4.13] | – |
| CBAK | 2.44 [2.40, 2.49] | 3.41 [3.37, 3.45] | – |
| COVL | 2.62 [2.57, 2.68] | 3.43 [3.39, 3.47] | – |
| SSNR (dB) | 1.68 | 9.85 | – |
| STOI | 0.921 [0.916, 0.926] | 0.939 [0.935, 0.942] | 0.940 |
| SI-SDR (dB) | 8.45 [8.06, 8.82] | 18.69 [18.42, 18.97] | 18.83 (SI-SNR) |
| DNSMOS SIG / BAK / OVRL | 3.34 / 3.12 / 2.70 | 3.36 / 4.00 / 3.07 | – |
| DNSMOS P.808 | 3.02 | 3.43 | – |

- The SSNR row is the segmental SNR of Loizou's `composite.m`, the column
  papers on this test set print; kudio's own `segsnr` frames differently
  and gives 1.56 and 10.03.
- The test set has 2 speakers, so the intervals resample sentences: they
  say how far the means would move over other sentences from p232 and
  p257, not how a third speaker would fare.
- The paper's 2.87 is outside the interval. The paper gives no crop length
  or epoch count for this dataset. These weights saw 2 s crops and stopped
  at epoch 61; its author has suggested crops of about 8 s and 200 epochs
  elsewhere. Whether that closes the gap has not been tried.
- STOI and SI-SDR are within 0.001 and 0.14 dB of the paper.

The signature kudio printed under the table, for the means:

```
kudio:4.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5,whole=pesq|snr|si_sdr|segsnr:frame_length=512,hop_length=256|pesq:mode=wb,scale=lqo|stoi|sdr|csig:mode=wb,scale=lqo,pesq=computed|cbak:mode=wb,scale=lqo,pesq=computed|covl:mode=wb,scale=lqo,pesq=computed|dnsmos:model=sig_bak_ovr.onnx@269fbebd,polyfit=707fde4b,personalized=no|dnsmos_p808:model=model_v8.onnx@9246480c,librosa=0.8|pkg:mir_eval=0.8.2,onnxruntime=1.32.0,pesq=0.0.4,pystoi=0.4.1|n:824|ci:kind=bootstrap_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=none
```

and for the differences, the same with `ci:kind=paired_ci`.

## Limits

- Trained on one corpus: read English sentences from 26 speakers, additive
  noise between 0 and 15 dB, no reverberation. Other noise, rooms,
  languages and recording chains have not been measured.
- 16 kHz only; a file at another rate is resampled to 16 kHz first.
- Each 30 s chunk of a long file starts the GRUs from rest. Against the
  whole file enhanced at once, the output differs by about 3% of the file's
  RMS for two seconds after a seam, about 1% from two to five seconds on,
  and 0.3–0.6% from six to ten. On four files of 96–98 s joined from test
  sentences, the chunked output's WB-PESQ was within 0.005 of the whole
  file's and its SI-SDR within 0.05 dB, higher on some files and lower on
  others.

## Citing

The weights: KudioEnhance (`CITATION.cff`). The architecture:

```bibtex
@inproceedings{rong2024gtcrn,
  author = {X. Rong and T. Sun and X. Zhang and Y. Hu and C. Zhu and J. Lu},
  title = {{GTCRN}: A speech enhancement model requiring ultralow
           computational resources},
  booktitle = {Proc. ICASSP},
  pages = {971--975},
  year = {2024},
  doi = {10.1109/ICASSP48485.2024.10448310},
}
```

The data, which CC BY 4.0 asks to be credited, as its DataShare record
cites it:

```bibtex
@misc{valentini2017noisy,
  author = {C. Valentini-Botinhao},
  title = {Noisy speech database for training speech enhancement
           algorithms and {TTS} models},
  publisher = {University of Edinburgh. School of Informatics. Centre for
               Speech Technology Research (CSTR)},
  year = {2017},
  doi = {10.7488/ds/2117},
}
```
