# kudio-enhance

[English](README.md) | **繁體中文**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![kudio](https://img.shields.io/pypi/v/kudio.svg?label=kudio)](https://pypi.org/project/kudio/)

**訓練並執行學習式語音增強（降噪）模型**，建立在
[`kudio`](https://pypi.org/project/kudio/) 音訊工具包之上。

三個階段，一個設定檔：

[![KudioEnhance workflow](docs/images/workflow.png)](https://raw.githack.com/recklight/KudioEnhance/master/docs/workflow.html?theme=dark)

這張圖連到互動版本——[線上開啟](https://raw.githack.com/recklight/KudioEnhance/master/docs/workflow.html?theme=dark)，或在 clone 下來的專案中開啟 [`docs/workflow.html`](docs/workflow.html)。可以縮放、搜尋節點、追蹤兩個節點之間的路徑，而且大多數方塊都連到它所依據的原始碼行，並固定在當時核對的 commit。

每個階段讀取前一個階段寫出的結果，所以可以不重新混音就重新訓練，也可以不重新訓練就重新評估。`run` 會依序執行三個階段。

模型在 **log-power 頻譜域** 運作，並沿用含噪訊號的相位——這是標準的「只處理幅度」增強做法。特徵擷取與重建都經過 `kudio`，所以模型看到的，和任何以 kudio 為基礎的工具產生的完全相同。

## 安裝

本套件不在 PyPI 上。請 clone 後從原始碼安裝：

```bash
git clone https://github.com/recklight/KudioEnhance.git
cd KudioEnhance
pip install -e ".[dev]"
```

需要 **Python 3.9+** 與 **PyTorch ≥ 2.4.1**。模型是單純的 `torch.nn` 模組，由本套件自己的訓練迴圈訓練——不用 Keras，也不用 Lightning。更舊的 torch wheel，以及 Windows 上的 2.4.0，是針對 NumPy 1 編譯的；pip 不會阻止它們和 NumPy 2 裝在一起，結果會在第一次 `torch.from_numpy` 時失敗。

在 Linux 上，預設的 `torch` wheel 內含 CUDA，大小達數 GB。沒有 NVIDIA GPU 的話，請**先**安裝 CPU 版，pip 在安裝本套件時會保留它：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

PyPI 上的 Windows 與 macOS wheel 不含 CUDA，所以體積很小。只要 torch 偵測得到 NVIDIA GPU，訓練就會使用它（`train` 區段的 `device: auto`），因此同一份設定在兩種機器上都能用。Apple 的 GPU 不會被使用：torch 可以透過 `mps` 使用它，但本套件不會要求這麼做，在 Mac 上訓練會跑在 CPU。

STOI 與 SDR 是選用的：`pip install -e ".[eval]"`。**PESQ 是另一個獨立的 extra**，`pip install "kudio[pesq]"`，因為它要從原始碼編譯——需要 C 編譯器，在 Windows 上是 MSVC Build Tools——所以刻意不放進 `[eval]` 和 `[all]`，否則它們會裝不起來。

PESQ 有兩種尺度。`evaluate` 預設回報 MOS-LQO；加上 `--pesq-scale raw` 則是多數語音增強論文引用的窄頻 P.862 分數。它們是不同的尺度，不是四捨五入的差別：raw 2.03 大約等於 MOS-LQO 1.65。

## 快速開始

```bash
kudio-enhance init config.yaml        # 產生起始設定檔
kudio-enhance run -c config.yaml -n exp1
```

這會混出資料集、訓練，並印出前後的分數：

```
si_sdr      -4.812 ->    6.933  (+11.745)
snr         -4.998 ->    5.221  (+10.219)
seg_snr     -6.114 ->    1.870  (+7.984)
report -> runs/exp1/report.csv
```

接著用訓練好的 run 對任何音訊降噪：

```bash
kudio-enhance denoise runs/exp1 noisy.wav enhanced.wav
kudio-enhance denoise runs/exp1 noisy/ clean/ --recursive --report
```

資料夾模式對整棵目錄樹只載入模型**一次**，輸出會鏡像輸入的目錄結構，加上 `--report` 會告訴你噪音底降了多少。

之後再確認它到底有沒有學到東西：

```bash
kudio-enhance curves -c config.yaml -n exp1
```

```
3 epoch(s), monitoring val_loss
  best   0.12172 at epoch 3
  first  0.28104   last 0.12172
```

訓練迴圈會交回訓練曲線，但幾乎每個腳本都把它丟掉，於是終端機一關，「它收斂了嗎？還是因為驗證損失已經在上升而提早停止？」就再也無從回答。曲線會寫到 `runs/exp1/history.json`（有 matplotlib 時另存 `history.png`）。如果最好的 epoch 是*第一個*，這個指令會明說並以非零狀態結束——那就是「沒有學到東西」的樣子。

### 逐階段執行

```bash
kudio-enhance synth    -c config.yaml -n exp1   # 混音並固定資料切分
kudio-enhance train    -c config.yaml -n exp1   # 訓練（--epochs 1 可做冒煙測試）
kudio-enhance evaluate -c config.yaml -n exp1   # 評分（--full 加上 PESQ/STOI）
```

每個階段讀取前一個階段寫出的結果，所以可以不重新混音就重新訓練，或不重新訓練就重新評估。

## 設定

```yaml
audio:                      # STFT 幾何參數，特徵擷取與重建共用
  sr: 16000
  n_fft: 512
  hop_length: 256
  win_length: 512

data:
  clean_dir: data/clean
  noise_dir: data/noise
  mixed_dir: runs/mixed
  snr_db: [-5, 0, 5]
  rt60: []                  # 殘響時間（秒）；[] = 不加房間
  drr_db: 0.0               # 麥克風有多近，單位 dB
  reverberant_target: false # 訓練成只去除噪音、保留房間
  channel: null             # telephone / voip / mu_law / a_law，或 null
  mode: regular             # 'regular' = 每個乾淨檔一個混音；'inc' = 所有組合
  seed: 17                  # 讓混音與資料切分可重現
  val_split: 0.1
  test_split: 0.1

model:
  name: ddae                # ddae | blstm | conv_ae | crn | unet
  target: spectrum          # spectrum | irm  -- 見「網路被要求輸出什麼」
  context: 2                # 逐幀模型：前後各堆疊幾幀進輸入
  n_frames: 64              # 序列模型：每個訓練視窗的幀數
  units: null               # 各層寬度；null = 架構的預設值
  dropout: 0.1

train:
  epochs: 50
  batch_size: 256           # ddae 需要 2 以上；序列模型可以用 1
  learning_rate: 0.001
  patience: 8
  out_dir: runs
  device: auto              # auto | cpu | cuda | cuda:1
  seed: null                # 權重、dropout 與批次順序；null = data.seed
  streaming: false          # true：每個 epoch 從磁碟讀取特徵
  shuffle_buffer: 0         # 僅串流模式：保留多少樣本用來打亂
```

每個鍵都是有型別的 dataclass 欄位。打錯字會在載入時失敗並指出是哪個鍵——不會在三個階段之後悄悄改用預設值。

## 模型

```bash
kudio-enhance models
```

```
blstm       stacked bidirectional LSTM                       units: [1024, 1024, 1024]
conv_ae     dilated Conv1D autoencoder, residual             units: [1024, 1024, 1024]
crn         causal conv-recurrent network (Tan & Wang 2018)  units: [16, 32, 64, 64, 64]
ddae        dense denoising autoencoder, frame-wise          units: [1024, 1024, 1024]
unet        2-D U-Net over time and frequency, non-causal    units: [16, 32, 64, 128]
```

| 名稱 | 輸入形狀 | 因果 | 是什麼、何時選用 |
|---|---|---|---|
| `ddae` | `(N, bins × (2·context+1))` | 否——看得到前後 `context` 幀 | 全連接降噪自編碼器。逐幀、快速，是穩健的基準。 |
| `blstm` | `(N, n_frames, bins)` | 否 | 堆疊的雙向 LSTM。雙向的時間上下文。 |
| `conv_ae` | `(N, n_frames, bins)` | 否 | 帶殘差連接的膨脹 Conv1D 自編碼器——從「讓訊號直接通過」開始學，而不是從雜訊開始。 |
| `crn` | `(N, n_frames, bins)` | **是** | 依 Tan & Wang 的卷積遞迴網路（*A convolutional recurrent neural network for real-time speech enhancement*，Interspeech 2018）：在時間與頻率上跨步的 2-D 卷積編碼器、沿時間的 LSTM，以及對稱的解碼器，把每層編碼器的輸出接回來。當輸出不能等待後面的內容時選它。 |
| `unet` | `(N, n_frames, bins)` | 否——約往後看 50 幀 | 時間與頻率上的 2-D U-Net：卷積區塊，往下時兩軸都減半，往上時以跳接接回。適合整檔離線增強；在預設寬度下約五十萬個參數，也是這裡最小的模型。 |

序列模型用固定長度的視窗訓練，但模型本身沒有固定時間軸長度，所以同一組權重可以一次處理整段語句。兩個 2-D 模型也接受任何 `n_fft`：它們在內部把時間與頻率補到減半所需的大小，再裁回來。

**因果**的意思是某一幀的輸出只取決於該幀及更早的幀——`crn` 的卷積只在過去那一側補值、它的 LSTM 只往前跑，並有測試把關。這是即時版本需要的性質，但本套件不做串流：`Enhancer` 仍一次增強整個檔案，而 STFT 本身也會帶來一個視窗的延遲。

**`model.units`** 是各層寬度的列表：全連接層的單元數、LSTM 的單元數、卷積的濾波器或通道數。設為 `null` 時採用上面列出的架構預設值。寬度 1024 對全連接層很合理，對 2-D 卷積卻很荒謬（它的每個通道都是一整張時頻圖），所以預設值屬於架構本身。設成列表就能自行指定寬度。訓練會把實際建出的寬度寫進 run 的 `config.yaml`，即使之後的版本改了預設值，舊的 run 仍能載入。

`crn` 的 LSTM 會讀取每層把頻率點減半後剩下的所有通道與頻率點，所以大小隨 `n_fft` 增長：`n_fft: 256` 時約 1.8M 參數、512 時 5.5M、1024 時 19M。`n_fft` 較大時，多加一層再減半一次——`units: [16, 32, 64, 64, 64, 64]` 能把 1024 時的大小降回 5.6M。`unet` 的大小與頻率點數無關。

新增架構只需要一個建構函式加上一筆 `REGISTRY` 項目——沒有要在三個地方擴充的 `if/elif` 鏈。項目中的 `default_units` 就是 `units: null` 時建出的寬度；省略它的話，架構會得到前三個模型共用的 1024 寬堆疊。

### 網路被要求輸出什麼

`model.target` 與 `model.name` 是分開的選擇，每種架構都支援兩種：

| 目標 | 網路輸出 | |
|---|---|---|
| `spectrum` | 直接輸出乾淨的 log-power 頻譜 | 輸出無界；除了形狀，還得學會*音量* |
| `irm` | 介於 `[0, 1]` 的**理想比例遮罩** | 有界，所以 sigmoid 不會產生不可能的答案，而音量來自輸入 |

```yaml
model:
  name: ddae
  target: irm
```

遮罩是 `sqrt(|S|² / (|S|² + |N|²))`——每個時頻點中有多少是語音——在對數域中套用它只是一個加法，所以網路永遠只需要決定*要保留多少*。

**在本專案自己的冒煙測試語料上，用小型 `ddae` 訓練三個 epoch 的實測：**

| 目標 | SI-SDR |
|---|---|
| `spectrum` | +0.05 → **−7.59 dB** |
| `irm` | +0.05 → **+2.61 dB** |
| 理想（oracle）遮罩 | −3.02 → **+14.16 dB** |

訓練不足的頻譜模型還沒學會音量，所以輸出又大聲又錯；訓練不足的遮罩模型已經有用，因為它最糟也只是保留或去掉原本就有的東西。oracle 那一列是這個目標所追求的上限。

這是小模型配小語料，不是基準測試——但它把「為什麼通常選遮罩目標」在大約二十秒內具體呈現出來。

## 房間

噪音是加性的，房間是卷積性的，而真實錄音兩者都有。`data.rt60` 會在混入噪音之前先讓語音經過一個房間：

```yaml
data:
  snr_db: [-5, 0, 5]
  rt60: [0.3, 0.6, 0.9]     # 第四個軸——在 `inc` 模式下檔案數 ×3
  drr_db: 0.0               # +10 是近距離麥克風，0 是一兩公尺
```

此時 SNR 是相對於**含殘響**的語音來量的，因為在麥克風那裡和噪音競爭的就是它。

**預設情況下，乾淨檔仍是目標。** 用這種方式訓練的模型被要求同時去除房間與噪音，這是標準做法，也是值得刻意選擇、而不是事後才發現的設定。

`reverberant_target: true` 會改變這點。`kudio.Synthesizer` 會在每個混音旁的 `TARGETS/` 寫出「含殘響但乾淨」的訊號——同一段語音、保留房間——訓練則要求模型去除噪音並**保留**房間。這是另一個比較容易的問題，而且在實際部署環境保有房間時才是誠實的設定。

它也讓 `irm` 目標變得正確。遮罩是由 `noisy - reference` 建出來的；語音上有房間時，乾的乾淨檔並不是實際被混進去的訊號，於是這個減法把殘響算成了*噪音*，遮罩就會要求模型把房間自己的殘響尾巴當成噪音去掉。改用含殘響的目標後，這個減法得到的才又是真正的噪音。

每個混音都會記錄它所用的房間——`Pair.rt60`，檔名中也有（`speech_white_0dB_rt600ms.wav`），所以 manifest 會說明每個樣本被要求承受什麼。

## 通道

第三種劣化，也是既非加性也非卷積性的那一種：編解碼器做量化、傳輸鏈路掉封包、轉換器丟掉位元。

```yaml
data:
  channel: telephone        # 8 kHz 下 300-3400 Hz，經過 G.711
```

可選 `telephone`、`telephone_a`、`voip`（G.711，5% 封包以每段四個的方式遺失並保留前一個封包）、`mu_law`、`a_law`。它套用在**混音**上，噪音也包含在內，因為線路傳送的是麥克風收到的一切。

**它不是一個軸。** 與 `snr_db` 和 `rt60` 不同，它對整個資料集固定不變，也不會出現在檔名中：一個語料要嘛是透過電話線錄的，要嘛不是。`Pair.channel` 記錄用了哪種鏈路，而目標不會經過它——要求模型重現 G.711 並不是降噪。

## 資料集

把 `clean_dir` 和 `noise_dir` 指向任意兩個資料夾即可。可直接使用的公開資料集：

- **[VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791)**——標準的語音增強基準（已預先混好；直接使用它的 clean/noisy 資料夾並略過 synth 階段）。
- **[LibriSpeech](https://www.openslr.org/12/)** + **[DEMAND](https://zenodo.org/record/1227121)** 或 **[MUSAN](https://www.openslr.org/17/)**——以你關心的 SNR 自行混音。
- **[ESC-50](https://github.com/karolpiczak/ESC-50)**——環境噪音，適合非語音的目標。

設定 `seed` 後混音即可重現：相同的噪音起點、相同的配對、相同的切分。

## 訓練資料：放在記憶體或串流

預設情況下，訓練會把特徵放在記憶體中：只計算一次，每個 epoch 都把整個資料集完整打亂。對這類模型通常使用的數十小時資料而言，這是正確的選擇。

逐幀模型（`ddae`）看的是每一幀加上前後 `context` 個鄰幀，過去事先為整個資料集堆疊這些鄰幀，會讓記憶體乘上 `2·context + 1` 倍——`context: 5` 時是十一倍。現在訓練集每一幀只存一份，放在 `kudio.ContextFrames` 中，在批次抽到某些列時才把它們的鄰幀堆起來。得到的列完全相同，訓練結果也相同：有測試確保它與堆疊矩陣逐列一致，而且用兩者訓練出的權重一模一樣。序列模型的視窗本來就不會比特徵大。

剩下的就是特徵本身，輸入與目標：在 16 kHz、預設 STFT 下，每小時訓練音訊約 0.46 GB。放不下時就用串流：

```yaml
train:
  streaming: true
  shuffle_buffer: 25000     # 約一百個 4 秒檔案的 ddae 幀數
```

這時什麼都不會常駐記憶體。每個 epoch 以新的順序重新讀取訓練檔，邊讀邊計算每個檔的樣本——與記憶體模式逐列相同的樣本。代價是每個 epoch 都要把音訊走一遍：讀檔、重新取樣與 STFT，由訓練行程在批次之間完成。

- **正規化**在第一個 epoch 之前以一次額外的掃描來擬合，一次處理一個檔案。用的是 kudio 的公式——平均值、母體標準差，以及加上同樣的 `1e-8`——並以 float64 累加；記憶體模式也以相同方式、一次一個檔案的頻譜來擬合，所以兩種模式使用的統計量逐位元相同。（kudio 自己的 `Standardizer.fit` 以 float32 累加，在一小時的幀上會與真實統計量相差約十萬分之幾；這裡的誤差在千萬分之一以內。）
- **驗證**也採串流，依 manifest 的順序，批次的切分位置與記憶體模式完全相同，所以分數也和記憶體模式一樣。
- **打亂緩衝區。** 每個檔案的樣本會加入一個池子，只要池中至少還會留下 `shuffle_buffer` 個樣本，每個批次就從池中所有樣本隨機抽取；檔案讀完後，剩下的以隨機順序送出。對 `ddae` 而言一個樣本是一幀，對序列模型則是一個 `n_frames` 視窗。要混得好，讓緩衝區涵蓋大約一百個檔案：典型檔案幀數的一百倍，或一個檔案中視窗數的一百倍。記憶體成本就是這麼多個輸入與目標：`ddae` 在 257 個頻率點、`context: 2` 時每個樣本 6 KB，64 幀視窗則是 132 KB。
- **`shuffle_buffer: 0` 不保留任何樣本。** 檔案仍以新的順序讀入，每個檔案內的樣本也會打亂，但每個批次只從正在讀取的檔案以及先前抽剩的（少於一個批次）中抽取——也就是同一段錄音的相鄰幀，訓練效果較差。要求這樣做時訓練會發出警告。記憶體模式不使用這個設定。
- **可重現。** 檔案順序與每一次從池中抽取都來自 `train.seed`，所以串流的訓練在 CPU 上可以精確重現到最後一位，和記憶體模式一樣。不過它與記憶體模式是不同的訓練：樣本相同，順序不同。

兩種模式都經過 `kudio_enhance.data.build_dataset`，而 `train.fit` 接受它的回傳值來取代陣列：

```python
from kudio_enhance.data import build_dataset
from kudio_enhance.train import fit

train_set, std = build_dataset(train_pairs, cfg, sequence=False, fit=True)
history = fit(model, train_set, None, cfg, "runs/exp1/ddae.pt")
```

`build_arrays` 仍會回傳單純的堆疊矩陣，給需要它的程式使用。

## Python API

```python
from kudio_enhance import Config, Enhancer, pipeline

cfg = Config.from_yaml("config.yaml")
summary = pipeline.run(cfg, "exp1")          # synth + train + evaluate

enh = Enhancer.load("runs/exp1")             # 模型 + 統計量 + 設定一起載入
clean = enh.enhance_file("noisy.wav", "enhanced.wav")
result = enh.enhance_folder("noisy/", "clean/", report=True)
print(result, result.mean_reduction_db())
```

整棵目錄樹只載入一次模型，結果是 `kudio.EnhanceFolderResult`——與 `kudio.convert_folder` 回傳的形狀相同，所以不論是誰處理的，「我處理了一個資料夾」都只有一種答案。

訓練好的 run 也能直接放進 kudio 的比較表，和統計式方法並列：

```python
import kudio
kudio.compare_enhancers(y, sr, ['logmmse', ('exp1', enh.enhance)],
                        reference=clean)
```

`Enhancer.load` 從同一個 run 目錄讀取模型、它的正規化統計量，**以及**產生它們的 STFT 幾何參數，所以推論永遠不會悄悄與訓練不一致。

### run 目錄裡有什麼

```
runs/exp1/
├── config.yaml      # 訓練時的實驗設定
├── ddae.pt          # 權重：<model.name>.pt
├── stats.npz        # 特徵經過的 Standardizer
├── history.json     # 訓練曲線（有 matplotlib 時另有 history.png）
├── train.json  val.json  test.json  manifest.json
└── report.csv       # evaluate 之後產生
```

`.pt` 檔是一個單純的字典——格式標記、訓練時的架構與目標，以及一個 `state_dict`——以 `torch.load(..., weights_only=True)` 讀回。裡面沒有任何被 pickle 的類別，所以打開別人傳來的 run 不會執行他們的程式碼；`Enhancer.load` 會先依 `config.yaml` 重建網路，再把權重灌進去。它也永遠是最好的那個 epoch：每當監控的損失改善時訓練就存檢查點，停止時再把那組權重放回模型。

設定 `train.seed`（或只設 `data.seed`），初始權重、dropout 遮罩與批次順序都由它決定，所以在 CPU 上訓練可以精確重現到最後一位。有些 GPU kernel 不是確定性的，所以在 GPU 上只會非常接近，而非完全相同。

## 專案結構

```
docs/
├── images/workflow.png  # README 的圖
└── workflow.html        # 同一張圖的互動版
src/kudio_enhance/
├── config.py        # 有型別的 dataclass、YAML 讀寫、run 路徑
├── data.py          # 混音、manifest、切分、訓練集
├── datasets.py      # torch 看到的訓練集：常駐或串流
├── features.py      # 頻譜 / 上下文堆疊 / 視窗 / Standardizer
├── models/
│   ├── registry.py      # 名稱 -> 建構函式 + 資料形狀
│   ├── dense.py         # ddae
│   ├── recurrent.py     # blstm
│   ├── convolutional.py # conv_ae
│   ├── crn.py           # crn
│   └── unet.py          # unet
├── train.py         # 訓練迴圈：Adam、檢查點、提早停止
├── checkpoint.py    # .pt 格式
├── inference.py     # Enhancer
├── evaluate.py      # 逐檔評分 + CSV 報告
├── pipeline.py      # 三個階段
└── cli.py           # kudio-enhance
```

模型建構函式回傳單純的 `torch.nn.Module`；`train.py` 負責優化器、損失函數與迴圈。這樣架構可以被檢視，而不必綁定某種優化設定。

## 測試

```bash
pytest -q
```

共 228 個測試。測試套件會在暫存目錄建立一個很小的合成語料，並在幾秒內跑完整個流程——混音、訓練一個 epoch、增強、評分。需要 PyTorch 的測試會明確要求它，沒安裝時會乾淨地跳過；其餘 105 個仍會執行。

遮罩測試檢查的是*公式*，而不只是能不能跑：全是語音時為 1、全是噪音時為 0、功率相等時為 1/√2，而 oracle 遮罩必須真的提升 SI-SDR。這些不成立，就代表目標是錯的，不管損失曲線怎麼說。架構測試也一樣：改動 *t* 之後的任何一幀，`crn` 到 *t* 為止的輸出必須逐位元不變；`unet` 的則必須改變，因為它的說明寫著它是非因果的。訓練集也是：記憶體模式必須逐位元交回堆疊矩陣的列，依 manifest 順序的串流掃描也是，而一個打亂的 epoch 必須讓每個樣本恰好出現一次，並且在相同 seed 下得到相同順序。兩種模式必須以逐位元相同的統計量做正規化。

## 已知限制

- 串流的 epoch 由訓練行程在批次之間逐一讀取並轉換檔案。沒有 loader 工作行程，所以 GPU 可能會等 CPU；記憶體模式只需付出一次這個成本。
- 只處理幅度：相位沿用含噪輸入的相位。考慮相位或時域的模型屬於另一個架構家族。
- 序列模型一次處理整個檔案。對 `crn` 和 `unet` 而言，這代表每一幀、每個頻率點、每個通道都有一張特徵圖：在預設寬度下，每分鐘 16 kHz 音訊約需半 GB，所以長錄音請先切段再增強。
- `irm` 目標需要噪音，並在時域中以 `noisy - reference` 取得。對 `kudio.Synthesizer` 混出來的資料這是精確的，但若語料的 noisy 與 clean 是分開的兩段錄音而不是相加而成，就是錯的。在含殘響的資料集中，reference 必須是含殘響的目標，否則房間的殘響尾巴會被算成噪音——見上面的 `reverberant_target`。

### 0.2.0 起的變更

模型改為**原生 PyTorch**；0.1 是 TensorFlow 與 Keras 3。設定檔沒有改變——`train.device` 和 `train.seed` 是新增的選用項目——架構也沒變，連 Keras 的初始化方式、BatchNorm momentum 與 LSTM forget-gate bias 都一樣，所以在 0.1 調好的設定在 0.2 仍保有相同意義——但數字不會完全一樣，因為兩個框架的 kernel 不同。

權重則不同。0.1 訓練出的 run 目錄裡是 `ddae.keras`，沒有 `ddae.pt`，`Enhancer.load` 會明說這點並要求重新訓練，而不是回報找不到檔案。其中的設定與已固定的資料切分可以原樣沿用：

```bash
kudio-enhance train -c runs/exp1/config.yaml -n exp1
```

`fit` 以一般的 dict 回傳訓練曲線，而不是 Keras 的 `History`；`compile_model` 成為已棄用、不做任何事的函式，保留它是為了讓 0.1 時代呼叫 `fit(compile_model(...))` 的腳本仍能執行。`build_callbacks` 已移除，匯入它會失敗：`fit` 自己處理檢查點、提早停止與學習率減半，所以原本為 `model.fit` 建立 Keras callback 的程式，請改呼叫 `kudio_enhance.train.fit`。

`ddae` 的 `train.batch_size` 必須是 2 以上，訓練會在讀取任何音訊之前就指出這點；只有一幀的 `ddae` 訓練集也會被拒絕。它的 BatchNorm 要對批次取變異數，而單一樣本沒有變異數。0.1 接受批次大小 1，但 Keras 在那種情況下會把每個單元正規化成常數，所以最後一個 BatchNorm 之前的層全都沒有被訓練到。序列模型仍可用一個視窗的批次訓練。

正規化統計量現在無論記憶體或串流模式都以 float64 累加，所以同一份訓練集得到的 `stats.npz` 與 0.1 略有不同：一小時音訊上約十萬分之幾，資料集較小時更少。

訓練改經過 `build_dataset` 而不是 `build_arrays`，所以逐幀模型的上下文不再為整個資料集堆疊在記憶體中；訓練結果精確相同。`fit` 可以用資料集取代 `x` 並傳入 `y=None`，也仍接受陣列。`train.streaming` 是新增的；`train.shuffle_buffer` 以前在設定裡卻沒有任何程式讀取，現在用來決定串流打亂的大小——也只用在這裡。

新增兩個架構 `crn` 與 `unet`，`model.units` 的預設值改為 `null`，代表架構自己的寬度。有設定 `units` 的設定檔建出的模型與以前相同，沒設定的也一樣：`ddae`、`blstm` 與 `conv_ae` 的預設仍是 `[1024, 1024, 1024]`。直接讀取 `cfg.model.units` 的程式現在可能拿到 `None`；`kudio_enhance.models.resolve_units(cfg)` 會回傳一份把寬度明確寫出的副本。

### 3.5.0 起的變更

`Pair`、`save_manifest`、`load_manifest` 與 `split_pairs` 現在來自 **kudio**——它們沒有任何降噪專屬的東西。磁碟上的 JSON 不變，這些名稱也仍可從 `kudio_enhance.data` 匯入，但 kudio 的 `split_pairs` 比較嚴格：`val_split + test_split >= 1.0` 會引發錯誤，而不是悄悄把所有資料都交給訓練。

## 相關專案

- **[kudio](https://pypi.org/project/kudio/)**——底層的音訊工具包。
- **[KudioStudio](https://github.com/recklight/KudioStudio)**——同一工具包的桌面工作台。與本套件一起安裝時，它的 Enhance 面板可以載入這裡產生的 run 目錄，讓你直接聆聽模型的輸出並互動評分，而不只是看 CSV。

## 授權

MIT——見 [LICENSE.txt](LICENSE.txt)。
