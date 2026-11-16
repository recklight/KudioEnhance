# KudioEnhance 使用指南

[English](https://github.com/recklight/KudioEnhance/blob/master/docs/guide.md) | **繁體中文**

[README](https://github.com/recklight/KudioEnhance/blob/master/README.zh-TW.md) 沒寫到的部分：`evaluate` 寫出什麼、模型和兩條路徑的細節、訓練資料放在記憶體或串流、`synth` 能加上的劣化、run 目錄、專案結構，以及測試檢查什麼。

## `evaluate` 寫出什麼

`evaluate` 把測試集的每個檔案增強後寫到 `runs/exp1/enhanced/`，存成 32 位元浮點，評分的對象就是模型輸出的原樣。接著把三邊交給 `kudio.score_dataset`：增強檔對乾淨檔評分，noisy 檔當 baseline。run 目錄裡除了 `report.csv`，還有結果表的三種格式 `results.md`、`results.tex`、`results.json`，以及 `results.bib`，裡面是 kudio 和表上每個指標的 BibTeX。表的每一列是 noisy 輸入的平均、增強後的平均和兩者的差，各附 95% BCa 區間，再來是這個差的雙尾符號翻轉檢定，和對所有列做 Holm 調整後的 p 值。表下的 signature 記下 kudio 版本、取樣率、每個指標的設定和背後套件的版本，數字是怎麼算出來的都查得回去。

不加 `--full` 時評分段落 SNR、SI-SDR 和 SNR，只需要 numpy。`--full` 再加上 PESQ、CSIG/CBAK/COVL、STOI 和 SDR，沒裝的就略過並發出警告。`--metrics pesq,stoi,si_sdr` 用 kudio 的名稱指定，只評這幾個。`--dnsmos-dir` 加上 DNSMOS P.835 和 P.808；這需要 `pip install "kudio[dnsmos]"`，以及 [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge) 的 ONNX 模型，kudio 沒有附。`--jobs 4` 一次評四個檔。

同一位說話者的句子彼此不獨立，所以檔名帶得出說話者時（像 VCTK 的 `p232_001.wav`），區間按整位說話者重抽。這要有夠多說話者：少於 10 位，區間會明顯達不到 95%，在 kudio 的模擬裡只有 2 位時，區間涵蓋真值的比例只有一半。VoiceBank-DEMAND 的測試集只有 2 位，p232 和 p257，所以這時區間改按句子重抽，表下會寫一行說明：區間反映的是這兩位說話者的其他句子，不代表第三位說話者。`--speakers` 接一個正規表示式，第一個群組就是說話者；給 `''` 則不分說話者。

## 模型細節

序列模型用固定長度的視窗訓練，但模型本身沒有固定時間軸長度，所以同一組權重可以一次處理整段語句。兩個 2-D 模型也接受任何 `n_fft`：它們在內部把時間與頻率補到減半所需的大小，再裁回來。

**因果**的意思是某一幀的輸出只取決於該幀及更早的幀——`crn` 與 `gtcrn` 的卷積只在過去那一側補值、沿時間的遞迴層只往前跑，兩者都有測試把關。這是即時版本需要的性質，但本套件不做串流：`Enhancer` 一次增強整個檔案（波形路徑上的長檔案一次 30 秒），而 STFT 本身也會帶來一個視窗的延遲。

**`model.units`** 是各層寬度的列表：全連接層的單元數、LSTM 的單元數、卷積的濾波器或通道數。設為 `null` 時採用 `kudioenhance models` 列出的架構預設值。寬度 1024 對全連接層很合理，對 2-D 卷積卻很荒謬（它的每個通道都是一整張時頻圖），所以預設值屬於架構本身。設成列表就能自行指定寬度。訓練會把實際建出的寬度寫進 run 的 `config.yaml`，即使之後的版本改了預設值，舊的 run 仍能載入。

`crn` 的 LSTM 會讀取每層把頻率點減半後剩下的所有通道與頻率點，所以大小隨 `n_fft` 增長：`n_fft: 256` 時約 1.8M 參數、512 時 5.5M、1024 時 19M。`n_fft` 較大時，多加一層再減半一次——`units: [16, 32, 64, 64, 64, 64]` 能把 1024 時的大小降回 5.6M。`unet` 的大小與頻率點數無關。

新增架構只需要一個建構函式加上一筆 `REGISTRY` 項目——沒有要在三個地方擴充的 `if/elif` 鏈。項目中的 `default_units` 就是 `units: null` 時建出的寬度；省略它的話，架構會得到前三個模型共用的 1024 寬堆疊。

## log-power 路徑

`ddae`、`blstm`、`conv_ae`、`crn` 和 `unet` 看的是 kudio 算的 log-power 頻譜，用在訓練集上擬合的統計量標準化，統計量存成權重旁邊的 `stats.npz`。它們學乾淨的頻譜或一個遮罩，增強後的幅度配上含噪的相位還原回去。訓練集放在記憶體或串流讀取（見「訓練資料：放在記憶體或串流」）。

### 網路被要求輸出什麼

`model.target` 與 `model.name` 是分開的選擇，這條路徑的每種架構都支援兩種：

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

`gtcrn` 一律輸出複數遮罩，不讀 `model.target`。

## 波形路徑：`gtcrn`

`gtcrn` 的訓練方式和另外五個不同。它讀含噪的複數頻譜、輸出複數比值遮罩，所以給回來的相位是它自己估的；它從波形學，每個 epoch 從每一對檔案重新切一段；損失算在它輸出的頻譜和該頻譜還原的波形上，所以不需要擬合正規化統計量。GTCRN 自己的 STFT，加上論文給 VoiceBank-DEMAND 的設定，寫成設定檔是這樣：

```yaml
audio:
  sr: 16000
  n_fft: 512
  hop_length: 256
  win_length: 512
  window: sqrt_hann         # Hann 窗開根號，分析與合成都用它
  pad_mode: reflect         # torch.stft 補兩端的方式，GTCRN 的做法
  compress: null            # 模型看到的是線性尺度的頻譜
data:
  prepared: data/vbd16k
  crop_seconds: 2.0         # 每個 epoch 在每對檔案的新位置切這麼長
model:
  name: gtcrn               # units: [16]，論文的通道數
train:
  batch_size: 4
  learning_rate: 0.001
  patience: 10              # 5 個 epoch 沒進步時學習率減半
  grad_clip: 3.0            # 梯度範數上限；null 不裁剪
  num_workers: 2            # 讀檔與切段的行程數
```

沒寫 `loss` 段時，損失就是 GTCRN repo 裡的那一個：壓縮到 `|S|^0.3`、保留相位的頻譜 MSE 乘 30，壓縮後幅度的 MSE 乘 70，再加 0.1 倍以 dB 計的負 SI-SNR，算 SI-SNR 前不去均值。這一段的鍵 `complex_mse`、`magnitude_mse`、`time_l1`、`neg_si_snr`、`compress`、`zero_mean` 可以改權重、指數和去均值與否。

這條路徑的 run 在第一個 epoch 之前寫 `config.yaml`，之後每個 epoch 寫目前最好的權重、`resume.pt`、`history.json` 和 `progress.json`。中途停掉的話，同一行指令加 `--resume` 就從最後一個跑完的 epoch 接著跑；只要 torch 的執行緒數相同，在 CPU 上續跑的結果和沒中斷過的逐位元相同。`--resume --epochs N` 可以延長已經跑完的 run。續跑時只有 `train.epochs`、`train.patience`、`train.num_workers`、`train.device`、`train.out_dir` 可以改，其他的改了會被拒絕並列出是哪些鍵，因為那已經是另一個 run。`progress.json` 寫著 run 跑到哪：epoch、損失和各項、目前最好的 epoch、學習率、每個 epoch 的秒數和剩餘時間。還在訓練的 run 就能拿去 evaluate 或降噪，用的是目前最好的那個 epoch。

`Enhancer` 對 30 秒以內的檔案一次處理完。更長的檔案切成 30 秒、彼此重疊 1 秒的段，在接縫處交叉淡化，所以檔案再長，記憶體用量都有上限。最後一段停在檔案結束的地方，所以檔尾經過 STFT 的方式和整檔處理時完全一樣。

在 VoiceBank-DEMAND 上用這組設定、2 個 CPU 執行緒訓練了 14 小時 45 分：61 個 epoch，最佳在第 51 個，每個約 14.5 分鐘。公開的權重就是這一組，分數見 [README](https://github.com/recklight/KudioEnhance/blob/master/README.zh-TW.md) 的「VoiceBank-DEMAND 上的結果」。

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

兩種模式都經過 `kudioenhance.data.build_dataset`，而 `train.fit` 接受它的回傳值來取代陣列：

```python
from kudioenhance.data import build_dataset
from kudioenhance.train import fit

train_set, std = build_dataset(train_pairs, cfg, sequence=False, fit=True)
history = fit(model, train_set, None, cfg, "runs/exp1/ddae.pt")
```

`build_arrays` 仍會回傳單純的堆疊矩陣，給需要它的程式使用。

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

## run 目錄裡有什麼

```
runs/exp1/
├── config.yaml      # 訓練時的實驗設定
├── ddae.pt          # 權重：<model.name>.pt
├── stats.npz        # 特徵經過的 Standardizer（gtcrn 沒有）
├── history.json     # 訓練曲線（有 matplotlib 時另有 history.png）
├── resume.pt  progress.json   # gtcrn：最近一個 epoch 的狀態，以及 run 跑到哪
├── train.json  val.json  test.json  manifest.json
├── enhanced/        # evaluate 之後產生：模型輸出的測試集
├── report.csv       # 逐檔分數
└── results.md  results.tex  results.json  results.bib   # 結果表
```

`.pt` 檔是一個單純的字典——格式標記、訓練時的架構與目標，以及一個 `state_dict`——以 `torch.load(..., weights_only=True)` 讀回。裡面沒有任何被 pickle 的類別，所以打開別人傳來的 run 不會執行他們的程式碼；`Enhancer.load` 會先依 `config.yaml` 重建網路，再把權重灌進去。它也永遠是最好的那個 epoch：每當監控的損失改善時訓練就存檢查點，停止時再把那組權重放回模型。

設定 `train.seed`（或只設 `data.seed`），初始權重、dropout 遮罩與批次順序都由它決定，所以在 CPU 上訓練可以精確重現到最後一位。有些 GPU kernel 不是確定性的，所以在 GPU 上只會非常接近，而非完全相同。

## 專案結構

```
docs/
├── images/workflow.png  # README 的圖
├── workflow.html        # 同一張圖的互動版
├── guide.zh-TW.md       # 這份指南（英文版是 guide.md）
└── models/              # 每組公開權重的模型卡
src/kudioenhance/
├── config.py        # 有型別的 dataclass、YAML 讀寫、run 路徑
├── data.py          # 混音、manifest、切分、訓練集
├── vbd.py           # VoiceBank-DEMAND：重抽樣、驗證集切分
├── datasets.py      # torch 看到的訓練集：常駐或串流
├── features.py      # 頻譜 / 上下文堆疊 / 視窗 / Standardizer
├── waveform.py      # clean/noisy 配對的隨機裁段、固定種子的 DataLoader
├── spectral.py      # torch 的複數 STFT、冪次壓縮
├── losses.py        # 複數、幅度、L1、SI-SNR 損失與加權組合
├── models/
│   ├── registry.py      # 名稱 -> 建構函式 + 資料形狀
│   ├── dense.py         # ddae
│   ├── recurrent.py     # blstm
│   ├── convolutional.py # conv_ae
│   ├── crn.py           # crn
│   ├── unet.py          # unet
│   └── gtcrn.py         # gtcrn
├── train.py         # 訓練迴圈：Adam、檢查點、提早停止
├── waveform_train.py # 波形路徑的訓練迴圈，中斷了能從停下的地方接著跑
├── progress.py      # 每個 epoch 重寫的 progress.json 與 history.json
├── checkpoint.py    # .pt 格式：權重，以及 run 的訓練狀態
├── inference.py     # Enhancer
├── pretrained.py    # 公開的權重：清單、快取、SHA-256 核對
├── evaluate.py      # 測試集交給 kudio.score_dataset，產生結果表
├── pipeline.py      # 三個階段
└── cli.py           # kudioenhance
```

模型建構函式回傳單純的 `torch.nn.Module`；`train.py` 負責優化器、損失函數與迴圈。這樣架構可以被檢視，而不必綁定某種優化設定。

## 測試檢查什麼

遮罩測試檢查的是*公式*，而不只是能不能跑：全是語音時為 1、全是噪音時為 0、功率相等時為 1/√2，而 oracle 遮罩必須真的提升 SI-SDR。這些不成立，就代表目標是錯的，不管損失曲線怎麼說。架構測試也一樣：改動 *t* 之後的任何一幀，`crn` 到 *t* 為止的輸出必須逐位元不變；`unet` 的則必須改變，因為它的說明寫著它是非因果的。`gtcrn` 也用同樣的測試確認不往後看，它的權重逐層數、要等於論文的 23,669，乘加次數逐幀數。訓練集也是：記憶體模式必須逐位元交回堆疊矩陣的列，依 manifest 順序的串流掃描也是，而一個打亂的 epoch 必須讓每個樣本恰好出現一次，並且在相同 seed 下得到相同順序。兩種模式必須以逐位元相同的統計量做正規化。
