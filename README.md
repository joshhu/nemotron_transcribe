# nemotron_transcribe

替 [Hermes Agent](https://hermes-agent.nousresearch.com/) 桌面版加上**語者辨識**：按下麥克風說話後，
輸入欄照常顯示你說的話，Hermes 的回答會以你的名字開頭，例如「Josh你好，台北市今天天氣……」。

## 運作方式

```
按麥克風說話 → identify.py
                ├─ 聲紋比對（WeSpeaker）→ 是誰 → 寫入 speaker_state.json
                └─ 語音轉文字（Nemotron 3.5 Transcribe）→ OpenCC 轉台灣繁體 → 輸入欄
送出訊息   → speaker-greeting plugin
                ├─ pre_llm_call：附註說話者給模型（使用者看不到）
                └─ transform_llm_output：回答開頭沒有「名字你好，」就自動補上
```

- **誰在說話：** [WeSpeaker CN-Celeb ResNet34](https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34)
  聲紋模型（ONNX，用中文資料訓練，CPU 執行）。每個人可以有多組聲紋，例如手機錄音一組、筆電麥克風一組，
  比對時取分數最高的那組。
- **說了什麼：** [NVIDIA Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)，
  由 [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) 執行（Windows ARM64 + CUDA 原生版），
  使用 NVFP4 解碼器。輸出用 OpenCC `s2twp` 轉成台灣繁體與台灣用語。
- **接進 Hermes 的兩個地方：**
  - **STT command provider：** `identify.py` 只輸出轉寫文字（也就是輸入欄的內容），說話者另外寫進
    `speaker_state.json`。
  - **`speaker-greeting` plugin**（`hermes_plugin/`）：只對「錄音後送出、且內容和轉寫相符」的那一則訊息生效，
    打字的訊息不會被加上名字。

在 RTX Spark N1X 上，每次說話（聲紋＋轉寫）約 1.1 秒；轉寫服務約佔 23GB GPU 記憶體。

## 需求

- [uv](https://docs.astral.sh/uv/)、Python 3.13
- ffmpeg（解碼 wav / aac / m4a / webm），建議裝在 `%LOCALAPPDATA%\Programs\ffmpeg`
- NVIDIA Blackwell GPU（NVFP4 解碼器）；其他 GPU 請改用 `AD-IQ4_NL` 檔案
- GPU 記憶體要同時放得下轉寫服務（約 23GB）和 Hermes 用的大語言模型。
  例如 Qwen3.8-27B Q4（16GB）可以，Nemotron 3.5 Super VL 120B（73GB）在 64GB 的 GPU 上放不下。

## 安裝

```powershell
uv sync
New-Item -ItemType Directory -Force models, bin
curl.exe -L -o models/cnceleb_resnet34.onnx https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34/resolve/main/cnceleb_resnet34.onnx

# Nemotron 3.5 Transcribe GGUF（解碼器＋編碼器，由 AtomicChat 用 NeMo-Speech.cpp 轉好）
hf download AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF --local-dir models `
  --include "*Encoder-F16.gguf" --include "*AD-NVFP4.gguf"

# NeMo-Speech.cpp 執行檔（請用旁邊的 .sha256 驗證）
gh release download v0.3.0 -R NVIDIA/NeMo-Speech.cpp -p "nemo-speech-0.3.0-windows-aarch64-cuda.zip*" -D bin
7z x bin/nemo-speech-0.3.0-windows-aarch64-cuda.zip -obin
Rename-Item bin/nemo-speech-0.3.0-windows-aarch64-cuda nemo-speech
```

想用 NVIDIA 官方流程的話，可以下載 BF16 原始檔（約 68GB），再用 NeMo-Speech.cpp 的 `convert_model.py`
自己轉成 GGUF，詳見其 [SALM 文件](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.3.0/docs/salm/models.md)。

## 啟動轉寫服務

```powershell
./start_asr.ps1   # http://127.0.0.1:8090，使用 Hermes 時必須開著
```

服務沒開時，轉寫文字會是空的，但說話者仍會被辨識。

## 登錄聲紋

每個人錄幾段 5～6 秒的話（16kHz 單聲道最好，ffmpeg 讀得到的格式都可以）：

```powershell
ffmpeg -f dshow -i audio="<你的麥克風名稱>" -t 6 -ar 16000 -ac 1 wavs/voices/josh1.wav
```

把一段長錄音切成每 6 秒一段：

```powershell
ffmpeg -i long.aac -f segment -segment_time 6 -segment_start_number 1 -ar 16000 -ac 1 wavs/voices/sandy%02d.wav
```

登錄（平均聲紋存在 `voiceprints.json`）：

```powershell
uv run enroll.py --name Josh wavs/voices/josh1.wav wavs/voices/josh2.wav ...
uv run enroll.py --name Sandy wavs/voices/sandy01.wav ...

# 加一組不同麥克風的聲紋（例如經由 Hermes 錄到的 logs/last.webm），不會覆蓋原本的
uv run enroll.py --name Sandy --add wavs/voices/sandy_laptop1.webm
```

不加 `--add` 時，重新登錄同一個名字會取代原本的聲紋。

測試：

```powershell
uv run identify.py some_recording.wav --verbose   # 每個人的分數會印在 stderr
# 嘿，你好，今天臺北市天氣怎麼樣？
```

每次執行都會把分數、說話者和轉寫寫進 `logs/identify.log`，並保留最後一段錄音 `logs/last.*`，方便除錯。

門檻預設 0.4（`--threshold`）。實測同一人的分數：乾淨錄音約 0.9、經由 Hermes 筆電麥克風約 0.7；
不同人之間都低於 0.1 左右。

## Hermes 設定

在 `%LOCALAPPDATA%\hermes\config.yaml` 加入：

```yaml
stt:
  enabled: true
  provider: speaker-id
  providers:
    speaker-id:
      type: command
      command: "C:/path/to/nemotron_transcribe/.venv/Scripts/python.exe C:/path/to/nemotron_transcribe/identify.py {input_path}"
      timeout: 60
tts:
  provider: edge
  edge:
    voice: zh-TW-HsiaoChenNeural
voice:
  auto_tts: true   # 用語音唸出回答
plugins:
  enabled:
    - speaker-greeting
```

安裝 plugin：

```powershell
Copy-Item -Recurse hermes_plugin/speaker-greeting $env:LOCALAPPDATA/hermes/plugins/
```

改完設定或 plugin 後，要把 Hermes 完全關掉再重開。

plugin 預設讀取本專案資料夾的 `speaker_state.json`，可用環境變數 `SPEAKER_STATE_PATH` 改路徑。
每段錄音只會用在一則訊息上，而且要在錄音後 180 秒內送出。

**注意：**

- plugin 為了保證開頭一定有「名字你好，」，使用了 `hold_stream`，所以 Hermes 的回答不會逐字顯示，
  會等整段寫完才一次出現（打字聊天也一樣）。
- Hermes 桌面版是 MSIX（Windows Store）套件，它把內附的 ffmpeg 放在 `WindowsApps` 資料夾並排在 PATH 最前面，
  其他程式無法執行它。所以 `speaker.py` 會優先使用 `%LOCALAPPDATA%\Programs\ffmpeg`，並跳過
  `WindowsApps` 裡的 ffmpeg。

## 隱私

錄音（`wavs/`、`logs/`）和聲紋（`voiceprints.json`）屬於生物特徵資料，都已列在 `.gitignore`，
只留在本機，不會上傳。
