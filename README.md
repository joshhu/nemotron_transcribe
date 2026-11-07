# nemotron_transcribe

Speaker identification for the [Hermes Agent](https://hermes-agent.nousresearch.com/) desktop app.
When you press the mic and speak, Hermes recognizes who is talking and greets them by name
("Josh你好，有什麼需要幫忙的嗎？").

It plugs into Hermes as an STT command provider: `identify.py` receives the recording and prints
`【Name】transcript` (or `【未知】transcript`), and a rule in Hermes's `SOUL.md` makes the agent greet the
speaker by name and then answer what they said.

- **Who:** [WeSpeaker CN-Celeb ResNet34](https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34) speaker
  embeddings (ONNX, trained on Mandarin, CPU).
- **What:** [NVIDIA Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
  served by [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) (native Windows ARM64 + CUDA build), NVFP4
  decoder. Output is converted to Taiwan Traditional Chinese with OpenCC `s2twp`.

About 1.1 s per push-to-talk clip on an RTX Spark N1X (speaker ID + transcription); the ASR server uses ~23 GB of
GPU memory.

## Requirements

- [uv](https://docs.astral.sh/uv/), Python 3.13
- ffmpeg on PATH (decodes wav / aac / m4a / webm)
- NVIDIA Blackwell GPU for the NVFP4 decoder (use the `AD-IQ4_NL` file on other GPUs)

## Setup

```powershell
uv sync
New-Item -ItemType Directory -Force models, bin
curl.exe -L -o models/cnceleb_resnet34.onnx https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34/resolve/main/cnceleb_resnet34.onnx

# Nemotron 3.5 Transcribe GGUF (decoder + encoder; pre-converted by AtomicChat with NeMo-Speech.cpp)
hf download AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF --local-dir models `
  --include "*Encoder-F16.gguf" --include "*AD-NVFP4.gguf"

# NeMo-Speech.cpp runtime (verify the .sha256 next to it)
gh release download v0.3.0 -R NVIDIA/NeMo-Speech.cpp -p "nemo-speech-0.3.0-windows-aarch64-cuda.zip*" -D bin
7z x bin/nemo-speech-0.3.0-windows-aarch64-cuda.zip -obin
Rename-Item bin/nemo-speech-0.3.0-windows-aarch64-cuda nemo-speech
```

## Run the ASR server

```powershell
./start_asr.ps1   # http://127.0.0.1:8090, must be running while Hermes is used
```

If it is not running, `identify.py` still prints the speaker name (with no transcript).

## Enroll speakers

Record a few 5–6 second clips per person (16 kHz mono is ideal; any ffmpeg-readable format works):

```powershell
ffmpeg -f dshow -i audio="<your microphone>" -t 6 -ar 16000 -ac 1 wavs/voices/josh1.wav
```

Split one long recording into 6-second clips:

```powershell
ffmpeg -i long.aac -f segment -segment_time 6 -segment_start_number 1 -ar 16000 -ac 1 wavs/voices/sandy%02d.wav
```

Then enroll (stores averaged voiceprints in `voiceprints.json`; re-running a name replaces it):

```powershell
uv run enroll.py --name Josh wavs/voices/josh1.wav wavs/voices/josh2.wav ...
uv run enroll.py --name Sandy wavs/voices/sandy01.wav ...
```

Test:

```powershell
uv run identify.py some_recording.wav --verbose   # prints per-speaker cosine scores to stderr
# 【Sandy】嘿，你好，今天臺北市天氣怎麼樣？
```

The default threshold is 0.5 (`--threshold`). In testing, same-speaker clips scored ~0.9 and
other speakers ~0.

## Hermes configuration

Add to `%LOCALAPPDATA%\hermes\config.yaml`:

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
  auto_tts: true   # speak the reply aloud
```

Append to `%LOCALAPPDATA%\hermes\SOUL.md`:

```
## 語者辨識問候
使用者的語音訊息格式是「【名字】說話內容」，【】裡是系統用聲紋辨識出的說話者（例如【Josh】、【Sandy】），後面是他說的話。
- 回覆一律用說話者的名字開頭：「名字你好，」，接著直接回答或處理他說的內容，用台灣繁體中文。
- 如果【】後面沒有內容，或只是打招呼，就回：「名字你好，有什麼需要幫忙的嗎？」
- 如果是【未知】，用「你好，」開頭（不要編造名字），其餘照常回答。
- 回覆中不要重複【名字】這個標記本身。
```

Restart Hermes after changing the config.

## Privacy

Voice recordings (`wavs/`) and voiceprints (`voiceprints.json`) are biometric data; they are
git-ignored and stay on the local machine.
