# nemotron_transcribe

Speaker identification for the [Hermes Agent](https://hermes-agent.nousresearch.com/) desktop app.
When you press the mic and speak, your words appear in the input box as usual, and Hermes's reply
starts with your name ("Josh你好，台北市今天天氣…").

It plugs into Hermes in two places:

- **STT command provider:** `identify.py` receives the recording, prints only the transcript (that is
  what appears in the input box), and records who spoke in `speaker_state.json`.
- **`speaker-greeting` plugin** (`hermes_plugin/`): a `pre_llm_call` hook that, for the turn right after a
  recording, adds a hidden context note naming the speaker, so the reply starts with "<name>你好，".

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

If it is not running, the transcript is empty (the speaker is still recorded).

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
# 嘿，你好，今天臺北市天氣怎麼樣？
```

Each call appends the scores, speaker and transcript to `logs/identify.log` and keeps the last
recording as `logs/last.*`.

The default threshold is 0.4 (`--threshold`). In testing, same-speaker clips scored ~0.9 (clean) or ~0.7
(through the Hermes desktop mic), and different speakers stayed below ~0.1.

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

Install the plugin and enable it in the same `config.yaml`:

```powershell
Copy-Item -Recurse hermes_plugin/speaker-greeting $env:LOCALAPPDATA/hermes/plugins/
```

```yaml
plugins:
  enabled:
    - speaker-greeting
```

The plugin reads `speaker_state.json` from this project folder; set the `SPEAKER_STATE_PATH` env var
to point elsewhere. Each recording is used for one turn only, and only within 180 s.

Restart Hermes after changing the config or the plugin.

The Hermes desktop app (an MSIX package) puts its bundled ffmpeg under `WindowsApps` first on PATH,
and other processes cannot execute it. `speaker.py` therefore prefers `%LOCALAPPDATA%\Programs\ffmpeg`
and skips anything under `WindowsApps`.

## Privacy

Voice recordings (`wavs/`) and voiceprints (`voiceprints.json`) are biometric data; they are
git-ignored and stay on the local machine.
