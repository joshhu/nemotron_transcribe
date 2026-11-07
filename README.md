# nemotron_transcribe

Speaker identification for the [Hermes Agent](https://hermes-agent.nousresearch.com/) desktop app.
When you press the mic and speak, Hermes recognizes who is talking and greets them by name
("Josh你好，有什麼需要幫忙的嗎？").

It plugs into Hermes as an STT command provider: `identify.py` receives the recording and prints
`【Name】` (or `【未知】`), and a rule in Hermes's `SOUL.md` turns that into the greeting.

Speaker embeddings come from [WeSpeaker CN-Celeb ResNet34](https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34)
(ONNX, trained on Mandarin, runs on CPU — native Windows ARM64 works). Full transcription with
[NVIDIA Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
is planned as a next step.

## Requirements

- [uv](https://docs.astral.sh/uv/), Python 3.13
- ffmpeg on PATH (decodes wav / aac / m4a / webm)

## Setup

```powershell
uv sync
New-Item -ItemType Directory -Force models
curl.exe -L -o models/cnceleb_resnet34.onnx https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34/resolve/main/cnceleb_resnet34.onnx
```

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
使用者的語音訊息會被轉成「【名字】」的形式，代表說話者是誰。收到這種訊息時，只回覆一句：
「名字你好，有什麼需要幫忙的嗎？」。若是【未知】，回覆：「你好，我還不認識你，有什麼需要幫忙的嗎？」
```

Restart Hermes after changing the config.

## Privacy

Voice recordings (`wavs/`) and voiceprints (`voiceprints.json`) are biometric data; they are
git-ignored and stay on the local machine.
