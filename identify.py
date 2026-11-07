"""Hermes STT command: who spoke + what they said.

Prints 【Name】transcript  (or 【未知】transcript). Speaker ID runs locally (WeSpeaker ONNX);
the transcript comes from a running `nemo-speech serve` (Nemotron 3.5 Transcribe), converted to
Taiwan Traditional Chinese with OpenCC.

identify.py <audio> [--threshold 0.5] [--asr-url http://127.0.0.1:8090] [--verbose]
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.request

from opencc import OpenCC

from speaker import _ffmpeg, embed, load_voiceprints

DEFAULT_THRESHOLD = 0.5
DEFAULT_ASR_URL = os.environ.get("NEMO_SPEECH_URL", "http://127.0.0.1:8090")
INSTRUCTION = (
    "Audio transcription. Language: as spoken (usually Mandarin Chinese, sometimes English). "
    "Write Mandarin in traditional Chinese characters as used in Taiwan, and English in Latin script. "
    "Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. "
    "Formatting: punctuation and capitalization. Output: transcript only."
)
_s2twp = OpenCC("s2twp")


def who(audio: str, threshold: float, verbose: bool, log=None) -> str:
    prints = load_voiceprints()
    if not prints:
        if log:
            log.write("no voiceprints enrolled")
        return "未知"
    try:
        emb = embed(audio)
    except Exception as exc:  # too short / undecodable -> unknown
        if log:
            log.write(f"embed failed: {exc!r}")
        print(f"identify: {exc}", file=sys.stderr)
        return "未知"
    scores = sorted(((float(emb @ vec), name) for name, vec in prints.items()), reverse=True)
    if log:
        log.write("scores: " + ", ".join(f"{n}={s:.3f}" for s, n in scores))
    if verbose:
        for score, name in scores:
            print(f"  {name}: {score:.3f}", file=sys.stderr)
    best_score, best_name = scores[0]
    return best_name if best_score >= threshold else "未知"


def transcribe(audio: str, asr_url: str) -> str:
    wav = subprocess.run([_ffmpeg(), "-v", "error", "-nostdin", "-i", audio, "-ac", "1", "-ar", "16000",
                          "-f", "wav", "-"], capture_output=True, check=True).stdout
    body = {"model": "salm", "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "text", "text": INSTRUCTION},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
    ]}]}
    req = urllib.request.Request(f"{asr_url.rstrip('/')}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = json.load(resp)["choices"][0]["message"]["content"] or ""
    return _s2twp.convert(text.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--asr-url", default=DEFAULT_ASR_URL)
    ap.add_argument("--no-asr", action="store_true", help="speaker only")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    log = _Log(args.audio)
    name = who(args.audio, args.threshold, args.verbose or True, log)
    text = ""
    if not args.no_asr:
        try:
            text = transcribe(args.audio, args.asr_url)
        except Exception as exc:  # ASR server down -> still report the speaker
            log.write(f"ASR failed: {exc!r}")
            print(f"identify: ASR failed: {exc}", file=sys.stderr)
    result = f"【{name}】{text}"
    log.write(f"result: {result}")
    print(result)


class _Log:
    """Keeps a copy of the last recording and appends a line per call to logs/identify.log (debugging)."""

    def __init__(self, audio: str) -> None:
        import datetime
        import shutil
        from pathlib import Path

        self.dir = Path(__file__).resolve().parent / "logs"
        self.dir.mkdir(exist_ok=True)
        self.stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            shutil.copyfile(audio, self.dir / ("last" + Path(audio).suffix))
        except OSError:
            pass
        self.write(f"input: {audio}")

    def write(self, line: str) -> None:
        with open(self.dir / "identify.log", "a", encoding="utf-8") as fh:
            fh.write(f"{self.stamp} {line}\n")


if __name__ == "__main__":
    main()
