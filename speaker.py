"""Shared speaker-embedding helpers (WeSpeaker CN-Celeb ResNet34, ONNX, CPU)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cnceleb_resnet34.onnx"
VOICEPRINTS_PATH = ROOT / "voiceprints.json"
SAMPLE_RATE = 16000

_session: ort.InferenceSession | None = None


def _ffmpeg() -> str:
    # Prefer the user's own install. When launched by the Hermes desktop app (an MSIX package), PATH
    # puts Hermes's bundled ffmpeg under C:\Program Files\WindowsApps first, which other processes
    # cannot execute (PermissionError / access denied) — so skip anything under WindowsApps.
    own = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if own.exists():
        return str(own)
    for d in os.environ.get("PATH", "").split(os.pathsep):
        exe = Path(d) / "ffmpeg.exe"
        if "windowsapps" not in d.lower() and exe.exists():
            return str(exe)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("ffmpeg not found")


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any format ffmpeg understands (wav/aac/m4a/webm/opus...) to 16 kHz mono float32."""
    cmd = [_ffmpeg(), "-v", "error", "-nostdin", "-i", str(path),
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)  # kept at int16 scale (WeSpeaker convention)


def fbank(samples: np.ndarray) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hamming"
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(SAMPLE_RATE, samples.tolist())
    fb.input_finished()
    feats = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)
    return feats - feats.mean(axis=0, keepdims=True)  # CMN


def embed(path: str | Path) -> np.ndarray:
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    samples = load_audio(path)
    if len(samples) < SAMPLE_RATE // 2:
        raise ValueError("audio too short (< 0.5 s)")
    emb = _session.run(None, {"feats": fbank(samples)[None]})[0][0]
    return emb / np.linalg.norm(emb)


def load_voiceprints() -> dict[str, np.ndarray]:
    """name -> (k, D) array: one or more centroids per person (e.g. phone mic and laptop mic)."""
    if not VOICEPRINTS_PATH.exists():
        return {}
    data = json.loads(VOICEPRINTS_PATH.read_text(encoding="utf-8"))
    return {name: np.atleast_2d(np.asarray(vec, dtype=np.float32)) for name, vec in data.items()}


def save_voiceprints(prints: dict[str, np.ndarray]) -> None:
    data = {name: [[round(float(x), 6) for x in row] for row in np.atleast_2d(vecs)]
            for name, vecs in prints.items()}
    VOICEPRINTS_PATH.write_text(json.dumps(data), encoding="utf-8")


def score(emb: np.ndarray, vecs: np.ndarray) -> float:
    """Best cosine similarity against any of a person's centroids."""
    return float(np.max(np.atleast_2d(vecs) @ emb))
