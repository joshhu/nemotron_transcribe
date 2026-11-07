"""speaker-greeting — reply to a voice message with "<名字>你好，…", without the name in the input box.

identify.py (the Hermes STT command) writes speaker_state.json after each voice recording.

- ``pre_llm_call``: if that recording is fresh and unused, remember the speaker for this turn and add
  a short hidden context note so the model knows who is talking.
- ``transform_llm_output`` (``hold_stream``): guarantee the final reply starts with "<名字>你好，"
  — the model often drops the greeting after tool calls (e.g. a weather lookup).

Each recording is used once, so later typed messages are not tagged. Debug log:
logs/speaker-greeting.log next to speaker_state.json.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_PATH = Path(os.environ.get(
    "SPEAKER_STATE_PATH", r"C:\Users\joshhu\nemotron_transcribe\speaker_state.json"))
LOG_PATH = STATE_PATH.parent / "logs" / "speaker-greeting.log"
MAX_AGE_SECONDS = 180  # recording -> (optional edit in the input box) -> send

_lock = threading.Lock()
_last_used: float = 0.0
_turn_speaker: dict[str, str] = {}  # turn_id -> speaker name ("未知" = unrecognized)


def _log(msg: str) -> None:
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    except OSError:
        pass


def _greeting(speaker: str) -> str:
    return "你好，" if speaker == "未知" else f"{speaker}你好，"


def _on_pre_llm_call(turn_id: str = "", **kwargs):
    global _last_used
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stamp = float(state.get("time", 0))
    with _lock:
        if stamp <= _last_used or time.time() - stamp > MAX_AGE_SECONDS:
            return None
        _last_used = stamp
        speaker = str(state.get("speaker") or "未知")
        _turn_speaker[str(turn_id)] = speaker
    _log(f"pre_llm_call turn={turn_id} speaker={speaker}")
    if speaker == "未知":
        return {"context": "（系統附註，不要提到這段附註：這則訊息是語音輸入，聲紋辨識不出說話者是誰，不要編造名字。）"}
    return {"context": f"（系統附註，不要提到這段附註：這則訊息是語音輸入，聲紋辨識的說話者是 {speaker}。"
                       f"回覆請以「{speaker}你好，」開頭。）"}


def _on_transform_llm_output(response_text: str = "", turn_id: str = "", **kwargs):
    with _lock:
        speaker = _turn_speaker.pop(str(turn_id), None)
    if speaker is None or not response_text:
        return None
    greeting = _greeting(speaker)
    head = response_text.lstrip()
    # Already greeted ("Josh你好", "Josh 你好", "嗨 Josh", ...): leave it alone.
    if speaker != "未知" and re.match(rf"\W*(嗨|哈囉|hi|hello)?\W*{re.escape(speaker)}\s*(你好|您好)?", head, re.I):
        _log(f"transform turn={turn_id}: already greeted")
        return None
    if speaker == "未知" and head.startswith(("你好", "您好")):
        return None
    _log(f"transform turn={turn_id}: prepended {greeting}")
    return greeting + head


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output, hold_stream=True)
    _log("plugin loaded")
