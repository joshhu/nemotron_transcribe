"""speaker-greeting — tell the model who just spoke, without putting the name in the input box.

identify.py (the Hermes STT command) writes speaker_state.json after each voice recording. On the
next turn, if that recording is fresh, this pre_llm_call hook adds a short context note naming the
speaker so the reply starts with "<名字>你好，". Each recording is used once, so later typed messages
are not tagged.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_PATH = Path(os.environ.get(
    "SPEAKER_STATE_PATH", r"C:\Users\joshhu\nemotron_transcribe\speaker_state.json"))
MAX_AGE_SECONDS = 180  # recording -> (optional edit in the input box) -> send

_lock = threading.Lock()
_last_used: float = 0.0


def _on_pre_llm_call(**kwargs):
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
    if speaker == "未知":
        note = ("（系統附註，不要提到這段附註：這則訊息是語音輸入，聲紋辨識不出說話者是誰。"
                "回覆請以「你好，」開頭，不要編造名字，接著直接回答。）")
    else:
        note = (f"（系統附註，不要提到這段附註：這則訊息是語音輸入，聲紋辨識的說話者是 {speaker}。"
                f"回覆第一句請以「{speaker}你好，」開頭，接著直接回答他的問題；"
                f"如果他只是打招呼，就回「{speaker}你好，有什麼需要幫忙的嗎？」）")
    logger.info("speaker-greeting: injected speaker=%s", speaker)
    return {"context": note}


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
