"""Meeting recording -> transcript for UC3 (Record). Live only: an audio-capable model on OpenRouter.

The output is a draft: the RM reviews and edits it before the Brief & Record Agent runs, and every summary line is
still traced to transcript lines. Offline mode never calls this; the workspace uses the client's demo transcript.
"""

import base64
import os
from typing import List, Optional

import httpx

from src.llm.openrouter import OPENROUTER_URL, LLMUnavailableError

AUDIO_FORMATS = ("mp3", "wav", "m4a", "aac", "ogg", "flac")
DEFAULT_TRANSCRIPTION_MODEL = "google/gemini-2.5-flash"  # override with OPENROUTER_MODEL_TRANSCRIPTION
MAX_AUDIO_BYTES = 20 * 1024 * 1024


def audio_format(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in AUDIO_FORMATS:
        raise ValueError(f"Unsupported recording format '{ext or filename}'. Use one of: {', '.join(AUDIO_FORMATS)}.")
    return ext


def transcription_prompt(speakers: List[str]) -> str:
    return ("Transcribe this client meeting recording verbatim.\n"
            f"Speakers: {', '.join(speakers)}.\n"
            "Write one line per speaker turn as 'Name: words', using the speaker names above.\n"
            "Do not summarise, translate, correct or add anything. Write [inaudible] for parts you cannot hear.\n"
            "Return only the transcript.")


def transcribe_recording(audio: bytes, filename: str, speakers: List[str], api_key: Optional[str] = None,
                         model: Optional[str] = None, timeout_s: float = 180.0,
                         transport: Optional[httpx.BaseTransport] = None) -> str:
    """Transcribe a meeting recording into 'Name: words' lines, the format transcript_parser reads."""
    fmt = audio_format(filename)
    if not audio:
        raise ValueError("The recording is empty.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError(f"The recording is larger than {MAX_AUDIO_BYTES // (1024 * 1024)} MB. Paste the transcript instead.")
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    model = model or os.environ.get("OPENROUTER_MODEL_TRANSCRIPTION", DEFAULT_TRANSCRIPTION_MODEL)
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": transcription_prompt(speakers)},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": fmt}},
        ]}],
    }
    with httpx.Client(timeout=timeout_s, transport=transport) as http:
        try:
            resp = http.post(OPENROUTER_URL, json=payload,
                             headers={"Authorization": f"Bearer {api_key}", "X-Title": "UBS OneView Advisory"})
            resp.raise_for_status()
            text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMUnavailableError(f"Transcription failed: {exc}") from exc
    if not text:
        raise LLMUnavailableError("Transcription returned no text.")
    return text
