"""Unit tests for meeting-recording transcription (UC3): request shape, format checks and fail-safe errors."""

import base64
import json

import httpx
import pytest

from src.llm.openrouter import LLMUnavailableError
from src.tools.transcript_tools import transcript_parser
from src.llm.transcription import MAX_AUDIO_BYTES, transcribe_recording


def transport(reply, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})
    return httpx.MockTransport(handler)


def test_sends_audio_and_returns_parseable_transcript():
    seen = []
    text = transcribe_recording(b"RIFFdata", "meeting.WAV", ["Rachel", "Sarah"], api_key="k", model="m",
                                transport=transport("Rachel: Shall we add S$50k?\nSarah: Yes.", seen))
    content = seen[0]["messages"][0]["content"]
    assert seen[0]["model"] == "m"
    assert content[1]["input_audio"] == {"data": base64.b64encode(b"RIFFdata").decode(), "format": "wav"}
    assert "Rachel, Sarah" in content[0]["text"]
    assert [(l.speaker, l.text) for l in transcript_parser(text)] == [("Rachel", "Shall we add S$50k?"), ("Sarah", "Yes.")]


@pytest.mark.parametrize("name,size,message", [
    ("notes.txt", 1, "Unsupported recording format"),
    ("meeting.mp3", 0, "empty"),
    ("meeting.mp3", MAX_AUDIO_BYTES + 1, "larger than"),
], ids=["format", "empty", "too-large"])
def test_rejects_bad_recordings_before_any_call(name, size, message):
    seen = []
    with pytest.raises(ValueError, match=message):
        transcribe_recording(b"x" * size, name, ["Rachel"], api_key="k", transport=transport("x", seen))
    assert seen == []


@pytest.mark.parametrize("reply", [500, ""])
def test_provider_failure_or_empty_text_is_surfaced(reply):
    with pytest.raises(LLMUnavailableError):
        transcribe_recording(b"x", "meeting.mp3", ["Rachel"], api_key="k", transport=transport(reply, []))
