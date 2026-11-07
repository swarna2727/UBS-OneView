"""Transcript parsing and tracing tools for the Brief & Record Agent (UC3, deck slides 14 and 23). Deterministic: no model calls."""

import hashlib
import re
from typing import Dict, List, Tuple

from src.schemas.tools import TranscriptLine

_SPEAKER = re.compile(r"^\s*([A-Za-z][A-Za-z .'-]{0,40}?)\s*[:>-]\s+(.*\S)\s*$")
_AMOUNT = re.compile(r"(?:S\$|US\$|\$|SGD|USD|HKD)\s?\d[\d,]*(?:\.\d+)?\s?[kKmM]?\b|\b\d[\d,]*(?:\.\d+)?\s?(?:k|K|m|M|percent|%)\b")

def transcript_parser(text: str) -> List[TranscriptLine]:
    """Split a transcript into numbered lines with speaker labels. Unlabelled lines keep the previous speaker."""
    lines: List[TranscriptLine] = []
    speaker = "Unknown"
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        m = _SPEAKER.match(raw)
        if m:
            speaker, body = m.group(1).strip(), m.group(2).strip()
        else:
            body = raw.strip()
        lines.append(TranscriptLine(line_no=len(lines) + 1, speaker=speaker, text=body))
    return lines

def transcript_fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def amounts_in(text: str) -> List[str]:
    """Money and percentage figures mentioned in a piece of text, normalised for comparison."""
    return [re.sub(r"[\s,]", "", m.group(0)).lower() for m in _AMOUNT.finditer(text or "")]

def lines_support_amounts(claim: str, cited: List[TranscriptLine]) -> List[str]:
    """Figures in a claim that no cited transcript line contains: the 'unclear amount' stress case."""
    available = {a for line in cited for a in amounts_in(line.text)}
    return [a for a in amounts_in(claim) if a not in available]

HONORIFICS = {"mr", "mrs", "ms", "mdm", "madam", "miss", "dr"}

def transcript_client_match(lines: List[TranscriptLine], client_name: str) -> Tuple[bool, str]:
    """Does the transcript belong to this client? The client must speak, or at least be named (slide 28: changes tied to
    the correct client). A transcript that never names the client is flagged for the RM, never guessed."""
    names = [w for w in (t.strip(".,").lower() for t in client_name.split()) if len(w) > 1 and w not in HONORIFICS]
    spoke = [l for l in lines if set(l.speaker.lower().replace(".", " ").split()) & set(names)]
    if spoke:
        return True, f"{client_name} speaks in {len(spoke)} transcript line(s)."
    if any(re.search(rf"\b{re.escape(n)}\b", l.text.lower()) for l in lines for n in names):
        return True, f"{client_name} is named in the transcript."
    return False, f"{client_name} is not named in the transcript. Check it is this client's meeting before confirming."

def index_lines(lines: List[TranscriptLine]) -> Dict[int, TranscriptLine]:
    return {l.line_no: l for l in lines}
