"""Append-only run log for after-the-fact audit (AS-8).

Each advisory session is one run ID. Every event (inputs, agent outputs with model and
prompt versions, sources, rule results, advisor decisions, PDF generation) is appended as
one JSON line to <run_dir>/<run_id>.jsonl.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

DEFAULT_RUN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "runs"
)

def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value

def file_fingerprint(path: str) -> str:
    """Short content hash used to version configuration files such as the suitability policy."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]

class RunLog:
    def __init__(self, run_dir: Optional[str] = None):
        self.run_dir = run_dir or os.environ.get("RUN_LOG_DIR") or DEFAULT_RUN_DIR
        os.makedirs(self.run_dir, exist_ok=True)

    def _path(self, run_id: str) -> str:
        safe = "".join(c for c in run_id if c.isalnum() or c in "-_")
        return os.path.join(self.run_dir, f"{safe}.jsonl")

    def record(self, run_id: str, event: str, **payload: Any) -> None:
        entry = {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **_jsonable(payload)}
        with open(self._path(run_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def read(self, run_id: str) -> List[Dict[str, Any]]:
        path = self._path(run_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
