"""Architectural boundary enforcement test: tools/ must never import llm/."""

import os
import re
import pytest

def test_tools_never_import_llm():
    """Deterministic layer guardrail: tools/ must never import llm/."""
    tools_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "src", "tools"
    )

    violations = []
    for root, _, files in os.walk(tools_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as file_content:
                    content = file_content.read()
                    if re.search(r"from\s+src\.llm|import\s+src\.llm|google\.genai|gemini", content):
                        violations.append(f"{f} imports LLM components!")

    assert len(violations) == 0, f"Architectural boundary violation detected: {violations}"
