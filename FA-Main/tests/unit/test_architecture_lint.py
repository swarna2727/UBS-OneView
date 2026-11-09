"""Architectural boundary enforcement test: tools/ must never import llm/."""

import os
import re
import pytest

def test_tools_never_import_llm():
    """Deterministic layer guardrail: tools/ must never import llm/. The MCP server, client and gateway that expose
    those tools (src/mcp_tools/) are held to the same rule."""
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")

    violations = []
    for root, _, files in (w for d in ("tools", "mcp_tools") for w in os.walk(os.path.join(src_dir, d))):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as file_content:
                    content = file_content.read()
                    if re.search(r"from\s+src\.llm|import\s+src\.llm|openrouter|httpx", content):
                        violations.append(f"{f} imports LLM components!")

    assert len(violations) == 0, f"Architectural boundary violation detected: {violations}"
