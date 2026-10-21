"""Pytest fixtures for UBS advisory test suite."""

import os
import json
import tempfile
import pytest

# Keep audit run logs from tests out of data/runs.
os.environ.setdefault("RUN_LOG_DIR", tempfile.mkdtemp(prefix="ubs_runlog_"))
from src.llm.fake import FakeAdvisoryLLMClient

@pytest.fixture
def fake_llm():
    return FakeAdvisoryLLMClient()

@pytest.fixture
def sample_profiles():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "samples", "client_profiles.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def balanced_client_data(sample_profiles):
    return sample_profiles["balanced_client"]

@pytest.fixture
def contradictory_client_data(sample_profiles):
    return sample_profiles["contradictory_client"]

@pytest.fixture
def incomplete_client_data(sample_profiles):
    return sample_profiles["incomplete_client"]
