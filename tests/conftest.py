"""
Shared fixtures for the Prompt2ML test suite.

The preprocessing tools read and write a module-level state file and resolve
their output paths through pipeline_state.get_run_dir(). Both are redirected
into a per-test temporary directory so tests never touch a real run.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """
    Point pipeline_state at a temp state file and run directory.

    Yields the pipeline_state module so tests can assert on what the tools wrote.
    """
    import pipeline_state

    state_file = tmp_path / "pipeline_state.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setattr(pipeline_state, "STATE_FILE", state_file)
    monkeypatch.setattr(pipeline_state, "MODIFIED_DATASETS_ROOT", tmp_path / "modified")
    monkeypatch.setattr(pipeline_state, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline_state, "_run_dir_cache", run_dir)
    monkeypatch.setattr(pipeline_state, "get_run_dir", lambda: run_dir)

    # The agent module imported get_run_dir by value, so patch its binding too.
    import data_preprocessing_agent.agent as agent_mod

    monkeypatch.setattr(agent_mod, "get_run_dir", lambda: run_dir)

    pipeline_state.save_state({})
    yield pipeline_state


@pytest.fixture
def agent():
    """The preprocessing agent module under test."""
    import data_preprocessing_agent.agent as agent_mod

    return agent_mod


@pytest.fixture
def csv_factory(tmp_path):
    """Write a DataFrame to a CSV in tmp_path and return its path as a string."""

    def _make(df, name="input.csv"):
        path = tmp_path / name
        df.to_csv(path, index=False)
        return str(path)

    return _make
