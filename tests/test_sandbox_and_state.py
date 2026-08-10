"""
Tests for run isolation, sandbox path handling, and state durability.
"""

import json

import pytest


# --------------------------------------------------------------------------
# Run isolation
# --------------------------------------------------------------------------

def test_reset_run_id_produces_a_new_run(tmp_path, monkeypatch):
    """
    reset_run_id() existed but was never called, so every run after the first
    reused the previous run_id and wrote into the same folders.
    """
    import pipeline_state

    monkeypatch.setattr(pipeline_state, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(pipeline_state, "MODIFIED_DATASETS_ROOT", tmp_path / "modified")
    monkeypatch.setattr(pipeline_state, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline_state, "_run_dir_cache", None)

    first = pipeline_state._get_or_create_run_id()
    assert pipeline_state._get_or_create_run_id() == first   # stable within a run

    monkeypatch.setattr(pipeline_state, "_run_dir_cache", None)
    second = pipeline_state.reset_run_id()

    assert second != first
    assert pipeline_state.load_state()["run_id"] == second


def test_orchestrator_starts_a_fresh_run():
    """The fix is only real if the orchestrator actually calls it."""
    import inspect

    import master_orchestrator.agent as orch

    source = inspect.getsource(orch.run_pipeline)
    assert "reset_run_id()" in source


# --------------------------------------------------------------------------
# Single sandbox module instance
# --------------------------------------------------------------------------

def test_sandbox_executor_is_a_single_module_instance():
    """
    Importing the executor as both `sandbox_executor` and
    `data_preprocessing_agent.sandbox_executor` created two module objects with
    independent _container/_started globals — so one phase could believe the
    sandbox was stopped while another still held it.
    """
    from data_preprocessing_agent import sandbox_executor as pkg_mod
    import machine_learning_agent.agent as ml

    assert ml.run_in_sandbox.__module__ == pkg_mod.__name__


def test_report_generator_has_exactly_one_parent():
    """
    report_generator_agent was a sub-agent of the Phase 3 SequentialAgent AND the
    root agent of a second Runner in Phase 4. Assigning sub_agents sets
    parent_agent, so it ended up with two — which ADK does not support, and the
    Phase 4 run was duplicated work regardless.
    """
    import data_preprocessing_agent.agent as prep
    import master_orchestrator.agent as orch

    # It still belongs to the preprocessing pipeline...
    names = [a.name for a in prep.data_preprocessing_agent.sub_agents]
    assert "report_generator_agent" in names

    # ...and the orchestrator no longer imports or re-runs it.
    assert not hasattr(orch, "report_generator_agent")


def test_orchestrator_has_no_duplicate_report_phase():
    import inspect

    import master_orchestrator.agent as orch

    source = inspect.getsource(orch.run_pipeline)
    assert "phase4_runner" not in source
    assert "SESSION_PHASE4" not in source


# --------------------------------------------------------------------------
# Docker path comparison
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("C:\\Users\\me\\run", "C:\\Users\\me\\run", True),
        ("C:\\Users\\me\\run", "C:/Users/me/run", True),
        ("/host_mnt/c/Users/me/run", "C:\\Users\\me\\run", True),
        ("/run/desktop/mnt/host/c/Users/me/run", "C:\\Users\\me\\run", True),
        ("C:\\Users\\me\\run_OLD", "C:\\Users\\me\\run", False),
        ("", "C:\\Users\\me\\run", False),
    ],
)
def test_workspace_mount_comparison(a, b, expected):
    """
    A crashed run leaves a container bound to the previous workspace. Detecting
    that requires comparing the mount Docker reports against the path we want —
    across the several spellings Docker Desktop uses for the same directory.
    """
    from data_preprocessing_agent.sandbox_executor import _same_path

    assert _same_path(a, b) is expected


def test_docker_error_detection_is_not_overbroad():
    """
    Matching a bare "docker" substring reported every unrelated failure as
    "Docker Desktop is not running", sending users off fixing the wrong thing.
    """
    from data_preprocessing_agent.sandbox_executor import _is_docker_daemon_error

    assert _is_docker_daemon_error(Exception("Error while fetching server API version"))
    assert _is_docker_daemon_error(ConnectionError("connection refused"))

    assert not _is_docker_daemon_error(ValueError("invalid docker image tag"))
    assert not _is_docker_daemon_error(KeyError("dockerfile"))


def test_exec_timeout_is_configured():
    """An exec with no ceiling hangs until the phase timeout and survives it."""
    from data_preprocessing_agent import sandbox_executor as sbx

    assert sbx.DEFAULT_EXEC_TIMEOUT_S > 0
    assert sbx.CONTAINER_MEM_LIMIT
    assert sbx.CONTAINER_CPUS > 0
    assert hasattr(sbx, "SandboxTimeout")


def test_image_provides_pkill_for_timeout_kills():
    """
    The timeout path runs `pkill` inside the container to stop the overrun
    process. pkill ships in procps, which python:3.12-slim omits — without it
    the kill silently no-ops and a runaway job keeps burning the container's
    CPU and memory after the timeout returns.
    """
    from pathlib import Path

    dockerfile = Path(__file__).resolve().parents[1] / "docker" / "Dockerfile"
    assert "procps" in dockerfile.read_text(encoding="utf-8"), (
        "docker/Dockerfile must install procps so pkill exists in the sandbox image"
    )


def test_failed_timeout_kill_is_reported_not_swallowed():
    """A silent `except: pass` here hides a surviving runaway process."""
    import inspect

    from data_preprocessing_agent import sandbox_executor as sbx

    src = inspect.getsource(sbx._exec_in_container)
    kill_block = src.split("if worker.is_alive():", 1)[1]
    assert "exit_code" in kill_block, "pkill result must be checked"
    assert "SANDBOX WARNING" in kill_block, "a failed kill must surface to the user"


# --------------------------------------------------------------------------
# State durability
# --------------------------------------------------------------------------

def test_corrupt_state_is_backed_up_not_fatal(tmp_path, monkeypatch):
    import pipeline_state

    state_file = tmp_path / "state.json"
    state_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(pipeline_state, "STATE_FILE", state_file)

    assert pipeline_state.load_state() == {}
    assert state_file.with_suffix(".corrupt.json").exists()


def test_checkpoint_timestamps_are_timezone_aware(tmp_path, monkeypatch):
    import pipeline_state

    monkeypatch.setattr(pipeline_state, "STATE_FILE", tmp_path / "state.json")
    pipeline_state.save_state({})
    pipeline_state.mark_checkpoint("phase_x")

    stamp = pipeline_state.get_all_checkpoints()["phase_x"]
    assert stamp.endswith("+00:00") or stamp.endswith("Z")
