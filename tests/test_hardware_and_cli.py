"""
Tests for hardware profiling, budget derivation, and the CLI (M1 / D6).

The budget tests matter most: a profile that only informs the planner's prompt
gets ignored, so these pin that VRAM actually changes the limits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt2ml.core import credentials, hardware
from prompt2ml.core.contracts import GPUInfo, HardwareProfile

# ---------------------------------------------------------------------------
# Budget derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "vram,expect_finetune,expect_lora",
    [
        (0, False, True),     # CPU-only
        (4, False, True),     # small laptop GPU
        (8, True, False),     # RTX 4060
        (16, True, False),    # RTX 4080
        (80, True, False),    # A100/H100
    ],
)
def test_budget_tightens_as_vram_shrinks(vram, expect_finetune, expect_lora):
    budget = hardware.derive_budget(vram, ram_gb=32, has_gpu=vram > 0)
    assert budget.allow_full_finetune is expect_finetune
    assert budget.prefer_lora is expect_lora


def test_budget_is_monotonic_in_vram():
    """More VRAM must never yield a smaller allowance."""
    sizes = [
        hardware.derive_budget(v, 32, has_gpu=True).max_params_m
        for v in (4, 8, 16, 40)
    ]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) > 1, "the budget must actually vary with hardware"


def test_cpu_only_never_allows_a_full_finetune():
    budget = hardware.derive_budget(vram_gb=0, ram_gb=64, has_gpu=False)
    assert budget.allow_full_finetune is False
    assert budget.precision == "fp32", "no GPU means no half precision"


def test_gpu_selection_picks_the_largest_vram():
    gpus = [
        GPUInfo(name="GTX 1650", vram_gb=4),
        GPUInfo(name="RTX 4090", vram_gb=24),
        GPUInfo(name="RTX 3060", vram_gb=12),
    ]
    assert hardware.select_gpu(gpus) == 1


def test_gpu_selection_with_no_gpus_is_none():
    assert hardware.select_gpu([]) is None


# ---------------------------------------------------------------------------
# Degradation — doctor must run when everything is broken
# ---------------------------------------------------------------------------

def test_probes_degrade_instead_of_raising(monkeypatch):
    """Every probe shells out; a missing binary must not crash the scan."""
    def missing(cmd, *args, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(hardware.subprocess, "run", missing)
    assert hardware.detect_gpus() == []
    assert hardware.detect_docker() is False
    assert hardware.detect_sandbox_image() is False


def test_passthrough_is_not_probed_without_docker(monkeypatch):
    """None means 'not probed' and must stay distinct from False."""
    monkeypatch.setattr(hardware, "detect_docker", lambda: False)
    monkeypatch.setattr(hardware, "detect_gpus", lambda: [])
    monkeypatch.setattr(hardware, "detect_sandbox_image", lambda *a, **k: False)

    hw = hardware.profile(probe_gpu_passthrough=True)
    assert hw.docker_gpu_passthrough is None


def test_profile_always_produces_a_budget(monkeypatch):
    monkeypatch.setattr(hardware, "detect_gpus", lambda: [])
    monkeypatch.setattr(hardware, "detect_docker", lambda: False)
    monkeypatch.setattr(hardware, "detect_sandbox_image", lambda *a, **k: False)

    hw = hardware.profile(probe_gpu_passthrough=False)
    assert hw.budget is not None
    assert hw.budget.max_train_minutes > 0


# ---------------------------------------------------------------------------
# Backend recommendation
# ---------------------------------------------------------------------------

def test_working_passthrough_recommends_the_sandbox():
    hw = HardwareProfile(
        gpus=[GPUInfo(name="RTX 4060", vram_gb=8)], selected_gpu=0,
        docker_available=True, docker_gpu_passthrough=True,
    )
    assert hardware.recommended_backend(hw) == "LocalDockerGPU"


def test_broken_passthrough_flags_the_isolation_tradeoff():
    """The consent path must never be recommended without naming what is lost."""
    hw = HardwareProfile(
        gpus=[GPUInfo(name="RTX 4060", vram_gb=8)], selected_gpu=0,
        docker_available=True, docker_gpu_passthrough=False,
    )
    rec = hardware.recommended_backend(hw)
    assert "LocalVenv" in rec
    assert "consent" in rec and "isolation" in rec


def test_no_docker_is_reported_as_no_backend():
    hw = HardwareProfile(docker_available=False)
    assert "Docker unavailable" in hardware.recommended_backend(hw)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def test_env_file_does_not_override_real_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=from-file\nTAVILY_API_KEY=tv-123\n")

    credentials.load_env_file(env)

    import os
    assert os.environ["GOOGLE_API_KEY"] == "from-shell"
    assert os.environ["TAVILY_API_KEY"] == "tv-123"


def test_credential_aliases_are_honoured(monkeypatch):
    """HF_TOKEN and HUGGING_FACE_TOKEN both have to work."""
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-abc123456")

    hf = next(c for c in credentials.CREDENTIALS if c.env_var == "HUGGING_FACE_TOKEN")
    assert hf.is_set
    assert "hf-a" in hf.masked()


def test_secrets_are_masked_not_printed(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyVERYSECRETVALUE")
    cred = next(c for c in credentials.CREDENTIALS if c.env_var == "GOOGLE_API_KEY")
    assert "VERYSECRET" not in cred.masked()


def test_optional_cloud_keys_do_not_block_a_run(monkeypatch):
    for cred in credentials.CREDENTIALS:
        if cred.required:
            monkeypatch.setenv(cred.env_var, "set")
        else:
            monkeypatch.delenv(cred.env_var, raising=False)
            for alias in cred.aliases:
                monkeypatch.delenv(alias, raising=False)
    assert credentials.missing_required() == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_every_command_is_reachable():
    from prompt2ml.cli.app import build_parser

    parser = build_parser()
    for command in ("init", "doctor", "run", "status", "clean"):
        args = parser.parse_args([command] + (["goal"] if command == "run" else []))
        assert hasattr(args, "func")


def test_run_creates_a_run_directory(tmp_path: Path, monkeypatch, capsys):
    from prompt2ml.cli.app import main
    from prompt2ml.core.runstore import RunStore

    monkeypatch.setattr(hardware, "detect_gpus", lambda: [])
    monkeypatch.setattr(hardware, "detect_docker", lambda: False)
    monkeypatch.setattr(hardware, "detect_sandbox_image", lambda *a, **k: False)

    code = main([
        "--runs-root", str(tmp_path), "run", "predict churn", "--fast", "--force",
    ])
    assert code == 0

    runs = RunStore.list_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].meta().goal == "predict churn"
    assert runs[0].exists(HardwareProfile), "the run must record the hardware it saw"


def test_run_refuses_without_credentials(tmp_path: Path, monkeypatch):
    """Failing at the start beats failing three phases in."""
    from prompt2ml.cli.app import main

    for cred in credentials.CREDENTIALS:
        monkeypatch.delenv(cred.env_var, raising=False)
        for alias in cred.aliases:
            monkeypatch.delenv(alias, raising=False)
    monkeypatch.setattr(credentials, "load_env_file", lambda path: 0)

    from prompt2ml.core.runstore import RunStore

    assert main(["--runs-root", str(tmp_path), "run", "x", "--fast"]) == 1
    assert RunStore.list_runs(tmp_path) == [], "a refused run must not leave a directory"


def test_status_on_an_empty_root_is_not_an_error(tmp_path: Path):
    from prompt2ml.cli.app import main

    assert main(["--runs-root", str(tmp_path), "status"]) == 0
