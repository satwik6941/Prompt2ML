"""
Hardware profiling and the training budget it implies.

A scan that only *informs* the planner's prompt gets ignored, so the profile
produces a ``TrainingBudget`` that the planner must obey and the codegen gate
enforces (D6).

Everything here degrades: a missing ``nvidia-smi``, absent Docker, or no psutil
yields a reduced profile rather than an exception, because ``prompt2ml doctor``
has to run precisely when the environment is broken.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from .contracts import GPUInfo, HardwareProfile, TrainingBudget

SANDBOX_IMAGE_CPU = "prompt2ml-sandbox:latest"
_PROBE_TIMEOUT_S = 20


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    """Run a command, returning (exit_code, combined output). Never raises."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 127, str(exc)


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------

def detect_cpu_cores() -> int:
    return os.cpu_count() or 1


def detect_ram_gb() -> float:
    """RAM in GB. Uses psutil when present, else per-platform fallbacks."""
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return round(pages * page_size / 1024**3, 1)
        except (ValueError, OSError):
            pass

    if platform.system() == "Windows":
        code, out = _run([
            "wmic", "computersystem", "get", "TotalPhysicalMemory", "/value",
        ])
        if code == 0:
            for line in out.splitlines():
                if "=" in line:
                    try:
                        return round(int(line.split("=", 1)[1].strip()) / 1024**3, 1)
                    except ValueError:
                        pass
    return 0.0


def detect_disk_free_gb(path: Path | str = ".") -> float:
    try:
        return round(shutil.disk_usage(str(path)).free / 1024**3, 1)
    except OSError:
        return 0.0


def detect_gpus() -> list[GPUInfo]:
    """
    Enumerate NVIDIA GPUs via nvidia-smi.

    nvidia-smi is preferred over torch.cuda because it reports the hardware even
    when the installed torch is a CPU-only build — which is the common Windows
    case and exactly the situation the user needs told about.
    """
    code, out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if code != 0:
        return []

    gpus: list[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vram_mb = float(parts[1])
        except ValueError:
            continue
        gpus.append(GPUInfo(
            name=parts[0],
            vram_gb=round(vram_mb / 1024, 1),
            driver=parts[2] if len(parts) > 2 else None,
            cuda=detect_cuda_version(),
        ))
    return gpus


def detect_cuda_version() -> str | None:
    code, out = _run(["nvidia-smi", "--query"], timeout=10)
    if code == 0:
        for line in out.splitlines():
            if "CUDA Version" in line:
                return line.split(":", 1)[-1].strip() or None
    code, out = _run(["nvidia-smi"])
    if code == 0 and "CUDA Version:" in out:
        return out.split("CUDA Version:", 1)[1].split()[0].strip()
    return None


def select_gpu(gpus: list[GPUInfo]) -> int | None:
    """Pick the GPU with the most VRAM — the constraint that actually binds."""
    if not gpus:
        return None
    return max(range(len(gpus)), key=lambda i: gpus[i].vram_gb)


def detect_docker() -> bool:
    code, _ = _run(["docker", "info"], timeout=15)
    return code == 0


def detect_sandbox_image(image: str = SANDBOX_IMAGE_CPU) -> bool:
    code, out = _run(["docker", "image", "inspect", image], timeout=15)
    return code == 0 and out.strip() not in ("", "[]")


def probe_docker_gpu_passthrough(image: str = SANDBOX_IMAGE_CPU) -> bool:
    """
    Actually try it, rather than inferring from the presence of a GPU.

    On Windows this needs the WSL2 backend plus the NVIDIA Container Toolkit,
    and both can be absent while ``nvidia-smi`` works fine on the host — so the
    only trustworthy answer comes from running a throwaway container.
    """
    code, _ = _run(
        ["docker", "run", "--rm", "--gpus", "all", image, "nvidia-smi", "-L"],
        timeout=_PROBE_TIMEOUT_S,
    )
    return code == 0


# ---------------------------------------------------------------------------
# Budget derivation
# ---------------------------------------------------------------------------

def derive_budget(vram_gb: float, ram_gb: float, has_gpu: bool) -> TrainingBudget:
    """
    Map available memory onto limits the planner must respect.

    Deliberately conservative: an over-generous budget produces an OOM twenty
    minutes into training, which costs far more than picking a smaller model.
    """
    if not has_gpu or vram_gb <= 0:
        # CPU-only: keep models small enough to finish, prefer classical ML.
        return TrainingBudget(
            max_params_m=25,
            max_batch_tokens=2048,
            precision="fp32",
            max_train_minutes=20,
            allow_full_finetune=False,
            prefer_lora=True,
            max_image_px=160,
        )
    if vram_gb < 6:
        return TrainingBudget(
            max_params_m=70, max_batch_tokens=2048, precision="fp16",
            max_train_minutes=30, allow_full_finetune=False, prefer_lora=True,
            max_image_px=192,
        )
    if vram_gb < 12:
        return TrainingBudget(
            max_params_m=200, max_batch_tokens=4096, precision="fp16",
            max_train_minutes=45, allow_full_finetune=True, prefer_lora=False,
            max_image_px=224,
        )
    if vram_gb < 24:
        return TrainingBudget(
            max_params_m=800, max_batch_tokens=8192, precision="bf16",
            max_train_minutes=90, allow_full_finetune=True, prefer_lora=False,
            max_image_px=320,
        )
    return TrainingBudget(
        max_params_m=3000, max_batch_tokens=16384, precision="bf16",
        max_train_minutes=180, allow_full_finetune=True, prefer_lora=False,
        max_image_px=384,
    )


# ---------------------------------------------------------------------------
# Full profile
# ---------------------------------------------------------------------------

def profile(probe_gpu_passthrough: bool = True) -> HardwareProfile:
    """
    Build the complete profile.

    ``probe_gpu_passthrough`` launches a throwaway container, so it is skipped
    when Docker is missing, no GPU exists, or the caller asks for a fast scan.
    """
    gpus = detect_gpus()
    selected = select_gpu(gpus)
    docker = detect_docker()
    image_present = detect_sandbox_image() if docker else False

    passthrough: bool | None = None
    if probe_gpu_passthrough and docker and gpus and image_present:
        passthrough = probe_docker_gpu_passthrough()

    vram = gpus[selected].vram_gb if selected is not None else 0.0

    return HardwareProfile(
        cpu_cores=detect_cpu_cores(),
        ram_gb=detect_ram_gb(),
        disk_free_gb=detect_disk_free_gb(),
        platform=f"{platform.system()} {platform.release()}",
        gpus=gpus,
        selected_gpu=selected,
        docker_available=docker,
        docker_gpu_passthrough=passthrough,
        sandbox_image_present=image_present,
        budget=derive_budget(vram, detect_ram_gb(), has_gpu=bool(gpus)),
    )


def recommended_backend(hw: HardwareProfile) -> str:
    """
    Where training should run given the profile alone (D7 preference order).

    The ComputeBroker makes the final call once it also knows the plan's
    resource requirement, residency policy, and cost ceiling — this is the
    local-only view that ``doctor`` reports.
    """
    if hw.gpus and hw.docker_gpu_passthrough:
        return "LocalDockerGPU"
    if hw.gpus and hw.docker_gpu_passthrough is False:
        return "LocalVenv (consent required — no container isolation)"
    if hw.docker_available:
        return "LocalDocker (CPU)"
    return "none — Docker unavailable"
