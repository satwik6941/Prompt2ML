"""
SafeExecute Executor — Secure code execution for data preprocessing agents.

Replaces OpenSandbox. Uses SafeExecute (https://github.com/Josh-XT/SafeExecute)
which manages Docker containers directly — no separate server process required.

Key difference from OpenSandbox:
  - The host's MODIFIED_DATASETS_DIR is mounted as /workspace inside the container.
  - Files written there by built-in tools are immediately visible to sandbox scripts.
  - No upload/download step needed for dataset files already on disk.

Tool functions exposed to ADK agents:
    start_sandbox()                          — pull image, warm up container
    stop_sandbox()                           — remove container, free resources
    run_in_sandbox(code)                     — execute Python code in container
    write_file_to_sandbox(filename, content) — write a script into /workspace
    read_file_from_sandbox(filename)         — read a file from /workspace

ERROR HANDLING:
  - start_sandbox returns {"error": "...", "error_type": "docker_not_running"} when
    Docker Desktop is not started. Agents MUST check for "error" key before proceeding.
  - run_in_sandbox returns {"success": False, "error_type": "..."} on failure.
  - Agents should degrade gracefully: skip sandbox steps if Docker unavailable,
    then note this in their output rather than crashing the pipeline.
"""

import builtins
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_state import get_run_dir, reset_run_dir_cache

# Resolved at sandbox start time (not import time) so the slug is always
# based on the user_goal that was already saved to pipeline_state.json.
# WORKSPACE is set by start_sandbox() and reused by run_in_sandbox().
WORKSPACE: str = ""

# Stable conversation ID — one container per pipeline run
_CONVERSATION_ID = "prompt2ml_pipeline"

# Module-level flag so start/stop are idempotent
_started = False

# (import_name, pip_install_name) pairs — only packages that are lightweight
# and expected to already be in the SafeExecute image. Heavy optional packages
# (xgboost, lightgbm) are NOT listed here; SafeExecute auto-installs them on
# first use so we don't block startup with a slow pip download.
_REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("openpyxl", "openpyxl"),
    ("xlrd", "xlrd"),
    ("xlsxwriter", "XlsxWriter"),
    ("statsmodels", "statsmodels"),
    ("imblearn", "imbalanced-learn"),
    ("joblib", "joblib"),
]

# PIP_TIMEOUT caps each pip install so a missing/slow package never blocks start_sandbox.
_PIP_TIMEOUT = 60  # seconds per pip install call

_DEPENDENCY_CHECK_SCRIPT = """\
import subprocess, sys, json as _json

required = {required}
pip_timeout = {pip_timeout}

missing = []
for import_name, pip_name in required:
    try:
        __import__(import_name)
    except ImportError:
        missing.append(pip_name)

if missing:
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-script-location"] + missing,
            capture_output=True, text=True,
            timeout=pip_timeout,
        )
        print(_json.dumps({{
            "installed": missing,
            "returncode": r.returncode,
            "errors": r.stderr.strip()[-400:] if r.stderr.strip() else "",
        }}))
    except subprocess.TimeoutExpired:
        print(_json.dumps({{
            "installed": [],
            "warning": f"pip timed out after {{pip_timeout}}s installing {{missing}}",
        }}))
else:
    print(_json.dumps({{"installed": [], "note": "all dependencies already present"}}))
"""

# safeexecute writes temp.py and run_wrapper.sh with open(..., "w") which uses
# \r\n on Windows. run_wrapper.sh then runs inside a Linux Docker container where
# bash sees the \r and fails with "$'\r': command not found" and
# "python: can't open file /workspace/temp.py\r". We fix this by temporarily
# replacing builtins.open so all text-mode writes use LF-only endings.
_original_open = builtins.open


def _lf_open(file, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, closefd=True, opener=None):
    if 'w' in mode and 'b' not in mode:
        newline = '\n'
    return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)


def _safeexecute_call(code: str) -> str:
    """Call execute_python_code with builtins.open patched to force LF line endings."""
    from safeexecute import execute_python_code
    builtins.open = _lf_open
    try:
        return execute_python_code(code=code, working_directory=WORKSPACE)
    finally:
        builtins.open = _original_open


def _ensure_workspace() -> str:
    """Resolve the run-specific workspace and ensure it exists. Returns the path string."""
    run_dir = get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def _is_docker_error(exc: Exception) -> bool:
    """Return True if the exception looks like Docker daemon is not running."""
    msg = str(exc).lower()
    return any(k in msg for k in (
        "pipe", "createfile", "cannot find the file", "docker", "connection refused",
        "fetching server api", "is the docker daemon running",
    ))


async def start_sandbox() -> str:
    """
    Pull the SafeExecute Docker image and warm up the sandbox container.
    The host's modified_datasets/ folder is mounted as /workspace inside the container.
    Call this ONCE before using run_in_sandbox or write_file_to_sandbox.

    Returns:
        JSON with "status": "sandbox_started" on success.
        JSON with "error" and "error_type" on failure — check for "error" key before proceeding.
        If error_type == "docker_not_running": start Docker Desktop and retry.
        If error_type == "import_error": run `pip install safeexecute`.
    """
    global _started, WORKSPACE
    WORKSPACE = _ensure_workspace()   # resolve run-dir now that user_goal is in state

    try:
        from safeexecute import execute_python_code as _  # noqa: F401 — verify import only
    except ImportError:
        msg = "safeexecute package not installed. Run: pip install safeexecute"
        print(f"[SANDBOX ERROR] {msg}", flush=True)
        return json.dumps({
            "error": msg,
            "error_type": "import_error",
            "action": "Run `pip install safeexecute` then retry.",
        })

    try:
        warmup_code = "import sys; print(f'Python {sys.version[:6]} ready')"
        output = _safeexecute_call(warmup_code)
        _started = True
        print(f"[SANDBOX] Started — workspace: {WORKSPACE}", flush=True)
        print(f"[SANDBOX] {output.strip()}", flush=True)

        # --- dependency pre-flight -------------------------------------------
        dep_script = _DEPENDENCY_CHECK_SCRIPT.format(
            required=repr(_REQUIRED_PACKAGES),
            pip_timeout=_PIP_TIMEOUT,
        )
        dep_raw = _safeexecute_call(dep_script).strip()
        # Extract the last JSON line (pip may emit extra lines before ours)
        dep_json: dict = {}
        for line in reversed(dep_raw.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    dep_json = json.loads(line)
                except json.JSONDecodeError:
                    pass
                break

        installed = dep_json.get("installed", [])
        if installed:
            print(f"[SANDBOX] Installed missing deps: {installed}", flush=True)
            if dep_json.get("returncode", 0) != 0:
                print(f"[SANDBOX WARNING] pip errors: {dep_json.get('errors', '')}", flush=True)
        else:
            print(f"[SANDBOX] All dependencies present.", flush=True)
        # ---------------------------------------------------------------------

        return json.dumps({
            "status": "sandbox_started",
            "workspace": WORKSPACE,
            "container_note": f"host {WORKSPACE} is mounted as /workspace inside the container",
            "deps_installed": installed,
        })
    except Exception as e:
        if _is_docker_error(e):
            msg = (
                "Docker Desktop is not running. "
                "Please start Docker Desktop and wait for it to fully initialise, then retry."
            )
            error_type = "docker_not_running"
        else:
            msg = f"Failed to start sandbox: {e}"
            error_type = "sandbox_start_error"

        print(f"[SANDBOX ERROR] {msg}", flush=True)
        return json.dumps({
            "error": msg,
            "error_type": error_type,
            "action": "Start Docker Desktop (system tray) and call start_sandbox again." if error_type == "docker_not_running" else "Check Docker logs.",
        })


async def stop_sandbox() -> str:
    """
    Remove the sandbox container and free Docker resources.
    Call this ONCE after all sandbox work is complete.
    Safe to call even if start_sandbox failed — will be a no-op.

    Returns:
        JSON confirmation.
    """
    global _started

    if not _started:
        # Nothing was running — avoid misleading "container removed" log
        print("[SANDBOX] stop_sandbox called but sandbox was not running — no-op.", flush=True)
        return json.dumps({"status": "sandbox_not_running", "note": "Nothing to stop."})

    try:
        from safeexecute import get_container_manager
        manager = get_container_manager()
        manager.remove_container(_CONVERSATION_ID)
        _started = False
        print("[SANDBOX] Container removed.", flush=True)
        return json.dumps({"status": "sandbox_stopped"})
    except Exception as e:
        _started = False
        note = str(e)
        print(f"[SANDBOX] stop_sandbox: container may already be gone ({note})", flush=True)
        return json.dumps({"status": "sandbox_stopped", "note": note})


async def run_in_sandbox(code: str) -> str:
    code = code.replace('\r\n', '\n').replace('\r', '\n')  # prevent \r from corrupting filenames inside Linux container
    """
    Execute Python code inside the sandbox container.

    The container's /workspace is your host's modified_datasets/ folder.
    Any CSV files already there (from built-in tools) are readable as:
        pd.read_csv('/workspace/<filename>.csv')

    Results saved to /workspace/ are immediately available on the host.

    SafeExecute automatically detects missing imports and installs them via pip.
    Pre-installed: numpy, pandas, scikit-learn, scipy, matplotlib, seaborn,
                   plotly, openpyxl, xlrd, xlsxwriter, statsmodels, imbalanced-learn.

    Args:
        code: Python code to execute. Use /workspace/ paths for file I/O.

    Returns:
        JSON with "success": true and "stdout" on success.
        JSON with "success": false, "error_type", and "stderr" on failure.
        Always check "success" before using the output.
    """
    if not _started:
        return json.dumps({
            "success": False,
            "stdout": "",
            "stderr": "Sandbox is not running.",
            "error_type": "not_started",
            "action": "Call start_sandbox first. If start_sandbox returned an error, check Docker Desktop is running.",
        })

    try:
        output = _safeexecute_call(code)
        return json.dumps({
            "success": True,
            "stdout": output.strip(),
        }, indent=2)
    except Exception as e:
        if _is_docker_error(e):
            error_type = "docker_not_running"
            action = "Start Docker Desktop and call start_sandbox again."
        else:
            error_type = "execution_error"
            action = "Check the code for syntax errors and try again."

        print(f"[SANDBOX ERROR] run_in_sandbox failed ({error_type}): {e}", flush=True)
        return json.dumps({
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "error_type": error_type,
            "action": action,
        })


async def write_file_to_sandbox(filename: str, content: str) -> str:
    """
    Write a Python script or text file into /workspace so run_in_sandbox can execute it.

    Because /workspace is the host's modified_datasets/ folder, this just writes
    the file to disk — no Docker API call needed. Works even if Docker is not running.

    Args:
        filename: Filename only (e.g. 'preprocess.py'). Written to modified_datasets/.
        content: Full text content of the file.

    Returns:
        JSON with the path written.
    """
    workspace = Path(_ensure_workspace())
    dest = workspace / filename
    dest.write_text(content.replace('\r\n', '\n').replace('\r', '\n'), encoding="utf-8", newline='\n')
    print(f"[SANDBOX] Wrote script → {dest}", flush=True)
    return json.dumps({
        "status": "written",
        "host_path": str(dest),
        "sandbox_path": f"/workspace/{filename}",
        "bytes": len(content),
    })


async def read_file_from_sandbox(filename: str) -> str:
    """
    Read a file that was written by a sandbox script to /workspace.

    Because /workspace is the host's modified_datasets/ folder, this just reads
    the file from disk.

    Args:
        filename: Filename only (e.g. 'result.csv'). Read from modified_datasets/.

    Returns:
        File content as string (truncated to 50000 chars if large).
    """
    src = Path(_ensure_workspace()) / filename
    if not src.exists():
        return json.dumps({"error": f"File not found: {src}"})

    content = src.read_text(encoding="utf-8", errors="ignore")
    if len(content) > 50000:
        content = content[:50000] + f"\n\n[TRUNCATED — showing first 50000 of {len(content)} chars]"
    return content


# ---------------------------------------------------------------------------
# Legacy compatibility shims
# With SafeExecute the shared /workspace means no transfer is needed —
# these functions just return the path that's already accessible.
# ---------------------------------------------------------------------------

async def upload_dataset_to_sandbox(local_file_path: str) -> str:
    """
    No-op shim for backward compatibility.
    With SafeExecute, modified_datasets/ IS /workspace, so any file already
    there is immediately accessible. Returns the /workspace path for the file.

    Args:
        local_file_path: Absolute host path to the file.

    Returns:
        JSON with the equivalent /workspace path inside the container.
    """
    local = Path(local_file_path)
    if not local.exists():
        return json.dumps({"error": f"File not found: {local_file_path}"})

    sandbox_path = f"/workspace/{local.name}"
    return json.dumps({
        "status": "available",
        "sandbox_path": sandbox_path,
        "local_path": local_file_path,
        "note": "No upload needed — modified_datasets/ is mounted as /workspace",
    })


async def download_from_sandbox(sandbox_filename: str, local_path: str) -> str:
    """
    No-op shim for backward compatibility.
    With SafeExecute, files written to /workspace by sandbox scripts are
    immediately on the host at modified_datasets/<filename>.
    This function just confirms the file exists.

    Args:
        sandbox_filename: Filename or /workspace/<filename> path.
        local_path: Expected local path (for logging).

    Returns:
        JSON with the local path.
    """
    filename = Path(sandbox_filename).name
    actual = Path(_ensure_workspace()) / filename
    if actual.exists():
        return json.dumps({
            "status": "available",
            "local_path": str(actual),
            "note": "File already on host — no download needed",
        })
    if Path(local_path).exists():
        return json.dumps({"status": "available", "local_path": local_path})
    return json.dumps({"error": f"File not found at {actual} or {local_path}"})
