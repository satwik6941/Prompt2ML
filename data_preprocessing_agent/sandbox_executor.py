"""
Docker SDK Executor — Secure code execution for data preprocessing agents.

Replaces the safeexecute package with a direct Docker SDK approach, eliminating
the Windows \r\n line-ending bugs that plagued the previous implementation.

Architecture:
  - One persistent container named `prompt2ml-pipeline` per pipeline run.
  - The host's run-specific workspace directory is mounted as /workspace inside
    the container. Files written there by built-in tools are immediately visible
    to sandbox scripts — no upload/download needed.
  - Code execution works by writing a temp .py file into /workspace on the host,
    running `docker exec <container> python /workspace/_exec_<uuid>.py`, then
    deleting the temp file.
  - Missing pip dependencies are detected by parsing import statements and
    auto-installed into the running container before code executes.

Public API (signatures are identical to the safeexecute-based version):
    start_sandbox()                          — build/start container, pre-check deps
    stop_sandbox()                           — stop and remove container
    run_in_sandbox(code)                     — execute Python code, returns JSON
    write_file_to_sandbox(filename, content) — write a file into /workspace
    read_file_from_sandbox(filename)         — read a file from /workspace
    upload_dataset_to_sandbox(local_path)    — no-op shim (volume-mounted)
    download_from_sandbox(sandbox_fn, local) — no-op shim (volume-mounted)

ERROR HANDLING:
  - start_sandbox returns {"error": "...", "error_type": "docker_not_running"} when
    Docker Desktop is not started.
  - run_in_sandbox returns {"success": false, "error_type": "..."} on failure.
  - Agents MUST check for "error" key before proceeding.
"""

import ast
import json
import os
import sys
import threading
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_state import get_run_dir, reset_run_dir_cache  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------

# Resolved at sandbox start time (not import time) so the slug is always
# based on the user_goal already saved to pipeline_state.json.
WORKSPACE: str = ""

# The docker Container object — set by start_sandbox(), used everywhere else.
_container = None  # type: ignore[assignment]

# Idempotency flag
_started: bool = False

# Docker image and container name
_IMAGE_NAME = "prompt2ml-sandbox:latest"
_CONTAINER_NAME = "prompt2ml-pipeline"

# ---------------------------------------------------------------------------
# Resource limits. LLM-generated code runs in here; an unbounded container can
# exhaust host RAM or spin forever. Override via environment for big machines.
# ---------------------------------------------------------------------------

# Wall-clock ceiling for a single exec. Without this a runaway script hangs
# until the phase timeout, and the container survives even that.
DEFAULT_EXEC_TIMEOUT_S = int(os.getenv("PROMPT2ML_EXEC_TIMEOUT", "1800"))

# Memory ceiling. Docker kills the process (exit 137) rather than the host swapping.
CONTAINER_MEM_LIMIT = os.getenv("PROMPT2ML_SANDBOX_MEM", "8g")

# CPU ceiling, in whole cores.
CONTAINER_CPUS = float(os.getenv("PROMPT2ML_SANDBOX_CPUS", "4"))

# Process-count ceiling — stops a fork bomb from taking the host down with it.
CONTAINER_PIDS_LIMIT = int(os.getenv("PROMPT2ML_SANDBOX_PIDS", "512"))

# ---------------------------------------------------------------------------
# Import → pip package name mapping
# stdlib modules are skipped (no pip install needed).
# ---------------------------------------------------------------------------

IMPORT_TO_PIP: dict[str, str] = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "imblearn": "imbalanced-learn",
    "xlrd": "xlrd",
    "openpyxl": "openpyxl",
    "xlsxwriter": "XlsxWriter",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "statsmodels": "statsmodels",
    "plotly": "plotly",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "joblib": "joblib",
    "tqdm": "tqdm",
    "nltk": "nltk",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "optuna": "optuna",
    "shap": "shap",
    "category_encoders": "category_encoders",
}

# Standard library modules — never attempt to pip-install these.
STDLIB_MODULES: set[str] = {
    "os", "sys", "json", "re", "math", "random", "time", "datetime",
    "pathlib", "collections", "itertools", "functools", "io", "abc",
    "copy", "gc", "glob", "hashlib", "logging", "pickle", "shutil",
    "socket", "string", "struct", "subprocess", "tempfile", "threading",
    "traceback", "typing", "unittest", "urllib", "uuid", "warnings",
    "csv", "ast", "base64", "binascii", "builtins", "codecs", "contextlib",
    "dataclasses", "decimal", "enum", "gzip", "html", "http", "inspect",
    "operator", "platform", "pprint", "queue", "signal", "stat", "textwrap",
    "tokenize", "types", "weakref", "zipfile", "zlib", "argparse", "getpass",
    "importlib", "numbers", "sqlite3", "xml", "xmlrpc", "email", "array",
    "bisect", "calendar", "cmath", "cProfile", "difflib", "dis", "fcntl",
    "fractions", "getopt", "heapq", "hmac", "imaplib", "ipaddress",
    "keyword", "linecache", "locale", "mimetypes", "multiprocessing",
    "netrc", "optparse", "posixpath", "profile", "pstats", "pty", "pwd",
    "readline", "reprlib", "rlcompleter", "sched", "select", "shelve",
    "shlex", "smtplib", "sndhdr", "spwd", "statistics", "string",
    "tabnanny", "telnetlib", "termios", "test", "token", "turtle",
    "turtledemo", "unicodedata",
    # commonly pre-installed heavy packages (no need to re-pip them)
    "numpy", "pandas", "matplotlib", "sklearn", "scipy", "seaborn",
    "plotly", "joblib",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_workspace() -> str:
    """Resolve the run-specific workspace and ensure the directory exists."""
    run_dir = get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def _same_path(a: str, b: str) -> bool:
    """
    Compare two host paths for equality across the spellings Docker reports.

    Docker Desktop on Windows may hand back `/host_mnt/c/...` or `C:\\...` for the
    same directory depending on the backend, so a plain string compare produces
    false mismatches and needlessly rebuilds the container.
    """
    if not a or not b:
        return False
    if a == b:
        return True

    def norm(p: str) -> str:
        p = p.replace("\\", "/").rstrip("/").lower()
        for prefix in ("/host_mnt/", "/mnt/", "/run/desktop/mnt/host/"):
            if p.startswith(prefix):
                rest = p[len(prefix):]
                if len(rest) > 1 and rest[1] == "/":       # "c/users/..." -> "c:/users/..."
                    rest = f"{rest[0]}:{rest[1:]}"
                return rest
        return p

    return norm(a) == norm(b)


def _is_docker_daemon_error(exc: Exception) -> bool:
    """
    Detect whether the exception means the Docker daemon isn't reachable.

    Deliberately narrow: matching a bare "docker" substring made every unrelated
    failure inside a docker-sdk call report itself as "Docker Desktop is not
    running", which sends the user off fixing the wrong thing.
    """
    import errno

    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, OSError) and exc.errno in (errno.ECONNREFUSED, errno.ENOENT, errno.EPIPE):
        return True

    msg = str(exc).lower()
    return any(k in msg for k in (
        "is the docker daemon running",
        "error while fetching server api version",
        "cannot connect to the docker daemon",
        "connection refused",
        "the system cannot find the file specified",   # Windows named-pipe absence
        "//./pipe/docker_engine",
        "dockerdesktoplinuxengine",
    ))


def _parse_imports(code: str) -> list[str]:
    """
    Parse Python source code and return a list of top-level module names
    referenced in import statements.

    Handles:
        import numpy
        import numpy as np
        from sklearn import metrics
        from sklearn.linear_model import ...
    """
    module_names: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If we can't parse, fall back to a quick line scan.
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("import "):
                rest = line[len("import "):].split("#")[0]
                for part in rest.split(","):
                    name = part.strip().split(" ")[0].split(".")[0]
                    if name:
                        module_names.append(name)
            elif line.startswith("from "):
                rest = line[len("from "):].split(" import")[0].strip()
                top = rest.split(".")[0]
                if top:
                    module_names.append(top)
        return module_names

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                module_names.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                module_names.append(top)

    return module_names


def _get_client():
    """Return a connected docker.DockerClient, or raise with a clear message."""
    try:
        import docker  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "docker package not installed. Run: pip install docker"
        ) from exc
    try:
        client = docker.from_env()
        client.ping()  # verify the daemon is actually reachable
        return client
    except Exception as exc:
        raise exc


class SandboxTimeout(Exception):
    """Raised when a container exec exceeds its wall-clock budget."""


def _exec_in_container(
    cmd: list[str],
    workdir: str = "/workspace",
    timeout_s: int | None = None,
    user: str | None = None,
):
    """
    Run a command inside the running container and return (exit_code, output).
    `output` is a single string combining stdout and stderr.

    docker-py's exec_run has no timeout parameter, so the call is run on a worker
    thread and abandoned if it overruns. Abandoning the thread is not enough on its
    own — the process inside the container keeps burning CPU — so on timeout we
    also kill the matching PIDs in the container before raising.
    """
    if _container is None:
        raise RuntimeError("Container is not started. Call start_sandbox() first.")

    budget = DEFAULT_EXEC_TIMEOUT_S if timeout_s is None else timeout_s
    box: dict = {}

    def _call():
        try:
            box["result"] = _container.exec_run(
                cmd, stdout=True, stderr=True, workdir=workdir, user=user or "",
            )
        except Exception as exc:  # surfaced on the calling thread below
            box["error"] = exc

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    worker.join(timeout=budget if budget > 0 else None)

    if worker.is_alive():
        # Kill the abandoned process so it stops consuming the container's budget.
        # pkill comes from procps, installed by docker/Dockerfile — if that ever
        # regresses the kill silently no-ops and a runaway job survives the
        # timeout, so a failure here is reported rather than swallowed.
        try:
            kill = _container.exec_run(
                ["pkill", "-9", "-f", cmd[-1]], stdout=True, stderr=True,
            )
            # pkill exits 1 when nothing matched (fine) and 127 when absent.
            if kill.exit_code not in (0, 1):
                print(
                    f"[SANDBOX WARNING] Could not kill the timed-out process "
                    f"(pkill exit {kill.exit_code}). It may still be running inside "
                    f"'{_CONTAINER_NAME}'; call stop_sandbox() to reclaim resources.",
                    flush=True,
                )
        except Exception as kill_exc:
            print(
                f"[SANDBOX WARNING] Kill after timeout failed ({kill_exc}). The "
                f"process may still be running inside '{_CONTAINER_NAME}'.",
                flush=True,
            )
        raise SandboxTimeout(
            f"Execution exceeded {budget}s and was terminated. "
            "Reduce the work per step, sample the data, or raise "
            "PROMPT2ML_EXEC_TIMEOUT if the job genuinely needs longer."
        )

    if "error" in box:
        raise box["error"]

    result = box["result"]
    output = result.output.decode("utf-8", errors="replace") if result.output else ""
    return result.exit_code, output


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------

def _check_and_install_deps(imports: list[str]) -> dict:
    """
    For each import name that isn't in stdlib:
      1. Try to import it inside the container.
      2. Collect failures, map them to pip package names.
      3. Run a single `pip install` for all missing packages.
      4. Return a dict describing what was done.
    """
    # De-duplicate and filter stdlib
    candidates = list(dict.fromkeys(
        name for name in imports
        if name and name not in STDLIB_MODULES
    ))

    if not candidates:
        return {"installed": [], "note": "no external imports detected"}

    missing_pip: list[str] = []
    for name in candidates:
        exit_code, _ = _exec_in_container(
            ["python", "-c", f"import {name}"], timeout_s=60
        )
        if exit_code != 0:
            # Map import name to pip package name (fall back to import name itself)
            pip_name = IMPORT_TO_PIP.get(name, name)
            missing_pip.append(pip_name)

    if not missing_pip:
        return {"installed": [], "note": "all imports already available"}

    print(f"[SANDBOX] Installing missing packages: {missing_pip}", flush=True)
    # The image runs as the non-root `sandbox` user, so installing into the system
    # site-packages fails with EACCES. --user targets the writable home directory,
    # which is on sys.path for that user.
    exit_code, output = _exec_in_container(
        ["pip", "install", "--user", "--quiet", "--no-warn-script-location"] + missing_pip,
        timeout_s=600,
    )
    if exit_code != 0:
        print(f"[SANDBOX WARNING] pip install returned {exit_code}: {output[-400:]}", flush=True)
        return {
            "installed": missing_pip,
            "returncode": exit_code,
            "errors": output.strip()[-400:],
        }

    print(f"[SANDBOX] Installed: {missing_pip}", flush=True)
    return {"installed": missing_pip, "returncode": 0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def start_sandbox() -> str:
    """
    Start the Docker sandbox container (or reuse one that's already running).

    Steps:
      1. Resolve the run-specific workspace directory.
      2. Connect to Docker daemon.
      3. Verify the image `prompt2ml-sandbox:latest` exists.
      4. Create/start the `prompt2ml-pipeline` container with /workspace mounted.
      5. Run a warmup Python snippet to verify the interpreter works.

    Returns:
        JSON {"status": "sandbox_started", "workspace": "..."} on success.
        JSON {"error": "...", "error_type": "<type>"} on failure.
        error_type values:
            "import_error"        — `docker` package missing
            "docker_not_running"  — Docker Desktop not started
            "image_not_found"     — image hasn't been built yet
            "sandbox_start_error" — anything else
    """
    global _container, _started, WORKSPACE

    WORKSPACE = _ensure_workspace()

    # --- 1. Import docker SDK ---
    try:
        import docker  # type: ignore
    except ImportError:
        msg = "docker package not installed. Run: pip install docker"
        print(f"[SANDBOX ERROR] {msg}", flush=True)
        return json.dumps({
            "error": msg,
            "error_type": "import_error",
            "action": "Run `pip install docker` then retry.",
        })

    # --- 2. Connect to Docker daemon ---
    try:
        client = _get_client()
    except ImportError:
        # Already handled above, but guard for completeness.
        pass
    except Exception as exc:
        if _is_docker_daemon_error(exc):
            msg = (
                "Docker Desktop is not running. "
                "Please start Docker Desktop and wait for it to fully initialise, then retry."
            )
            error_type = "docker_not_running"
        else:
            msg = f"Cannot connect to Docker: {exc}"
            error_type = "sandbox_start_error"
        print(f"[SANDBOX ERROR] {msg}", flush=True)
        return json.dumps({
            "error": msg,
            "error_type": error_type,
            "action": (
                "Start Docker Desktop (system tray) and call start_sandbox again."
                if error_type == "docker_not_running"
                else "Check Docker installation."
            ),
        })

    # --- 3. Verify image exists ---
    try:
        client.images.get(_IMAGE_NAME)
    except docker.errors.ImageNotFound:
        msg = (
            f"Image `{_IMAGE_NAME}` not found. "
            "Build it first by running `docker/build.ps1` from the project root."
        )
        print(f"[SANDBOX ERROR] {msg}", flush=True)
        return json.dumps({
            "error": msg,
            "error_type": "image_not_found",
            "action": "Run `docker/build.ps1` (or `docker build -t prompt2ml-sandbox:latest docker/`) to build the image.",
        })
    except Exception as exc:
        msg = f"Failed to check Docker image: {exc}"
        print(f"[SANDBOX ERROR] {msg}", flush=True)
        return json.dumps({"error": msg, "error_type": "sandbox_start_error"})

    # --- 4. Create or reuse container ---
    try:
        existing = client.containers.get(_CONTAINER_NAME)
        existing.reload()

        # A container left behind by a crashed run is still bound to THAT run's
        # workspace. Reusing it silently sends every sandbox write into the wrong
        # run directory, and the resulting files look plausible enough that the
        # mistake surfaces much later. Verify the mount and rebuild on mismatch.
        bound = ""
        for mount in existing.attrs.get("Mounts", []):
            if mount.get("Destination") == "/workspace":
                bound = mount.get("Source", "")
                break
        if not _same_path(bound, WORKSPACE):
            print(
                f"[SANDBOX] Existing container is mounted at '{bound}' but this run "
                f"needs '{WORKSPACE}'. Removing and recreating.",
                flush=True,
            )
            try:
                existing.remove(force=True)
            except Exception as exc:
                return json.dumps({
                    "error": f"Could not remove stale container: {exc}",
                    "error_type": "sandbox_start_error",
                    "action": f"Run `docker rm -f {_CONTAINER_NAME}` and retry.",
                })
            raise docker.errors.NotFound("recreating after workspace mismatch")

        if existing.status == "running":
            _container = existing
            print(f"[SANDBOX] Reusing running container '{_CONTAINER_NAME}'.", flush=True)
        else:
            # Container exists but isn't running — start it.
            print(f"[SANDBOX] Container '{_CONTAINER_NAME}' found (status={existing.status}), starting...", flush=True)
            existing.start()
            existing.reload()
            _container = existing
    except docker.errors.NotFound:
        # Container doesn't exist yet — create it fresh.
        print(f"[SANDBOX] Creating container '{_CONTAINER_NAME}'...", flush=True)
        try:
            _container = client.containers.run(
                _IMAGE_NAME,
                name=_CONTAINER_NAME,
                detach=True,
                volumes={
                    WORKSPACE: {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                },
                working_dir="/workspace",
                # Keep container alive indefinitely with a no-op tail command.
                command="tail -f /dev/null",
                remove=False,
                # Resource ceilings — this container runs LLM-written code.
                mem_limit=CONTAINER_MEM_LIMIT,
                nano_cpus=int(CONTAINER_CPUS * 1_000_000_000),
                pids_limit=CONTAINER_PIDS_LIMIT,
                # Network stays ON: _check_and_install_deps pip-installs missing
                # packages at run time. Isolating the network is tracked for the
                # M3 codegen gate, which will pre-resolve dependencies so the
                # container can run with network_disabled=True.
            )
            print(f"[SANDBOX] Container '{_CONTAINER_NAME}' created.", flush=True)
        except Exception as exc:
            msg = f"Failed to create container: {exc}"
            print(f"[SANDBOX ERROR] {msg}", flush=True)
            return json.dumps({"error": msg, "error_type": "sandbox_start_error"})
    except Exception as exc:
        msg = f"Error accessing container: {exc}"
        print(f"[SANDBOX ERROR] {msg}", flush=True)
        return json.dumps({"error": msg, "error_type": "sandbox_start_error"})

    # --- 5. Warmup verification ---
    try:
        exit_code, output = _exec_in_container(
            ["python", "-c", "import sys; print(f'Python {sys.version[:6]} ready')"],
            timeout_s=60,
        )
        if exit_code != 0:
            msg = f"Container warmup failed (exit {exit_code}): {output}"
            print(f"[SANDBOX ERROR] {msg}", flush=True)
            return json.dumps({"error": msg, "error_type": "sandbox_start_error"})
        _started = True
        print(f"[SANDBOX] Started — workspace: {WORKSPACE}", flush=True)
        print(f"[SANDBOX] {output.strip()}", flush=True)
    except Exception as exc:
        msg = f"Warmup exec failed: {exc}"
        print(f"[SANDBOX ERROR] {msg}", flush=True)
        return json.dumps({"error": msg, "error_type": "sandbox_start_error"})

    return json.dumps({
        "status": "sandbox_started",
        "workspace": WORKSPACE,
        "container_name": _CONTAINER_NAME,
        "container_note": f"host {WORKSPACE} is mounted as /workspace inside the container",
    })


async def stop_sandbox() -> str:
    """
    Stop and remove the sandbox container, freeing Docker resources.

    Safe to call even if start_sandbox() failed — no-op in that case.

    Returns:
        JSON {"status": "sandbox_stopped"} on success.
        JSON {"status": "sandbox_not_running"} if sandbox was never started.
    """
    global _container, _started

    if not _started or _container is None:
        print("[SANDBOX] stop_sandbox called but sandbox was not running — no-op.", flush=True)
        return json.dumps({"status": "sandbox_not_running", "note": "Nothing to stop."})

    try:
        _container.reload()
        if _container.status == "running":
            _container.stop(timeout=10)
        _container.remove()
        print(f"[SANDBOX] Container '{_CONTAINER_NAME}' stopped and removed.", flush=True)
    except Exception as exc:
        note = str(exc)
        print(f"[SANDBOX] stop_sandbox: container may already be gone ({note})", flush=True)
        return json.dumps({"status": "sandbox_stopped", "note": note})
    finally:
        _container = None
        _started = False

    return json.dumps({"status": "sandbox_stopped"})


def _is_shell_command(code: str) -> tuple[bool, str]:
    """
    Detect the pattern `python <filename>.py` (single-line, no source code).
    Returns (True, filename) when the input is a shell-style invocation of a
    workspace file rather than inline Python source.
    """
    stripped = code.strip()
    if "\n" in stripped:
        return False, ""
    parts = stripped.split()
    if len(parts) >= 2 and parts[0] in ("python", "python3") and parts[1].endswith(".py"):
        return True, parts[1]
    return False, ""


def _timeout_response(exc: "SandboxTimeout") -> str:
    """Return a standard error JSON for an execution that blew its time budget."""
    print(f"[SANDBOX ERROR] {exc}", flush=True)
    return json.dumps({
        "success": False,
        "stdout": "",
        "stderr": str(exc),
        "error_type": "timeout",
        "action": (
            "The step ran too long and was killed. Sample the data, split the work into "
            "smaller steps, or reduce the search budget — then run it again."
        ),
    })


def _exec_error_response(exec_exc: Exception) -> str:
    """Return a standard error JSON for a failed container exec call."""
    if _is_docker_daemon_error(exec_exc):
        error_type = "docker_not_running"
        action = "Start Docker Desktop and call start_sandbox() again."
    else:
        error_type = "execution_error"
        action = "Check the code for syntax errors and try again."
    print(f"[SANDBOX ERROR] exec failed ({error_type}): {exec_exc}", flush=True)
    return json.dumps({
        "success": False,
        "stdout": "",
        "stderr": str(exec_exc),
        "error_type": error_type,
        "action": action,
    })


async def run_in_sandbox(code: str) -> str:
    """
    Execute Python code — or a pre-written workspace script — inside the sandbox.

    Supports two calling styles:

    Style A — inline source code (preprocessing agents):
        run_in_sandbox("import pandas as pd\\n...")
        Writes a temp file and executes it.

    Style B — run a file already written via write_file_to_sandbox (ML trainer):
        run_in_sandbox("python train_RandomForest.py")
        Runs /workspace/train_RandomForest.py directly.

    Missing pip packages are auto-installed in both styles.

    Args:
        code: Python source code OR a single-line 'python <filename>.py' command.

    Returns:
        JSON {"success": true, "stdout": "..."} on success.
        JSON {"success": false, "stderr": "...", "error_type": "..."} on failure.
    """
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    if not _started or _container is None:
        return json.dumps({
            "success": False,
            "stdout": "",
            "stderr": "Sandbox is not running.",
            "error_type": "not_started",
            "action": (
                "Call start_sandbox() first. "
                "If start_sandbox returned an error, check that Docker Desktop is running."
            ),
        })

    is_shell, filename = _is_shell_command(code)

    if is_shell:
        # Style B: file already in /workspace — run it directly.
        host_path = Path(WORKSPACE) / filename
        if not host_path.exists():
            return json.dumps({
                "success": False,
                "stdout": "",
                "stderr": (
                    f"Script '{filename}' not found in workspace. "
                    f"Call write_file_to_sandbox('{filename}', code) first."
                ),
                "error_type": "file_not_found",
                "action": f"write_file_to_sandbox('{filename}', script_code)",
            })

        try:
            src = host_path.read_text(encoding="utf-8", errors="ignore")
            dep_result = _check_and_install_deps(_parse_imports(src))
            if dep_result.get("installed"):
                print(f"[SANDBOX] Auto-installed: {dep_result['installed']}", flush=True)
        except Exception as dep_exc:
            print(f"[SANDBOX WARNING] Dependency check failed: {dep_exc}", flush=True)

        print(f"[SANDBOX] Running /workspace/{filename}", flush=True)
        try:
            exit_code, output = _exec_in_container(["python", f"/workspace/{filename}"])
        except SandboxTimeout as timeout_exc:
            return _timeout_response(timeout_exc)
        except Exception as exec_exc:
            return _exec_error_response(exec_exc)

    else:
        # Style A: inline source — write temp file, run, delete.
        try:
            dep_result = _check_and_install_deps(_parse_imports(code))
            if dep_result.get("installed"):
                print(f"[SANDBOX] Auto-installed packages: {dep_result['installed']}", flush=True)
        except Exception as dep_exc:
            print(f"[SANDBOX WARNING] Dependency check failed: {dep_exc}", flush=True)

        script_name = f"_exec_{uuid.uuid4().hex[:8]}.py"
        script_host_path = Path(WORKSPACE) / script_name
        try:
            script_host_path.write_text(code, encoding="utf-8", newline="\n")
        except Exception as write_exc:
            return json.dumps({
                "success": False,
                "stdout": "",
                "stderr": f"Failed to write temp script: {write_exc}",
                "error_type": "io_error",
                "action": f"Check that {WORKSPACE} is writable.",
            })

        try:
            exit_code, output = _exec_in_container(
                ["python", f"/workspace/{script_name}"]
            )
        except SandboxTimeout as timeout_exc:
            return _timeout_response(timeout_exc)
        except Exception as exec_exc:
            return _exec_error_response(exec_exc)
        finally:
            try:
                script_host_path.unlink(missing_ok=True)
            except Exception:
                pass

    if exit_code == 0:
        return json.dumps({"success": True, "stdout": output.strip()}, indent=2)

    print(f"[SANDBOX] Code exited {exit_code}. stderr:\n{output[-600:]}", flush=True)
    return json.dumps({
        "success": False,
        "stdout": "",
        "stderr": output.strip(),
        "error_type": "execution_error",
        "action": (
            "Review the stderr above, fix the code, and call run_in_sandbox again. "
            "Common causes: wrong file paths (use /workspace/), missing imports, "
            "or a package that needs installing."
        ),
    })


async def write_file_to_sandbox(filename: str, content: str) -> str:
    """
    Write a Python script or text file into /workspace so run_in_sandbox can use it.

    Because /workspace is the host's workspace folder, this just writes the file
    to disk — no Docker API call needed. Works even if Docker is not running.

    Args:
        filename: Filename only (e.g. 'preprocess.py'). Written to WORKSPACE/.
        content:  Full text content of the file.

    Returns:
        JSON with the host path and equivalent /workspace path.
    """
    workspace = Path(_ensure_workspace())
    dest = workspace / filename
    # Normalise line endings before writing — avoids \r\n issues inside Linux containers.
    clean_content = content.replace("\r\n", "\n").replace("\r", "\n")
    dest.write_text(clean_content, encoding="utf-8", newline="\n")
    print(f"[SANDBOX] Wrote file -> {dest}", flush=True)
    return json.dumps({
        "status": "written",
        "host_path": str(dest),
        "sandbox_path": f"/workspace/{filename}",
        "bytes": len(clean_content),
    })


async def read_file_from_sandbox(filename: str) -> str:
    """
    Read a file that was written by a sandbox script to /workspace.

    Because /workspace is the host's workspace folder, this just reads
    the file from disk.

    Args:
        filename: Filename only (e.g. 'result.csv'). Read from WORKSPACE/.

    Returns:
        File content as a string (truncated to 50 000 chars if large).
        JSON {"error": "..."} if the file does not exist.
    """
    src = Path(_ensure_workspace()) / filename
    if not src.exists():
        return json.dumps({"error": f"File not found: {src}"})

    content = src.read_text(encoding="utf-8", errors="ignore")
    if len(content) > 50_000:
        content = content[:50_000] + f"\n\n[TRUNCATED — showing first 50 000 of {len(content)} chars]"
    return content


# ---------------------------------------------------------------------------
# Legacy compatibility shims
# With volume mounting the shared /workspace means no transfer is needed —
# these functions just return the path that's already accessible.
# ---------------------------------------------------------------------------

async def upload_dataset_to_sandbox(local_file_path: str) -> str:
    """
    No-op shim for backward compatibility.

    With the volume-mounted /workspace, any file already in the workspace is
    immediately accessible from inside the container. No upload step is needed.

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
        "note": "No upload needed — workspace directory is mounted as /workspace",
    })


async def download_from_sandbox(sandbox_filename: str, local_path: str) -> str:
    """
    No-op shim for backward compatibility.

    With the volume-mounted /workspace, files written by sandbox scripts are
    immediately on the host. This function just confirms the file exists.

    Args:
        sandbox_filename: Filename or /workspace/<filename> path.
        local_path:       Expected local path (for logging/confirmation).

    Returns:
        JSON with the local path, or an error if neither location has the file.
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
