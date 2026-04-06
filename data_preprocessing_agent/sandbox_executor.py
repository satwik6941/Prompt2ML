"""
OpenSandbox Executor — Secure code execution for data preprocessing agents.

Provides three ADK-compatible async tool functions:
    - run_in_sandbox(command)      → execute shell commands
    - write_file_to_sandbox(path, content) → write files into sandbox
    - read_file_from_sandbox(path) → read files from sandbox

The sandbox is a Docker container with Python + data science libraries.
It is created once and reused across all tool calls, then killed at the end.

Usage:
    executor = SandboxExecutor()
    await executor.start()
    # ... use executor.run_in_sandbox, executor.write_file_to_sandbox, etc.
    await executor.stop()
"""

import os
import json
from datetime import timedelta
from pathlib import Path
from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig


# Default sandbox image with Python + data science libs
DEFAULT_IMAGE = "opensandbox/code-interpreter:v1.0.2"

# Paths inside the sandbox
SANDBOX_DATA_DIR = "/workspace/data"
SANDBOX_OUTPUT_DIR = "/workspace/output"


class SandboxExecutor:
    """Manages an OpenSandbox container lifecycle and provides tool functions."""

    def __init__(
        self,
        domain: str = None,
        api_key: str = None,
        image: str = None,
        timeout_minutes: int = 30,
    ):
        self.domain = domain or os.getenv("SANDBOX_DOMAIN", "localhost:8080")
        self.api_key = api_key or os.getenv("SANDBOX_API_KEY")
        self.image = image or os.getenv("SANDBOX_IMAGE", DEFAULT_IMAGE)
        self.timeout_minutes = timeout_minutes
        self.sandbox = None

    async def start(self):
        """Create and start the sandbox container."""
        config = ConnectionConfig(
            domain=self.domain,
            api_key=self.api_key,
            request_timeout=timedelta(seconds=120),
        )

        self.sandbox = await Sandbox.create(
            self.image,
            connection_config=config,
            entrypoint=["/opt/opensandbox/code-interpreter.sh"],
            env={"PYTHON_VERSION": "3.11"},
            timeout=timedelta(minutes=self.timeout_minutes),
        )

        # Create working directories inside sandbox
        await self.sandbox.commands.run(f"mkdir -p {SANDBOX_DATA_DIR} {SANDBOX_OUTPUT_DIR}")
        print(f"[SANDBOX] Started sandbox container", flush=True)

    async def stop(self):
        """Kill and clean up the sandbox container."""
        if self.sandbox:
            try:
                await self.sandbox.kill()
                await self.sandbox.close()
                print("[SANDBOX] Sandbox container stopped", flush=True)
            except Exception as e:
                print(f"[SANDBOX] Warning during cleanup: {e}", flush=True)
            self.sandbox = None

    async def upload_dataset(self, local_path: str) -> str:
        """
        Upload a local dataset file into the sandbox.
        Returns the path inside the sandbox.
        """
        local_file = Path(local_path)
        if not local_file.exists():
            return json.dumps({"error": f"Local file not found: {local_path}"})

        sandbox_path = f"{SANDBOX_DATA_DIR}/{local_file.name}"
        content = local_file.read_text(encoding="utf-8", errors="ignore")
        await self.sandbox.files.write_file(sandbox_path, content)
        print(f"[SANDBOX] Uploaded {local_file.name} → {sandbox_path}", flush=True)
        return sandbox_path

    async def download_file(self, sandbox_path: str, local_path: str) -> str:
        """
        Download a file from the sandbox to the local filesystem.
        Returns the local path.
        """
        content = await self.sandbox.files.read_file(sandbox_path)
        local_file = Path(local_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text(content, encoding="utf-8")
        print(f"[SANDBOX] Downloaded {sandbox_path} → {local_path}", flush=True)
        return local_path


# ============================================================
# Module-level singleton — shared across all tool calls
# ============================================================

_executor = SandboxExecutor()


async def start_sandbox() -> str:
    """
    Start the OpenSandbox container. Call this ONCE at the beginning
    of preprocessing before using any other sandbox tools.

    Returns:
        Confirmation message with sandbox status.
    """
    await _executor.start()
    return json.dumps({"status": "sandbox_started", "data_dir": SANDBOX_DATA_DIR, "output_dir": SANDBOX_OUTPUT_DIR})


async def stop_sandbox() -> str:
    """
    Stop and clean up the OpenSandbox container. Call this ONCE
    after all preprocessing is complete.

    Returns:
        Confirmation message.
    """
    await _executor.stop()
    return json.dumps({"status": "sandbox_stopped"})


async def upload_dataset_to_sandbox(local_file_path: str) -> str:
    """
    Upload a local dataset file into the sandbox environment.
    The file will be available at /workspace/data/<filename> inside the sandbox.

    Args:
        local_file_path: Absolute path to the local file to upload.

    Returns:
        JSON with the sandbox path where the file was uploaded.
    """
    sandbox_path = await _executor.upload_dataset(local_file_path)
    if sandbox_path.startswith("{"):
        return sandbox_path  # Error JSON
    return json.dumps({"status": "uploaded", "sandbox_path": sandbox_path, "local_path": local_file_path})


async def run_in_sandbox(command: str) -> str:
    """
    Run a shell command inside the OpenSandbox container.
    Use this to execute Python scripts, install packages, or run any shell command.

    The sandbox has Python with pandas, numpy, scikit-learn, scipy, matplotlib, seaborn.
    Data files are in /workspace/data/ and outputs go to /workspace/output/.

    Args:
        command: Shell command to execute (e.g. 'python3 /workspace/data/preprocess.py'
                 or 'pip install xgboost' or 'ls /workspace/data/').

    Returns:
        JSON with stdout, stderr, and success status.
    """
    if not _executor.sandbox:
        return json.dumps({"error": "Sandbox not started. Call start_sandbox first."})

    try:
        execution = await _executor.sandbox.commands.run(command)
        stdout = "\n".join(msg.text for msg in execution.logs.stdout)
        stderr = "\n".join(msg.text for msg in execution.logs.stderr)

        if execution.error:
            stderr = "\n".join([
                stderr,
                f"[error] {execution.error.name}: {execution.error.value}",
            ]).strip()

        return json.dumps({
            "success": not bool(execution.error),
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "stdout": "", "stderr": str(e)})


async def write_file_to_sandbox(sandbox_path: str, content: str) -> str:
    """
    Write a file inside the sandbox environment.
    Use this to create Python scripts, config files, or any text file.

    Args:
        sandbox_path: Path inside the sandbox (e.g. '/workspace/data/preprocess.py').
        content: The full text content to write.

    Returns:
        Confirmation with bytes written.
    """
    if not _executor.sandbox:
        return json.dumps({"error": "Sandbox not started. Call start_sandbox first."})

    try:
        await _executor.sandbox.files.write_file(sandbox_path, content)
        return json.dumps({
            "status": "written",
            "path": sandbox_path,
            "bytes": len(content),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def read_file_from_sandbox(sandbox_path: str) -> str:
    """
    Read a file from the sandbox environment.
    Use this to read results, processed data, or any file.

    Args:
        sandbox_path: Path inside the sandbox (e.g. '/workspace/output/result.csv').

    Returns:
        The file content as a string (first 50000 chars if large).
    """
    if not _executor.sandbox:
        return json.dumps({"error": "Sandbox not started. Call start_sandbox first."})

    try:
        content = await _executor.sandbox.files.read_file(sandbox_path)
        # Truncate very large files to avoid overwhelming the LLM context
        if len(content) > 50000:
            content = content[:50000] + f"\n\n[TRUNCATED — showing first 50000 of {len(content)} chars]"
        return content
    except Exception as e:
        return json.dumps({"error": str(e)})


async def download_from_sandbox(sandbox_path: str, local_path: str) -> str:
    """
    Download a file from the sandbox to the local filesystem.
    Use this to save preprocessed datasets back to the local machine.

    Args:
        sandbox_path: Path inside the sandbox (e.g. '/workspace/output/preprocessed.csv').
        local_path: Absolute local path to save the file to.

    Returns:
        JSON with the local path where the file was saved.
    """
    if not _executor.sandbox:
        return json.dumps({"error": "Sandbox not started. Call start_sandbox first."})

    try:
        result = await _executor.download_file(sandbox_path, local_path)
        return json.dumps({"status": "downloaded", "sandbox_path": sandbox_path, "local_path": result})
    except Exception as e:
        return json.dumps({"error": str(e)})