"""
Shared pipeline state manager.
Import this in ANY file across the codebase to read/write shared data.
"""
import re
import json
import shutil
import logging
from pathlib import Path

STATE_FILE = Path(__file__).parent / "pipeline_state.json"
MODIFIED_DATASETS_ROOT = Path(__file__).parent / "modified_datasets"
OUTPUTS_ROOT = Path(__file__).parent / "outputs"

logger = logging.getLogger(__name__)

# Module-level cache so every agent in the same process resolves the same run dir
_run_dir_cache: "Path | None" = None


def load_state() -> dict:
    """
    Load the current pipeline state.
    Returns {} if the file doesn't exist.
    If the file is corrupt (invalid JSON), backs it up and returns {} so the
    pipeline can start fresh rather than crashing.
    """
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        backup = STATE_FILE.with_suffix(".corrupt.json")
        try:
            shutil.copy2(STATE_FILE, backup)
        except Exception:
            pass
        logger.warning(
            "[PIPELINE] pipeline_state.json is corrupt (%s). "
            "Backed up to %s and starting with empty state.",
            e, backup,
        )
        print(
            f"[PIPELINE WARNING] State file corrupt — backed up to {backup}, starting fresh.",
            flush=True,
        )
        return {}
    except OSError as e:
        logger.error("[PIPELINE] Could not read state file: %s", e)
        print(f"[PIPELINE ERROR] Could not read state file: {e}", flush=True)
        return {}


def save_state(data: dict) -> bool:
    """
    Merge data into the shared pipeline state and persist to disk.
    Returns True on success, False on failure (logs the error but does NOT raise).

    Uses json default=str so non-serializable objects become strings
    rather than crashing the entire pipeline on serialization edge cases.
    """
    try:
        current = load_state()
        current.update(data)
        # Write to a temp file first, then atomically rename — avoids corrupt
        # state file if the process is killed mid-write.
        tmp = STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=4, default=str)
        tmp.replace(STATE_FILE)
        print(f"[PIPELINE] State saved to {STATE_FILE}", flush=True)
        return True
    except OSError as e:
        logger.error("[PIPELINE] Could not write state file: %s", e)
        print(f"[PIPELINE ERROR] Could not write state file: {e}", flush=True)
        return False
    except Exception as e:
        logger.error("[PIPELINE] Unexpected error saving state: %s", e)
        print(f"[PIPELINE ERROR] Unexpected error saving state: {e}", flush=True)
        return False


def get(key: str, default=None):
    """Get a specific value from the pipeline state."""
    return load_state().get(key, default)


# ── Run-specific working directory ───────────────────────────────────────────

def _get_or_create_run_id() -> str:
    """
    Return the run_id for the current pipeline session.

    On the first call for a fresh state, generates a timestamp-based ID
    (``YYYYMMDD_HHMMSS``) and persists it so every agent in the same run
    resolves the same ID.  Subsequent calls just read it from state.

    This means folders look like:
        modified_datasets/20240605_153042/
        outputs/20240605_153042/
    instead of the old prompt-derived slugs like
        modified_datasets/i_want_to_build_a_ml_model_for_...
    """
    import datetime
    state = load_state()
    if "run_id" in state:
        return state["run_id"]
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_state({"run_id": run_id})
    return run_id


def get_run_dir() -> Path:
    """
    Return the run-specific subdirectory inside modified_datasets/.

        modified_datasets/<run_id>/

    The run_id is a timestamp (``YYYYMMDD_HHMMSS``) stored in pipeline_state.json
    so all agents in the same pipeline run share the same folder.

    This function lives in pipeline_state.py (not agent.py) so both
    agent.py and sandbox_executor.py can import it without circular dependencies.
    """
    global _run_dir_cache
    if _run_dir_cache is not None:
        return _run_dir_cache

    run_id = _get_or_create_run_id()
    run_dir = MODIFIED_DATASETS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _run_dir_cache = run_dir
    print(f"[PIPELINE] Run directory: {run_dir}", flush=True)
    return run_dir


def get_outputs_dir() -> Path:
    """
    Return the run-specific outputs directory.

        outputs/<run_id>/

    All reports, model files, plots, and other pipeline outputs for this run
    land here.  Keeps each run's files isolated so agents never accidentally
    read files from a previous run.
    """
    run_id = _get_or_create_run_id()
    out_dir = OUTPUTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def reset_run_dir_cache():
    """
    Clear the in-process run-dir cache so the next call to get_run_dir()
    re-reads the run_id from state.  The persisted run_id in pipeline_state.json
    is NOT cleared — use reset_run_id() for a truly fresh pipeline start.
    """
    global _run_dir_cache
    _run_dir_cache = None


def reset_run_id() -> str:
    """
    Force-generate a brand-new run_id and clear the directory cache.
    Call this at the very start of a fresh pipeline run (before saving user_goal)
    so the new run gets its own isolated folders.
    Returns the new run_id.
    """
    import datetime
    global _run_dir_cache
    _run_dir_cache = None
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_state({"run_id": run_id})
    print(f"[PIPELINE] New run started: {run_id}", flush=True)
    return run_id


# ── Checkpoint tracking ───────────────────────────────────────────────────────

def mark_checkpoint(name: str) -> None:
    """Record that a named pipeline step completed successfully with a UTC timestamp."""
    import datetime
    state = load_state()
    checkpoints = state.get("pipeline_checkpoints", {})
    checkpoints[name] = datetime.datetime.utcnow().isoformat()
    save_state({"pipeline_checkpoints": checkpoints})
    print(f"[PIPELINE] Checkpoint marked: {name}", flush=True)


def is_checkpoint_done(name: str) -> bool:
    """Return True if the named checkpoint was previously completed."""
    return name in load_state().get("pipeline_checkpoints", {})


def get_all_checkpoints() -> dict:
    """Return all completed checkpoints and their timestamps."""
    return load_state().get("pipeline_checkpoints", {})


# ── State backup ──────────────────────────────────────────────────────────────

def backup_state() -> bool:
    """
    Copy pipeline_state.json → pipeline_state.backup.json.
    Called after each phase completes so there's always a last-good snapshot.
    Returns True on success, False on failure (non-fatal).
    """
    if not STATE_FILE.exists():
        return False
    try:
        backup = STATE_FILE.with_name("pipeline_state.backup.json")
        shutil.copy2(STATE_FILE, backup)
        print(f"[PIPELINE] State backed up to {backup}", flush=True)
        return True
    except Exception as e:
        logger.warning("[PIPELINE] Could not backup state: %s", e)
        return False
