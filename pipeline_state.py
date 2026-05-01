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

def _slugify(text: str, max_len: int = 40) -> str:
    """Convert a free-form problem statement into a filesystem-safe folder name."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s_-]", "", text)   # keep alphanum, space, dash, underscore
    text = re.sub(r"\s+", "_", text)              # spaces → underscores
    text = re.sub(r"_+", "_", text).strip("_")   # collapse repeated underscores
    return text[:max_len] or "run"


def get_run_dir() -> Path:
    """
    Return the run-specific subdirectory inside modified_datasets/.

        modified_datasets/<problem-slug>/

    The slug is derived from user_goal in pipeline_state.json and cached for
    the lifetime of the process so all agents in the same run share the same folder.

    Falls back to modified_datasets/run/ if no goal is set yet.

    This function lives in pipeline_state.py (not agent.py) so both
    agent.py and sandbox_executor.py can import it without circular dependencies.
    """
    global _run_dir_cache
    if _run_dir_cache is not None:
        return _run_dir_cache

    state = load_state()
    goal = state.get("user_goal", "").strip()
    slug = _slugify(goal) if goal else "run"
    run_dir = MODIFIED_DATASETS_ROOT / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    _run_dir_cache = run_dir
    print(f"[PIPELINE] Run directory: {run_dir}", flush=True)
    return run_dir


def reset_run_dir_cache():
    """
    Clear the cached run directory so the next call to get_run_dir() re-reads
    the goal from state. Call this after saving user_goal if agent.py and
    pipeline_state.py are in the same process and the goal wasn't set at import time.
    """
    global _run_dir_cache
    _run_dir_cache = None
