"""
Shared pipeline state manager.
Import this in ANY file across the codebase to read/write shared data.
"""
import json
from pathlib import Path

STATE_FILE = Path(__file__).parent / "pipeline_state.json"

def save_state(data: dict):
    """Merge and save data into the shared pipeline state."""
    current = load_state()
    current.update(data)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(current, f, indent=4)
    print(f"[PIPELINE] State saved to {STATE_FILE}", flush=True)

def load_state() -> dict:
    """Load the current pipeline state."""
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get(key: str, default=None):
    """Get a specific value from the pipeline state."""
    return load_state().get(key, default)
