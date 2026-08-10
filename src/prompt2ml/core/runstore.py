"""
Per-run document store.

One directory per run, one file per document, atomic writes guarded by an
inter-process lock. Replaces the global ``pipeline_state.json`` — which made
concurrent runs impossible, grew without bound, and had to be fully re-read and
re-serialized on every tool call.

    runs/<run_id>/
      run.json  requirements.json  dataset_manifest.json
      data_contract.json  plan.json  hardware.json
      logs/  artifacts/

See docs/IMPLEMENTATION_PLAN.md — D2.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import TypeVar

from .contracts import Phase, RunMeta, _Doc

D = TypeVar("D", bound=_Doc)

DEFAULT_RUNS_ROOT = Path("runs")
_LOCK_STALE_S = 60.0


class RunStoreError(RuntimeError):
    pass


class SchemaMismatch(RunStoreError):
    """A document was written by a different build of the schema."""


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

class _DirLock:
    """
    Cross-platform advisory lock via atomic ``O_CREAT | O_EXCL``.

    fcntl is POSIX-only and msvcrt locking is awkward across processes, so this
    uses exclusive file creation, which is atomic on both NTFS and POSIX
    filesystems. A lock older than ``_LOCK_STALE_S`` is treated as abandoned by
    a killed process and broken, so a crash cannot wedge a run permanently.
    """

    def __init__(self, path: Path, timeout: float = 10.0) -> None:
        self.path = path
        self.timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> "_DirLock":
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                if self._break_if_stale():
                    continue
                if time.monotonic() >= deadline:
                    raise RunStoreError(
                        f"Timed out after {self.timeout}s waiting for {self.path}. "
                        "Another Prompt2ML process may be writing this run; if none "
                        "is running, delete the lock file."
                    )
                time.sleep(0.05)

    def _break_if_stale(self) -> bool:
        try:
            if time.time() - self.path.stat().st_mtime > _LOCK_STALE_S:
                self.path.unlink(missing_ok=True)
                return True
        except OSError:
            return True   # vanished between check and stat — retry
        return False

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def new_run_id(now: _dt.datetime | None = None) -> str:
    return (now or _dt.datetime.now()).strftime("%Y%m%d_%H%M%S")


class RunStore:
    """Reads and writes the typed documents of a single run."""

    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir)

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, root: Path | str = DEFAULT_RUNS_ROOT, goal: str = "") -> "RunStore":
        """Create a fresh run directory. Collides safely if two start in one second."""
        root = Path(root)
        run_id = new_run_id()
        candidate = root / run_id
        suffix = 1
        while candidate.exists():
            candidate = root / f"{run_id}_{suffix}"
            suffix += 1

        store = cls(candidate)
        store.dir.mkdir(parents=True, exist_ok=True)
        store.logs_dir.mkdir(exist_ok=True)
        store.artifacts_dir.mkdir(exist_ok=True)
        store.write(RunMeta(run_id=candidate.name, goal=goal))
        return store

    @classmethod
    def open(cls, root: Path | str, run_id: str) -> "RunStore":
        store = cls(Path(root) / run_id)
        if not store.dir.exists():
            raise RunStoreError(f"No such run: {store.dir}")
        # A directory without run.json is an orphan from a process killed
        # between mkdir and the first write. Say that, rather than letting the
        # caller hit a confusing "the phase that produces it has not completed".
        if not (store.dir / RunMeta.filename).exists():
            raise RunStoreError(
                f"{store.dir} has no {RunMeta.filename} — it is an incomplete run "
                f"directory left by an interrupted start. Delete it, or run "
                f"'prompt2ml clean --orphans'."
            )
        return store

    @classmethod
    def list_runs(cls, root: Path | str = DEFAULT_RUNS_ROOT) -> list["RunStore"]:
        """All runs, newest first."""
        root = Path(root)
        if not root.exists():
            return []
        dirs = [d for d in root.iterdir() if d.is_dir() and (d / RunMeta.filename).exists()]
        return [cls(d) for d in sorted(dirs, key=lambda d: d.name, reverse=True)]

    @classmethod
    def latest(cls, root: Path | str = DEFAULT_RUNS_ROOT) -> "RunStore | None":
        runs = cls.list_runs(root)
        return runs[0] if runs else None

    # -- paths -------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self.dir.name

    @property
    def logs_dir(self) -> Path:
        return self.dir / "logs"

    @property
    def artifacts_dir(self) -> Path:
        return self.dir / "artifacts"

    @property
    def _lock_path(self) -> Path:
        return self.dir / ".lock"

    def path_for(self, doc: type[_Doc]) -> Path:
        return self.dir / doc.filename

    # -- document IO -------------------------------------------------------

    def exists(self, doc: type[_Doc]) -> bool:
        return self.path_for(doc).exists()

    def read(self, doc: type[D]) -> D | None:
        """Load a document, or None if it hasn't been written yet."""
        path = self.path_for(doc)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunStoreError(f"{path} is not valid JSON: {exc}") from exc

        found = raw.get("schema_version")
        expected = doc.model_fields["schema_version"].default
        if found != expected:
            raise SchemaMismatch(
                f"{path.name} has schema_version {found}, this build expects "
                f"{expected}. The run was created by a different version of "
                f"Prompt2ML — start a new run rather than resuming this one."
            )
        return doc.model_validate(raw)

    def require(self, doc: type[D]) -> D:
        """Load a document that must exist, with an actionable error if it doesn't."""
        loaded = self.read(doc)
        if loaded is None:
            raise RunStoreError(
                f"{doc.filename} is missing from run {self.run_id}. "
                f"The phase that produces it has not completed."
            )
        return loaded

    def write(self, document: _Doc) -> Path:
        """
        Persist a document atomically.

        Writes to a temp file in the same directory then renames, so a process
        killed mid-write leaves the previous version intact rather than a
        truncated file.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(type(document))
        payload = document.model_dump_json(indent=2)

        with _DirLock(self._lock_path):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
        return path

    def update(self, doc: type[D], **changes: object) -> D:
        """Read-modify-write under one lock. Use for phase transitions."""
        with _DirLock(self._lock_path):
            path = self.path_for(doc)
            current = doc.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else doc()  # type: ignore[call-arg]
            for key, value in changes.items():
                setattr(current, key, value)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(current.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, path)
        return current

    # -- phase helpers -----------------------------------------------------

    def meta(self) -> RunMeta:
        return self.require(RunMeta)

    def start_phase(self, phase: Phase) -> RunMeta:
        m = self.meta()
        m.start(phase)
        self.write(m)
        return m

    def complete_phase(self, phase: Phase) -> RunMeta:
        m = self.meta()
        m.complete(phase)
        self.write(m)
        return m

    def fail_phase(self, phase: Phase, error: str) -> RunMeta:
        m = self.meta()
        m.fail(phase, error)
        self.write(m)
        return m

    def __repr__(self) -> str:
        return f"<RunStore {self.run_id}>"
