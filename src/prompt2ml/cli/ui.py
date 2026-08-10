"""
Terminal rendering helpers.

Deliberately stdlib-only. ``prompt2ml doctor`` has to run when the environment
is broken — including when dependencies failed to install — so the command that
diagnoses a broken install must not itself import third-party packages.
"""

from __future__ import annotations

import os
import sys

OK = "ok"
WARN = "warn"
FAIL = "fail"

_GLYPH = {OK: "+", WARN: "!", FAIL: "x"}
_COLOR = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def supports_color(stream=sys.stdout) -> bool:
    """
    Colour only when it will actually render.

    Windows terminals older than Windows Terminal do not interpret ANSI, and a
    redirected stream should stay plain so logs are greppable.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name == "nt":
        return bool(os.environ.get("WT_SESSION") or os.environ.get("TERM"))
    return True


class Console:
    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self.color = supports_color(self.stream)

    def _wrap(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self.color else text

    def print(self, text: str = "") -> None:
        print(text, file=self.stream)

    def title(self, text: str) -> None:
        self.print()
        self.print(self._wrap(text, _BOLD))
        self.print(self._wrap("─" * min(len(text), 62), _DIM))

    def dim(self, text: str) -> None:
        self.print(self._wrap(text, _DIM))

    def row(self, status: str, label: str, detail: str = "", fix: str = "") -> None:
        """One readiness row: marker, label, detail, and the fix when not ok."""
        glyph = self._wrap(_GLYPH.get(status, "?"), _COLOR.get(status, ""))
        self.print(f"  {glyph}  {label:<26} {detail}")
        if fix and status != OK:
            self.print(f"     {self._wrap('→ ' + fix, _DIM)}")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        if not rows:
            self.dim("  (none)")
            return
        widths = [len(h) for h in headers]
        for r in rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(str(cell)))
        head = "  ".join(h.upper().ljust(widths[i]) for i, h in enumerate(headers))
        self.print("  " + self._wrap(head, _DIM))
        for r in rows:
            self.print("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))

    def error(self, text: str) -> None:
        print(self._wrap(f"error: {text}", _COLOR[FAIL]), file=sys.stderr)
