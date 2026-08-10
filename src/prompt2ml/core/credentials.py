"""
Credential inventory and validation.

The CLI asks for every credential up front (``prompt2ml init``) rather than
failing three phases into a run because a Kaggle token was missing. Each
credential knows how to describe itself, where to obtain it, and which phase
breaks without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Credential:
    env_var: str
    label: str
    obtain_url: str
    needed_for: str
    required: bool = True
    aliases: tuple[str, ...] = ()

    def value(self) -> str | None:
        for name in (self.env_var, *self.aliases):
            val = os.environ.get(name)
            if val and val.strip():
                return val.strip()
        return None

    @property
    def is_set(self) -> bool:
        return self.value() is not None

    def masked(self) -> str:
        val = self.value()
        if not val:
            return "not set"
        return f"{val[:4]}…{val[-2:]}" if len(val) > 8 else "set"


CREDENTIALS: tuple[Credential, ...] = (
    Credential(
        env_var="GOOGLE_API_KEY",
        label="Gemini API key",
        obtain_url="https://aistudio.google.com/apikey",
        needed_for="every agent — the pipeline cannot run without it",
    ),
    Credential(
        env_var="TAVILY_API_KEY",
        label="Tavily search key",
        obtain_url="https://tavily.com/",
        needed_for="dataset discovery and the research tool",
    ),
    Credential(
        env_var="KAGGLE_KEY",
        label="Kaggle API token",
        obtain_url="https://www.kaggle.com/settings",
        needed_for="downloading tabular datasets",
        aliases=("KAGGLE_API_TOKEN",),
    ),
    Credential(
        env_var="HUGGING_FACE_TOKEN",
        label="HuggingFace token",
        obtain_url="https://huggingface.co/settings/tokens",
        needed_for="text and vision datasets, and pretrained backbones",
        aliases=("HF_TOKEN",),
    ),
    Credential(
        env_var="JARVISLABS_API_KEY",
        label="JarvisLabs key",
        obtain_url="https://jarvislabs.ai/",
        needed_for="renting a cloud GPU (optional)",
        required=False,
        aliases=("JL_API_KEY",),
    ),
    Credential(
        env_var="RUNPOD_API_KEY",
        label="RunPod key",
        obtain_url="https://www.runpod.io/console/user/settings",
        needed_for="renting a cloud GPU (optional)",
        required=False,
    ),
)


def load_env_file(path: Path) -> int:
    """
    Read a ``.env`` file into ``os.environ`` without overwriting real env vars.

    A tiny parser rather than python-dotenv, so ``doctor`` still works when
    dependency installation is the thing that is broken.
    """
    if not path.exists():
        return 0
    loaded = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def missing_required() -> list[Credential]:
    return [c for c in CREDENTIALS if c.required and not c.is_set]


def write_env_template(path: Path) -> Path:
    """Write a .env.example listing every credential with its source URL."""
    lines = ["# Prompt2ML credentials — copy to .env and fill in.", ""]
    for cred in CREDENTIALS:
        tag = "" if cred.required else "  (optional)"
        lines += [
            f"# {cred.label}{tag} — {cred.needed_for}",
            f"# obtain: {cred.obtain_url}",
            f"{cred.env_var}=",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
