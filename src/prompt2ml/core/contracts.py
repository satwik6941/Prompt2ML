"""
Typed documents that flow between pipeline phases.

Replaces the single ``pipeline_state.json`` blob (one overloaded ``status``
string, list-or-dict ambiguity, unbounded growth, no concurrency) with a set of
small validated documents, one per concern, stored in a per-run directory.

Every model carries ``schema_version`` so a run written by an older build is
detected rather than silently misread.

See docs/IMPLEMENTATION_PLAN.md — D2.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class _Doc(BaseModel):
    """Base for every run document. Subclasses set ``filename``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: int = SCHEMA_VERSION

    # ClassVar, not a field: pydantic's metaclass intercepts attribute access
    # for declared fields, so a plain annotation here would make the class-level
    # ``SomeDoc.filename`` lookup raise AttributeError.
    filename: ClassVar[str] = ""

    def __init_subclass__(cls, filename: str = "", **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if filename:
            cls.filename = filename


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Modality(str, Enum):
    """What kind of data this is — the pipeline routing key (D1)."""

    TABULAR = "tabular"
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    TIMESERIES = "timeseries"


class Phase(str, Enum):
    INTERVIEW = "interview"
    EXTRACT = "extract"
    PREPROCESS = "preprocess"
    TRAIN = "train"
    REPORT = "report"


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"


class Residency(str, Enum):
    """Whether the dataset may leave this machine (D7)."""

    LOCAL_ONLY = "local_only"
    ASK = "ask"
    ALLOW = "allow"


# ---------------------------------------------------------------------------
# run.json — phase state machine
# ---------------------------------------------------------------------------

class PhaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    status: PhaseStatus = PhaseStatus.PENDING
    started_at: _dt.datetime | None = None
    completed_at: _dt.datetime | None = None
    attempts: int = 0
    error: str | None = None

    @property
    def is_done(self) -> bool:
        return self.status in (PhaseStatus.COMPLETE, PhaseStatus.SKIPPED)


class RunMeta(_Doc, filename="run.json"):
    """
    Per-run metadata and the phase state machine.

    Status is tracked per phase. There is deliberately no single global status
    string — that is what let ``"pipeline_complete"`` mean both "preprocessing
    finished" and "the whole pipeline finished" in the previous design.
    """

    run_id: str
    created_at: _dt.datetime = Field(default_factory=_utcnow)
    goal: str = ""
    phases: dict[Phase, PhaseRecord] = Field(
        default_factory=lambda: {p: PhaseRecord() for p in Phase}
    )
    cost_usd: float = 0.0

    def phase(self, phase: Phase) -> PhaseRecord:
        return self.phases.setdefault(phase, PhaseRecord())

    def start(self, phase: Phase) -> None:
        rec = self.phase(phase)
        rec.status = PhaseStatus.RUNNING
        rec.started_at = _utcnow()
        rec.attempts += 1
        rec.error = None

    def complete(self, phase: Phase) -> None:
        rec = self.phase(phase)
        rec.status = PhaseStatus.COMPLETE
        rec.completed_at = _utcnow()
        rec.error = None

    def fail(self, phase: Phase, error: str) -> None:
        rec = self.phase(phase)
        rec.status = PhaseStatus.FAILED
        rec.completed_at = _utcnow()
        rec.error = error

    def next_phase(self) -> Phase | None:
        """First phase not yet complete — the resume point."""
        for p in Phase:
            if not self.phase(p).is_done:
                return p
        return None


# ---------------------------------------------------------------------------
# requirements.json — the interview's structured output (D-interview)
# ---------------------------------------------------------------------------

REQUIRED_SLOTS: tuple[str, ...] = (
    "task_intent",
    "model_output",
    "success_definition",
    "deployment_context",
)


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: str
    value: str
    source: str = "user"            # "user" | "assumed" | "researched"
    confidence: str = "high"        # high | medium | low
    follow_up_to: str | None = None  # slot this question built on


class Requirements(_Doc, filename="requirements.json"):
    """
    Structured interview output. Downstream agents read slots, not a 1500-word
    essay, so a phase can consume one fact without parsing prose.
    """

    goal: str = ""
    facts: list[Fact] = Field(default_factory=list)
    narrative: str = ""

    def slot(self, name: str) -> Fact | None:
        for f in reversed(self.facts):
            if f.slot == name:
                return f
        return None

    def missing_required(self) -> list[str]:
        """Empty required slots. The interview cannot finalize while non-empty."""
        return [s for s in REQUIRED_SLOTS if self.slot(s) is None]

    @property
    def assumptions(self) -> list[Fact]:
        """Facts the system chose on the user's behalf — surfaced in the report."""
        return [f for f in self.facts if f.source == "assumed"]


# ---------------------------------------------------------------------------
# dataset_manifest.json — what extraction produced (D1)
# ---------------------------------------------------------------------------

class DataFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    role: str = "train"        # train | validation | test | index | all
    rows: int | None = None
    size_bytes: int | None = None


class DatasetManifest(_Doc, filename="dataset_manifest.json"):
    """
    Extraction's typed output. Declaring modality here is what makes CV/NLP
    routable — previously extraction returned a bare path and everything
    downstream assumed a rectangular CSV.
    """

    name: str
    source: str                       # kaggle | huggingface | local
    modality: Modality
    files: list[DataFile] = Field(default_factory=list)
    label_column: str | None = None
    label_layout: str | None = None   # e.g. "column", "folder_per_class"
    text_column: str | None = None
    time_column: str | None = None
    license: str | None = None
    notes: str = ""

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes or 0 for f in self.files)


# ---------------------------------------------------------------------------
# data_contract.json — preprocessing's output, trainer's input (D5)
# ---------------------------------------------------------------------------

class DeferredTransform(BaseModel):
    """
    A transform that learns from data and therefore must be fitted inside the
    training split, never at preprocessing time.

    This is the machine-readable half of the leakage rule: preprocessing records
    the intent, the trainer rebuilds it inside a Pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str          # imputation | scaling | encoding | vectorization | augmentation
    column: str | None = None
    method: str        # e.g. "SimpleImputer(strategy='median')"
    reason: str = ""


class DataContract(_Doc, filename="data_contract.json"):
    """The single interface between any preprocessing pipeline and the trainer."""

    modality: Modality
    dataset_path: str
    features: list[str] = Field(default_factory=list)
    target: str | None = None
    problem_type: ProblemType | None = None
    dtypes: dict[str, str] = Field(default_factory=dict)
    n_rows: int | None = None
    split_strategy: str = "random"   # random | stratified | chronological | group
    group_column: str | None = None
    deferred: list[DeferredTransform] = Field(default_factory=list)
    class_balance: dict[str, int] = Field(default_factory=dict)
    notes: str = ""

    def deferred_columns(self) -> set[str]:
        """Columns whose nulls are expected — the validator must not 'fix' them."""
        return {d.column for d in self.deferred if d.column}

    @property
    def is_timeseries(self) -> bool:
        return self.modality is Modality.TIMESERIES or self.split_strategy == "chronological"


# ---------------------------------------------------------------------------
# hardware.json — profile plus the budget it implies (D6)
# ---------------------------------------------------------------------------

class GPUInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    vram_gb: float
    driver: str | None = None
    cuda: str | None = None


class TrainingBudget(BaseModel):
    """
    Derived limits the planner must obey and the codegen gate enforces.

    A profile that only informs the prompt gets ignored; these are checked.
    """

    model_config = ConfigDict(extra="forbid")

    max_params_m: int
    max_batch_tokens: int
    precision: str                 # fp32 | fp16 | bf16
    max_train_minutes: int
    allow_full_finetune: bool
    prefer_lora: bool
    max_image_px: int = 224


class HardwareProfile(_Doc, filename="hardware.json"):
    cpu_cores: int = 1
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    platform: str = ""
    gpus: list[GPUInfo] = Field(default_factory=list)
    selected_gpu: int | None = None
    docker_available: bool = False
    docker_gpu_passthrough: bool | None = None   # None = not probed
    sandbox_image_present: bool = False
    budget: TrainingBudget | None = None
    probed_at: _dt.datetime = Field(default_factory=_utcnow)

    @property
    def gpu(self) -> GPUInfo | None:
        if self.selected_gpu is None:
            return None
        if 0 <= self.selected_gpu < len(self.gpus):
            return self.gpus[self.selected_gpu]
        return None

    @property
    def vram_gb(self) -> float:
        g = self.gpu
        return g.vram_gb if g else 0.0


# ---------------------------------------------------------------------------
# plan.json — the training plan
# ---------------------------------------------------------------------------

class ModelCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    rationale: str = ""
    hyperparameter_ranges: dict[str, str] = Field(default_factory=dict)
    is_baseline: bool = False


class ComputeRequirement(BaseModel):
    """
    What the planner declares it needs. It never names a provider — the
    ComputeBroker maps this to a backend (D7).
    """

    model_config = ConfigDict(extra="forbid")

    vram_gb: float = 0.0
    est_minutes: int = 10
    needs_cuda: bool = False


class MLPlan(_Doc, filename="plan.json"):
    problem_type: ProblemType
    target: str | None = None
    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list)
    cv_strategy: str = "KFold(5)"
    candidates: list[ModelCandidate] = Field(default_factory=list)
    compute: ComputeRequirement = Field(default_factory=ComputeRequirement)
    narrative: str = ""

    @property
    def baseline(self) -> ModelCandidate | None:
        return next((c for c in self.candidates if c.is_baseline), None)


ALL_DOCS: tuple[type[_Doc], ...] = (
    RunMeta,
    Requirements,
    DatasetManifest,
    DataContract,
    HardwareProfile,
    MLPlan,
)
