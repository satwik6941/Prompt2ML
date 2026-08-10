"""
Tests for the typed run contract and per-run store (M1 / D2).

These pin the properties the old ``pipeline_state.json`` design lacked:
per-phase status instead of one overloaded string, atomic and locked writes,
schema-version detection, and run isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from prompt2ml.core.contracts import (
    DataContract,
    DatasetManifest,
    DeferredTransform,
    Fact,
    Modality,
    Phase,
    PhaseStatus,
    Requirements,
    RunMeta,
)
from prompt2ml.core.runstore import RunStore, RunStoreError, SchemaMismatch

# ---------------------------------------------------------------------------
# Phase state machine
# ---------------------------------------------------------------------------

def test_each_phase_tracks_its_own_status():
    """
    The previous design had one global `status` string, so "pipeline_complete"
    meant both "preprocessing finished" and "everything finished".
    """
    meta = RunMeta(run_id="r1")
    meta.complete(Phase.PREPROCESS)

    assert meta.phase(Phase.PREPROCESS).status is PhaseStatus.COMPLETE
    assert meta.phase(Phase.TRAIN).status is PhaseStatus.PENDING
    assert meta.phase(Phase.REPORT).status is PhaseStatus.PENDING


def test_next_phase_is_the_resume_point():
    meta = RunMeta(run_id="r1")
    assert meta.next_phase() is Phase.INTERVIEW

    meta.complete(Phase.INTERVIEW)
    meta.complete(Phase.EXTRACT)
    assert meta.next_phase() is Phase.PREPROCESS

    for p in Phase:
        meta.complete(p)
    assert meta.next_phase() is None


def test_failure_records_the_error_and_counts_attempts():
    meta = RunMeta(run_id="r1")
    meta.start(Phase.TRAIN)
    meta.fail(Phase.TRAIN, "CUDA out of memory")

    rec = meta.phase(Phase.TRAIN)
    assert rec.status is PhaseStatus.FAILED
    assert rec.error == "CUDA out of memory"
    assert rec.attempts == 1

    meta.start(Phase.TRAIN)
    assert meta.phase(Phase.TRAIN).attempts == 2
    assert meta.phase(Phase.TRAIN).error is None, "retry must clear the stale error"


# ---------------------------------------------------------------------------
# Interview slots
# ---------------------------------------------------------------------------

def test_interview_gates_on_slot_coverage_not_question_count():
    """The fix for a hardcoded 7-question interview."""
    req = Requirements(goal="predict churn")
    assert req.missing_required(), "a fresh interview cannot be complete"

    for slot in ("task_intent", "model_output", "success_definition"):
        req.facts.append(Fact(slot=slot, value="x"))
    assert req.missing_required() == ["deployment_context"]

    req.facts.append(Fact(slot="deployment_context", value="internal dashboard"))
    assert req.missing_required() == []


def test_latest_answer_for_a_slot_wins():
    req = Requirements()
    req.facts.append(Fact(slot="model_output", value="a number"))
    req.facts.append(Fact(slot="model_output", value="a probability"))
    assert req.slot("model_output").value == "a probability"


def test_assumptions_are_distinguishable_from_user_answers():
    """'I don't know' answers get decided for the user, and must be reportable."""
    req = Requirements()
    req.facts.append(Fact(slot="task_intent", value="churn", source="user"))
    req.facts.append(
        Fact(slot="success_definition", value="AUC > 0.75",
             source="assumed", confidence="low")
    )
    assert [f.slot for f in req.assumptions] == ["success_definition"]


# ---------------------------------------------------------------------------
# Data contract / leakage bookkeeping
# ---------------------------------------------------------------------------

def test_deferred_columns_are_queryable():
    """The validator must not 'fix' nulls that were deliberately left."""
    contract = DataContract(
        modality=Modality.TABULAR,
        dataset_path="d.csv",
        deferred=[
            DeferredTransform(kind="imputation", column="age", method="SimpleImputer(median)"),
            DeferredTransform(kind="scaling", column="income", method="StandardScaler()"),
            DeferredTransform(kind="augmentation", method="RandomFlip()"),
        ],
    )
    assert contract.deferred_columns() == {"age", "income"}


def test_chronological_split_marks_the_contract_as_timeseries():
    contract = DataContract(
        modality=Modality.TABULAR, dataset_path="d.csv", split_strategy="chronological",
    )
    assert contract.is_timeseries, "row-dropping steps must be refused for these"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_runs_are_isolated_from_each_other(tmp_path: Path):
    """reset_run_id() was never called, so every run reused one folder."""
    a = RunStore.create(tmp_path, goal="first")
    b = RunStore.create(tmp_path, goal="second")

    assert a.dir != b.dir
    assert a.meta().goal == "first"
    assert b.meta().goal == "second"


def test_round_trip_preserves_every_field(tmp_path: Path):
    store = RunStore.create(tmp_path)
    manifest = DatasetManifest(
        name="tickets", source="huggingface", modality=Modality.TEXT,
        text_column="body", label_column="urgency",
    )
    store.write(manifest)

    loaded = store.require(DatasetManifest)
    assert loaded.modality is Modality.TEXT
    assert loaded.text_column == "body"
    assert loaded.label_column == "urgency"


def test_reading_an_unwritten_document_returns_none(tmp_path: Path):
    store = RunStore.create(tmp_path)
    assert store.read(DataContract) is None
    assert not store.exists(DataContract)


def test_require_names_the_missing_document(tmp_path: Path):
    store = RunStore.create(tmp_path)
    with pytest.raises(RunStoreError, match="data_contract.json"):
        store.require(DataContract)


def test_unknown_fields_are_rejected(tmp_path: Path):
    """extra='forbid' turns a typo into an error instead of a silent no-op."""
    store = RunStore.create(tmp_path)
    path = store.path_for(RunMeta)
    raw = json.loads(path.read_text())
    raw["totally_new_key"] = 1
    path.write_text(json.dumps(raw))

    with pytest.raises(ValidationError, match="totally_new_key"):
        store.require(RunMeta)


def test_schema_version_mismatch_is_detected(tmp_path: Path):
    """A run from an older build must fail loudly, not be half-read."""
    store = RunStore.create(tmp_path)
    path = store.path_for(RunMeta)
    raw = json.loads(path.read_text())
    raw["schema_version"] = 99
    path.write_text(json.dumps(raw))

    with pytest.raises(SchemaMismatch, match="99"):
        store.require(RunMeta)


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path):
    store = RunStore.create(tmp_path)
    store.write(Requirements(goal="x"))
    assert list(store.dir.glob("*.tmp")) == []


def test_interrupted_write_leaves_the_previous_version_intact(tmp_path: Path, monkeypatch):
    """A killed process must not leave a truncated document."""
    store = RunStore.create(tmp_path)
    store.write(Requirements(goal="good"))

    import prompt2ml.core.runstore as rs

    def boom(src, dst):
        raise KeyboardInterrupt("killed mid-write")

    monkeypatch.setattr(rs.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        store.write(Requirements(goal="bad"))

    monkeypatch.undo()
    assert store.require(Requirements).goal == "good"


def test_lock_is_released_after_write(tmp_path: Path):
    store = RunStore.create(tmp_path)
    store.write(Requirements(goal="one"))
    store.write(Requirements(goal="two"))
    assert not (store.dir / ".lock").exists()


def test_stale_lock_does_not_wedge_the_run(tmp_path: Path):
    """A lock left by a killed process must expire rather than block forever."""
    import os
    import time

    import prompt2ml.core.runstore as rs

    store = RunStore.create(tmp_path)
    lock = store.dir / ".lock"
    lock.write_text("99999")
    old = time.time() - (rs._LOCK_STALE_S + 10)
    os.utime(lock, (old, old))

    store.write(Requirements(goal="after stale lock"))
    assert store.require(Requirements).goal == "after stale lock"


def test_orphan_run_directory_is_reported_clearly(tmp_path: Path):
    """A dir with no run.json is an interrupted start, not a corrupt run."""
    (tmp_path / "20990101_000000").mkdir()
    with pytest.raises(RunStoreError, match="incomplete run directory"):
        RunStore.open(tmp_path, "20990101_000000")


def test_listing_skips_orphans_and_sorts_newest_first(tmp_path: Path):
    RunStore.create(tmp_path, goal="older")
    (tmp_path / "19700101_000000").mkdir()
    newer = RunStore.create(tmp_path, goal="newer")

    runs = RunStore.list_runs(tmp_path)
    assert all((r.dir / "run.json").exists() for r in runs)
    assert runs[0].dir == newer.dir


def test_phase_helpers_persist_through_the_store(tmp_path: Path):
    store = RunStore.create(tmp_path)
    store.start_phase(Phase.EXTRACT)
    store.complete_phase(Phase.EXTRACT)

    reopened = RunStore.open(tmp_path, store.run_id)
    assert reopened.meta().phase(Phase.EXTRACT).status is PhaseStatus.COMPLETE
    assert reopened.meta().next_phase() is Phase.INTERVIEW
