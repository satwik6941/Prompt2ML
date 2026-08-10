"""
Regression tests for the data-leakage fixes.

Each test here corresponds to a defect that silently produced optimistic metrics:
a statistic fitted across the full dataset before the train/test split leaks
held-out information into the training features, and nothing downstream can
detect it after the fact.
"""

import json

import numpy as np
import pandas as pd
import pytest


def test_mean_imputation_is_deferred_not_applied(isolated_state, agent, csv_factory):
    """fill_mean must NOT be applied here — it fits a statistic over held-out rows."""
    df = pd.DataFrame({"age": [10.0, 20.0, np.nan, 40.0]})
    path = csv_factory(df)

    result = json.loads(
        agent.handle_missing_values(path, json.dumps({"age": "fill_mean"}))
    )

    assert result["status"] == "success"
    # The null survives on purpose.
    out = pd.read_parquet(result["output_path"])
    assert out["age"].isnull().sum() == 1

    # And the intent is recorded for the training pipeline.
    assert "age" in result["deferred_to_training_pipeline"]
    deferred = isolated_state.load_state()["deferred_transforms"]
    assert any(
        d["column"] == "age" and d["kind"] == "imputation" and "mean" in d["method"]
        for d in deferred
    )


@pytest.mark.parametrize("strategy", ["fill_mean", "fill_median", "fill_mode"])
def test_all_fitted_imputations_are_deferred(isolated_state, agent, csv_factory, strategy):
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 2.0]})
    path = csv_factory(df)

    result = json.loads(agent.handle_missing_values(path, json.dumps({"x": strategy})))

    assert "x" in result["deferred_to_training_pipeline"]
    assert pd.read_parquet(result["output_path"])["x"].isnull().sum() == 1


def test_row_local_imputation_is_applied_immediately(isolated_state, agent, csv_factory):
    """fill_zero depends on no other row, so there is nothing to leak."""
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    path = csv_factory(df)

    result = json.loads(agent.handle_missing_values(path, json.dumps({"x": "fill_zero"})))

    out = pd.read_parquet(result["output_path"])
    assert out["x"].isnull().sum() == 0
    assert out["x"].tolist() == [1.0, 0.0, 3.0]
    assert result["deferred_to_training_pipeline"] == {}


def test_frequency_encoding_is_deferred(isolated_state, agent, csv_factory):
    """A frequency map over the full column encodes the held-out distribution."""
    df = pd.DataFrame({"city": ["a", "a", "b", "c"]})
    path = csv_factory(df)

    result = json.loads(
        agent.encode_categorical_columns(path, json.dumps({"city": "frequency"}))
    )

    assert result["encoding_details"]["city"]["status"] == "deferred"
    out = pd.read_parquet(result["output_path"])
    assert out["city"].tolist() == ["a", "a", "b", "c"]   # untouched

    deferred = isolated_state.load_state()["deferred_transforms"]
    assert any(d["column"] == "city" and d["kind"] == "encoding" for d in deferred)


def test_target_encoding_still_refused(isolated_state, agent, csv_factory):
    df = pd.DataFrame({"city": ["a", "b"], "y": [0, 1]})
    path = csv_factory(df)

    result = json.loads(
        agent.encode_categorical_columns(path, json.dumps({"city": "target:y"}))
    )
    assert result["encoding_details"]["city"]["status"] == "refused"


def test_validator_ignores_nulls_it_deferred(isolated_state, agent, csv_factory):
    """
    The deferral must not send Agent 3 into a retry loop: a no_nulls check has to
    tolerate exactly the columns whose imputation was postponed.
    """
    df = pd.DataFrame({"age": [1.0, np.nan], "keep": [1.0, 2.0]})
    path = csv_factory(df, "validate.csv")

    agent.record_deferred_transform(
        kind="imputation",
        column="age",
        method="SimpleImputer(strategy='median')",
        reason="test",
    )

    result = json.loads(agent.validate_dataset(path, json.dumps({"no_nulls": True})))

    assert result["no_nulls"]["passed"] is True
    assert result["no_nulls"]["ignored_deferred"] == ["age"]


def test_validator_still_fails_on_undeferred_nulls(isolated_state, agent, csv_factory):
    """The tolerance must be scoped — an unrelated null column still fails."""
    df = pd.DataFrame({"age": [1.0, np.nan], "other": [np.nan, 2.0]})
    path = csv_factory(df, "validate2.csv")

    agent.record_deferred_transform(
        kind="imputation", column="age", method="SimpleImputer()", reason="test"
    )

    result = json.loads(agent.validate_dataset(path, json.dumps({"no_nulls": True})))

    assert result["no_nulls"]["passed"] is False
    assert "other" in result["no_nulls"]["null_columns"]
    assert "age" not in result["no_nulls"]["null_columns"]
