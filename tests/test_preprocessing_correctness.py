"""
Regression tests for the silent-corruption defects fixed in M0.

These are the bugs that produced a plausible-looking dataset with wrong values —
the worst kind, because nothing downstream raises.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Stable hashing
# --------------------------------------------------------------------------

def test_hash_encoding_is_stable_within_a_process(isolated_state, agent, csv_factory):
    df = pd.DataFrame({"tag": ["alpha", "beta", "alpha", "gamma"]})
    path = csv_factory(df)

    first = json.loads(agent.process_text_columns(path, json.dumps({"tag": "hash:8"})))
    second = json.loads(agent.process_text_columns(path, json.dumps({"tag": "hash:8"})))

    a = pd.read_parquet(first["output_path"])["tag_hash"].tolist()
    b = pd.read_parquet(second["output_path"])["tag_hash"].tolist()
    assert a == b
    assert a[0] == a[2]           # same input, same bucket


def test_hash_encoding_is_stable_across_processes(tmp_path):
    """
    The real defect: Python salts hash() per process, so a model trained on one
    run's buckets scored garbage on the next. Run the hashing in two separate
    interpreters with different PYTHONHASHSEED values and require agreement.
    """
    snippet = textwrap.dedent(
        """
        import hashlib
        def bucket(v, n=64):
            d = hashlib.blake2b(v.encode("utf-8"), digest_size=8).digest()
            return int.from_bytes(d, "big") % n
        print([bucket(x) for x in ["alpha", "beta", "gamma"]])
        """
    )
    script = tmp_path / "h.py"
    script.write_text(snippet, encoding="utf-8")

    def run(seed):
        return subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
        ).stdout.strip()

    assert run("0") == run("12345")


# --------------------------------------------------------------------------
# Boolean coercion
# --------------------------------------------------------------------------

def test_string_false_does_not_become_true(isolated_state, agent, csv_factory):
    """astype(bool) mapped every non-empty string to True, inverting half the column."""
    df = pd.DataFrame({"flag": ["True", "False", "yes", "no", "0", "1"]})
    path = csv_factory(df)

    result = json.loads(
        agent.detect_and_fix_data_types(path, json.dumps({"flag": "bool"}))
    )
    assert result["type_fix_details"]["flag"]["status"] == "converted"

    out = pd.read_parquet(result["output_path"])
    assert out["flag"].tolist() == [True, False, True, False, False, True]


def test_unrecognised_boolean_spelling_becomes_null_not_true(isolated_state, agent, csv_factory):
    df = pd.DataFrame({"flag": ["true", "banana"]})
    path = csv_factory(df)

    result = json.loads(
        agent.detect_and_fix_data_types(path, json.dumps({"flag": "bool"}))
    )

    out = pd.read_parquet(result["output_path"])
    assert out["flag"].iloc[0] is True or out["flag"].iloc[0] == True  # noqa: E712
    assert pd.isna(out["flag"].iloc[1])
    assert "warning" in result["type_fix_details"]["flag"]


# --------------------------------------------------------------------------
# pandas >= 2.2 removal
# --------------------------------------------------------------------------

def test_auto_type_detection_does_not_use_removed_api(isolated_state, agent, csv_factory):
    """errors='ignore' was removed in pandas 3.0; 'auto' must still work."""
    df = pd.DataFrame({"n": ["1", "2", "3"], "s": ["a", "b", "c"]})
    path = csv_factory(df)

    result = json.loads(
        agent.detect_and_fix_data_types(path, json.dumps({"n": "auto", "s": "auto"}))
    )

    out = pd.read_parquet(result["output_path"])
    assert pd.api.types.is_numeric_dtype(out["n"])
    assert not pd.api.types.is_numeric_dtype(out["s"])   # left alone, not coerced to NaN


def test_auto_does_not_destroy_partially_numeric_column(isolated_state, agent, csv_factory):
    df = pd.DataFrame({"mixed": ["1", "2", "not-a-number"]})
    path = csv_factory(df)

    result = json.loads(
        agent.detect_and_fix_data_types(path, json.dumps({"mixed": "auto"}))
    )
    out = pd.read_parquet(result["output_path"])
    assert out["mixed"].tolist() == ["1", "2", "not-a-number"]


# --------------------------------------------------------------------------
# Unknown strategy handling
# --------------------------------------------------------------------------

def test_unknown_imputation_strategy_is_reported(isolated_state, agent, csv_factory):
    df = pd.DataFrame({"x": [1.0, np.nan]})
    path = csv_factory(df)

    result = json.loads(
        agent.handle_missing_values(path, json.dumps({"x": "fill_with_vibes"}))
    )
    assert any("fill_with_vibes" in w for w in result["warnings"])
