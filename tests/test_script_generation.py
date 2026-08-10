"""
The generated validation script must stay syntactically valid no matter what
the LLM wrote into the preprocessing plan.

Plan fields are free-text. They were previously interpolated straight into
single-quoted Python string literals, so one apostrophe turned the whole script
into a SyntaxError — and the failure surfaced only as the useless message
"Validation script failed", never as "your plan contained a quote".
"""

import ast
import json

import pytest


def _lit(value) -> str:
    """Mirror of the helper used inside run_plan_aware_validation."""
    return json.dumps("" if value is None else str(value))


HOSTILE_VALUES = [
    "drop_keep_first",
    "it's a plan",                             # apostrophe
    'he said "drop them"',                     # double quotes
    "line one\nline two",                      # newline
    "backslash \\ here",                       # backslash
    "'''triple quoted'''",                     # triple quotes
    "'; import os; os.system('echo pwned'); '",  # injection attempt
    "unicode — em dash and ünïcødé",
]


@pytest.mark.parametrize("value", HOSTILE_VALUES)
def test_literal_helper_produces_parseable_python(value):
    source = f"planned = {_lit(value)}\n"
    tree = ast.parse(source)                        # must not raise
    assert isinstance(tree.body[0], ast.Assign)


@pytest.mark.parametrize("value", HOSTILE_VALUES)
def test_literal_helper_round_trips_the_value(value):
    namespace: dict = {}
    exec(f"planned = {_lit(value)}", {}, namespace)  # noqa: S102 - testing the escaping
    assert namespace["planned"] == value


def test_injection_attempt_stays_inert_data():
    """The classic payload must land as a string, not as executed code."""
    payload = "'; raise RuntimeError('executed'); '"
    namespace: dict = {}
    exec(f"planned = {_lit(payload)}", {}, namespace)  # noqa: S102
    assert namespace["planned"] == payload


@pytest.mark.parametrize(
    "step_name",
    ["custom smote", "step-1", "2nd_pass", "class", "with.dot", ""],
)
def test_sandbox_step_names_are_not_used_as_identifiers(step_name):
    """
    Sandbox step names came from the LLM and were spliced in as Python variable
    names. Anything with a space, hyphen, leading digit, or reserved word broke
    the script. They must be dict keys instead.
    """
    source = (
        "_sandbox_check = {}\n"
        f"_sandbox_check['step'] = {_lit(step_name)}\n"
        f"results = {{}}\n"
        f"results[{_lit('check_sandbox_' + str(step_name))}] = _sandbox_check\n"
    )
    ast.parse(source)   # must not raise


def test_column_names_with_quotes_are_safe():
    """Column names come from real datasets and are not always well behaved."""
    col = "it's a column"
    source = (
        f"enc_check = {{}}\n"
        f"enc_check['col'] = {_lit(col)}\n"
        f"enc_check['gone'] = {_lit(col)} not in pre.columns\n"
        f"enc_check['dummies'] = [c for c in pre.columns if c.startswith({_lit(col + '_')})]\n"
    )
    ast.parse(source)
