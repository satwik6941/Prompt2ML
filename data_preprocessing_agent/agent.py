"""
Data Preprocessing Multi-Agent System (Google ADK)

Architecture:
    SequentialAgent (root orchestrator)
    ├── Agent 1: Dataset Analyzer — selects the best dataset
    ├── Agent 2: Preprocessing Strategist — creates a preprocessing plan
    ├── LoopAgent (max 5 iterations)
    │   ├── Agent 3: Preprocessing Executor — runs preprocessing code
    │   ├── Agent 4: Validation Agent — validates results
    │   └── LoopEscalationChecker — escalates if PASS or max retries
    └── Agent 5: Report Generator — writes A-to-Z preprocessing report

Data flow uses output_key → {state_key} templating between agents.
Agents also read/write pipeline_state.json for persistent cross-session state.
Report is saved to outputs/ folder as a detailed markdown file.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from pipeline_state import load_state, save_state, get_run_dir, reset_run_dir_cache

load_dotenv()

# ============================================================
# CONSTANTS
# ============================================================

MODEL = "gemini-3.1-flash-lite"
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODIFIED_DATASETS_DIR = PROJECT_ROOT / "modified_datasets"   # base root — never written to directly
# All step files are written to modified_datasets/<problem-slug>/ via get_run_dir()

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".parquet", ".xlsx", ".xls"}
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
    ".pkl", ".pickle", ".h5", ".hdf5", ".pt", ".pth", ".bin",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".mp3", ".mp4", ".wav", ".avi",
    ".pdf", ".doc", ".docx",
    ".pyc", ".exe", ".dll", ".so",
    ".md", ".yml", ".yaml", ".cfg", ".ini", ".gitignore",
}


# ============================================================
# ==================  AGENT 1 TOOLS  =========================
# ============================================================

def get_project_context() -> str:
    """
    Load the current pipeline state from pipeline_state.json and return
    the user's project goal and report as a JSON string.
    Use this to understand what the user wants to build so you can
    select the most relevant dataset.
    """
    state = load_state()
    context = {
        "user_goal": state.get("user_goal", ""),
        "report": state.get("report", "")[:3000] if state.get("report") else "",
        "status": state.get("status", ""),
    }
    return json.dumps(context, indent=2)


def scan_datasets_folder() -> str:
    """
    Scan the 'datasets' folder in the project root.
    For each subfolder, list all files and read a sample (first 5 rows)
    of each supported data file (csv, tsv, json, jsonl, parquet, xlsx, txt).
    Returns a JSON string with folder summaries, file metadata, column names,
    dtypes, missing values, and a text preview of each file.
    """
    import pandas as pd

    if not DATASETS_DIR.exists():
        return json.dumps({"error": f"Datasets folder not found: {DATASETS_DIR}"})

    folder_summaries = []

    for folder in sorted(DATASETS_DIR.iterdir()):
        if not folder.is_dir():
            continue

        folder_info = {
            "folder_name": folder.name,
            "folder_path": str(folder),
            "files": [],
            "skipped_files": [],
        }

        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()

            if ext in SKIP_EXTENSIONS or ext not in SUPPORTED_EXTENSIONS:
                folder_info["skipped_files"].append(file_path.name)
                continue

            file_info = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": ext,
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "success": False,
                "preview": "",
                "columns": [],
                "num_columns": 0,
                "num_rows_hint": "unknown",
            }

            try:
                if ext == ".csv":
                    df = pd.read_csv(file_path, nrows=5)
                    with open(file_path, encoding="utf-8", errors="ignore") as row_f:
                        total_rows = sum(1 for _ in row_f) - 1
                    file_info["num_rows_hint"] = str(total_rows)
                elif ext == ".tsv":
                    df = pd.read_csv(file_path, sep="\t", nrows=5)
                    with open(file_path, encoding="utf-8", errors="ignore") as row_f:
                        total_rows = sum(1 for _ in row_f) - 1
                    file_info["num_rows_hint"] = str(total_rows)
                elif ext == ".jsonl":
                    df = pd.read_json(file_path, lines=True, nrows=5)
                elif ext == ".json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    if isinstance(raw, list):
                        df = pd.DataFrame(raw[:5])
                    else:
                        file_info["preview"] = json.dumps(raw, indent=2)[:2000]
                        file_info["success"] = True
                        folder_info["files"].append(file_info)
                        continue
                elif ext == ".parquet":
                    df = pd.read_parquet(file_path).head(5)
                    total_rows = pd.read_parquet(file_path, columns=[]).shape[0]
                    file_info["num_rows_hint"] = str(total_rows)
                elif ext in (".xlsx", ".xls"):
                    df = pd.read_excel(file_path, nrows=5)
                elif ext == ".txt":
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [f.readline() for _ in range(5)]
                    file_info["preview"] = "".join(lines)
                    file_info["success"] = True
                    folder_info["files"].append(file_info)
                    continue
                else:
                    continue

                file_info["preview"] = df.to_string(index=False)
                file_info["columns"] = list(df.columns)
                file_info["num_columns"] = len(df.columns)
                file_info["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                file_info["missing_values"] = df.isnull().sum().to_dict()
                file_info["success"] = True

            except Exception as e:
                file_info["preview"] = f"[Error reading file: {str(e)}]"

            folder_info["files"].append(file_info)

        if folder_info["files"]:
            folder_summaries.append(folder_info)

    if not folder_summaries:
        return json.dumps({"error": "No readable datasets found in any subfolder."})

    return json.dumps(folder_summaries, indent=2)


def save_selected_dataset(
    selected_folder: str,
    selected_file: str,
    selected_file_path: str,
    reason: str,
    columns: str,
    num_rows: str,
    potential_issues: str,
) -> str:
    """
    Save the selected dataset info to pipeline_state.json so that
    Agent 2 can read it. Call this AFTER you have analyzed all datasets
    and decided which one is the best.

    Args:
        selected_folder: Name of the folder containing the best dataset.
        selected_file: Name of the selected file (e.g. 'data.csv').
        selected_file_path: Full absolute path to the selected file.
        reason: Detailed reason why this dataset was selected.
        columns: Comma-separated list of column names in the dataset.
        num_rows: Approximate number of rows as a string.
        potential_issues: Comma-separated list of potential data quality issues.

    Returns:
        Confirmation message.
    """
    agent1_output = {
        "selected_dataset": {
            "selected_folder": selected_folder,
            "selected_file": selected_file,
            "selected_file_path": selected_file_path,
            "reason": reason,
            "columns": [c.strip() for c in columns.split(",") if c.strip()],
            "num_rows": num_rows,
            "potential_issues": [i.strip() for i in potential_issues.split(",") if i.strip()],
        }
    }

    existing = load_state().get("agent1_output", [])
    if not isinstance(existing, list):
        existing = [existing]
    existing.append(agent1_output)

    save_state({
        "agent1_output": existing,
        "selected_dataset_path": selected_file_path,
        "status": "agent1_done",
    })

    return f"Selected dataset saved: {selected_folder}/{selected_file}"


# ============================================================
# ==================  AGENT 2 TOOLS  =========================
# ============================================================

def load_dataset_profile() -> str:
    """
    Load the selected dataset from pipeline_state.json (Agent 1's output),
    then perform a deep statistical profile of the data.
    Returns JSON with: shape, dtypes, missing values per column,
    unique counts, basic stats (mean/median/std/min/max),
    sample rows, and detected data quality issues.
    """
    import pandas as pd
    import numpy as np

    state = load_state()
    agent1_out = _latest(state.get("agent1_output"))
    selected = agent1_out.get("selected_dataset", {})
    file_path = selected.get("selected_file_path", "")

    if not file_path or not Path(file_path).exists():
        return json.dumps({"error": f"Selected dataset not found: {file_path}"})

    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t")
        elif ext == ".json":
            df = pd.read_json(file_path)
        elif ext == ".jsonl":
            df = pd.read_json(file_path, lines=True)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            return json.dumps({"error": f"Unsupported file type: {ext}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to read dataset: {str(e)}"})

    profile = {
        "file_path": file_path,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "unique_counts": df.nunique().to_dict(),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }

    # Numeric stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        profile["numeric_stats"] = df[numeric_cols].describe().round(4).to_dict()

        # Outlier detection via IQR
        outlier_cols = []
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
                if outlier_count > 0:
                    outlier_cols.append({"column": col, "outlier_count": outlier_count, "pct": round(outlier_count / len(df) * 100, 2)})
        profile["potential_outliers"] = outlier_cols

        # Highly correlated pairs (>0.9)
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs()
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    if corr.iloc[i, j] > 0.9:
                        high_corr.append({"col1": corr.columns[i], "col2": corr.columns[j], "correlation": round(float(corr.iloc[i, j]), 4)})
            profile["high_correlations"] = high_corr

    # Categorical stats
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        cat_stats = {}
        for col in cat_cols:
            top_vals = df[col].value_counts().head(10).to_dict()
            cat_stats[col] = {
                "unique": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in top_vals.items()},
                "avg_length": round(float(df[col].dropna().astype(str).str.len().mean()), 1),
            }
        profile["categorical_stats"] = cat_stats

    # Constant columns
    near_constant = [col for col in df.columns if df[col].nunique() <= 1]
    if near_constant:
        profile["constant_columns"] = near_constant

    # Duplicates
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # Unparsed date columns
    date_candidates = []
    for col in cat_cols:
        sample = df[col].dropna().head(20)
        try:
            pd.to_datetime(sample)
            date_candidates.append(col)
        except (ValueError, TypeError):
            pass
    if date_candidates:
        profile["unparsed_date_columns"] = date_candidates

    return json.dumps(profile, indent=2, default=str)


def get_user_requirements() -> str:
    """
    Load the user's original goal, Q&A pairs, report, and Agent 1's
    selection reason from pipeline_state.json. This gives full context
    for creating a preprocessing plan aligned with the user's ML goal.
    """
    state = load_state()
    requirements = {
        "user_goal": state.get("user_goal", ""),
        "qa_pairs": state.get("qa_pairs", []),
        "report_summary": state.get("report", "")[:4000] if state.get("report") else "",
        "agent1_selection": _latest(state.get("agent1_output")).get("selected_dataset", {}),
    }
    return json.dumps(requirements, indent=2, default=str)


def save_preprocessing_plan(
    plan_summary: str,
    target_column: str,
    columns_to_drop: str,
    missing_value_strategy: str,
    encoding_strategy: str,
    scaling_strategy: str,
    outlier_strategy: str,
    feature_engineering: str,
    text_processing: str,
    datetime_processing: str,
    duplicate_strategy: str,
    final_validation_checks: str,
    step_by_step_order: str,
) -> str:
    """
    Save the complete preprocessing plan to pipeline_state.json.
    Agent 3 will read this plan and execute each step.

    Args:
        plan_summary: A 2-3 sentence summary of the overall preprocessing approach.
        target_column: The target/label column name for ML (or 'none' if unsupervised).
        columns_to_drop: Comma-separated column names to drop and why (format: 'col:reason,col:reason').
        missing_value_strategy: Detailed strategy per column (format: 'col:strategy,col:strategy').
            Strategies: drop_rows, fill_mean, fill_median, fill_mode, fill_zero, fill_ffill, fill_custom.
        encoding_strategy: How to encode categorical columns (format: 'col:method,col:method').
            Methods: label_encode, onehot_encode, ordinal_encode, target_encode, frequency_encode, drop.
        scaling_strategy: How to scale numeric columns (format: 'col:method,col:method' or 'all:method').
            Methods: standard, minmax, robust, log_transform, none.
        outlier_strategy: How to handle outliers (format: 'col:method,col:method').
            Methods: clip_iqr, remove_iqr, clip_zscore, remove_zscore, log_transform, keep.
        feature_engineering: New features to create (format: 'new_col:formula_or_description,...').
        text_processing: How to handle text columns (format: 'col:method,...').
            Methods: tfidf, countvec, length_feature, word_count, drop, keep.
        datetime_processing: How to handle datetime columns (format: 'col:extractions,...').
            Extractions: year, month, day, dayofweek, hour, is_weekend, time_since.
        duplicate_strategy: How to handle duplicate rows: 'drop_all', 'drop_keep_first', 'drop_keep_last', 'keep'.
        final_validation_checks: Comma-separated list of checks to run after preprocessing.
        step_by_step_order: Ordered comma-separated list of steps (e.g., 'drop_columns,handle_missing,encode,...').

    Returns:
        Confirmation message.
    """
    plan = {
        "plan_summary": plan_summary,
        "target_column": target_column,
        "columns_to_drop": columns_to_drop,
        "missing_value_strategy": missing_value_strategy,
        "encoding_strategy": encoding_strategy,
        "scaling_strategy": scaling_strategy,
        "outlier_strategy": outlier_strategy,
        "feature_engineering": feature_engineering,
        "text_processing": text_processing,
        "datetime_processing": datetime_processing,
        "duplicate_strategy": duplicate_strategy,
        "final_validation_checks": final_validation_checks,
        "step_by_step_order": step_by_step_order,
    }

    existing = load_state().get("agent2_output", [])
    if not isinstance(existing, list):
        existing = [existing]
    existing.append({"preprocessing_plan": plan})

    save_state({
        "agent2_output": existing,
        "status": "agent2_done",
    })

    return f"Preprocessing plan saved. Steps: {step_by_step_order}"


# ============================================================
# ==================  AGENT 3 TOOLS  =========================
# ============================================================

def _resolve_local_path(p: str) -> str:
    """
    Defensive path translator for built-in (host-side) preprocessing tools.

    The LLM sometimes hands a sandbox path like '/workspace/output/foo.csv'
    to a built-in tool that runs on the host. Pandas would then crash with
    [Errno 2] No such file or directory. This helper rewrites such paths to
    the equivalent local file under modified_datasets/ if it exists, and
    raises a CLEAR error otherwise so the LLM can recover instead of looping.
    """
    if not p:
        raise FileNotFoundError(
            "Empty dataset_path. Built-in tools need an absolute LOCAL path."
        )

    # Sandbox path? Try to remap to the run-specific dir first, then base dir
    if p.startswith("/workspace/") or p.startswith("\\workspace\\"):
        basename = os.path.basename(p.replace("\\", "/"))
        run_candidate = get_run_dir() / basename
        if run_candidate.exists():
            return str(run_candidate)
        base_candidate = MODIFIED_DATASETS_DIR / basename
        if base_candidate.exists():
            return str(base_candidate)
        raise FileNotFoundError(
            f"Path '{p}' is a SANDBOX path. Built-in tools run on the HOST and "
            f"cannot read sandbox files. Either (1) call download_from_sandbox "
            f"first to copy the file to '{run_candidate}', or "
            f"(2) skip the sandbox entirely and use the local path returned by "
            f"the previous built-in tool. Built-in tools never accept /workspace/... paths."
        )

    if not os.path.isabs(p):
        # Resolve relative to project root
        candidate = PROJECT_ROOT / p
        if candidate.exists():
            return str(candidate)

    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Local file not found: {p}. Use the path returned by the previous "
            f"built-in tool, or the 'dataset_path' from get_preprocessing_context."
        )
    return p


def _next_step_path(tool_name: str) -> str:
    """
    Generate the next auto-named step output path and increment the step counter.
    Files are written to modified_datasets/<problem-slug>/step_N_<tool_name>.csv
    so each pipeline run is isolated in its own subfolder.
    """
    state = load_state()
    step_num = state.get("current_step", 0) + 1
    save_state({"current_step": step_num})
    run_dir = get_run_dir()
    return str(run_dir / f"step_{step_num}_{tool_name}.csv")


def _latest(value) -> dict:
    """
    Safely get the latest entry from a field that may be a list (new format)
    or a plain dict (old format from pipeline_state.json written before this change).
    Returns an empty dict if the value is missing or empty.
    """
    if isinstance(value, list):
        return value[-1] if value else {}
    if isinstance(value, dict):
        return value
    return {}


def _parse_json_param(param: str, param_name: str) -> tuple:
    """
    Safely parse a JSON string parameter from an agent tool call.
    Returns (parsed_value, None) on success, or (None, error_json_string) on failure.
    The caller should check: if err is not None: return err
    """
    try:
        return json.loads(param), None
    except (json.JSONDecodeError, TypeError) as e:
        err = json.dumps({
            "error": f"Invalid JSON in parameter '{param_name}': {e}",
            "received": str(param)[:200],
            "fix": f"Pass a valid JSON string for '{param_name}'.",
        }, indent=2)
        return None, err


def _safe_read_csv(path: str):
    """
    Read a CSV file with graceful error handling for encoding issues and corruption.
    Returns (dataframe, None) on success, or (None, error_json_string) on failure.
    """
    import pandas as pd
    encodings = ["utf-8", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc), None
        except UnicodeDecodeError:
            last_err = f"Encoding error with {enc}"
        except Exception as e:
            last_err = str(e)
            break
    err = json.dumps({
        "error": f"Could not read CSV: {last_err}",
        "path": path,
        "fix": "Ensure the file exists and is a valid CSV. Check encoding.",
    }, indent=2)
    return None, err


def _safe_write_csv(df, output_path: str) -> str | None:
    """
    Write a DataFrame to CSV with error handling.
    Returns None on success, or an error JSON string on failure.
    Saves last_step_output_path to state on every successful write for crash recovery.
    """
    try:
        df.to_csv(output_path, index=False)
        save_state({"last_step_output_path": output_path})
        return None
    except OSError as e:
        return json.dumps({
            "error": f"Could not write output file: {e}",
            "path": output_path,
            "fix": "Check disk space and write permissions for the modified_datasets/ folder.",
        }, indent=2)


def get_preprocessing_context() -> str:
    """
    Load everything Agent 3 needs: user goal, Agent 1's dataset selection,
    Agent 2's preprocessing plan, the dataset file path, and any previous
    validation feedback from Agent 4 (for retry loops).
    """
    state = load_state()
    agent1_output = state.get("agent1_output", [])
    agent2_output = state.get("agent2_output", [])
    context = {
        "user_goal": state.get("user_goal", ""),
        "selected_dataset": _latest(agent1_output).get("selected_dataset", {}),
        "preprocessing_plan": _latest(agent2_output).get("preprocessing_plan", {}),
        "dataset_path": state.get("selected_dataset_path", ""),
        "validation_feedback": state.get("agent4_feedback", ""),
        "iteration": state.get("loop_iteration", 0),
        "previous_attempts": state.get("agent3_iterations", []),
        "previous_validations": state.get("agent4_iterations", []),
    }
    return json.dumps(context, indent=2, default=str)


def handle_missing_values(
    dataset_path: str,
    strategy_json: str,
) -> str:
    """
    Handle missing values in the dataset using column-specific strategies.

    Args:
        dataset_path: Absolute path to the input CSV/Parquet file.
        strategy_json: JSON string mapping column names to strategies.
            Example: '{"age": "fill_median", "city": "fill_mode", "temp": "fill_mean", "id": "drop_rows"}'
            Supported strategies: fill_mean, fill_median, fill_mode, fill_zero,
            fill_ffill, fill_bfill, fill_interpolate, drop_rows, drop_column.

    Returns:
        JSON with before/after missing counts and rows affected.
    """
    import pandas as pd

    output_path = _next_step_path("handle_missing")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    before_missing = df.isnull().sum().to_dict()
    before_rows = len(df)
    warnings = []

    strategies, err = _parse_json_param(strategy_json, "strategy_json")
    if err:
        return err

    for col, strategy in strategies.items():
        if col not in df.columns:
            continue
        if strategy == "fill_mean":
            mean_val = df[col].mean()
            if pd.isna(mean_val):
                warnings.append(f"{col}: fill_mean skipped — column is entirely null (mean is NaN)")
                continue
            df[col] = df[col].fillna(mean_val)
        elif strategy == "fill_median":
            median_val = df[col].median()
            if pd.isna(median_val):
                warnings.append(f"{col}: fill_median skipped — column is entirely null")
                continue
            df[col] = df[col].fillna(median_val)
        elif strategy == "fill_mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
        elif strategy == "fill_zero":
            df[col] = df[col].fillna(0)
        elif strategy == "fill_ffill":
            df[col] = df[col].ffill()
        elif strategy == "fill_bfill":
            df[col] = df[col].bfill()
        elif strategy == "fill_interpolate":
            df[col] = df[col].interpolate(method="linear")
        elif strategy == "drop_rows":
            df = df.dropna(subset=[col])
        elif strategy == "drop_column":
            df = df.drop(columns=[col])

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    after_missing = df.isnull().sum().to_dict()

    return json.dumps({
        "status": "success",
        "before_missing": {k: v for k, v in before_missing.items() if v > 0},
        "after_missing": {k: v for k, v in after_missing.items() if v > 0},
        "rows_before": before_rows,
        "rows_after": len(df),
        "warnings": warnings,
        "output_path": output_path,
    }, indent=2)


def remove_duplicates(
    dataset_path: str,
    strategy: str,
    subset_columns: str,
) -> str:
    """
    Remove duplicate rows from the dataset.

    Args:
        dataset_path: Absolute path to the input CSV file.
        strategy: One of 'keep_first', 'keep_last', 'drop_all'.
        subset_columns: Comma-separated column names to check for duplicates,
            or 'all' to check all columns.

    Returns:
        JSON with duplicate count before/after and rows removed.
    """
    import pandas as pd

    output_path = _next_step_path("remove_duplicates")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    if strategy not in ("keep_first", "keep_last", "drop_all"):
        return json.dumps({
            "error": f"Unknown strategy '{strategy}'. Use: keep_first, keep_last, drop_all.",
        }, indent=2)

    subset = None if subset_columns.strip().lower() == "all" else [c.strip() for c in subset_columns.split(",") if c.strip()]
    # Validate subset columns exist
    if subset:
        missing = [c for c in subset if c not in df.columns]
        if missing:
            return json.dumps({
                "error": f"Subset columns not found: {missing}. Available: {list(df.columns)}",
            }, indent=2)

    before_dupes = int(df.duplicated(subset=subset).sum())

    if strategy == "keep_first":
        df = df.drop_duplicates(subset=subset, keep="first")
    elif strategy == "keep_last":
        df = df.drop_duplicates(subset=subset, keep="last")
    elif strategy == "drop_all":
        df = df.drop_duplicates(subset=subset, keep=False)

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    return json.dumps({
        "status": "success",
        "duplicates_found": before_dupes,
        "duplicates_removed": before_dupes - int(df.duplicated(subset=subset).sum()),
        "rows_after": len(df),
        "output_path": output_path,
    }, indent=2)


def drop_columns(
    dataset_path: str,
    columns_to_drop: str,
) -> str:
    """
    Drop specified columns from the dataset.

    Args:
        dataset_path: Absolute path to the input CSV file.
        columns_to_drop: Comma-separated column names to drop.

    Returns:
        JSON with columns dropped and remaining columns.
    """
    import pandas as pd

    output_path = _next_step_path("drop_columns")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    cols = [c.strip() for c in columns_to_drop.split(",") if c.strip()]
    if not cols:
        return json.dumps({"error": "columns_to_drop is empty. Provide at least one column name."}, indent=2)

    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    df = df.drop(columns=existing)

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    return json.dumps({
        "status": "success",
        "dropped": existing,
        "not_found": missing,
        "remaining_columns": list(df.columns),
        "output_path": output_path,
    }, indent=2)


def encode_categorical_columns(
    dataset_path: str,
    encoding_json: str,
) -> str:
    """
    Encode categorical columns using specified methods.

    Args:
        dataset_path: Absolute path to the input CSV file.
        encoding_json: JSON string mapping column names to encoding methods.
            Example: '{"city": "onehot", "grade": "label", "browser": "frequency", "size": "ordinal:S,M,L,XL"}'
            Supported: 'label', 'onehot', 'frequency', 'ordinal:val1,val2,...' (ordered),
            'binary' (for 2-class columns), 'target:target_col' (target/mean encoding).

    Returns:
        JSON with encoding details per column and new column list.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    output_path = _next_step_path("encode_categoricals")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    encodings, err = _parse_json_param(encoding_json, "encoding_json")
    if err:
        return err

    details = {}

    for col, method in encodings.items():
        if col not in df.columns:
            details[col] = {"status": "skipped", "reason": "column not found"}
            continue

        if method == "label":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            details[col] = {"method": "label", "classes": list(le.classes_)}

        elif method == "onehot":
            n_unique = df[col].nunique()
            if n_unique > 50:
                details[col] = {
                    "status": "skipped",
                    "reason": f"onehot skipped — {n_unique} unique values would create {n_unique} columns. "
                               "Use 'frequency' or 'hash:32' for high-cardinality columns instead.",
                }
                continue
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            details[col] = {"method": "onehot", "new_columns": list(dummies.columns)}

        elif method == "frequency":
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map)
            details[col] = {"method": "frequency", "top_mappings": dict(list(freq_map.items())[:5])}

        elif method.startswith("ordinal:"):
            order = method.split(":", 1)[1].split(",")
            mapping = {val.strip(): i for i, val in enumerate(order)}
            df[col] = df[col].map(mapping)
            details[col] = {"method": "ordinal", "mapping": mapping}

        elif method == "binary":
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                df[col] = (df[col] == unique_vals[1]).astype(int)
                details[col] = {"method": "binary", "mapping": {str(unique_vals[0]): 0, str(unique_vals[1]): 1}}
            else:
                details[col] = {"status": "skipped", "reason": f"not binary — has {len(unique_vals)} unique values"}

        elif method.startswith("target:"):
            target_col = method.split(":", 1)[1]
            if target_col in df.columns:
                means = df.groupby(col)[target_col].mean()
                df[col] = df[col].map(means)
                details[col] = {"method": "target_encoding", "target": target_col}
            else:
                details[col] = {"status": "skipped", "reason": f"target column '{target_col}' not found"}

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    return json.dumps({
        "status": "success",
        "encoding_details": details,
        "final_columns": list(df.columns),
        "output_path": output_path,
    }, indent=2)


def scale_numeric_columns(
    dataset_path: str,
    scaling_json: str,
    exclude_columns: str,
) -> str:
    """
    Scale/normalize numeric columns using specified methods.

    Args:
        dataset_path: Absolute path to the input CSV file.
        scaling_json: JSON string mapping column names to scaling methods.
            Use '__all_numeric__' as key to apply one method to all numeric columns.
            Example: '{"__all_numeric__": "standard"}' or '{"price": "minmax", "age": "robust"}'
            Supported: 'standard' (z-score), 'minmax' (0-1), 'robust' (IQR-based),
            'log' (log1p transform), 'maxabs' (scale by max absolute value),
            'power_yeo' (Yeo-Johnson power transform for skewed data).
        exclude_columns: Comma-separated column names to NEVER scale (e.g., target column, IDs).

    Returns:
        JSON with scaling details per column.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer

    output_path = _next_step_path("scale_numerics")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    scalings, err = _parse_json_param(scaling_json, "scaling_json")
    if err:
        return err

    exclude = {c.strip() for c in exclude_columns.split(",") if c.strip()}
    details = {}
    warnings = []

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if "__all_numeric__" in scalings:
        method = scalings["__all_numeric__"]
        col_methods = {col: method for col in numeric_cols}
    else:
        col_methods = {col: m for col, m in scalings.items() if col in numeric_cols and col not in exclude}

    for col, method in col_methods.items():
        if col not in df.columns:
            continue
        col_std = float(df[col].std())
        col_range = float(df[col].max() - df[col].min()) if len(df[col].dropna()) > 0 else 0
        before_stats = {"mean": round(float(df[col].mean()), 4), "std": round(col_std, 4)}

        # Guard: constant column — scaling would produce NaN
        if method in ("standard", "minmax", "maxabs") and col_std == 0:
            warnings.append(f"{col}: skipped '{method}' — column is constant (std=0), scaling would produce NaN.")
            details[col] = {"method": method, "status": "skipped", "reason": "constant column (std=0)"}
            continue

        try:
            if method == "standard":
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
            elif method == "minmax":
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
            elif method == "robust":
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(df[[col]])
            elif method == "log":
                n_negative = int((df[col] < 0).sum())
                if n_negative > 0:
                    warnings.append(f"{col}: log scaling — {n_negative} negative values clipped to 0 before log1p.")
                df[col] = np.log1p(df[col].clip(lower=0))
            elif method == "maxabs":
                scaler = MaxAbsScaler()
                df[col] = scaler.fit_transform(df[[col]])
            elif method == "power_yeo":
                pt = PowerTransformer(method="yeo-johnson")
                df[col] = pt.fit_transform(df[[col]])
        except Exception as e:
            warnings.append(f"{col}: {method} failed — {e}")
            details[col] = {"method": method, "status": "error", "reason": str(e)}
            continue

        after_stats = {"mean": round(float(df[col].mean()), 4), "std": round(float(df[col].std()), 4)}
        details[col] = {"method": method, "before": before_stats, "after": after_stats}

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    return json.dumps({
        "status": "success",
        "scaling_details": details,
        "excluded": list(exclude),
        "warnings": warnings,
        "output_path": output_path,
    }, indent=2)


def handle_outliers(
    dataset_path: str,
    outlier_json: str,
) -> str:
    """
    Detect and handle outliers in numeric columns.

    Args:
        dataset_path: Absolute path to the input CSV file.
        outlier_json: JSON string mapping column names to outlier strategies.
            Example: '{"price": "clip_iqr", "age": "remove_zscore:3", "salary": "winsorize:5,95"}'
            Supported:
            - 'clip_iqr' or 'clip_iqr:1.5' — clip values to [Q1-k*IQR, Q3+k*IQR]
            - 'remove_iqr' or 'remove_iqr:1.5' — remove rows outside IQR bounds
            - 'clip_zscore:3' — clip values beyond z-score threshold
            - 'remove_zscore:3' — remove rows beyond z-score threshold
            - 'winsorize:5,95' — clip to given percentiles
            - 'log_transform' — apply log1p (compresses extreme values)
            - 'keep' — do nothing

    Returns:
        JSON with outlier counts before/after per column.
    """
    import pandas as pd
    import numpy as np

    output_path = _next_step_path("handle_outliers")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    strategies, err = _parse_json_param(outlier_json, "outlier_json")
    if err:
        return err

    details = {}
    warnings = []
    before_rows = len(df)

    for col, strategy in strategies.items():
        if col not in df.columns or strategy == "keep":
            continue

        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        before_outliers = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0

        if strategy.startswith("clip_iqr"):
            k = float(strategy.split(":")[1]) if ":" in strategy else 1.5
            lower, upper = q1 - k * iqr, q3 + k * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)

        elif strategy.startswith("remove_iqr"):
            k = float(strategy.split(":")[1]) if ":" in strategy else 1.5
            lower, upper = q1 - k * iqr, q3 + k * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper)]

        elif strategy.startswith("clip_zscore"):
            threshold = float(strategy.split(":")[1]) if ":" in strategy else 3
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df[col] = df[col].clip(lower=mean - threshold * std, upper=mean + threshold * std)

        elif strategy.startswith("remove_zscore"):
            threshold = float(strategy.split(":")[1]) if ":" in strategy else 3
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df = df[((df[col] - mean) / std).abs() <= threshold]

        elif strategy.startswith("winsorize"):
            parts = strategy.split(":")[1].split(",") if ":" in strategy else ["5", "95"]
            lower_p, upper_p = float(parts[0]), float(parts[1])
            lower_val = df[col].quantile(lower_p / 100)
            upper_val = df[col].quantile(upper_p / 100)
            df[col] = df[col].clip(lower=lower_val, upper=upper_val)

        elif strategy == "log_transform":
            df[col] = np.log1p(df[col].clip(lower=0))

        after_outliers = 0
        if iqr > 0:
            new_q1, new_q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            new_iqr = new_q3 - new_q1
            if new_iqr > 0:
                after_outliers = int(((df[col] < new_q1 - 1.5 * new_iqr) | (df[col] > new_q3 + 1.5 * new_iqr)).sum())

        details[col] = {"strategy": strategy, "outliers_before": before_outliers, "outliers_after": after_outliers}

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    return json.dumps({
        "status": "success",
        "outlier_details": details,
        "rows_before": before_rows,
        "rows_after": len(df),
        "warnings": warnings,
        "output_path": output_path,
    }, indent=2)


def parse_datetime_columns(
    dataset_path: str,
    datetime_json: str,
) -> str:
    """
    Parse datetime columns and extract useful time features.

    Args:
        dataset_path: Absolute path to the input CSV file.
        datetime_json: JSON mapping column names to comma-separated features to extract.
            Example: '{"publish_date": "year,month,day,dayofweek,hour,is_weekend", "trending_date": "year,month,dayofweek"}'
            Supported extractions: year, month, day, dayofweek (0=Mon), hour, minute,
            is_weekend (0/1), quarter, week_of_year, days_since_earliest, time_of_day (morning/afternoon/evening/night).
            The original column will be dropped after extraction.

    Returns:
        JSON with new columns created per datetime column.
    """
    import pandas as pd

    output_path = _next_step_path("parse_datetime")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    configs, err = _parse_json_param(datetime_json, "datetime_json")
    if err:
        return err

    details = {}

    for col, features_str in configs.items():
        if col not in df.columns:
            details[col] = {"status": "skipped", "reason": "column not found"}
            continue

        try:
            dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        except Exception as e:
            details[col] = {"status": "error", "reason": str(e)}
            continue

        features = [f.strip() for f in features_str.split(",")]
        new_cols = []

        for feat in features:
            new_col = f"{col}_{feat}"
            if feat == "year":
                df[new_col] = dt.dt.year
            elif feat == "month":
                df[new_col] = dt.dt.month
            elif feat == "day":
                df[new_col] = dt.dt.day
            elif feat == "dayofweek":
                df[new_col] = dt.dt.dayofweek
            elif feat == "hour":
                df[new_col] = dt.dt.hour
            elif feat == "minute":
                df[new_col] = dt.dt.minute
            elif feat == "is_weekend":
                df[new_col] = (dt.dt.dayofweek >= 5).astype(int)
            elif feat == "quarter":
                df[new_col] = dt.dt.quarter
            elif feat == "week_of_year":
                df[new_col] = dt.dt.isocalendar().week.astype(int)
            elif feat == "days_since_earliest":
                earliest = dt.min()
                df[new_col] = (dt - earliest).dt.days
            elif feat == "time_of_day":
                df[new_col] = pd.cut(
                    dt.dt.hour,
                    bins=[-1, 6, 12, 18, 24],
                    labels=["night", "morning", "afternoon", "evening"],
                )
            new_cols.append(new_col)

        df = df.drop(columns=[col])
        null_dates = int(dt.isnull().sum())
        detail = {"status": "success", "new_columns": new_cols, "null_dates": null_dates}
        if null_dates > 0:
            detail["warning"] = f"{null_dates} dates could not be parsed and became NaT."
        details[col] = detail

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    return json.dumps({
        "status": "success",
        "datetime_details": details,
        "final_columns": list(df.columns),
        "output_path": output_path,
    }, indent=2)


def engineer_features(
    dataset_path: str,
    features_json: str,
) -> str:
    """
    Create new engineered features from existing columns.

    Args:
        dataset_path: Absolute path to the input CSV file.
        features_json: JSON mapping new column names to feature definitions.
            Example: '{
                "engagement_rate": "formula:(likes + comments) / views",
                "title_length": "str_len:title",
                "title_word_count": "word_count:title",
                "log_views": "log:views",
                "views_per_day": "ratio:views,days_since_publish",
                "is_popular": "threshold_above:likes,1000",
                "category_group": "bin:views,5",
                "interaction": "multiply:likes,comments"
            }'
            Supported operations:
            - 'formula:<pandas_expression>' — evaluate a pandas expression
            - 'str_len:<col>' — string length
            - 'word_count:<col>' — word count
            - 'log:<col>' — log1p transform
            - 'ratio:<col1>,<col2>' — col1 / col2 (safe division)
            - 'threshold_above:<col>,<value>' — binary: 1 if col > value
            - 'threshold_below:<col>,<value>' — binary: 1 if col < value
            - 'bin:<col>,<n_bins>' — equal-frequency binning
            - 'multiply:<col1>,<col2>' — product of two columns
            - 'add:<col1>,<col2>' — sum of two columns
            - 'subtract:<col1>,<col2>' — difference: col1 - col2

    Returns:
        JSON with created features and their basic stats.
    """
    import pandas as pd
    import numpy as np

    output_path = _next_step_path("engineer_features")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    features, err = _parse_json_param(features_json, "features_json")
    if err:
        return err

    details = {}

    for new_col, definition in features.items():
        try:
            if definition.startswith("formula:"):
                expr = definition.split(":", 1)[1]
                df[new_col] = df.eval(expr)

            elif definition.startswith("str_len:"):
                src = definition.split(":")[1]
                df[new_col] = df[src].astype(str).str.len()

            elif definition.startswith("word_count:"):
                src = definition.split(":")[1]
                df[new_col] = df[src].astype(str).str.split().str.len()

            elif definition.startswith("log:"):
                src = definition.split(":")[1]
                df[new_col] = np.log1p(df[src].clip(lower=0))

            elif definition.startswith("ratio:"):
                cols = definition.split(":")[1].split(",")
                col1, col2 = cols[0].strip(), cols[1].strip()
                df[new_col] = df[col1] / df[col2].replace(0, np.nan)

            elif definition.startswith("threshold_above:"):
                parts = definition.split(":")[1].split(",")
                col, val = parts[0].strip(), float(parts[1].strip())
                df[new_col] = (df[col] > val).astype(int)

            elif definition.startswith("threshold_below:"):
                parts = definition.split(":")[1].split(",")
                col, val = parts[0].strip(), float(parts[1].strip())
                df[new_col] = (df[col] < val).astype(int)

            elif definition.startswith("bin:"):
                parts = definition.split(":")[1].split(",")
                col, n_bins = parts[0].strip(), int(parts[1].strip())
                df[new_col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")

            elif definition.startswith("multiply:"):
                cols = definition.split(":")[1].split(",")
                df[new_col] = df[cols[0].strip()] * df[cols[1].strip()]

            elif definition.startswith("add:"):
                cols = definition.split(":")[1].split(",")
                df[new_col] = df[cols[0].strip()] + df[cols[1].strip()]

            elif definition.startswith("subtract:"):
                cols = definition.split(":")[1].split(",")
                df[new_col] = df[cols[0].strip()] - df[cols[1].strip()]

            else:
                op_prefix = definition.split(":")[0] if ":" in definition else definition
                details[new_col] = {
                    "status": "unsupported_operation",
                    "definition": definition,
                    "reason": (
                        f"Operation '{op_prefix}' is not a built-in supported operation. "
                        "Group-aware ops (shift by group, rolling by group, etc.) must be done "
                        "via run_in_sandbox. Skipping this feature — pipeline continues."
                    ),
                }
                continue

            stats = {}
            if df[new_col].dtype in ["int64", "float64"]:
                stats = {
                    "mean": round(float(df[new_col].mean()), 4),
                    "min": round(float(df[new_col].min()), 4),
                    "max": round(float(df[new_col].max()), 4),
                }
            details[new_col] = {
                "status": "created",
                "dtype": str(df[new_col].dtype),
                "null_count": int(df[new_col].isnull().sum()),
                "stats": stats,
            }

        except Exception as e:
            details[new_col] = {"status": "error", "reason": str(e)}

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    return json.dumps({
        "status": "success",
        "feature_details": details,
        "total_columns": len(df.columns),
        "output_path": output_path,
    }, indent=2)


def process_text_columns(
    dataset_path: str,
    text_json: str,
) -> str:
    """
    Process text/string columns — extract features or transform them for ML.

    Args:
        dataset_path: Absolute path to the input CSV file.
        text_json: JSON mapping column names to processing methods.
            Example: '{"title": "length_and_words", "tags": "count_delimiter:|", "description": "tfidf:50"}'
            Supported methods:
            - 'length_and_words' — creates <col>_length and <col>_word_count, drops original
            - 'count_delimiter:<sep>' — count items in delimited string (e.g., tags separated by |)
            - 'tfidf:<n_features>' — TF-IDF vectorization, creates top N features, drops original
            - 'contains:<keyword>' — binary: 1 if column contains keyword (case-insensitive)
            - 'extract_numbers' — extract first number found in the string
            - 'clean_and_keep' — lowercase, strip whitespace, remove special chars, keep column
            - 'hash:<n_buckets>' — hash encoding into N buckets (for high-cardinality columns)
            - 'drop' — simply drop the column

    Returns:
        JSON with processing details per text column.
    """
    import pandas as pd
    import numpy as np
    import re

    output_path = _next_step_path("process_text")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    configs, err = _parse_json_param(text_json, "text_json")
    if err:
        return err

    details = {}

    for col, method in configs.items():
        if col not in df.columns:
            details[col] = {"status": "skipped", "reason": "column not found"}
            continue

        try:
            if method == "length_and_words":
                df[f"{col}_length"] = df[col].astype(str).str.len()
                df[f"{col}_word_count"] = df[col].astype(str).str.split().str.len()
                df = df.drop(columns=[col])
                details[col] = {"status": "success", "new_columns": [f"{col}_length", f"{col}_word_count"]}

            elif method.startswith("count_delimiter:"):
                sep = method.split(":", 1)[1]
                df[f"{col}_count"] = df[col].astype(str).str.split(sep).str.len()
                df = df.drop(columns=[col])
                details[col] = {"status": "success", "new_columns": [f"{col}_count"], "delimiter": sep}

            elif method.startswith("tfidf:"):
                from sklearn.feature_extraction.text import TfidfVectorizer
                n_features = int(method.split(":")[1])
                tfidf = TfidfVectorizer(max_features=n_features, stop_words="english")
                text_data = df[col].fillna("").astype(str)
                matrix = tfidf.fit_transform(text_data)
                tfidf_cols = [f"{col}_tfidf_{w}" for w in tfidf.get_feature_names_out()]
                tfidf_df = pd.DataFrame(matrix.toarray(), columns=tfidf_cols, index=df.index)
                df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)
                details[col] = {"status": "success", "new_columns": tfidf_cols[:5], "total_tfidf_features": len(tfidf_cols)}

            elif method.startswith("contains:"):
                keyword = method.split(":", 1)[1]
                df[f"{col}_has_{keyword}"] = df[col].astype(str).str.contains(keyword, case=False, na=False).astype(int)
                details[col] = {"status": "success", "new_columns": [f"{col}_has_{keyword}"]}

            elif method == "extract_numbers":
                df[f"{col}_number"] = df[col].astype(str).apply(
                    lambda x: float(re.findall(r"[\d.]+", x)[0]) if re.findall(r"[\d.]+", x) else np.nan
                )
                df = df.drop(columns=[col])
                details[col] = {"status": "success", "new_columns": [f"{col}_number"]}

            elif method == "clean_and_keep":
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].str.replace(r"[^a-z0-9\s]", "", regex=True)
                details[col] = {"status": "success", "method": "cleaned in-place"}

            elif method.startswith("hash:"):
                n_buckets = int(method.split(":")[1])
                df[f"{col}_hash"] = df[col].astype(str).apply(lambda x: hash(x) % n_buckets)
                df = df.drop(columns=[col])
                details[col] = {"status": "success", "new_columns": [f"{col}_hash"], "n_buckets": n_buckets}

            elif method == "drop":
                df = df.drop(columns=[col])
                details[col] = {"status": "dropped"}

        except Exception as e:
            details[col] = {"status": "error", "reason": str(e)}

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    return json.dumps({
        "status": "success",
        "text_processing_details": details,
        "final_columns": list(df.columns),
        "output_path": output_path,
    }, indent=2)


def detect_and_fix_data_types(
    dataset_path: str,
    type_fixes_json: str,
) -> str:
    """
    Detect mistyped columns and cast them to correct types.

    Args:
        dataset_path: Absolute path to the input CSV file.
        type_fixes_json: JSON mapping column names to target types.
            Example: '{"price": "float", "age": "int", "is_active": "bool", "date": "datetime", "category": "category"}'
            Supported types: 'int', 'float', 'str', 'bool', 'datetime', 'category'.
            Use 'auto' to let pandas infer the best type for a column.

    Returns:
        JSON with type changes per column and any conversion errors.
    """
    import pandas as pd

    output_path = _next_step_path("fix_data_types")
    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    fixes, err = _parse_json_param(type_fixes_json, "type_fixes_json")
    if err:
        return err

    details = {}

    for col, target_type in fixes.items():
        if col not in df.columns:
            details[col] = {"status": "skipped", "reason": "column not found"}
            continue

        before_type = str(df[col].dtype)
        try:
            if target_type == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif target_type == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_type == "str":
                df[col] = df[col].astype(str)
            elif target_type == "bool":
                df[col] = df[col].astype(bool)
            elif target_type == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif target_type == "category":
                df[col] = df[col].astype("category")
            elif target_type == "auto":
                df[col] = pd.to_numeric(df[col], errors="ignore")

            details[col] = {
                "status": "converted",
                "from": before_type,
                "to": str(df[col].dtype),
                "nulls_after": int(df[col].isnull().sum()),
            }
        except Exception as e:
            details[col] = {"status": "error", "from": before_type, "reason": str(e)}

    write_err = _safe_write_csv(df, output_path)
    if write_err:
        return write_err

    return json.dumps({
        "status": "success",
        "type_fix_details": details,
        "output_path": output_path,
    }, indent=2)


def validate_dataset(
    dataset_path: str,
    checks_json: str,
) -> str:
    """
    Run validation checks on a preprocessed dataset to verify quality.

    Args:
        dataset_path: Absolute path to the CSV file to validate.
        checks_json: JSON string with validation checks to perform.
            Example: '{"no_nulls": true, "no_duplicates": true, "min_rows": 100,
                       "expected_columns": "col1,col2,col3", "no_infinite": true,
                       "target_column": "label", "numeric_only": false,
                       "max_null_pct": 5}'
            Supported checks:
            - 'no_nulls': true — fail if ANY null exists
            - 'max_null_pct': N — fail if any column has > N% nulls
            - 'no_duplicates': true — fail if duplicate rows exist
            - 'min_rows': N — fail if fewer than N rows
            - 'expected_columns': 'col1,col2,...' — fail if these columns are missing
            - 'no_infinite': true — fail if infinite values exist in numeric columns
            - 'target_column': 'col' — verify target column exists and has no nulls
            - 'numeric_only': true — fail if non-numeric columns remain (except target)

    Returns:
        JSON with pass/fail per check and an overall verdict.
    """
    import pandas as pd
    import numpy as np

    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    df, err = _safe_read_csv(dataset_path)
    if err:
        return err

    checks, err = _parse_json_param(checks_json, "checks_json")
    if err:
        return err

    results = {}
    all_passed = True

    if checks.get("no_nulls"):
        null_cols = {col: int(v) for col, v in df.isnull().sum().items() if v > 0}
        passed = len(null_cols) == 0
        results["no_nulls"] = {"passed": passed, "null_columns": null_cols}
        if not passed:
            all_passed = False

    if "max_null_pct" in checks:
        max_pct = checks["max_null_pct"]
        pct = (df.isnull().sum() / len(df) * 100).round(2)
        bad_cols = {col: float(v) for col, v in pct.items() if v > max_pct}
        passed = len(bad_cols) == 0
        results["max_null_pct"] = {"passed": passed, "threshold": max_pct, "violations": bad_cols}
        if not passed:
            all_passed = False

    if checks.get("no_duplicates"):
        dup_count = int(df.duplicated().sum())
        passed = dup_count == 0
        results["no_duplicates"] = {"passed": passed, "duplicate_count": dup_count}
        if not passed:
            all_passed = False

    if "min_rows" in checks:
        min_r = checks["min_rows"]
        passed = len(df) >= min_r
        results["min_rows"] = {"passed": passed, "expected": min_r, "actual": len(df)}
        if not passed:
            all_passed = False

    if "expected_columns" in checks:
        expected = [c.strip() for c in checks["expected_columns"].split(",")]
        missing_cols = [c for c in expected if c not in df.columns]
        passed = len(missing_cols) == 0
        results["expected_columns"] = {"passed": passed, "missing": missing_cols, "actual_columns": list(df.columns)}
        if not passed:
            all_passed = False

    if checks.get("no_infinite"):
        numeric = df.select_dtypes(include=[np.number])
        inf_cols = {col: int(np.isinf(numeric[col]).sum()) for col in numeric.columns if np.isinf(numeric[col]).any()}
        passed = len(inf_cols) == 0
        results["no_infinite"] = {"passed": passed, "infinite_columns": inf_cols}
        if not passed:
            all_passed = False

    if "target_column" in checks:
        target = checks["target_column"]
        exists = target in df.columns
        nulls = int(df[target].isnull().sum()) if exists else -1
        passed = exists and nulls == 0
        results["target_column"] = {"passed": passed, "exists": exists, "null_count": nulls}
        if not passed:
            all_passed = False

    if checks.get("numeric_only"):
        target = checks.get("target_column", "")
        non_numeric = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != target]
        passed = len(non_numeric) == 0
        results["numeric_only"] = {"passed": passed, "non_numeric_columns": non_numeric}
        if not passed:
            all_passed = False

    results["_summary"] = {
        "all_passed": all_passed,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
    }

    return json.dumps(results, indent=2)


# NOTE: Custom code execution is handled by SafeExecute tools
# (run_in_sandbox, write_file_to_sandbox, read_file_from_sandbox).
# These are regular async functions that CAN be mixed with other tools.


def log_sandbox_step(
    step_name: str,
    script: str,
    stdout: str,
    purpose: str,
) -> str:
    """
    Log a sandbox execution step so Agent 4 can verify it.
    Call this AFTER every run_in_sandbox call, passing the script that ran and its stdout.

    Args:
        step_name: Short name for this step (e.g. 'custom_smote', 'custom_feature_engineer').
        script: The full Python script that was written to the sandbox and executed.
        stdout: The stdout output returned by run_in_sandbox.
        purpose: One sentence describing what this step was supposed to do.

    Returns:
        Confirmation message.
    """
    state = load_state()
    iteration = state.get("loop_iteration", 0)
    entry = {
        "iteration": iteration,
        "step_name": step_name,
        "script": script,
        "stdout": stdout,
        "purpose": purpose,
    }
    existing = state.get("agent3_sandbox_log", [])
    existing.append(entry)
    save_state({"agent3_sandbox_log": existing})
    return json.dumps({"status": "logged", "step_name": step_name}, indent=2)


def save_preprocessed_output(
    dataset_path: str,
    output_filename: str,
    summary: str,
    steps_completed: str,
) -> str:
    """
    Copy the final preprocessed dataset to the modified_datasets/ folder
    and save a processing summary to pipeline_state.json.

    Args:
        dataset_path: Absolute path to the final preprocessed CSV file.
        output_filename: Desired filename for the output (e.g. 'preprocessed_youtube.csv').
        summary: A paragraph summarizing all preprocessing steps performed.
        steps_completed: Comma-separated list of steps completed (e.g. 'drop_columns,handle_missing,...').

    Returns:
        Confirmation with output file path.
    """
    import shutil

    run_dir = get_run_dir()
    dest = run_dir / output_filename

    try:
        dataset_path = _resolve_local_path(dataset_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, indent=2)

    try:
        shutil.copy2(dataset_path, dest)
    except OSError as e:
        return json.dumps({
            "error": f"Could not copy preprocessed file to output: {e}",
            "source": dataset_path,
            "destination": str(dest),
        }, indent=2)

    iteration = load_state().get("loop_iteration", 0)
    agent3_entry = {
        "iteration": iteration,
        "preprocessed_file": str(dest),
        "summary": summary,
        "steps_completed": [s.strip() for s in steps_completed.split(",")],
    }
    existing_iterations = load_state().get("agent3_iterations", [])
    existing_iterations.append(agent3_entry)

    save_state({
        "agent3_iterations": existing_iterations,
        "preprocessed_dataset_path": str(dest),
        "status": "agent3_done",
    })
    save_state({"current_step": 0})

    return json.dumps({
        "status": "success",
        "output_path": str(dest),
        "summary": summary,
    }, indent=2)


# ============================================================
# ==================  AGENT 4 TOOLS  =========================
# ============================================================

def load_validation_context() -> str:
    """
    Load everything Agent 4 needs for validation: original dataset info,
    preprocessing plan, Agent 3's output summary, and the preprocessed file path.
    """
    state = load_state()
    agent1_output = state.get("agent1_output", [])
    agent2_output = state.get("agent2_output", [])
    agent3_iterations = state.get("agent3_iterations", [])
    context = {
        "user_goal": state.get("user_goal", ""),
        "original_dataset": _latest(agent1_output).get("selected_dataset", {}),
        "preprocessing_plan": _latest(agent2_output).get("preprocessing_plan", {}),
        "agent3_latest_output": agent3_iterations[-1] if agent3_iterations else {},
        "agent3_all_attempts": agent3_iterations,
        "preprocessed_path": state.get("preprocessed_dataset_path", ""),
        "iteration": state.get("loop_iteration", 0),
        "previous_validations": state.get("agent4_iterations", []),
    }
    return json.dumps(context, indent=2, default=str)


def compare_before_after(
    original_path: str,
    preprocessed_path: str,
) -> str:
    """
    Compare the original and preprocessed datasets side-by-side.
    Returns detailed comparison: shape changes, column changes,
    dtype changes, missing value changes, and basic stat shifts.

    Args:
        original_path: Absolute path to the original dataset.
        preprocessed_path: Absolute path to the preprocessed dataset.

    Returns:
        JSON with detailed before/after comparison.
    """
    import pandas as pd
    import numpy as np

    try:
        original_path = _resolve_local_path(original_path)
        preprocessed_path = _resolve_local_path(preprocessed_path)
        orig = pd.read_csv(original_path)
        proc = pd.read_csv(preprocessed_path)
    except Exception as e:
        return json.dumps({"error": str(e)})

    comparison = {
        "shape": {"original": list(orig.shape), "preprocessed": list(proc.shape)},
        "columns_added": [c for c in proc.columns if c not in orig.columns],
        "columns_removed": [c for c in orig.columns if c not in proc.columns],
        "columns_kept": [c for c in orig.columns if c in proc.columns],
        "missing_values": {
            "original_total": int(orig.isnull().sum().sum()),
            "preprocessed_total": int(proc.isnull().sum().sum()),
        },
        "duplicates": {
            "original": int(orig.duplicated().sum()),
            "preprocessed": int(proc.duplicated().sum()),
        },
    }

    # Dtype changes for kept columns
    dtype_changes = {}
    for col in comparison["columns_kept"]:
        if str(orig[col].dtype) != str(proc[col].dtype):
            dtype_changes[col] = {"from": str(orig[col].dtype), "to": str(proc[col].dtype)}
    comparison["dtype_changes"] = dtype_changes

    # Numeric stats shifts
    orig_numeric = orig.select_dtypes(include=[np.number]).columns
    proc_numeric = proc.select_dtypes(include=[np.number]).columns
    common_numeric = [c for c in orig_numeric if c in proc_numeric]

    if common_numeric:
        stat_shifts = {}
        for col in common_numeric[:10]:
            stat_shifts[col] = {
                "mean_shift": round(float(proc[col].mean() - orig[col].mean()), 4),
                "std_shift": round(float(proc[col].std() - orig[col].std()), 4),
            }
        comparison["stat_shifts"] = stat_shifts

    comparison["preprocessed_dtypes"] = proc.dtypes.astype(str).value_counts().to_dict()

    return json.dumps(comparison, indent=2)


async def run_plan_aware_validation(
    original_path: str,
    preprocessed_path: str,
) -> str:
    """
    Upload both original and preprocessed datasets to the sandbox,
    generate a plan-aware validation script from Agent 2's preprocessing plan,
    run it, and return a structured checklist of pass/fail results per step.

    This verifies that each planned transformation was actually applied correctly
    by checking the actual numeric properties of the data (not Agent 3's self-report).

    Args:
        original_path: Absolute local path to the original raw dataset CSV.
        preprocessed_path: Absolute local path to the final preprocessed CSV.

    Returns:
        JSON checklist with pass/fail per planned step plus sandbox step verifications.
    """
    state = load_state()
    agent2_output = state.get("agent2_output", [])
    plan = _latest(agent2_output).get("preprocessing_plan", {})
    sandbox_log = state.get("agent3_sandbox_log", [])
    iteration = state.get("loop_iteration", 0)

    # With SafeExecute, modified_datasets/ IS /workspace — no upload needed
    orig_sandbox = f"/workspace/{Path(original_path).name}"
    pre_sandbox = f"/workspace/{Path(preprocessed_path).name}"

    # Build the validation script from the plan
    checks = []

    # Always check: row count, column count, null count, duplicate count
    checks.append("""
results['shape_original'] = {'rows': len(orig), 'cols': len(orig.columns)}
results['shape_preprocessed'] = {'rows': len(pre), 'cols': len(pre.columns)}
results['nulls_original'] = int(orig.isnull().sum().sum())
results['nulls_preprocessed'] = int(pre.isnull().sum().sum())
results['dupes_original'] = int(orig.duplicated().sum())
results['dupes_preprocessed'] = int(pre.duplicated().sum())
""")

    # duplicate_strategy
    dup_strategy = plan.get("duplicate_strategy", "none")
    if dup_strategy and dup_strategy.lower() not in ("none", "keep", ""):
        checks.append(f"""
# Check: duplicates were actually removed
dup_check = {{}}
dup_check['planned'] = '{dup_strategy}'
dup_check['dupes_before'] = int(orig.duplicated().sum())
dup_check['dupes_after'] = int(pre.duplicated().sum())
dup_check['rows_removed'] = len(orig) - len(pre)
dup_check['passed'] = pre.duplicated().sum() == 0
results['check_drop_duplicates'] = dup_check
""")

    # missing_value_strategy
    missing_strategy = plan.get("missing_value_strategy", "none")
    if missing_strategy and missing_strategy.lower() not in ("none", ""):
        checks.append(f"""
# Check: missing values were reduced
missing_check = {{}}
missing_check['planned'] = '{missing_strategy}'
missing_check['nulls_before'] = int(orig.isnull().sum().sum())
missing_check['nulls_after'] = int(pre.isnull().sum().sum())
missing_check['passed'] = pre.isnull().sum().sum() <= orig.isnull().sum().sum()
results['check_handle_missing'] = missing_check
""")

    # columns_to_drop
    cols_to_drop = plan.get("columns_to_drop", "none")
    if cols_to_drop and cols_to_drop.lower() not in ("none", ""):
        drop_cols = [c.strip().split(":")[0] for c in cols_to_drop.split(",") if c.strip() and c.strip().lower() != "none"]
        if drop_cols:
            checks.append(f"""
# Check: specified columns were dropped
drop_check = {{}}
drop_check['planned_drops'] = {drop_cols}
drop_check['still_present'] = [c for c in {drop_cols} if c in pre.columns]
drop_check['passed'] = len(drop_check['still_present']) == 0
results['check_drop_columns'] = drop_check
""")

    # scaling_strategy — parse "col:method,col:method" or "col:method"
    scaling_strategy = plan.get("scaling_strategy", "none")
    if scaling_strategy and scaling_strategy.lower() not in ("none", ""):
        # Build per-column scaling checks
        scale_checks_code = []
        parts = [p.strip() for p in scaling_strategy.split(",") if p.strip()]
        for part in parts:
            tokens = part.split(":")
            if len(tokens) < 2:
                continue
            col = tokens[0].strip()
            method = tokens[1].strip().lower()
            if col.lower() == "none" or method.lower() == "none":
                continue
            if method == "robust":
                scale_checks_code.append(f"""
if '{col}' in pre.columns:
    median_val = float(pre['{col}'].median())
    q75 = float(pre['{col}'].quantile(0.75))
    q25 = float(pre['{col}'].quantile(0.25))
    iqr_val = q75 - q25
    results['check_scale_{col}_robust'] = {{
        'planned': 'robust',
        'median': round(median_val, 4),
        'iqr': round(iqr_val, 4),
        'passed': abs(median_val) < 0.5 and 0.5 < iqr_val < 2.0
    }}
""")
            elif method == "standard":
                scale_checks_code.append(f"""
if '{col}' in pre.columns:
    mean_val = float(pre['{col}'].mean())
    std_val = float(pre['{col}'].std())
    results['check_scale_{col}_standard'] = {{
        'planned': 'standard',
        'mean': round(mean_val, 4),
        'std': round(std_val, 4),
        'passed': abs(mean_val) < 0.1 and abs(std_val - 1.0) < 0.2
    }}
""")
            elif method == "minmax":
                scale_checks_code.append(f"""
if '{col}' in pre.columns:
    min_val = float(pre['{col}'].min())
    max_val = float(pre['{col}'].max())
    results['check_scale_{col}_minmax'] = {{
        'planned': 'minmax',
        'min': round(min_val, 4),
        'max': round(max_val, 4),
        'passed': min_val >= -0.01 and max_val <= 1.01
    }}
""")
        if scale_checks_code:
            checks.extend(scale_checks_code)

    # encoding_strategy — parse "col:method,col:method"
    encoding_strategy = plan.get("encoding_strategy", "none")
    if encoding_strategy and encoding_strategy.lower() not in ("none", ""):
        parts = [p.strip() for p in encoding_strategy.split(",") if p.strip()]
        for part in parts:
            tokens = part.split(":")
            if len(tokens) < 2:
                continue
            col = tokens[0].strip()
            method = tokens[1].strip().lower()
            if col.lower() == "none" or method.lower() == "none":
                continue
            if method == "onehot":
                checks.append(f"""
enc_check = {{}}
enc_check['planned'] = 'onehot'
enc_check['col'] = '{col}'
enc_check['original_col_gone'] = '{col}' not in pre.columns
enc_check['dummy_cols_present'] = [c for c in pre.columns if c.startswith('{col}_')]
enc_check['passed'] = '{col}' not in pre.columns and len(enc_check['dummy_cols_present']) > 0
results['check_encode_{col}_onehot'] = enc_check
""")
            elif method in ("label", "binary"):
                checks.append(f"""
enc_check = {{}}
enc_check['planned'] = '{method}'
enc_check['col'] = '{col}'
if '{col}' in pre.columns:
    enc_check['dtype'] = str(pre['{col}'].dtype)
    enc_check['passed'] = str(pre['{col}'].dtype) in ('int64', 'int32', 'float64', 'int8')
else:
    enc_check['passed'] = False
    enc_check['reason'] = 'column not found'
results['check_encode_{col}_{method}'] = enc_check
""")

    # outlier_strategy — parse "col:method,col:method"
    outlier_strategy = plan.get("outlier_strategy", "none")
    if outlier_strategy and outlier_strategy.lower() not in ("none", ""):
        parts = [p.strip() for p in outlier_strategy.split(",") if p.strip()]
        for part in parts:
            tokens = part.split(":")
            if len(tokens) < 2:
                continue
            col = tokens[0].strip()
            method = tokens[1].strip().lower()
            if col.lower() == "none" or method.lower() in ("none", "keep"):
                continue
            checks.append(f"""
if '{col}' in orig.columns and '{col}' in pre.columns:
    q1 = float(orig['{col}'].quantile(0.25))
    q3 = float(orig['{col}'].quantile(0.75))
    iqr = q3 - q1
    expected_upper = q3 + 1.5 * iqr
    expected_lower = q1 - 1.5 * iqr
    actual_max = float(pre['{col}'].max())
    actual_min = float(pre['{col}'].min())
    results['check_outlier_{col}'] = {{
        'planned': '{method}',
        'original_max': round(float(orig['{col}'].max()), 4),
        'preprocessed_max': round(actual_max, 4),
        'expected_upper_bound': round(expected_upper, 4),
        'passed': actual_max <= expected_upper * 1.05
    }}
""")

    # feature_engineering
    feat_engineering = plan.get("feature_engineering", "none")
    if feat_engineering and feat_engineering.lower() not in ("none", ""):
        checks.append(f"""
feat_check = {{}}
feat_check['planned'] = '{feat_engineering[:100]}'
feat_check['cols_added'] = len(pre.columns) - len(orig.columns)
feat_check['passed'] = len(pre.columns) > len(orig.columns)
results['check_feature_engineering'] = feat_check
""")

    # datetime_processing
    datetime_processing = plan.get("datetime_processing", "none")
    if datetime_processing and datetime_processing.lower() not in ("none", ""):
        checks.append(f"""
dt_check = {{}}
dt_check['planned'] = '{datetime_processing[:100]}'
dt_check['cols_added'] = len(pre.columns) - len(orig.columns)
dt_check['passed'] = len(pre.columns) > len(orig.columns)
results['check_datetime_processing'] = dt_check
""")

    # Sandbox log checks — verify each custom sandbox step
    current_iter_logs = [s for s in sandbox_log if s.get("iteration") == iteration]
    for log_entry in current_iter_logs:
        step_name = log_entry.get("step_name", "unknown")
        purpose = log_entry.get("purpose", "")
        stdout = log_entry.get("stdout", "")
        checks.append(f"""
sandbox_check_{step_name} = {{}}
sandbox_check_{step_name}['step'] = '{step_name}'
sandbox_check_{step_name}['purpose'] = '''{purpose[:100]}'''
sandbox_check_{step_name}['stdout_preview'] = '''{stdout[:200]}'''
sandbox_check_{step_name}['rows_after'] = len(pre)
sandbox_check_{step_name}['passed'] = len(pre) > 0
results['check_sandbox_{step_name}'] = sandbox_check_{step_name}
""")

    # Assemble the full script
    checks_code = "\n".join(checks)
    script = f"""
import pandas as pd
import numpy as np
import json

orig = pd.read_csv('{orig_sandbox}')
pre = pd.read_csv('{pre_sandbox}')

results = {{}}

{checks_code}

print(json.dumps(results, indent=2, default=str))
"""

    # Write and run via SafeExecute
    await write_file_to_sandbox("validate.py", script)
    raw = await run_in_sandbox("exec(open('/workspace/validate.py').read())")
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({
            "error": "run_in_sandbox returned unexpected output (not JSON)",
            "raw": str(raw)[:500],
        }, indent=2)
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")

    if not result.get("success") or not stdout.strip():
        return json.dumps({
            "error": "Validation script failed",
            "stderr": stderr,
            "script": script,
        }, indent=2)

    try:
        checklist = json.loads(stdout)
    except Exception:
        return json.dumps({"error": "Could not parse validation output", "raw_stdout": stdout}, indent=2)

    # Compute overall pass/fail summary
    check_keys = [k for k in checklist if k.startswith("check_")]
    passed = [k for k in check_keys if checklist[k].get("passed") is True]
    failed = [k for k in check_keys if checklist[k].get("passed") is False]

    return json.dumps({
        "status": "ok",
        "iteration": iteration,
        "total_checks": len(check_keys),
        "passed": len(passed),
        "failed": len(failed),
        "failed_checks": failed,
        "checklist": checklist,
    }, indent=2, default=str)


def save_validation_result(
    verdict: str,
    issues_found: str,
    feedback_for_agent3: str,
    quality_score: str,
    validation_summary: str,
    checklist_json: str,
) -> str:
    """
    Save the validation result to pipeline_state.json.
    If verdict is 'FAIL', Agent 3 will retry with the feedback.
    If verdict is 'PASS', the pipeline finishes.

    Args:
        verdict: Either 'PASS' or 'FAIL'.
        issues_found: Comma-separated list of issues found (empty if PASS).
        feedback_for_agent3: Specific instructions for Agent 3 to fix issues (empty if PASS).
        quality_score: A score from 1-10 rating the preprocessing quality.
        validation_summary: A paragraph summarizing the validation results.
        checklist_json: JSON string of the plan-aware checklist from run_plan_aware_validation. Pass '{}' if not available.

    Returns:
        Confirmation message.
    """
    if verdict.upper() not in ("PASS", "FAIL"):
        return json.dumps({
            "error": f"Invalid verdict '{verdict}'. Must be 'PASS' or 'FAIL'.",
        }, indent=2)

    try:
        checklist = json.loads(checklist_json) if checklist_json and checklist_json.strip() else {}
    except (json.JSONDecodeError, TypeError):
        checklist = {}

    result = {
        "verdict": verdict.upper(),
        "issues_found": [i.strip() for i in issues_found.split(",") if i.strip()],
        "feedback_for_agent3": feedback_for_agent3,
        "quality_score": quality_score,
        "validation_summary": validation_summary,
        "checklist": checklist,
    }

    current_state = load_state()
    iteration = current_state.get("loop_iteration", 0)
    result["iteration"] = iteration

    existing_iterations = current_state.get("agent4_iterations", [])
    existing_iterations.append(result)

    state_update = {
        "agent4_iterations": existing_iterations,
        "agent4_feedback": feedback_for_agent3 if verdict.upper() == "FAIL" else "",
        "loop_iteration": iteration + 1,
    }

    if verdict.upper() == "PASS":
        state_update["status"] = "preprocessing_complete"
    else:
        state_update["status"] = "agent4_needs_retry"

    save_state(state_update)

    return json.dumps({"status": "saved", "verdict": verdict.upper(), "quality_score": quality_score}, indent=2)


# ============================================================
# ===============  REPORT GENERATOR TOOLS  ===================
# ============================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_full_pipeline_context() -> str:
    """
    Load the ENTIRE pipeline state — every agent's output, every decision,
    every validation result. This is used by the Report Generator to write
    a comprehensive A-to-Z preprocessing report.
    Returns all agent outputs, the plan, validation details, and file paths.
    """
    import pandas as pd
    import numpy as np

    state = load_state()

    # --- Original dataset stats ---
    orig_path = state.get("selected_dataset_path", "")
    orig_stats = {}
    if orig_path and Path(orig_path).exists():
        try:
            df_orig = pd.read_csv(orig_path)
            orig_stats = {
                "shape": list(df_orig.shape),
                "columns": list(df_orig.columns),
                "dtypes": {c: str(d) for c, d in df_orig.dtypes.items()},
                "missing_total": int(df_orig.isnull().sum().sum()),
                "missing_per_column": {c: int(v) for c, v in df_orig.isnull().sum().items() if v > 0},
                "duplicates": int(df_orig.duplicated().sum()),
                "numeric_stats": df_orig.describe().round(4).to_dict() if len(df_orig.select_dtypes(include=[np.number]).columns) > 0 else {},
            }
        except Exception:
            orig_stats = {"error": "could not read original dataset"}

    # --- Preprocessed dataset stats ---
    proc_path = state.get("preprocessed_dataset_path", "")
    proc_stats = {}
    if proc_path and Path(proc_path).exists():
        try:
            df_proc = pd.read_csv(proc_path)
            proc_stats = {
                "shape": list(df_proc.shape),
                "columns": list(df_proc.columns),
                "dtypes": {c: str(d) for c, d in df_proc.dtypes.items()},
                "missing_total": int(df_proc.isnull().sum().sum()),
                "missing_per_column": {c: int(v) for c, v in df_proc.isnull().sum().items() if v > 0},
                "duplicates": int(df_proc.duplicated().sum()),
                "numeric_stats": df_proc.describe().round(4).to_dict() if len(df_proc.select_dtypes(include=[np.number]).columns) > 0 else {},
            }
        except Exception:
            proc_stats = {"error": "could not read preprocessed dataset"}

    context = {
        "user_goal": state.get("user_goal", ""),
        "qa_pairs": state.get("qa_pairs", []),
        "agent1_output": _latest(state.get("agent1_output")),
        "agent2_output": _latest(state.get("agent2_output")),
        "agent3_iterations": state.get("agent3_iterations", []),
        "agent4_iterations": state.get("agent4_iterations", []),
        "original_dataset_path": orig_path,
        "preprocessed_dataset_path": proc_path,
        "original_dataset_stats": orig_stats,
        "preprocessed_dataset_stats": proc_stats,
        "loop_iterations": state.get("loop_iteration", 0),
        "final_status": state.get("status", ""),
    }
    return json.dumps(context, indent=2, default=str)


def save_preprocessing_report(report_content: str, report_filename: str) -> str:
    """
    Save the final preprocessing report to the outputs/ folder.

    Args:
        report_content: The full markdown report content. This should be a comprehensive,
            detailed report covering every single step of the preprocessing pipeline.
        report_filename: The filename for the report (e.g. 'preprocessing_report.md').

    Returns:
        Confirmation with the output file path.
    """
    if not report_content or len(report_content) < 100:
        return json.dumps({
            "error": "report_content is empty or too short. Write the full report before calling this tool.",
            "received_length": len(report_content) if report_content else 0,
        }, indent=2)

    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = OUTPUTS_DIR / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except OSError as e:
        return json.dumps({
            "error": f"Could not write preprocessing report: {e}",
            "path": str(OUTPUTS_DIR / report_filename),
            "fix": "Check write permissions on the outputs/ folder.",
        }, indent=2)

    save_state({
        "report_path": str(report_path),
        "status": "pipeline_complete",
    })

    return json.dumps({
        "status": "success",
        "report_path": str(report_path),
        "report_length": len(report_content),
    }, indent=2)


# ============================================================
# ==================  AGENT DEFINITIONS  =====================
# ============================================================

from google.adk.agents import Agent, SequentialAgent, LoopAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext

# SafeExecute tools for secure code execution
from data_preprocessing_agent.sandbox_executor import (
    start_sandbox,
    stop_sandbox,
    upload_dataset_to_sandbox,
    run_in_sandbox,
    write_file_to_sandbox,
    read_file_from_sandbox,
    download_from_sandbox,
)


# ── Resume-state tool (shared by Agents 1, 2, 3) ─────────────────────────────

def get_preprocessing_resume_state() -> str:
    """
    Check which sub-agents in the preprocessing pipeline have already completed
    their work in a previous run. Call this FIRST before doing any expensive work
    to avoid re-running steps that already succeeded.

    Returns a JSON object with:
      - agent1_done: True if Agent 1 already selected a dataset AND the file exists
      - agent2_done: True if Agent 2 already saved a preprocessing plan
      - agent3_has_partial_output: True if a step file from a crashed Agent 3 run exists
      - preprocessing_complete: True if the full preprocessed dataset exists on disk
      - last_step_output_path: path to the last successfully written step file (empty if none)
      - current_step: how many preprocessing steps completed in the last run
    """
    state = load_state()

    selected_path = state.get("selected_dataset_path", "")
    preprocessed_path = state.get("preprocessed_dataset_path", "")
    last_step_path = state.get("last_step_output_path", "")

    selected_exists = bool(selected_path) and Path(selected_path).exists()
    preprocessed_exists = bool(preprocessed_path) and Path(preprocessed_path).exists()
    last_step_exists = bool(last_step_path) and Path(last_step_path).exists()

    return json.dumps({
        "agent1_done": bool(state.get("agent1_output")) and selected_exists,
        "agent2_done": bool(state.get("agent2_output")),
        "agent3_has_partial_output": last_step_exists,
        "preprocessing_complete": preprocessed_exists,
        "status": state.get("status", ""),
        "selected_dataset_path": selected_path,
        "selected_file_exists": selected_exists,
        "preprocessed_dataset_path": preprocessed_path,
        "last_step_output_path": last_step_path if last_step_exists else "",
        "loop_iteration": state.get("loop_iteration", 0),
        "current_step": state.get("current_step", 0),
    }, indent=2)


# --- Agent 1: Dataset Analyzer ---
dataset_analyzer_agent = Agent(
    model=MODEL,
    name="dataset_analyzer_agent",
    description="Analyzes all available datasets and selects the best one for the user's ML project.",
    output_key="agent1_result",
    instruction="""You are Agent 1: the Dataset Analyzer & Selector.

RESUME CHECK — do this FIRST, before anything else:
1. Call get_preprocessing_resume_state().
   - If agent1_done is True → your work already completed in a previous run.
     Do NOT call scan_datasets_folder or save_selected_dataset again.
     Respond: "Agent 1 already completed. Selected: <selected_dataset_path from resume state>"
     Then STOP — do no further work.
   - If agent1_done is False → proceed with the normal workflow below.

NORMAL WORKFLOW (only if agent1_done is False):
1. Call get_project_context to understand the user's goal and project plan.
2. Call scan_datasets_folder to discover and preview all available datasets.
3. Analyze EVERY dataset returned — look at columns, dtypes, row counts, missing values, and previews.
4. Select the SINGLE BEST dataset file that is most relevant to the user's project goal.
5. Call save_selected_dataset with all the details of your selection.

SELECTION CRITERIA (in order of importance):
- Relevance to the user's stated goal and project plan
- Data quality (fewer missing values, clean column names, appropriate dtypes)
- Dataset size (prefer larger datasets when relevance is equal)
- File format (prefer CSV/Parquet over JSON/Excel for ML pipelines)

RULES:
- You MUST call all tools in order: get_project_context → scan_datasets_folder → save_selected_dataset
- You MUST select exactly ONE file, not a folder
- If multiple files exist in one folder (e.g. train.csv + test.csv), pick the main/largest one
- Provide a detailed reason for your selection
- List ALL potential data quality issues you notice

After saving, respond with a brief summary of which dataset you selected and why.
""",
    tools=[get_preprocessing_resume_state, get_project_context, scan_datasets_folder, save_selected_dataset],
)


# --- Agent 2: Preprocessing Strategist ---
preprocessing_strategist_agent = Agent(
    model=MODEL,
    name="preprocessing_strategist_agent",
    description="Analyzes the selected dataset deeply and creates a detailed preprocessing plan.",
    output_key="agent2_result",
    instruction="""You are Agent 2: the Preprocessing Strategist.

RESUME CHECK — do this FIRST, before anything else:
1. Call get_preprocessing_resume_state().
   - If agent2_done is True → your work already completed in a previous run.
     Do NOT call load_dataset_profile or save_preprocessing_plan again.
     Respond: "Agent 2 already completed. Preprocessing plan exists in pipeline state."
     Then STOP — do no further work.
   - If agent2_done is False → proceed with the normal workflow below.

NORMAL WORKFLOW (only if agent2_done is False):
1. Call get_user_requirements to understand what the user is building and what Agent 1 selected.
2. Call load_dataset_profile to get deep statistics about the dataset.
3. Analyze the profile carefully — look at missing values, dtypes, distributions, outliers, correlations, duplicates.
4. Design a preprocessing strategy tailored to the user's ML goal.
5. Call save_preprocessing_plan with every detail filled in.

PLANNING PRINCIPLES:
- The plan must be SPECIFIC — name exact columns and exact methods, never say "handle appropriately"
- Consider the ML task type (classification, regression, clustering) when choosing strategies
- Preserve information — prefer imputation over dropping rows when possible
- Be careful with target leakage — don't use the target column in feature engineering
- Consider column relationships — if two columns are 95%+ correlated, suggest dropping one
- For text columns, choose methods based on cardinality and relevance
- Order matters: always handle missing values BEFORE encoding, and encoding BEFORE scaling
- Include validation checks that verify the preprocessing didn't corrupt the data

STEP ORDER (follow this standard order, skip steps that don't apply):
1. drop_columns → 2. handle_duplicates → 3. fix_data_types → 4. handle_missing →
5. parse_dates → 6. process_text → 7. handle_outliers → 8. encode_categoricals →
9. feature_engineering → 10. scale_numerics → 11. final_validation

RULES:
- You MUST call all tools in order
- Every column must be accounted for in the plan
- The target column must NEVER be scaled or encoded (unless it's a label that needs encoding)
- Be specific about WHY you chose each strategy

After saving, respond with a concise summary of your plan.
""",
    tools=[get_preprocessing_resume_state, get_user_requirements, load_dataset_profile, save_preprocessing_plan],
)


# --- Agent 3: Preprocessing Executor ---
preprocessing_executor_agent = Agent(
    model=MODEL,
    name="preprocessing_executor_agent",
    description="Executes the preprocessing plan step by step using specialized tools and a SafeExecute sandbox environment.",
    output_key="agent3_result",
    instruction="""You are Agent 3: the Preprocessing Executor.

Your job is to execute the preprocessing plan step by step.

================================================================
CRITICAL — TWO KINDS OF TOOLS, TWO KINDS OF PATHS
================================================================
You have TWO categories of tools and they DO NOT share filesystems:

A) BUILT-IN TOOLS (run on the HOST machine, use LOCAL Windows paths)
   - handle_missing_values, remove_duplicates, drop_columns,
     encode_categorical_columns, scale_numeric_columns,
     handle_outliers, parse_datetime_columns, engineer_features,
     process_text_columns, detect_and_fix_data_types,
     validate_dataset, save_preprocessed_output
   - These read/write files in: modified_datasets/<problem-slug>/ (a subfolder named after your project)
   - NEVER pass a /workspace/... path to these tools — it will crash
     with [Errno 2] No such file or directory.

B) SANDBOX TOOLS (run inside the Docker container, use /workspace/... paths)
   - start_sandbox, stop_sandbox, upload_dataset_to_sandbox,
     run_in_sandbox, write_file_to_sandbox, read_file_from_sandbox,
     download_from_sandbox
   - Files inside the sandbox live in /workspace/data/ and /workspace/output/
   - The host filesystem is NOT visible inside the sandbox.

================================================================
CRASH RECOVERY — check this BEFORE calling get_preprocessing_context
================================================================
1. Call get_preprocessing_resume_state() first.
   - If preprocessing_complete is True → preprocessing is fully done from a previous run.
     Call save_preprocessed_output with preprocessed_dataset_path to re-confirm, then STOP.
   - If agent3_has_partial_output is True → a previous run saved partial progress.
     Use last_step_output_path as the starting input to the NEXT step in the plan.
     The current_step value tells you how many steps already completed.
     Skip those steps and continue from there.
   - If neither → no partial state; proceed with the full normal workflow below.

================================================================
DEFAULT WORKFLOW (use built-in tools — no sandbox needed)
================================================================
1. Call get_preprocessing_context to load the plan, dataset path,
   AND any validation feedback from Agent 4 (for retries).
2. The 'selected_dataset_path' from context is a LOCAL Windows path.
   Use it as the input to the FIRST built-in tool.
   EXCEPTION: if crash recovery above found last_step_output_path, use THAT instead.
3. For each step in step_by_step_order, call the matching built-in tool.
   - Each built-in tool returns the LOCAL path of its output file.
   - Use that returned path as the input to the next tool.
   - Built-in tools automatically name their output files as step_N_<name>.csv — you do NOT specify output_path.
4. After the last step, call save_preprocessed_output with the
   final LOCAL path. This finalizes Agent 3's work.

DO NOT call start_sandbox / upload_dataset_to_sandbox unless step (B)
below applies.

================================================================
WHEN TO USE THE SANDBOX (only for complex custom transforms)
================================================================
Only spin up the sandbox if a step in the plan CANNOT be done by
any built-in tool — for example: a custom domain-specific transform,
running a library not exposed as a built-in tool, etc.

In that case, do this MINI-LOOP for that one step ONLY:
  1. start_sandbox
     → CHECK THE RESULT. If it contains "error", the sandbox is unavailable
       (Docker Desktop is likely not running). In that case:
       - Skip all sandbox steps entirely.
       - Use the closest built-in tool equivalent instead.
       - Note "sandbox unavailable" in save_preprocessed_output.
       - DO NOT crash the pipeline — continue with built-in tools.
  2. write_file_to_sandbox('script.py', '<python code>')
     → The file is already in /workspace inside the container — no upload needed.
  3. run_in_sandbox("exec(open('/workspace/script.py').read())")
     → CHECK the result's "success" field. If False:
       - If error_type == "docker_not_running": sandbox went down — skip remaining sandbox steps.
       - If error_type == "execution_error": fix your Python code and retry.
  4. read_file_from_sandbox('<output_filename>.csv')
     → Results written to /workspace by the script are immediately on disk.
  5. stop_sandbox
Then resume the built-in tool chain on the local file in modified_datasets/.

================================================================
TOOL SELECTION GUIDE FOR PLAN STEPS
================================================================
- drop_columns → drop_columns tool
- handle_duplicates → remove_duplicates tool
- fix_data_types → detect_and_fix_data_types tool
- handle_missing → handle_missing_values tool
- parse_dates → parse_datetime_columns tool
- process_text → process_text_columns tool
- handle_outliers → handle_outliers tool
- encode_categoricals → encode_categorical_columns tool
- feature_engineering → engineer_features tool
- scale_numerics → scale_numeric_columns tool
- final_validation → validate_dataset tool

RETRY HANDLING:
- If agent4_feedback is not empty, focus ONLY on fixing those issues.
- Re-run only the affected steps, starting from the previously
  preprocessed local file (not from the original).

RULES:
- NEVER mix path types. Built-in tools = local paths only.
  Sandbox tools = /workspace/... paths only.
- ALWAYS chain tools: output of step N = input of step N+1.
- Check each tool's return status before proceeding.
- Keep track of row counts — never lose more than 20% without good reason.
- If you started the sandbox, you MUST stop it before finishing.
- After EVERY run_in_sandbox call, immediately call log_sandbox_step with the script you wrote and the stdout returned. Agent 4 needs this to verify your sandbox work.
- NEVER choose your own output filenames. Built-in tools generate step_N_<name>.csv automatically.

HANDLING PLAN-DATA MISMATCHES (CRITICAL — do NOT stall):
- Before executing any plan step, check if the referenced columns exist in the current dataset.
  If a plan step references columns that are not present in the current CSV, SKIP that step entirely.
  Record what was skipped in save_preprocessed_output. Move on to the next step immediately.
- If engineer_features returns a result where all features have "status": "unsupported_operation",
  that step is complete (no crash) — move on to the next step. Do NOT retry.
- If scale_numeric_columns or encode_categorical_columns reports "column not found" or produces
  no changes because all target columns are missing, that is OK — skip and continue.
- The goal is to finish ALL steps (even if some are skipped), then call save_preprocessed_output.
  A pipeline that completes with some skipped steps is far better than a stalled pipeline.
""",
    tools=[
        get_preprocessing_resume_state,
        get_preprocessing_context,
        # Built-in preprocessing tools
        handle_missing_values,
        remove_duplicates,
        drop_columns,
        encode_categorical_columns,
        scale_numeric_columns,
        handle_outliers,
        parse_datetime_columns,
        engineer_features,
        process_text_columns,
        detect_and_fix_data_types,
        validate_dataset,
        log_sandbox_step,
        save_preprocessed_output,
        # SafeExecute tools
        start_sandbox,
        stop_sandbox,
        upload_dataset_to_sandbox,
        run_in_sandbox,
        write_file_to_sandbox,
        read_file_from_sandbox,
        download_from_sandbox,
    ],
)


# --- Agent 4: Validation Agent ---
validation_agent = Agent(
    model=MODEL,
    name="validation_agent",
    description="Validates preprocessing quality and provides feedback or approves the result.",
    output_key="agent4_validation",
    instruction="""You are Agent 4: the Preprocessing Validator.

Your job is to thoroughly validate the preprocessed dataset.

WORKFLOW:
1. Call load_validation_context to understand the plan, original path, and preprocessed path.
2. Call start_sandbox.
   → CHECK THE RESULT. If it contains "error":
     - The sandbox (Docker) is unavailable. Note this.
     - Skip steps 3–4 entirely.
     - Set checklist_json = '{}' when calling save_validation_result.
     - Proceed directly to step 5 (structural checks are still possible without Docker).
3. Call run_plan_aware_validation(original_path, preprocessed_path).
   → If the result contains "error": treat as if sandbox was unavailable (skip plan-aware checks).
4. Call stop_sandbox.
5. Call validate_dataset for structural checks (nulls, dupes, inf, dtypes).
6. Call compare_before_after for shape/stat diff.
7. Make your verdict based on available evidence:
   - If plan-aware checklist is available: primary weight on it.
   - If sandbox unavailable: base verdict ONLY on structural checks + compare_before_after.
     Reduce quality score by 1 point and note "plan-aware validation skipped (Docker unavailable)".
8. Call save_validation_result with your verdict, feedback, and the checklist_json from step 3
   (or '{}' if sandbox was unavailable).

VALIDATION CRITERIA:
PRIMARY — Plan-aware checks (from run_plan_aware_validation checklist, if available):
- Each planned step has a corresponding PASSED check in the checklist
- If any check is FAILED, that is a specific issue for Agent 3 to fix
- Use the checklist's evidence (actual numbers) in your feedback — not vague descriptions

SECONDARY — Structural checks (from validate_dataset, always run):
- No null values remain (unless the plan explicitly allows it)
- No duplicate rows remain
- No infinite values in numeric columns
- Row count hasn't dropped by more than 20%
- Target column exists and has no nulls

PASS CONDITIONS:
- All critical checks pass
- Quality score >= 7/10
- The preprocessing aligns with the user's ML goal

FAIL CONDITIONS:
- Any critical check fails (nulls, missing target, data corruption)
- Quality score < 7/10
- Major deviation from the preprocessing plan

When you FAIL:
- Provide SPECIFIC, ACTIONABLE feedback for Agent 3
- Say exactly which tool to use and with what parameters
- Don't be vague — say "fill column X with median" not "fix missing values"

When you PASS:
- Provide a quality summary the user can review
- Highlight any minor concerns that don't block approval

RULES:
- You MUST call all four tools
- Be thorough but fair — don't fail for minor cosmetic issues
- If this is iteration 3+, be more lenient on non-critical issues
- Always provide a quality score with justification
""",
    tools=[
        load_validation_context,
        # Plan-aware sandbox validation
        start_sandbox,
        stop_sandbox,
        upload_dataset_to_sandbox,
        run_plan_aware_validation,
        # Structural checks
        validate_dataset,
        compare_before_after,
        save_validation_result,
    ],
)


# --- Loop Escalation Checker ---
class LoopEscalationChecker(BaseAgent):
    """Checks if Agent 4 passed the validation or max retries reached."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        try:
            state = load_state()
            agent4_iterations = state.get("agent4_iterations", [])
            verdict = agent4_iterations[-1].get("verdict", "") if agent4_iterations else ""
            iteration = state.get("loop_iteration", 0)
        except Exception as e:
            # State is unreadable — stop the loop to avoid infinite retries on broken state
            print(f"[LOOP] Could not read pipeline state: {e} — stopping loop.", flush=True)
            save_state({"status": "preprocessing_complete_with_warnings", "loop_error": str(e)})
            yield Event(author=self.name, actions=EventActions(escalate=True))
            return

        status = state.get("status", "")
        # If Agent 4 never ran but Agent 3 already finished successfully,
        # escalate so the pipeline can proceed to the report generator.
        agent3_done_no_validation = (status in ("agent3_done", "preprocessing_complete")) and not agent4_iterations

        should_stop = (verdict == "PASS") or (iteration >= 5) or agent3_done_no_validation

        if should_stop and verdict != "PASS":
            final_status = "preprocessing_complete" if agent3_done_no_validation else "preprocessing_complete_with_warnings"
            save_state({"status": final_status})
            print(f"[LOOP] Escalating — reason: verdict={verdict!r}, iteration={iteration}, agent3_done_no_validation={agent3_done_no_validation}", flush=True)

        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop),
        )


# --- Code Gen + Validation Loop ---
code_gen_validation_loop = LoopAgent(
    name="code_gen_validation_loop",
    description="Runs Agent 3 (preprocessing) and Agent 4 (validation) in a loop until validation passes or max retries.",
    max_iterations=5,
    sub_agents=[
        preprocessing_executor_agent,
        validation_agent,
        LoopEscalationChecker(name="loop_escalation_checker"),
    ],
)


# --- Agent 5: Report Generator ---
report_generator_agent = Agent(
    model=MODEL,
    name="report_generator_agent",
    description="Generates a comprehensive A-to-Z preprocessing report documenting every decision, transformation, and result.",
    output_key="final_report",
    instruction="""You are Agent 5: the Preprocessing Report Generator.

Your job is to generate a COMPREHENSIVE, DETAILED preprocessing report that documents
EVERY SINGLE THING that happened in the pipeline — from the very first dataset scan
to the final validation verdict. Nothing should be left out.

WORKFLOW:
1. Call load_full_pipeline_context to get all pipeline data, stats, and file paths.
2. Write the report following the EXACT structure below.
3. Call save_preprocessing_report to save it.

REPORT STRUCTURE (follow this EXACTLY):

```
# Data Preprocessing Report
Generated: <current date/time>
Project: <user's goal>

---

## 1. Executive Summary
- One paragraph overview of what was done, the dataset used, and the final outcome.
- Final quality score from validation.
- Key numbers: original rows/cols → preprocessed rows/cols.

## 2. Project Context
- User's stated goal (verbatim from pipeline state)
- Key requirements gathered from Q&A session
- ML task type (classification/regression/etc.)
- Target variable identified

## 3. Dataset Selection (Agent 1)
- Total datasets scanned (list every folder and file found)
- For EACH dataset scanned:
  - File name, path, size, format
  - Number of rows and columns
  - Column names and types
  - Missing values found
- **Selected dataset**: name, path, reason for selection
- Potential data quality issues identified
- Why other datasets were NOT selected

## 4. Original Dataset Profile
- Full shape (rows × columns)
- Complete column inventory table:
  | Column | Data Type | Unique Values | Missing Count | Missing % | Sample Values |
- Numeric column statistics (mean, std, min, max, quartiles)
- Categorical column distribution summaries
- Duplicate row count
- Outlier summary per numeric column
- High correlation pairs (>0.9)
- Unparsed date columns detected
- Constant/near-constant columns

## 5. Preprocessing Plan (Agent 2)
- Plan summary
- Target column identified
- Step-by-step execution order planned
- For EACH step in the plan, document:
  - What: the operation
  - Why: the reasoning
  - How: the specific method/parameters chosen
  - Which columns: affected columns

### 5.1 Columns to Drop
- List each column and the reason for dropping

### 5.2 Duplicate Handling Strategy
- Method chosen and why

### 5.3 Data Type Fixes
- Each column that needed type conversion and the target type

### 5.4 Missing Value Strategy
- Per-column strategy with reasoning

### 5.5 DateTime Processing
- Columns parsed and features extracted

### 5.6 Text Processing
- Per-column method (TF-IDF, length, word count, hash, etc.)

### 5.7 Outlier Handling
- Per-column strategy with reasoning

### 5.8 Categorical Encoding
- Per-column encoding method with reasoning

### 5.9 Feature Engineering
- Each new feature: name, formula/method, reasoning

### 5.10 Scaling Strategy
- Per-column scaling method with reasoning

## 6. Preprocessing Execution (Agent 3)
- Number of loop iterations needed
- For EACH step executed:
  - Tool used
  - Input file → Output file
  - Rows before → Rows after
  - Columns before → Columns after
  - Specific changes made (e.g., "filled 234 nulls in column X with median value 45.2")
  - Any errors encountered and how they were resolved

## 7. Validation Results (Agent 4)
- Final verdict: PASS / FAIL
- Quality score: X/10
- Each validation check performed:
  | Check | Result | Details |
- Before vs. After comparison:
  - Shape change
  - Columns added/removed
  - Missing values eliminated
  - Duplicates eliminated
  - Data type changes
  - Statistical shifts in key columns
- If retries occurred:
  - What failed in each iteration
  - What feedback was given
  - How it was fixed

## 8. Final Dataset Summary
- Preprocessed file location
- Final shape (rows × columns)
- Complete column inventory of the FINAL dataset:
  | Column | Data Type | Non-Null Count | Min | Max | Mean/Mode |
- Data type distribution (how many int, float, object columns)
- Confirmation: zero nulls, zero duplicates, zero infinites

## 9. Recommendations for Next Steps
- Based on the preprocessed data, suggest:
  - Suitable ML algorithms for this dataset
  - Potential additional feature engineering
  - Cross-validation strategy
  - Any concerns about the data that the user should monitor

## 10. Appendix
- Full list of all files created during preprocessing (intermediate + final)
- Pipeline state keys and their final values
- Tool execution log (which tools were called in what order)
```

RULES:
- Do NOT skip any section. Every section must have content.
- Do NOT summarize — be EXHAUSTIVE. List every column, every change, every number.
- Use markdown tables wherever possible for readability.
- Include actual numbers, not vague descriptions like "several columns" — say "7 columns".
- If a section doesn't apply (e.g., no datetime columns), explicitly say "Not applicable — no datetime columns found in this dataset."
- The report should be self-contained — someone reading ONLY this report should fully understand what happened.
""",
    tools=[load_full_pipeline_context, save_preprocessing_report],
)


# ============================================================
# ==================  ROOT ORCHESTRATOR  =====================
# ============================================================
data_preprocessing_agent = SequentialAgent(
    name="data_preprocessing_orchestrator",
    description="Orchestrates the full data preprocessing pipeline: dataset selection → planning → execution+validation loop → final report.",
    sub_agents=[
        dataset_analyzer_agent,
        preprocessing_strategist_agent,
        code_gen_validation_loop,
        report_generator_agent,
    ],
)

root_agent = data_preprocessing_agent