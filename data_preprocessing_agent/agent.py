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
from pipeline_state import load_state, save_state

load_dotenv()

# ============================================================
# CONSTANTS
# ============================================================

MODEL = "gemini-3.1-flash-lite-preview"
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODIFIED_DATASETS_DIR = PROJECT_ROOT / "modified_datasets"

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

    save_state({
        "agent1_output": agent1_output,
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
    agent1_out = state.get("agent1_output", {})
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
        "agent1_selection": state.get("agent1_output", {}).get("selected_dataset", {}),
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

    save_state({
        "agent2_output": {"preprocessing_plan": plan},
        "status": "agent2_done",
    })

    return f"Preprocessing plan saved. Steps: {step_by_step_order}"


# ============================================================
# ==================  AGENT 3 TOOLS  =========================
# ============================================================

def get_preprocessing_context() -> str:
    """
    Load everything Agent 3 needs: user goal, Agent 1's dataset selection,
    Agent 2's preprocessing plan, the dataset file path, and any previous
    validation feedback from Agent 4 (for retry loops).
    """
    state = load_state()
    context = {
        "user_goal": state.get("user_goal", ""),
        "selected_dataset": state.get("agent1_output", {}).get("selected_dataset", {}),
        "preprocessing_plan": state.get("agent2_output", {}).get("preprocessing_plan", {}),
        "dataset_path": state.get("selected_dataset_path", ""),
        "validation_feedback": state.get("agent4_feedback", ""),
        "iteration": state.get("loop_iteration", 0),
    }
    return json.dumps(context, indent=2, default=str)


def handle_missing_values(
    dataset_path: str,
    strategy_json: str,
    output_path: str,
) -> str:
    """
    Handle missing values in the dataset using column-specific strategies.

    Args:
        dataset_path: Absolute path to the input CSV/Parquet file.
        strategy_json: JSON string mapping column names to strategies.
            Example: '{"age": "fill_median", "city": "fill_mode", "temp": "fill_mean", "id": "drop_rows"}'
            Supported strategies: fill_mean, fill_median, fill_mode, fill_zero,
            fill_ffill, fill_bfill, fill_interpolate, drop_rows, drop_column.
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with before/after missing counts and rows affected.
    """
    import pandas as pd

    df = pd.read_csv(dataset_path)
    before_missing = df.isnull().sum().to_dict()
    before_rows = len(df)

    strategies = json.loads(strategy_json)

    for col, strategy in strategies.items():
        if col not in df.columns:
            continue
        if strategy == "fill_mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "fill_median":
            df[col] = df[col].fillna(df[col].median())
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

    df.to_csv(output_path, index=False)
    after_missing = df.isnull().sum().to_dict()

    return json.dumps({
        "status": "success",
        "before_missing": {k: v for k, v in before_missing.items() if v > 0},
        "after_missing": {k: v for k, v in after_missing.items() if v > 0},
        "rows_before": before_rows,
        "rows_after": len(df),
        "output_path": output_path,
    }, indent=2)


def remove_duplicates(
    dataset_path: str,
    strategy: str,
    subset_columns: str,
    output_path: str,
) -> str:
    """
    Remove duplicate rows from the dataset.

    Args:
        dataset_path: Absolute path to the input CSV file.
        strategy: One of 'keep_first', 'keep_last', 'drop_all'.
        subset_columns: Comma-separated column names to check for duplicates,
            or 'all' to check all columns.
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with duplicate count before/after and rows removed.
    """
    import pandas as pd

    df = pd.read_csv(dataset_path)
    subset = None if subset_columns == "all" else [c.strip() for c in subset_columns.split(",")]

    before_dupes = int(df.duplicated(subset=subset).sum())

    if strategy == "keep_first":
        df = df.drop_duplicates(subset=subset, keep="first")
    elif strategy == "keep_last":
        df = df.drop_duplicates(subset=subset, keep="last")
    elif strategy == "drop_all":
        df = df.drop_duplicates(subset=subset, keep=False)

    df.to_csv(output_path, index=False)

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
    output_path: str,
) -> str:
    """
    Drop specified columns from the dataset.

    Args:
        dataset_path: Absolute path to the input CSV file.
        columns_to_drop: Comma-separated column names to drop.
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with columns dropped and remaining columns.
    """
    import pandas as pd

    df = pd.read_csv(dataset_path)
    cols = [c.strip() for c in columns_to_drop.split(",") if c.strip()]
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    df = df.drop(columns=existing)
    df.to_csv(output_path, index=False)

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
    output_path: str,
) -> str:
    """
    Encode categorical columns using specified methods.

    Args:
        dataset_path: Absolute path to the input CSV file.
        encoding_json: JSON string mapping column names to encoding methods.
            Example: '{"city": "onehot", "grade": "label", "browser": "frequency", "size": "ordinal:S,M,L,XL"}'
            Supported: 'label', 'onehot', 'frequency', 'ordinal:val1,val2,...' (ordered),
            'binary' (for 2-class columns), 'target:target_col' (target/mean encoding).
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with encoding details per column and new column list.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(dataset_path)
    encodings = json.loads(encoding_json)
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

    df.to_csv(output_path, index=False)

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
    output_path: str,
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
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with scaling details per column.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer

    df = pd.read_csv(dataset_path)
    scalings = json.loads(scaling_json)
    exclude = {c.strip() for c in exclude_columns.split(",") if c.strip()}
    details = {}

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if "__all_numeric__" in scalings:
        method = scalings["__all_numeric__"]
        col_methods = {col: method for col in numeric_cols}
    else:
        col_methods = {col: m for col, m in scalings.items() if col in numeric_cols and col not in exclude}

    for col, method in col_methods.items():
        if col not in df.columns:
            continue
        before_stats = {"mean": round(float(df[col].mean()), 4), "std": round(float(df[col].std()), 4)}

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
            df[col] = np.log1p(df[col].clip(lower=0))
        elif method == "maxabs":
            scaler = MaxAbsScaler()
            df[col] = scaler.fit_transform(df[[col]])
        elif method == "power_yeo":
            pt = PowerTransformer(method="yeo-johnson")
            df[col] = pt.fit_transform(df[[col]])

        after_stats = {"mean": round(float(df[col].mean()), 4), "std": round(float(df[col].std()), 4)}
        details[col] = {"method": method, "before": before_stats, "after": after_stats}

    df.to_csv(output_path, index=False)

    return json.dumps({
        "status": "success",
        "scaling_details": details,
        "excluded": list(exclude),
        "output_path": output_path,
    }, indent=2)


def handle_outliers(
    dataset_path: str,
    outlier_json: str,
    output_path: str,
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
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with outlier counts before/after per column.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(dataset_path)
    strategies = json.loads(outlier_json)
    details = {}
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

    df.to_csv(output_path, index=False)

    return json.dumps({
        "status": "success",
        "outlier_details": details,
        "rows_before": before_rows,
        "rows_after": len(df),
        "output_path": output_path,
    }, indent=2)


def parse_datetime_columns(
    dataset_path: str,
    datetime_json: str,
    output_path: str,
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
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with new columns created per datetime column.
    """
    import pandas as pd

    df = pd.read_csv(dataset_path)
    configs = json.loads(datetime_json)
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
        details[col] = {"status": "success", "new_columns": new_cols, "null_dates": int(dt.isnull().sum())}

    df.to_csv(output_path, index=False)

    return json.dumps({
        "status": "success",
        "datetime_details": details,
        "final_columns": list(df.columns),
        "output_path": output_path,
    }, indent=2)


def engineer_features(
    dataset_path: str,
    features_json: str,
    output_path: str,
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
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with created features and their basic stats.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(dataset_path)
    features = json.loads(features_json)
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

    df.to_csv(output_path, index=False)

    return json.dumps({
        "status": "success",
        "feature_details": details,
        "total_columns": len(df.columns),
        "output_path": output_path,
    }, indent=2)


def process_text_columns(
    dataset_path: str,
    text_json: str,
    output_path: str,
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
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with processing details per text column.
    """
    import pandas as pd
    import numpy as np
    import re

    df = pd.read_csv(dataset_path)
    configs = json.loads(text_json)
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

    df.to_csv(output_path, index=False)

    return json.dumps({
        "status": "success",
        "text_processing_details": details,
        "final_columns": list(df.columns),
        "output_path": output_path,
    }, indent=2)


def detect_and_fix_data_types(
    dataset_path: str,
    type_fixes_json: str,
    output_path: str,
) -> str:
    """
    Detect mistyped columns and cast them to correct types.

    Args:
        dataset_path: Absolute path to the input CSV file.
        type_fixes_json: JSON mapping column names to target types.
            Example: '{"price": "float", "age": "int", "is_active": "bool", "date": "datetime", "category": "category"}'
            Supported types: 'int', 'float', 'str', 'bool', 'datetime', 'category'.
            Use 'auto' to let pandas infer the best type for a column.
        output_path: Absolute path to save the result CSV.

    Returns:
        JSON with type changes per column and any conversion errors.
    """
    import pandas as pd

    df = pd.read_csv(dataset_path)
    fixes = json.loads(type_fixes_json)
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

    df.to_csv(output_path, index=False)

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

    df = pd.read_csv(dataset_path)
    checks = json.loads(checks_json)
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


# NOTE: Custom code execution is handled by OpenSandbox tools
# (run_in_sandbox, write_file_to_sandbox, read_file_from_sandbox).
# These are regular async functions that CAN be mixed with other tools.


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

    MODIFIED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODIFIED_DATASETS_DIR / output_filename

    shutil.copy2(dataset_path, dest)

    save_state({
        "agent3_output": {
            "preprocessed_file": str(dest),
            "summary": summary,
            "steps_completed": [s.strip() for s in steps_completed.split(",")],
        },
        "preprocessed_dataset_path": str(dest),
        "status": "agent3_done",
    })

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
    context = {
        "user_goal": state.get("user_goal", ""),
        "original_dataset": state.get("agent1_output", {}).get("selected_dataset", {}),
        "preprocessing_plan": state.get("agent2_output", {}).get("preprocessing_plan", {}),
        "agent3_output": state.get("agent3_output", {}),
        "preprocessed_path": state.get("preprocessed_dataset_path", ""),
        "iteration": state.get("loop_iteration", 0),
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


def save_validation_result(
    verdict: str,
    issues_found: str,
    feedback_for_agent3: str,
    quality_score: str,
    validation_summary: str,
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

    Returns:
        Confirmation message.
    """
    result = {
        "verdict": verdict.upper(),
        "issues_found": [i.strip() for i in issues_found.split(",") if i.strip()],
        "feedback_for_agent3": feedback_for_agent3,
        "quality_score": quality_score,
        "validation_summary": validation_summary,
    }

    state_update = {
        "agent4_output": result,
        "agent4_feedback": feedback_for_agent3 if verdict.upper() == "FAIL" else "",
        "loop_iteration": load_state().get("loop_iteration", 0) + 1,
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
        "agent1_output": state.get("agent1_output", {}),
        "agent2_output": state.get("agent2_output", {}),
        "agent3_output": state.get("agent3_output", {}),
        "agent4_output": state.get("agent4_output", {}),
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
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / report_filename

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

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

# OpenSandbox tools for secure code execution
from data_preprocessing_agent.sandbox_executor import (
    start_sandbox,
    stop_sandbox,
    upload_dataset_to_sandbox,
    run_in_sandbox,
    write_file_to_sandbox,
    read_file_from_sandbox,
    download_from_sandbox,
)


# --- Agent 1: Dataset Analyzer ---
dataset_analyzer_agent = Agent(
    model=MODEL,
    name="dataset_analyzer_agent",
    description="Analyzes all available datasets and selects the best one for the user's ML project.",
    output_key="agent1_result",
    instruction="""You are Agent 1: the Dataset Analyzer & Selector.

Your job is to:
1. First, call get_project_context to understand the user's goal and project plan.
2. Then, call scan_datasets_folder to discover and preview all available datasets.
3. Analyze EVERY dataset returned — look at columns, dtypes, row counts, missing values, and previews.
4. Select the SINGLE BEST dataset file that is most relevant to the user's project goal.
5. Finally, call save_selected_dataset with all the details of your selection.

SELECTION CRITERIA (in order of importance):
- Relevance to the user's stated goal and project plan
- Data quality (fewer missing values, clean column names, appropriate dtypes)
- Dataset size (prefer larger datasets when relevance is equal)
- File format (prefer CSV/Parquet over JSON/Excel for ML pipelines)

RULES:
- You MUST call all three tools in order: get_project_context → scan_datasets_folder → save_selected_dataset
- You MUST select exactly ONE file, not a folder
- If multiple files exist in one folder (e.g. train.csv + test.csv), pick the main/largest one
- Provide a detailed reason for your selection
- List ALL potential data quality issues you notice

After saving, respond with a brief summary of which dataset you selected and why.
""",
    tools=[get_project_context, scan_datasets_folder, save_selected_dataset],
)


# --- Agent 2: Preprocessing Strategist ---
preprocessing_strategist_agent = Agent(
    model=MODEL,
    name="preprocessing_strategist_agent",
    description="Analyzes the selected dataset deeply and creates a detailed preprocessing plan.",
    output_key="agent2_result",
    instruction="""You are Agent 2: the Preprocessing Strategist.

Your job is to create a comprehensive, actionable preprocessing plan for the selected dataset.

WORKFLOW:
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
- You MUST call all three tools in order
- Every column must be accounted for in the plan
- The target column must NEVER be scaled or encoded (unless it's a label that needs encoding)
- Be specific about WHY you chose each strategy

After saving, respond with a concise summary of your plan.
""",
    tools=[get_user_requirements, load_dataset_profile, save_preprocessing_plan],
)


# --- Agent 3: Preprocessing Executor ---
preprocessing_executor_agent = Agent(
    model=MODEL,
    name="preprocessing_executor_agent",
    description="Executes the preprocessing plan step by step using specialized tools and an OpenSandbox environment.",
    output_key="agent3_result",
    instruction="""You are Agent 3: the Preprocessing Executor.

Your job is to execute the preprocessing plan step by step.

WORKFLOW:
1. Call get_preprocessing_context to load the plan, dataset path, AND any validation feedback from Agent 4 (for retries).
2. Call start_sandbox to start the secure execution environment.
3. Call upload_dataset_to_sandbox to upload the selected dataset into the sandbox.
4. Follow the step_by_step_order from the plan EXACTLY.
5. For each step, use the appropriate tool.
6. Each tool outputs a file — use that output file as input for the next tool.
7. After all steps, call download_from_sandbox to save the final preprocessed file locally.
8. Call save_preprocessed_output with the local file path.
9. Call stop_sandbox to clean up.

TOOL SELECTION GUIDE (use these for standard preprocessing steps):
- drop_columns → use drop_columns tool
- handle_duplicates → use remove_duplicates tool
- fix_data_types → use detect_and_fix_data_types tool
- handle_missing → use handle_missing_values tool
- parse_dates → use parse_datetime_columns tool
- process_text → use process_text_columns tool
- handle_outliers → use handle_outliers tool
- encode_categoricals → use encode_categorical_columns tool
- feature_engineering → use engineer_features tool
- scale_numerics → use scale_numeric_columns tool
- final_validation → use validate_dataset tool

SANDBOX TOOLS (use these for complex/custom transformations):
- write_file_to_sandbox → write a Python script into the sandbox
- run_in_sandbox → execute the script (e.g. 'python3 /workspace/data/script.py')
- read_file_from_sandbox → read results or processed files from sandbox
- download_from_sandbox → save sandbox files back to your local machine

For complex steps that the built-in tools can't handle:
1. Write a Python script using write_file_to_sandbox
2. Run it using run_in_sandbox
3. Read/download the results

SANDBOX FILE PATHS:
- Data files go in: /workspace/data/
- Output files go in: /workspace/output/

LOCAL FILE PATH CONVENTION:
- Use the modified_datasets folder for intermediate files
- Name intermediate files: step_N_<step_name>.csv
- Final file: preprocessed_<original_filename>.csv
- The modified_datasets folder is at: """ + str(MODIFIED_DATASETS_DIR) + """

RETRY HANDLING:
- If agent4_feedback is not empty, focus ONLY on fixing the issues mentioned
- Re-run only the affected steps, not the entire pipeline
- Use the previously preprocessed file as starting point for fixes

RULES:
- ALWAYS start the sandbox before any processing and stop it when done
- ALWAYS chain tools: output of step N = input of step N+1
- Check each tool's return status before proceeding
- If a built-in tool fails, write a custom Python script and run it in the sandbox
- Keep track of row count changes — never lose more than 20% of rows without good reason
""",
    tools=[
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
        save_preprocessed_output,
        # OpenSandbox tools
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
1. Call load_validation_context to understand the plan and what Agent 3 did.
2. Call validate_dataset with comprehensive checks on the preprocessed file.
3. Call compare_before_after to see what changed between original and preprocessed.
4. Make your verdict: PASS or FAIL.
5. Call save_validation_result with your verdict and detailed feedback.

VALIDATION CRITERIA:
- No null values remain (unless the plan explicitly says to keep some)
- No duplicate rows remain
- No infinite values in numeric columns
- Row count hasn't dropped by more than 20% (unless plan required aggressive filtering)
- All expected columns from the plan exist
- Target column exists and has no nulls
- Data types are appropriate for ML (mostly numeric after encoding)
- Feature engineering produced meaningful features (no all-null or all-same columns)
- Scaling was applied correctly (check mean/std shifts)
- No data leakage (target-derived features shouldn't exist)

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
    tools=[load_validation_context, validate_dataset, compare_before_after, save_validation_result],
)


# --- Loop Escalation Checker ---
class LoopEscalationChecker(BaseAgent):
    """Checks if Agent 4 passed the validation or max retries reached."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        state = load_state()
        verdict = state.get("agent4_output", {}).get("verdict", "")
        iteration = state.get("loop_iteration", 0)

        should_stop = (verdict == "PASS") or (iteration >= 5)

        if should_stop and verdict != "PASS":
            save_state({"status": "preprocessing_complete_with_warnings"})

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

root_agent = SequentialAgent(
    name="data_preprocessing_orchestrator",
    description="Orchestrates the full data preprocessing pipeline: dataset selection → planning → execution+validation loop → final report.",
    sub_agents=[
        dataset_analyzer_agent,
        preprocessing_strategist_agent,
        code_gen_validation_loop,
        report_generator_agent,
    ],
)