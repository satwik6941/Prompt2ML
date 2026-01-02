import asyncio
from google.adk.agents import Agent
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from dotenv import load_dotenv
import sys
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

# Global variable to track current working dataset
CURRENT_DATASET_PATH = None

# ========== PHASE 1: UNDERSTANDING TOOLS ==========

def load_user_context() -> str:
    """Load and return the user's problem statement and dataset information from JSON file."""
    try:
        print("[INFO] Loading user context from user_input.json...", flush=True)

        project_dir = Path(__file__).parent.parent
        json_file_path = project_dir / "user_input.json"

        if not json_file_path.exists():
            return "Error: user_input.json not found. Please run the data extractor agent first."

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        context = {
            "problem_statement": data.get("problem_statement", "Not provided"),
            "dataset_source": data.get("downloaded_dataset", {}).get("source", "Unknown"),
            "dataset_name": data.get("downloaded_dataset", {}).get("dataset_name", "Unknown"),
            "dataset_location": data.get("downloaded_dataset", {}).get("save_location", "Unknown")
        }

        result = f"""
USER CONTEXT:
=============
Problem Statement: {context['problem_statement']}
Dataset Source: {context['dataset_source']}
Dataset Name: {context['dataset_name']}
Dataset Location: {context['dataset_location']}

You should now analyze this dataset and preprocess it according to the problem statement.
"""
        print(f"[SUCCESS] User context loaded!", flush=True)
        return result

    except Exception as e:
        error_msg = f"Error loading user context: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def discover_and_analyze_dataset(dataset_path: str) -> str:
    """
    Recursively discover all CSV files in the dataset folder and analyze their structure.
    Returns column names, data types, row counts, and sample data without loading full datasets.
    """
    try:
        print(f"[INFO] Discovering CSV files in: {dataset_path}", flush=True)

        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            return f"Error: Dataset path does not exist: {dataset_path}"

        # Find all CSV files recursively
        csv_files = list(dataset_dir.rglob("*.csv"))

        if not csv_files:
            return f"No CSV files found in {dataset_path}"

        print(f"[INFO] Found {len(csv_files)} CSV file(s)", flush=True)

        results = []
        results.append(f"DATASET DISCOVERY REPORT")
        results.append(f"=" * 80)
        results.append(f"Total CSV files found: {len(csv_files)}\n")

        for idx, csv_file in enumerate(csv_files, 1):
            relative_path = csv_file.relative_to(dataset_dir)
            results.append(f"\n[{idx}] FILE: {relative_path}")
            results.append("-" * 80)

            try:
                # Read only first few rows to get structure
                df_sample = pd.read_csv(csv_file, nrows=5)
                total_rows = sum(1 for _ in open(csv_file, encoding='utf-8', errors='ignore')) - 1

                results.append(f"Location: {csv_file}")
                results.append(f"Size: {csv_file.stat().st_size / 1024:.2f} KB")
                results.append(f"Rows: {total_rows:,}")
                results.append(f"Columns: {len(df_sample.columns)}")

                # Column information
                results.append(f"\nColumn Details:")
                for col in df_sample.columns:
                    dtype = str(df_sample[col].dtype)
                    sample_val = str(df_sample[col].iloc[0])[:50] if not pd.isna(df_sample[col].iloc[0]) else "NaN"
                    results.append(f"  - {col} ({dtype}): {sample_val}...")

                # Sample rows
                results.append(f"\nFirst 3 rows:")
                results.append(df_sample.head(3).to_string(index=False, max_cols=10))

            except Exception as e:
                results.append(f"Error reading file: {str(e)}")

        result_text = "\n".join(results)
        print(f"[SUCCESS] Dataset discovery complete!", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error discovering datasets: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def get_column_details(csv_path: str, columns: str) -> str:
    """
    Get detailed statistics for specific columns in a CSV file.

    Args:
        csv_path: Path to the CSV file
        columns: Comma-separated list of column names (e.g., "views,likes,comments")
    """
    try:
        print(f"[INFO] Analyzing columns: {columns} in {Path(csv_path).name}", flush=True)

        # Parse columns
        column_list = [col.strip() for col in columns.split(",")]

        # Read CSV with chunking for large files
        file_size = Path(csv_path).stat().st_size / (1024 * 1024)  # Size in MB

        if file_size > 100:
            # For large files, use sampling
            df = pd.read_csv(csv_path, nrows=10000)
            print(f"[INFO] Large file detected ({file_size:.1f}MB). Using sample of 10,000 rows.", flush=True)
        else:
            df = pd.read_csv(csv_path)

        results = []
        results.append(f"DETAILED COLUMN ANALYSIS")
        results.append(f"=" * 80)

        for col in column_list:
            if col not in df.columns:
                results.append(f"\n[ERROR] Column '{col}' not found in dataset!")
                continue

            results.append(f"\n📊 Column: {col}")
            results.append("-" * 80)
            results.append(f"Data Type: {df[col].dtype}")
            results.append(f"Non-Null Count: {df[col].notna().sum():,} / {len(df):,}")
            results.append(f"Missing Values: {df[col].isna().sum():,} ({df[col].isna().sum()/len(df)*100:.2f}%)")

            if pd.api.types.is_numeric_dtype(df[col]):
                # Numerical column statistics
                results.append(f"\nNumerical Statistics:")
                results.append(f"  Min: {df[col].min()}")
                results.append(f"  Max: {df[col].max()}")
                results.append(f"  Mean: {df[col].mean():.2f}")
                results.append(f"  Median: {df[col].median():.2f}")
                results.append(f"  Std Dev: {df[col].std():.2f}")
                results.append(f"  25th Percentile: {df[col].quantile(0.25):.2f}")
                results.append(f"  75th Percentile: {df[col].quantile(0.75):.2f}")
            else:
                # Categorical column statistics
                results.append(f"\nCategorical Statistics:")
                results.append(f"  Unique Values: {df[col].nunique():,}")
                results.append(f"  Most Common:")
                value_counts = df[col].value_counts().head(5)
                for val, count in value_counts.items():
                    results.append(f"    - {val}: {count:,} ({count/len(df)*100:.2f}%)")

        result_text = "\n".join(results)
        print(f"[SUCCESS] Column analysis complete!", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error analyzing columns: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


# ========== PHASE 2: INTELLIGENT ANALYSIS TOOLS ==========

def detect_anomalies(csv_path: str) -> str:
    """
    Detect data quality issues and anomalies in the dataset.
    Returns recommendations for preprocessing.
    """
    try:
        print(f"[INFO] Detecting anomalies in {Path(csv_path).name}...", flush=True)

        # Load dataset
        file_size = Path(csv_path).stat().st_size / (1024 * 1024)
        if file_size > 100:
            df = pd.read_csv(csv_path, nrows=10000)
            print(f"[INFO] Using sample for large file ({file_size:.1f}MB)", flush=True)
        else:
            df = pd.read_csv(csv_path)

        results = []
        results.append(f"ANOMALY DETECTION REPORT")
        results.append(f"=" * 80)
        results.append(f"Dataset: {Path(csv_path).name}")
        results.append(f"Rows analyzed: {len(df):,}\n")

        issues_found = 0

        # 1. Missing values
        results.append(f"🔍 MISSING VALUES:")
        missing_cols = df.columns[df.isna().sum() > 0]
        if len(missing_cols) > 0:
            for col in missing_cols:
                missing_pct = df[col].isna().sum() / len(df) * 100
                if missing_pct > 20:
                    results.append(f"  ⚠️  {col}: {missing_pct:.2f}% missing (HIGH - consider dropping or imputing)")
                    issues_found += 1
                else:
                    results.append(f"  ℹ️  {col}: {missing_pct:.2f}% missing")
        else:
            results.append(f"  ✅ No missing values detected")

        # 2. Duplicate rows
        results.append(f"\n🔍 DUPLICATE ROWS:")
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results.append(f"  ⚠️  {duplicates:,} duplicate rows found ({duplicates/len(df)*100:.2f}%)")
            results.append(f"     Recommendation: Consider removing duplicates")
            issues_found += 1
        else:
            results.append(f"  ✅ No duplicate rows detected")

        # 3. Outliers in numerical columns
        results.append(f"\n🔍 OUTLIERS (IQR Method):")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = []

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_pct = outliers / len(df) * 100
                if outlier_pct > 5:
                    results.append(f"  ⚠️  {col}: {outliers:,} outliers ({outlier_pct:.2f}%)")
                    outlier_cols.append(col)
                    issues_found += 1

        if not outlier_cols:
            results.append(f"  ✅ No significant outliers detected")

        # 4. Constant columns (single unique value)
        results.append(f"\n🔍 CONSTANT COLUMNS:")
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            results.append(f"  ⚠️  Columns with single value (consider dropping):")
            for col in constant_cols:
                results.append(f"     - {col}")
                issues_found += 1
        else:
            results.append(f"  ✅ No constant columns detected")

        # Summary
        results.append(f"\n" + "=" * 80)
        results.append(f"SUMMARY: {issues_found} issue(s) detected")

        result_text = "\n".join(results)
        print(f"[SUCCESS] Anomaly detection complete! Found {issues_found} issues.", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error detecting anomalies: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def suggest_feature_engineering(csv_path: str, problem_type: str) -> str:
    """
    Suggest new columns to create based on existing columns and problem type.

    Args:
        csv_path: Path to CSV file
        problem_type: Type of problem (e.g., "youtube trend analysis", "cricket analysis", "classification")
    """
    try:
        print(f"[INFO] Suggesting feature engineering for: {problem_type}", flush=True)

        df = pd.read_csv(csv_path, nrows=100)  # Just need column names
        columns = df.columns.tolist()

        results = []
        results.append(f"FEATURE ENGINEERING SUGGESTIONS")
        results.append(f"=" * 80)
        results.append(f"Problem Type: {problem_type}")
        results.append(f"Existing Columns: {', '.join(columns)}\n")

        suggestions = []

        # Generic suggestions based on column patterns
        if any('view' in col.lower() for col in columns) and any('like' in col.lower() for col in columns):
            suggestions.append("engagement_rate = likes / views (measure user engagement)")

        if any('view' in col.lower() for col in columns) and any('comment' in col.lower() for col in columns):
            suggestions.append("comment_rate = comments / views (measure discussion activity)")

        if any('date' in col.lower() or 'time' in col.lower() for col in columns):
            suggestions.append("Extract date features: year, month, day, day_of_week, hour")
            suggestions.append("Calculate time_since_published (days/hours since publication)")

        if any('run' in col.lower() for col in columns) and any('ball' in col.lower() for col in columns):
            suggestions.append("strike_rate = (runs / balls) * 100 (cricket performance metric)")

        if any('match' in col.lower() for col in columns) and any('inns' in col.lower() for col in columns):
            suggestions.append("average_performance = total_value / matches (consistency metric)")

        # Numerical columns that could be combined
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            suggestions.append(f"Consider creating ratio features from numeric columns: {', '.join(numeric_cols[:5])}")

        if suggestions:
            results.append(f"💡 RECOMMENDED NEW COLUMNS:\n")
            for i, suggestion in enumerate(suggestions, 1):
                results.append(f"{i}. {suggestion}")
        else:
            results.append(f"No specific feature engineering suggestions for this dataset.")
            results.append(f"Consider domain-specific features based on your problem.")

        result_text = "\n".join(results)
        print(f"[SUCCESS] Generated {len(suggestions)} feature suggestions!", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error suggesting features: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def identify_redundant_columns(csv_path: str) -> str:
    """
    Identify columns that can potentially be dropped (all null, constant, high correlation).
    """
    try:
        print(f"[INFO] Identifying redundant columns in {Path(csv_path).name}...", flush=True)

        file_size = Path(csv_path).stat().st_size / (1024 * 1024)
        if file_size > 100:
            df = pd.read_csv(csv_path, nrows=10000)
            print(f"[INFO] Using sample for analysis", flush=True)
        else:
            df = pd.read_csv(csv_path)

        results = []
        results.append(f"REDUNDANT COLUMN ANALYSIS")
        results.append(f"=" * 80)

        redundant_cols = []

        # 1. All null columns
        results.append(f"\n🔍 ALL NULL COLUMNS:")
        all_null = [col for col in df.columns if df[col].isna().all()]
        if all_null:
            results.append(f"  ⚠️  Columns with 100% missing values:")
            for col in all_null:
                results.append(f"     - {col}")
                redundant_cols.append(col)
        else:
            results.append(f"  ✅ No completely null columns")

        # 2. Constant columns
        results.append(f"\n🔍 CONSTANT COLUMNS:")
        constant = [col for col in df.columns if df[col].nunique() == 1]
        if constant:
            results.append(f"  ⚠️  Columns with only one unique value:")
            for col in constant:
                results.append(f"     - {col} (value: {df[col].iloc[0]})")
                redundant_cols.append(col)
        else:
            results.append(f"  ✅ No constant columns")

        # 3. High correlation (for numeric columns only)
        results.append(f"\n🔍 HIGHLY CORRELATED COLUMNS (>0.95):")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            high_corr_pairs = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

            if high_corr_pairs:
                results.append(f"  ⚠️  Highly correlated column pairs (consider keeping only one):")
                for col1, col2, corr in high_corr_pairs:
                    results.append(f"     - {col1} ↔ {col2} (correlation: {corr:.3f})")
            else:
                results.append(f"  ✅ No highly correlated columns")
        else:
            results.append(f"  ℹ️  Not enough numeric columns for correlation analysis")

        # Summary
        results.append(f"\n" + "=" * 80)
        if redundant_cols:
            results.append(f"💡 RECOMMENDATION: Consider dropping these {len(redundant_cols)} columns:")
            for col in redundant_cols:
                results.append(f"   - {col}")
        else:
            results.append(f"✅ No obviously redundant columns detected")

        result_text = "\n".join(results)
        print(f"[SUCCESS] Redundant column analysis complete!", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error identifying redundant columns: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


# ========== PHASE 3: TRANSFORMATION TOOLS ==========

def create_new_column(csv_path: str, column_name: str, formula: str) -> str:
    """
    Create a new column in the dataset using a formula.

    Args:
        csv_path: Path to CSV file
        column_name: Name of the new column to create
        formula: Python expression to calculate the new column (e.g., "df['col1'] / df['col2']")
    """
    try:
        print(f"[INFO] Creating new column '{column_name}' in {Path(csv_path).name}...", flush=True)

        df = pd.read_csv(csv_path)

        # Evaluate the formula safely
        try:
            df[column_name] = eval(formula, {'df': df, 'pd': pd, 'np': np})
        except Exception as eval_error:
            return f"Error evaluating formula: {str(eval_error)}"

        # Save back to CSV
        df.to_csv(csv_path, index=False)

        success_msg = f"Successfully created column '{column_name}' using formula: {formula}"
        print(f"[SUCCESS] {success_msg}", flush=True)
        print(f"[INFO] Sample values: {df[column_name].head(3).tolist()}", flush=True)
        return success_msg

    except Exception as e:
        error_msg = f"Error creating column: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def drop_columns(csv_path: str, columns: str) -> str:
    """
    Drop specified columns from the dataset.

    Args:
        csv_path: Path to CSV file
        columns: Comma-separated list of column names to drop
    """
    try:
        column_list = [col.strip() for col in columns.split(",")]
        print(f"[INFO] Dropping columns: {column_list} from {Path(csv_path).name}...", flush=True)

        df = pd.read_csv(csv_path)

        # Check which columns exist
        existing_cols = [col for col in column_list if col in df.columns]
        missing_cols = [col for col in column_list if col not in df.columns]

        if missing_cols:
            print(f"[WARNING] Columns not found: {missing_cols}", flush=True)

        if not existing_cols:
            return f"Error: None of the specified columns exist in the dataset"

        df = df.drop(columns=existing_cols)
        df.to_csv(csv_path, index=False)

        success_msg = f"Successfully dropped {len(existing_cols)} column(s): {existing_cols}"
        print(f"[SUCCESS] {success_msg}", flush=True)
        print(f"[INFO] Remaining columns ({len(df.columns)}): {df.columns.tolist()}", flush=True)
        return success_msg

    except Exception as e:
        error_msg = f"Error dropping columns: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def handle_missing_values(csv_path: str, column: str, strategy: str) -> str:
    """
    Handle missing values in a column using specified strategy.

    Args:
        csv_path: Path to CSV file
        column: Column name
        strategy: One of: drop_rows, fill_mean, fill_median, fill_mode, fill_zero, fill_forward, fill_backward
    """
    try:
        print(f"[INFO] Handling missing values in '{column}' using strategy '{strategy}'...", flush=True)

        df = pd.read_csv(csv_path)

        if column not in df.columns:
            return f"Error: Column '{column}' not found in dataset"

        missing_before = df[column].isna().sum()

        if missing_before == 0:
            return f"No missing values found in column '{column}'"

        if strategy == "drop_rows":
            df = df.dropna(subset=[column])
        elif strategy == "fill_mean":
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                return f"Error: Cannot calculate mean for non-numeric column '{column}'"
        elif strategy == "fill_median":
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].median(), inplace=True)
            else:
                return f"Error: Cannot calculate median for non-numeric column '{column}'"
        elif strategy == "fill_mode":
            mode_value = df[column].mode()[0] if not df[column].mode().empty else 0
            df[column].fillna(mode_value, inplace=True)
        elif strategy == "fill_zero":
            df[column].fillna(0, inplace=True)
        elif strategy == "fill_forward":
            df[column].fillna(method='ffill', inplace=True)
        elif strategy == "fill_backward":
            df[column].fillna(method='bfill', inplace=True)
        else:
            return f"Error: Unknown strategy '{strategy}'. Use: drop_rows, fill_mean, fill_median, fill_mode, fill_zero, fill_forward, fill_backward"

        missing_after = df[column].isna().sum()
        df.to_csv(csv_path, index=False)

        success_msg = f"Successfully handled missing values in '{column}' using '{strategy}'. Before: {missing_before}, After: {missing_after}"
        print(f"[SUCCESS] {success_msg}", flush=True)
        return success_msg

    except Exception as e:
        error_msg = f"Error handling missing values: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def remove_outliers(csv_path: str, column: str, method: str) -> str:
    """
    Remove outliers from a numerical column.

    Args:
        csv_path: Path to CSV file
        column: Column name
        method: One of: iqr (Interquartile Range), zscore, percentile
    """
    try:
        print(f"[INFO] Removing outliers from '{column}' using '{method}' method...", flush=True)

        df = pd.read_csv(csv_path)

        if column not in df.columns:
            return f"Error: Column '{column}' not found"

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Error: Column '{column}' is not numeric"

        rows_before = len(df)

        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        elif method == "zscore":
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            df = df[z_scores < 3]
        elif method == "percentile":
            lower = df[column].quantile(0.01)
            upper = df[column].quantile(0.99)
            df = df[(df[column] >= lower) & (df[column] <= upper)]
        else:
            return f"Error: Unknown method '{method}'. Use: iqr, zscore, percentile"

        rows_after = len(df)
        removed = rows_before - rows_after

        df.to_csv(csv_path, index=False)

        success_msg = f"Removed {removed} outlier rows ({removed/rows_before*100:.2f}%) from '{column}' using '{method}' method"
        print(f"[SUCCESS] {success_msg}", flush=True)
        return success_msg

    except Exception as e:
        error_msg = f"Error removing outliers: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def encode_categorical(csv_path: str, columns: str, method: str) -> str:
    """
    Encode categorical columns as numerical values.

    Args:
        csv_path: Path to CSV file
        columns: Comma-separated list of column names
        method: One of: label_encoding, onehot_encoding
    """
    try:
        column_list = [col.strip() for col in columns.split(",")]
        print(f"[INFO] Encoding columns {column_list} using '{method}'...", flush=True)

        df = pd.read_csv(csv_path)

        for col in column_list:
            if col not in df.columns:
                print(f"[WARNING] Column '{col}' not found, skipping", flush=True)
                continue

            if method == "label_encoding":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                print(f"[INFO] Label encoded '{col}': {len(le.classes_)} unique values", flush=True)
            elif method == "onehot_encoding":
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
                print(f"[INFO] One-hot encoded '{col}': created {len(dummies.columns)} new columns", flush=True)
            else:
                return f"Error: Unknown method '{method}'. Use: label_encoding, onehot_encoding"

        df.to_csv(csv_path, index=False)

        success_msg = f"Successfully encoded columns using '{method}'"
        print(f"[SUCCESS] {success_msg}", flush=True)
        return success_msg

    except Exception as e:
        error_msg = f"Error encoding columns: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def normalize_data(csv_path: str, columns: str, method: str) -> str:
    """
    Normalize/scale numerical columns.

    Args:
        csv_path: Path to CSV file
        columns: Comma-separated list of column names
        method: One of: standardization, minmax, robust
    """
    try:
        column_list = [col.strip() for col in columns.split(",")]
        print(f"[INFO] Normalizing columns {column_list} using '{method}'...", flush=True)

        df = pd.read_csv(csv_path)

        for col in column_list:
            if col not in df.columns:
                print(f"[WARNING] Column '{col}' not found, skipping", flush=True)
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"[WARNING] Column '{col}' is not numeric, skipping", flush=True)
                continue

            if method == "standardization":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
            elif method == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
            elif method == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(df[[col]])
            else:
                return f"Error: Unknown method '{method}'. Use: standardization, minmax, robust"

        df.to_csv(csv_path, index=False)

        success_msg = f"Successfully normalized {len(column_list)} column(s) using '{method}'"
        print(f"[SUCCESS] {success_msg}", flush=True)
        return success_msg

    except Exception as e:
        error_msg = f"Error normalizing data: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


# ========== PHASE 4: SAVING & REPORTING TOOLS ==========

def save_preprocessed_dataset(csv_path: str, output_name: str) -> str:
    """
    Save the preprocessed dataset to the modified_datasets folder.

    Args:
        csv_path: Path to the current CSV file
        output_name: Name for the output file (without .csv extension)
    """
    try:
        print(f"[INFO] Saving preprocessed dataset as '{output_name}.csv'...", flush=True)

        df = pd.read_csv(csv_path)

        # Create modified_datasets directory
        project_dir = Path(__file__).parent.parent
        output_dir = project_dir / "modified_datasets"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save preprocessed data
        output_path = output_dir / f"{output_name}.csv"
        df.to_csv(output_path, index=False)

        # Create metadata file
        metadata = {
            "original_file": str(csv_path),
            "output_file": str(output_path),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "timestamp": pd.Timestamp.now().isoformat()
        }

        metadata_path = output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        # Update user_input.json with preprocessing info
        json_file_path = project_dir / "user_input.json"
        if json_file_path.exists():
            with open(json_file_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)

            user_data["preprocessed_dataset"] = {
                "output_file": str(output_path),
                "metadata_file": str(metadata_path),
                "rows": len(df),
                "columns": len(df.columns)
            }

            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, indent=4)

        success_msg = f"""
Successfully saved preprocessed dataset!
- Output file: {output_path}
- Metadata: {metadata_path}
- Rows: {len(df):,}
- Columns: {len(df.columns)}
"""
        print(f"[SUCCESS] {success_msg}", flush=True)
        return success_msg

    except Exception as e:
        error_msg = f"Error saving preprocessed dataset: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def generate_summary_report() -> str:
    """
    Generate a summary report of all preprocessing activities.
    """
    try:
        print(f"[INFO] Generating preprocessing summary report...", flush=True)

        project_dir = Path(__file__).parent.parent
        json_file_path = project_dir / "user_input.json"

        if not json_file_path.exists():
            return "Error: user_input.json not found"

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        report = []
        report.append("=" * 80)
        report.append("DATA PREPROCESSING SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"\nPROBLEM STATEMENT:")
        report.append(f"  {data.get('problem_statement', 'N/A')}")

        report.append(f"\nORIGINAL DATASET:")
        dataset_info = data.get('downloaded_dataset', {})
        report.append(f"  Source: {dataset_info.get('source', 'N/A')}")
        report.append(f"  Name: {dataset_info.get('dataset_name', 'N/A')}")

        if 'preprocessed_dataset' in data:
            prep_info = data['preprocessed_dataset']
            report.append(f"\nPREPROCESSED DATASET:")
            report.append(f"  Output File: {prep_info.get('output_file', 'N/A')}")
            report.append(f"  Rows: {prep_info.get('rows', 0):,}")
            report.append(f"  Columns: {prep_info.get('columns', 0)}")
        else:
            report.append(f"\nNo preprocessed dataset saved yet.")

        report.append(f"\n" + "=" * 80)
        report.append(f"Report generated at: {pd.Timestamp.now()}")
        report.append("=" * 80)

        result = "\n".join(report)
        print(f"[SUCCESS] Report generated!", flush=True)
        return result

    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


async def main():
    # Load user context to get problem statement and dataset
    project_dir = Path(__file__).parent.parent
    json_file_path = project_dir / "user_input.json"

    with open(json_file_path, 'r', encoding='utf-8') as f:
        user_data = json.load(f)

    problem_statement = user_data.get("problem_statement", "")
    dataset_info = user_data.get("downloaded_dataset", {})

    print(f"[INFO] Starting Data Preprocessing Agent", flush=True)
    print(f"[INFO] Problem: {problem_statement}", flush=True)
    print(f"[INFO] Dataset: {dataset_info.get('dataset_name', 'Unknown')}", flush=True)

    root_agent = Agent(
        model=LiteLlm(model="openai/gpt-4o-mini"),
        name="data_preprocessing_agent",
        description='An expert data preprocessing agent that analyzes and transforms datasets based on user requirements',
        instruction=f"""
        You are an expert data preprocessing agent. Your task is to analyze and preprocess datasets intelligently.

        CONTEXT:
        - Problem Statement: {problem_statement}
        - Dataset: {dataset_info.get('dataset_name', 'Unknown')}

        YOUR WORKFLOW:

        PHASE 1 - UNDERSTANDING:
        1. Call load_user_context() to understand the problem and dataset location
        2. Call discover_and_analyze_dataset() with the dataset path to find all CSV files
        3. Use get_column_details() to deep-dive into important columns

        PHASE 2 - ANALYSIS:
        4. Call detect_anomalies() to find data quality issues
        5. Call suggest_feature_engineering() with the problem type to get new column ideas
        6. Call identify_redundant_columns() to find columns to drop

        PHASE 3 - TRANSFORMATION (apply these based on your analysis):
        7. Use create_new_column() to add engineered features (e.g., engagement_rate, strike_rate)
        8. Use drop_columns() to remove redundant or irrelevant columns
        9. Use handle_missing_values() to fix missing data (choose appropriate strategy)
        10. Use remove_outliers() for numerical columns with extreme values
        11. Use encode_categorical() if you need to convert text to numbers
        12. Use normalize_data() if columns have different scales

        PHASE 4 - SAVING:
        13. Call save_preprocessed_dataset() to save the final preprocessed data
        14. Call generate_summary_report() to create a final report

        IMPORTANT GUIDELINES:
        - Think like a data scientist - match preprocessing to the problem statement
        - Be selective about which columns are useful for the analysis goal
        - Create meaningful features based on domain knowledge (youtube metrics, cricket stats, etc.)
        - Apply transformations iteratively and validate results
        - Focus on data quality first, then feature engineering, then normalization

        Start by calling load_user_context() to begin your analysis.
        """,
        tools=[
            # Phase 1: Understanding
            load_user_context,
            discover_and_analyze_dataset,
            get_column_details,
            # Phase 2: Analysis
            detect_anomalies,
            suggest_feature_engineering,
            identify_redundant_columns,
            # Phase 3: Transformation
            create_new_column,
            drop_columns,
            handle_missing_values,
            remove_outliers,
            encode_categorical,
            normalize_data,
            # Phase 4: Saving & Reporting
            save_preprocessed_dataset,
            generate_summary_report
        ]
    )

    app_name = 'preprocessing_agent_app'
    user_id = 'user1'

    runner = InMemoryRunner(
        agent=root_agent,
        app_name=app_name,
    )

    my_session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id
    )

    # Start the agent
    content = types.Content(
        role='user',
        parts=[types.Part.from_text(text="Please analyze the dataset and provide preprocessing recommendations.")]
    )
    print('** User says:', content.model_dump(exclude_none=True))

    async for event in runner.run_async(
        user_id=user_id,
        session_id=my_session.id,
        new_message=content,
    ):
        if event.content.parts and event.content.parts[0].text:
            print(f'** {event.author}: {event.content.parts[0].text}')

if __name__ == "__main__":
    asyncio.run(main())