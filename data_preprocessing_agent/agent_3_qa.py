"""
Agent 3: Quality Assurance Agent
Validates preprocessing results and provides feedback
"""
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
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

sys.stdout.reconfigure(line_buffering=True)


def validate_data_quality(csv_path: str) -> str:
    """
    Comprehensive data quality validation after preprocessing.

    Args:
        csv_path: Path to preprocessed CSV file

    Returns:
        Detailed validation report
    """
    try:
        print(f"[INFO] [Agent 3] Validating data quality for {Path(csv_path).name}...", flush=True)

        df = pd.read_csv(csv_path)

        results = []
        results.append(f"QUALITY ASSURANCE REPORT")
        results.append(f"=" * 80)
        results.append(f"File: {Path(csv_path).name}")
        results.append(f"Rows: {len(df):,}")
        results.append(f"Columns: {len(df.columns)}")

        issues_found = 0
        warnings_found = 0

        # 1. Check for NaN/Inf values
        results.append(f"\n✓ CHECK 1: Invalid Values")
        results.append("-" * 80)

        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            results.append(f"⚠️ WARNING: Found NaN values in columns:")
            for col in nan_cols:
                nan_count = df[col].isna().sum()
                nan_pct = (nan_count / len(df)) * 100
                results.append(f"  - {col}: {nan_count:,} ({nan_pct:.2f}%)")
                warnings_found += 1
        else:
            results.append(f"✅ PASS: No NaN values found")

        # Check for infinity
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_found = False
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                results.append(f"❌ ERROR: Infinite values found in column '{col}'")
                issues_found += 1
                inf_found = True

        if not inf_found and len(numeric_cols) > 0:
            results.append(f"✅ PASS: No infinite values")

        # 2. Check data types
        results.append(f"\n✓ CHECK 2: Data Types")
        results.append("-" * 80)

        type_issues = []
        for col in df.columns:
            dtype = df[col].dtype

            # Check for object columns that might should be numeric
            if dtype == 'object':
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    type_issues.append(f"⚠️ Column '{col}' is object but contains numeric values")
                except:
                    pass

        if type_issues:
            for issue in type_issues:
                results.append(issue)
                warnings_found += 1
        else:
            results.append(f"✅ PASS: Data types are appropriate")

        # 3. Check for duplicate rows
        results.append(f"\n✓ CHECK 3: Duplicate Rows")
        results.append("-" * 80)

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            if dup_pct > 10:
                results.append(f"❌ ERROR: {duplicates:,} duplicate rows ({dup_pct:.2f}%) - Too many!")
                issues_found += 1
            else:
                results.append(f"⚠️ WARNING: {duplicates:,} duplicate rows ({dup_pct:.2f}%)")
                warnings_found += 1
        else:
            results.append(f"✅ PASS: No duplicate rows")

        # 4. Check value ranges for numeric columns
        results.append(f"\n✓ CHECK 4: Value Ranges")
        results.append("-" * 80)

        range_issues = []
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()

            # Check for suspicious ranges
            if min_val == max_val:
                range_issues.append(f"⚠️ Column '{col}' has constant value: {min_val}")

            # Check for negative values in columns that shouldn't have them
            if 'count' in col.lower() or 'rate' in col.lower() or 'ratio' in col.lower():
                if min_val < 0:
                    range_issues.append(f"❌ Column '{col}' has negative values (min: {min_val})")
                    issues_found += 1

        if range_issues:
            for issue in range_issues:
                results.append(issue)
                if '❌' in issue:
                    issues_found += 1
                else:
                    warnings_found += 1
        else:
            results.append(f"✅ PASS: Value ranges look reasonable")

        # 5. Check for empty DataFrame
        results.append(f"\n✓ CHECK 5: Data Integrity")
        results.append("-" * 80)

        if len(df) == 0:
            results.append(f"❌ CRITICAL: DataFrame is empty!")
            issues_found += 1
        elif len(df.columns) == 0:
            results.append(f"❌ CRITICAL: No columns in DataFrame!")
            issues_found += 1
        else:
            results.append(f"✅ PASS: DataFrame has data ({len(df):,} rows, {len(df.columns)} columns)")

        # Summary
        results.append(f"\n" + "=" * 80)
        results.append(f"VALIDATION SUMMARY:")
        results.append(f"  Critical Issues: {issues_found}")
        results.append(f"  Warnings: {warnings_found}")

        if issues_found == 0 and warnings_found == 0:
            results.append(f"\n✅ ALL CHECKS PASSED! Data quality is excellent.")
            verdict = "APPROVED"
        elif issues_found == 0:
            results.append(f"\n✅ APPROVED with {warnings_found} warning(s). Data is usable but could be improved.")
            verdict = "APPROVED_WITH_WARNINGS"
        else:
            results.append(f"\n❌ FAILED! Found {issues_found} critical issue(s). Please fix before proceeding.")
            verdict = "REJECTED"

        results.append(f"\nVerdict: {verdict}")
        results.append("=" * 80)

        result_text = "\n".join(results)
        print(f"[SUCCESS] [Agent 3] Validation complete! Verdict: {verdict}", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error validating data quality: {str(e)}"
        print(f"[ERROR] [Agent 3] {error_msg}", flush=True)
        return f"❌ Validation failed: {error_msg}"


def check_data_leakage(csv_path: str, target_column: str = None) -> str:
    """
    Check for potential data leakage issues.

    Args:
        csv_path: Path to CSV file
        target_column: Target/label column name (if applicable)

    Returns:
        Data leakage analysis report
    """
    try:
        print(f"[INFO] [Agent 3] Checking for data leakage...", flush=True)

        df = pd.read_csv(csv_path)

        results = []
        results.append(f"DATA LEAKAGE ANALYSIS")
        results.append(f"=" * 80)

        issues = []

        # Check for ID columns that might leak information
        suspicious_cols = []
        for col in df.columns:
            col_lower = col.lower()

            # Check for exact duplicates of target
            if target_column and col != target_column:
                correlation = df[col].corr(df[target_column]) if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_column]) else None
                if correlation is not None and abs(correlation) > 0.99:
                    suspicious_cols.append(f"⚠️ Column '{col}' has very high correlation ({correlation:.3f}) with target '{target_column}'")

            # Check for future information
            if 'future' in col_lower or 'next' in col_lower or 'forecast' in col_lower:
                suspicious_cols.append(f"⚠️ Column '{col}' might contain future information")

        if suspicious_cols:
            results.append("\n⚠️ POTENTIAL LEAKAGE ISSUES:")
            for issue in suspicious_cols:
                results.append(f"  {issue}")
        else:
            results.append("\n✅ No obvious data leakage detected")

        result_text = "\n".join(results)
        print(f"[SUCCESS] [Agent 3] Data leakage check complete!", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error checking data leakage: {str(e)}"
        print(f"[ERROR] [Agent 3] {error_msg}", flush=True)
        return error_msg


def compare_distributions(original_csv_path: str, processed_csv_path: str) -> str:
    """
    Compare distributions between original and processed datasets.

    Args:
        original_csv_path: Path to original CSV
        processed_csv_path: Path to processed CSV

    Returns:
        Distribution comparison report
    """
    try:
        print(f"[INFO] [Agent 3] Comparing distributions...", flush=True)

        df_original = pd.read_csv(original_csv_path, nrows=10000)
        df_processed = pd.read_csv(processed_csv_path, nrows=10000)

        results = []
        results.append(f"DISTRIBUTION COMPARISON")
        results.append(f"=" * 80)
        results.append(f"Original rows: {len(df_original):,}")
        results.append(f"Processed rows: {len(df_processed):,}")
        results.append(f"Row difference: {len(df_processed) - len(df_original):+,}\n")

        # Compare common columns
        common_cols = set(df_original.columns) & set(df_processed.columns)
        numeric_common = [col for col in common_cols if pd.api.types.is_numeric_dtype(df_original[col])]

        if numeric_common:
            results.append(f"Numeric Column Statistics:")
            for col in numeric_common[:5]:  # First 5 columns
                orig_mean = df_original[col].mean()
                proc_mean = df_processed[col].mean()
                mean_change = ((proc_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0

                results.append(f"\n  {col}:")
                results.append(f"    Original mean: {orig_mean:.2f}")
                results.append(f"    Processed mean: {proc_mean:.2f}")
                results.append(f"    Change: {mean_change:+.2f}%")

        result_text = "\n".join(results)
        print(f"[SUCCESS] [Agent 3] Distribution comparison complete!", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error comparing distributions: {str(e)}"
        print(f"[ERROR] [Agent 3] {error_msg}", flush=True)
        return error_msg


def suggest_improvements(csv_path: str) -> str:
    """
    Suggest improvements for the preprocessed dataset.

    Args:
        csv_path: Path to preprocessed CSV

    Returns:
        Improvement suggestions
    """
    try:
        print(f"[INFO] [Agent 3] Generating improvement suggestions...", flush=True)

        df = pd.read_csv(csv_path, nrows=10000)

        suggestions = []
        suggestions.append(f"IMPROVEMENT SUGGESTIONS")
        suggestions.append(f"=" * 80)

        # Check for high cardinality categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                suggestions.append(f"💡 Column '{col}' has high cardinality ({df[col].nunique()} unique values)")
                suggestions.append(f"   Consider: Target encoding, hashing, or grouping rare categories\n")

        # Check for imbalanced features
        for col in df.select_dtypes(include=[np.number]).columns:
            skewness = df[col].skew()
            if abs(skewness) > 2:
                suggestions.append(f"💡 Column '{col}' is highly skewed (skewness: {skewness:.2f})")
                suggestions.append(f"   Consider: Log transformation, Box-Cox, or quantile transformation\n")

        # Check for missing values
        missing_cols = df.columns[df.isna().any()]
        if len(missing_cols) > 0:
            suggestions.append(f"💡 {len(missing_cols)} column(s) still have missing values")
            suggestions.append(f"   Consider: More sophisticated imputation methods\n")

        if len(suggestions) == 2:  # Only header
            suggestions.append("✅ No major improvements needed. Dataset looks good!")

        result_text = "\n".join(suggestions)
        print(f"[SUCCESS] [Agent 3] Suggestions generated!", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error suggesting improvements: {str(e)}"
        print(f"[ERROR] [Agent 3] {error_msg}", flush=True)
        return error_msg


def provide_feedback_to_agent_2(validation_result: str, suggestions: str) -> str:
    """
    Provide structured feedback to Agent 2 based on validation.

    Args:
        validation_result: Results from validate_data_quality
        suggestions: Improvement suggestions

    Returns:
        Formatted feedback for Agent 2
    """
    try:
        print(f"[INFO] [Agent 3] Providing feedback to Agent 2...", flush=True)

        feedback = f"""
FEEDBACK TO AGENT 2:
====================

VALIDATION RESULTS:
{validation_result}

IMPROVEMENT SUGGESTIONS:
{suggestions}

If there are critical issues (❌), Agent 2 should regenerate code to fix them.
If there are only warnings (⚠️), the preprocessing is acceptable but could be improved.
"""
        print(f"[SUCCESS] [Agent 3] Feedback prepared for Agent 2", flush=True)
        return feedback

    except Exception as e:
        error_msg = f"Error providing feedback: {str(e)}"
        print(f"[ERROR] [Agent 3] {error_msg}", flush=True)
        return error_msg


def generate_final_report(csv_path: str) -> str:
    """
    Generate comprehensive final report for the preprocessing pipeline.

    Args:
        csv_path: Path to final preprocessed dataset

    Returns:
        Final report
    """
    try:
        print(f"[INFO] [Agent 3] Generating final report...", flush=True)

        df = pd.read_csv(csv_path)

        project_dir = Path(__file__).parent.parent
        json_file_path = project_dir / "user_input.json"

        with open(json_file_path, 'r', encoding='utf-8') as f:
            user_data = json.load(f)

        report = []
        report.append("=" * 80)
        report.append("PREPROCESSING PIPELINE - FINAL REPORT")
        report.append("=" * 80)

        report.append(f"\n📋 PROBLEM STATEMENT:")
        report.append(f"  {user_data.get('problem_statement', 'N/A')}")

        report.append(f"\n📊 FINAL DATASET SUMMARY:")
        report.append(f"  File: {Path(csv_path).name}")
        report.append(f"  Rows: {len(df):,}")
        report.append(f"  Columns: {len(df.columns)}")
        report.append(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        report.append(f"\n📈 COLUMN BREAKDOWN:")
        report.append(f"  Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
        report.append(f"  Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
        report.append(f"  Datetime columns: {len(df.select_dtypes(include=['datetime64']).columns)}")

        report.append(f"\n✅ QUALITY STATUS:")
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        report.append(f"  Missing values: {missing_pct:.2f}%")
        report.append(f"  Duplicate rows: {df.duplicated().sum():,}")

        report.append(f"\n" + "=" * 80)
        report.append(f"✅ PREPROCESSING COMPLETE!")
        report.append(f"Dataset is ready for model training.")
        report.append("=" * 80)

        result = "\n".join(report)
        print(f"[SUCCESS] [Agent 3] Final report generated!", flush=True)
        return result

    except Exception as e:
        error_msg = f"Error generating final report: {str(e)}"
        print(f"[ERROR] [Agent 3] {error_msg}", flush=True)
        return error_msg


# Agent 3 will be created in the orchestrator