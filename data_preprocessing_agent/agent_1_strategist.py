"""
Agent 1: Data Strategist Agent
High-level planning and strategy for data preprocessing
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
from typing import Dict, List
import warnings
import requests

warnings.filterwarnings('ignore')
load_dotenv()

sys.stdout.reconfigure(line_buffering=True)


def load_user_context() -> str:
    """Load and return the user's problem statement and dataset information from JSON file."""
    try:
        print("[INFO] [Agent 1] Loading user context...", flush=True)

        project_dir = Path(__file__).parent.parent
        json_file_path = project_dir / "user_input.json"

        if not json_file_path.exists():
            return "Error: user_input.json not found. Please run the requirement gathering agent first."

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        context = {
            "problem_statement": data.get("problem_statement", "Not provided"),
            "dataset_source": data.get("downloaded_dataset", {}).get("source", "Unknown"),
            "dataset_name": data.get("downloaded_dataset", {}).get("dataset_name", "Unknown"),
            "dataset_location": data.get("downloaded_dataset", {}).get("save_location", "Unknown")
        }

        result = f"""
USER CONTEXT LOADED:
====================
Problem Statement: {context['problem_statement']}
Dataset Source: {context['dataset_source']}
Dataset Name: {context['dataset_name']}
Dataset Location: {context['dataset_location']}

Now analyze the dataset structure and create a preprocessing strategy.
"""
        print(f"[SUCCESS] [Agent 1] Context loaded!", flush=True)
        return result

    except Exception as e:
        error_msg = f"Error loading user context: {str(e)}"
        print(f"[ERROR] [Agent 1] {error_msg}", flush=True)
        return error_msg


def discover_dataset_structure(dataset_path: str) -> str:
    """
    Discover all CSV files and their structure in the dataset directory.
    Returns column names, data types, row counts.
    """
    try:
        print(f"[INFO] [Agent 1] Discovering dataset structure in: {dataset_path}", flush=True)

        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            return f"Error: Dataset path does not exist: {dataset_path}"

        csv_files = list(dataset_dir.rglob("*.csv"))

        if not csv_files:
            return f"No CSV files found in {dataset_path}"

        print(f"[INFO] [Agent 1] Found {len(csv_files)} CSV file(s)", flush=True)

        results = []
        results.append(f"DATASET STRUCTURE ANALYSIS")
        results.append(f"=" * 80)
        results.append(f"Total CSV files: {len(csv_files)}\n")

        for idx, csv_file in enumerate(csv_files, 1):
            relative_path = csv_file.relative_to(dataset_dir)
            results.append(f"\n[File {idx}] {relative_path}")
            results.append("-" * 80)

            try:
                df_sample = pd.read_csv(csv_file, nrows=5)
                total_rows = sum(1 for _ in open(csv_file, encoding='utf-8', errors='ignore')) - 1

                results.append(f"Path: {csv_file}")
                results.append(f"Size: {csv_file.stat().st_size / (1024*1024):.2f} MB")
                results.append(f"Rows: {total_rows:,}")
                results.append(f"Columns: {len(df_sample.columns)}")

                results.append(f"\nColumns:")
                for col in df_sample.columns:
                    dtype = str(df_sample[col].dtype)
                    non_null = df_sample[col].notna().sum()
                    results.append(f"  - {col} ({dtype}) - {non_null}/5 non-null")

                results.append(f"\nSample (first 3 rows):")
                results.append(df_sample.head(3).to_string(index=False, max_cols=8))

            except Exception as e:
                results.append(f"Error reading file: {str(e)}")

        result_text = "\n".join(results)
        print(f"[SUCCESS] [Agent 1] Dataset structure analyzed!", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error discovering dataset: {str(e)}"
        print(f"[ERROR] [Agent 1] {error_msg}", flush=True)
        return error_msg


def analyze_data_quality(csv_path: str) -> str:
    """
    Quick data quality analysis: missing values, duplicates, outliers.
    """
    try:
        print(f"[INFO] [Agent 1] Analyzing data quality for {Path(csv_path).name}...", flush=True)

        file_size = Path(csv_path).stat().st_size / (1024 * 1024)
        if file_size > 100:
            df = pd.read_csv(csv_path, nrows=10000)
            print(f"[INFO] [Agent 1] Large file - using 10K row sample", flush=True)
        else:
            df = pd.read_csv(csv_path)

        results = []
        results.append(f"DATA QUALITY REPORT")
        results.append(f"=" * 80)
        results.append(f"File: {Path(csv_path).name}")
        results.append(f"Rows analyzed: {len(df):,}\n")

        # Missing values
        results.append(f"📊 Missing Values:")
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_cols = missing[missing > 0]

        if len(missing_cols) > 0:
            for col in missing_cols.index:
                results.append(f"  - {col}: {missing[col]:,} ({missing_pct[col]:.1f}%)")
        else:
            results.append(f"  ✅ No missing values")

        # Duplicates
        results.append(f"\n📊 Duplicate Rows:")
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results.append(f"  ⚠️ {duplicates:,} duplicates ({duplicates/len(df)*100:.1f}%)")
        else:
            results.append(f"  ✅ No duplicates")

        # Outliers in numeric columns
        results.append(f"\n📊 Outliers (IQR method):")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_found = False

        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

            if outliers > 0 and outliers / len(df) > 0.05:  # More than 5%
                results.append(f"  ⚠️ {col}: {outliers:,} outliers ({outliers/len(df)*100:.1f}%)")
                outlier_found = True

        if not outlier_found:
            results.append(f"  ✅ No significant outliers")

        result_text = "\n".join(results)
        print(f"[SUCCESS] [Agent 1] Quality analysis complete!", flush=True)
        return result_text

    except Exception as e:
        error_msg = f"Error analyzing quality: {str(e)}"
        print(f"[ERROR] [Agent 1] {error_msg}", flush=True)
        return error_msg


def web_search(query: str) -> str:
    """
    Search the web for information using Tavily API.

    Args:
        query: Search query string
    """
    try:
        print(f"[INFO] [Agent 1] Searching web for: {query}", flush=True)

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY not found in environment variables"

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            return f"No results found for: {query}"

        output = [f"SEARCH RESULTS FOR: {query}", "=" * 80]
        for i, result in enumerate(results, 1):
            output.append(f"\n[{i}] {result.get('title', 'No title')}")
            output.append(f"URL: {result.get('url', 'No URL')}")
            output.append(f"Content: {result.get('content', 'No content')[:300]}...")

        print(f"[SUCCESS] [Agent 1] Found {len(results)} results", flush=True)
        return "\n".join(output)

    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        print(f"[ERROR] [Agent 1] {error_msg}", flush=True)
        return error_msg


def delegate_to_agent_2(task_description: str, csv_path: str, code_instructions: str) -> str:
    """
    Delegate code generation task to Agent 2 (Code Generator).

    Args:
        task_description: High-level description of what needs to be done
        csv_path: Path to CSV file to process
        code_instructions: Specific instructions for code generation

    Returns:
        Delegation message that will be stored in session state
    """
    try:
        print(f"[INFO] [Agent 1] Delegating to Agent 2: {task_description}", flush=True)

        delegation = {
            "task": task_description,
            "csv_path": csv_path,
            "instructions": code_instructions,
            "from_agent": "agent_1_strategist"
        }

        result = f"""
DELEGATED TO AGENT 2 (Code Generator):
=======================================
Task: {task_description}
CSV Path: {csv_path}

Instructions:
{code_instructions}

Agent 2 will generate and execute Python code to complete this task.
"""
        print(f"[SUCCESS] [Agent 1] Task delegated to Agent 2", flush=True)
        return result

    except Exception as e:
        error_msg = f"Error delegating to Agent 2: {str(e)}"
        print(f"[ERROR] [Agent 1] {error_msg}", flush=True)
        return error_msg


# Agent 1 will be created in the orchestrator