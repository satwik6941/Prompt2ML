"""
Agent 2: Code Generator Agent
Generates and executes custom Python code for data preprocessing
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
import warnings
from docker_executor import DockerCodeExecutor

warnings.filterwarnings('ignore')
load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

# Global Docker executor instance
DOCKER_EXECUTOR = None


def initialize_docker_executor() -> str:
    """Initialize Docker executor and build image if needed."""
    global DOCKER_EXECUTOR

    try:
        print("[INFO] [Agent 2] Initializing Docker executor...", flush=True)

        DOCKER_EXECUTOR = DockerCodeExecutor()

        # Check if image exists, build if not
        if not DOCKER_EXECUTOR.check_image_exists():
            print("[INFO] [Agent 2] Docker image not found. Building...", flush=True)
            if DOCKER_EXECUTOR.build_image():
                return "✅ Docker executor initialized and image built successfully!"
            else:
                return "⚠️ Docker executor initialized but image build failed. Will retry on first execution."
        else:
            print("[SUCCESS] [Agent 2] Docker image already exists!", flush=True)
            return "✅ Docker executor initialized. Image already exists."

    except Exception as e:
        error_msg = f"Error initializing Docker: {str(e)}"
        print(f"[ERROR] [Agent 2] {error_msg}", flush=True)
        return f"❌ {error_msg}. Make sure Docker Desktop is running!"


def execute_preprocessing_code(python_code: str, csv_path: str) -> str:
    """
    Execute custom Python preprocessing code in Docker container.

    Args:
        python_code: Python code to execute (must use /workspace/data/ paths)
        csv_path: Path to the CSV file to process

    Returns:
        Execution result with stdout, stderr, and success status
    """
    global DOCKER_EXECUTOR

    try:
        print(f"[INFO] [Agent 2] Executing preprocessing code...", flush=True)

        if DOCKER_EXECUTOR is None:
            init_result = initialize_docker_executor()
            if "Error" in init_result or "failed" in init_result:
                return init_result

        # Get dataset directory (parent of CSV file)
        dataset_dir = str(Path(csv_path).parent.absolute())

        # Execute code in Docker
        result = DOCKER_EXECUTOR.execute_code(
            code=python_code,
            dataset_dir=dataset_dir,
            timeout=120  # 2 minute timeout
        )

        if result['success']:
            output = f"""
✅ CODE EXECUTION SUCCESSFUL!
==============================
Output:
{result['stdout']}

The code has been executed successfully in an isolated Docker container.
"""
            print(f"[SUCCESS] [Agent 2] Code executed successfully!", flush=True)
            return output
        else:
            error_output = f"""
❌ CODE EXECUTION FAILED!
=========================
Error: {result.get('error', 'Unknown error')}

Stdout:
{result.get('stdout', 'No output')}

Stderr:
{result.get('stderr', 'No errors')}

Please fix the code and try again.
"""
            print(f"[ERROR] [Agent 2] Code execution failed!", flush=True)
            return error_output

    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        print(f"[ERROR] [Agent 2] {error_msg}", flush=True)
        return f"❌ {error_msg}"


def generate_feature_engineering_code(csv_path: str, feature_description: str) -> str:
    """
    Generate Python code for feature engineering based on description.

    Args:
        csv_path: Path to CSV file
        feature_description: Description of features to create (e.g., "engagement_rate from likes/views")

    Returns:
        Generated Python code as a string
    """
    try:
        print(f"[INFO] [Agent 2] Generating feature engineering code...", flush=True)

        # Get filename for Docker path
        filename = Path(csv_path).name

        code_template = f'''
import pandas as pd
import numpy as np

# Read dataset (in Docker, files are at /workspace/data/)
df = pd.read_csv('/workspace/data/{filename}')

print(f"Original shape: {{df.shape}}")
print(f"Original columns: {{df.columns.tolist()}}")

# Feature engineering based on: {feature_description}
# TODO: Agent should generate specific feature engineering code here

print(f"\\nNew shape: {{df.shape}}")
print(f"New columns: {{df.columns.tolist()}}")

# Save processed dataset
df.to_csv('/workspace/data/{filename}', index=False)
print(f"\\n✅ Dataset saved successfully!")
'''

        print(f"[SUCCESS] [Agent 2] Code template generated!", flush=True)
        return code_template

    except Exception as e:
        error_msg = f"Error generating code: {str(e)}"
        print(f"[ERROR] [Agent 2] {error_msg}", flush=True)
        return f"# Error: {error_msg}"


def validate_generated_code(python_code: str) -> str:
    """
    Validate Python code for security and correctness before execution.

    Args:
        python_code: Python code to validate

    Returns:
        Validation result message
    """
    try:
        print(f"[INFO] [Agent 2] Validating generated code...", flush=True)

        issues = []

        # Check for dangerous operations
        dangerous_keywords = ['os.system', 'subprocess', 'eval(', 'exec(', '__import__', 'open(']
        for keyword in dangerous_keywords:
            if keyword in python_code:
                issues.append(f"⚠️ Potentially dangerous operation: {keyword}")

        # Check for required imports
        if 'import pandas' not in python_code and 'pd.' in python_code:
            issues.append("⚠️ Using 'pd' but pandas not imported")

        # Check for file path usage
        if '/workspace/data/' not in python_code:
            issues.append("⚠️ Code should use /workspace/data/ paths for Docker execution")

        if issues:
            result = f"CODE VALIDATION WARNINGS:\n" + "\n".join(issues)
        else:
            result = "✅ Code validation passed! No issues found."

        print(f"[SUCCESS] [Agent 2] Code validated!", flush=True)
        return result

    except Exception as e:
        error_msg = f"Error validating code: {str(e)}"
        print(f"[ERROR] [Agent 2] {error_msg}", flush=True)
        return error_msg


def get_dataset_info(csv_path: str) -> str:
    """
    Get quick information about a dataset to help with code generation.

    Args:
        csv_path: Path to CSV file

    Returns:
        Dataset information (columns, types, sample data)
    """
    try:
        print(f"[INFO] [Agent 2] Getting dataset info for code generation...", flush=True)

        import pandas as pd

        # Read small sample
        df = pd.read_csv(csv_path, nrows=5)

        info = f"""
DATASET INFO (for code generation):
====================================
File: {Path(csv_path).name}
Columns: {df.columns.tolist()}
Data types: {df.dtypes.to_dict()}

Sample data (first 3 rows):
{df.head(3).to_string(index=False)}

Use this information to generate appropriate preprocessing code.
"""
        print(f"[SUCCESS] [Agent 2] Dataset info retrieved!", flush=True)
        return info

    except Exception as e:
        error_msg = f"Error getting dataset info: {str(e)}"
        print(f"[ERROR] [Agent 2] {error_msg}", flush=True)
        return error_msg


def request_agent_3_validation(csv_path: str, operation_description: str) -> str:
    """
    Request Agent 3 (QA) to validate the preprocessing results.

    Args:
        csv_path: Path to processed CSV file
        operation_description: Description of what operations were performed

    Returns:
        Message requesting validation from Agent 3
    """
    try:
        print(f"[INFO] [Agent 2] Requesting validation from Agent 3...", flush=True)

        message = f"""
VALIDATION REQUEST TO AGENT 3:
===============================
Please validate the preprocessing results for:
File: {csv_path}
Operations performed: {operation_description}

Agent 3, please check:
1. Data quality after preprocessing
2. No invalid values introduced
3. Column types are correct
4. No data leakage issues
5. Distribution changes are reasonable
"""
        print(f"[SUCCESS] [Agent 2] Validation request prepared for Agent 3", flush=True)
        return message

    except Exception as e:
        error_msg = f"Error requesting validation: {str(e)}"
        print(f"[ERROR] [Agent 2] {error_msg}", flush=True)
        return error_msg


# Agent 2 will be created in the orchestrator