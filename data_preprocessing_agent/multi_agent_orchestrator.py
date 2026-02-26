"""
Multi-Agent Orchestrator
Coordinates the 3-agent preprocessing pipeline:
- Agent 1: Data Strategist (Planning & Delegation)
- Agent 2: Code Generator (Code Execution in Docker)
- Agent 3: Quality Assurance (Validation & Feedback)
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

# Import agent tools
from agent_1_strategist import (
    load_user_context,
    discover_dataset_structure,
    analyze_data_quality,
    web_search,
    delegate_to_agent_2
)

from agent_2_code_generator import (
    initialize_docker_executor,
    execute_preprocessing_code,
    generate_feature_engineering_code,
    validate_generated_code,
    get_dataset_info,
    request_agent_3_validation
)

from agent_3_qa import (
    validate_data_quality,
    check_data_leakage,
    compare_distributions,
    suggest_improvements,
    provide_feedback_to_agent_2,
    generate_final_report
)

warnings.filterwarnings('ignore')
load_dotenv()

sys.stdout.reconfigure(line_buffering=True)


async def main():
    """Main orchestrator for multi-agent preprocessing pipeline"""

    print("\n" + "="*80)
    print("🚀 MULTI-AGENT PREPROCESSING PIPELINE")
    print("="*80)
    print("\n💡 This pipeline uses 3 specialized agents:")
    print("   1. Data Strategist - Analyzes data and creates strategy")
    print("   2. Code Generator - Writes and executes custom Python code in Docker")
    print("   3. Quality Assurance - Validates results and provides feedback")
    print("\n" + "="*80 + "\n")

    # Load user context
    project_dir = Path(__file__).parent.parent
    json_file_path = project_dir / "user_input.json"

    if not json_file_path.exists():
        print("❌ ERROR: user_input.json not found!")
        print("Please run the requirement gathering agent first.")
        return

    with open(json_file_path, 'r', encoding='utf-8') as f:
        user_data = json.load(f)

    problem_statement = user_data.get("problem_statement", "")
    dataset_info = user_data.get("downloaded_dataset", {})
    dataset_path = dataset_info.get("save_location", "")

    print(f"📋 Problem: {problem_statement[:100]}...")
    print(f"📊 Dataset: {dataset_info.get('dataset_name', 'Unknown')}")
    print(f"📁 Location: {dataset_path}\n")

    # ========================
    # AGENT 1: DATA STRATEGIST
    # ========================

    print("\n" + "="*80)
    print("🎯 STARTING AGENT 1: DATA STRATEGIST")
    print("="*80 + "\n")

    agent_1 = Agent(
        model=LiteLlm(model="openai/gpt-4o-mini"),
        name="data_strategist",
        description="High-level data preprocessing strategist and planner",
        instruction=f"""
You are Agent 1: Data Strategist, an expert in ML/DL data preprocessing strategy.

PROBLEM STATEMENT:
{problem_statement}

YOUR ROLE:
You are the strategic planner. You DON'T execute preprocessing - you analyze, plan, and delegate.

YOUR WORKFLOW:

PHASE 1 - UNDERSTANDING (Required):
1. Call load_user_context() to understand the full problem
2. Call discover_dataset_structure() with the dataset path to see all CSV files
3. Call analyze_data_quality() on the main CSV file to understand issues

PHASE 2 - RESEARCH (If needed):
4. Use web_search() to research domain-specific preprocessing techniques
   Example: "youtube engagement rate feature engineering"
   Example: "cricket statistics feature engineering techniques"

PHASE 3 - STRATEGY (Required):
5. Based on your analysis, create a preprocessing strategy:
   - What features need to be created? (engagement_rate, time features, etc.)
   - What columns should be dropped?
   - What missing values need handling?
   - What transformations are needed?

PHASE 4 - DELEGATION (Required):
6. Call delegate_to_agent_2() with clear instructions for code generation
   Be specific: "Create engagement_rate = likes / views, drop columns X,Y,Z"

IMPORTANT:
- You are the PLANNER, not the EXECUTOR
- Be strategic and think about the ML/DL problem domain
- Delegate specific, actionable tasks to Agent 2
- Use web search to research best practices for the problem domain

Start by calling load_user_context().
""",
        tools=[
            load_user_context,
            discover_dataset_structure,
            analyze_data_quality,
            web_search,
            delegate_to_agent_2
        ],
        output_key="agent_1_strategy"
    )

    # ========================
    # AGENT 2: CODE GENERATOR
    # ========================

    agent_2 = Agent(
        model=LiteLlm(model="openai/gpt-4o-mini"),
        name="code_generator",
        description="Python code generator with Docker execution capabilities",
        instruction="""
You are Agent 2: Code Generator, an expert at writing Python data preprocessing code.

YOUR ROLE:
You receive instructions from Agent 1 and write custom Python code to execute them.

CONTEXT FROM AGENT 1:
{agent_1_strategy}

YOUR WORKFLOW:

STEP 1 - INITIALIZE:
1. Call initialize_docker_executor() to set up Docker environment

STEP 2 - UNDERSTAND DATA:
2. Call get_dataset_info() to see column names and data types

STEP 3 - GENERATE CODE:
3. Write Python code based on Agent 1's instructions
   Important: Use /workspace/data/ paths (Docker container paths)

   Example code structure:
   ```python
   import pandas as pd
   import numpy as np

   # Read data (use /workspace/data/ prefix!)
   df = pd.read_csv('/workspace/data/filename.csv')

   # Your preprocessing logic here
   df['new_feature'] = df['col1'] / df['col2']

   # Save back
   df.to_csv('/workspace/data/filename.csv', index=False)

   print(f"Shape: {{df.shape}}")
   print(f"Columns: {{df.columns.tolist()}}")
   ```

STEP 4 - VALIDATE CODE:
4. Call validate_generated_code() to check for issues

STEP 5 - EXECUTE:
5. Call execute_preprocessing_code() with your code and CSV path
   This runs in an isolated Docker container

STEP 6 - REQUEST VALIDATION:
6. Call request_agent_3_validation() to ask Agent 3 to validate results

IMPORTANT RULES:
- Always use /workspace/data/ paths in your code (Docker requirement)
- Test your logic before execution
- Handle edge cases (division by zero, missing values, etc.)
- Print informative messages about what you're doing
- If execution fails, fix the code and try again

Start by calling initialize_docker_executor().
""",
        tools=[
            initialize_docker_executor,
            execute_preprocessing_code,
            generate_feature_engineering_code,
            validate_generated_code,
            get_dataset_info,
            request_agent_3_validation
        ],
        output_key="agent_2_execution"
    )

    # ========================
    # AGENT 3: QUALITY ASSURANCE
    # ========================

    agent_3 = Agent(
        model=LiteLlm(model="openai/gpt-4o-mini"),
        name="quality_assurance",
        description="Quality validation and feedback specialist",
        instruction="""
You are Agent 3: Quality Assurance, an expert at validating data preprocessing results.

YOUR ROLE:
You validate the preprocessing performed by Agent 2 and provide feedback.

CONTEXT FROM AGENT 2:
{agent_2_execution}

YOUR WORKFLOW:

STEP 1 - VALIDATE QUALITY:
1. Call validate_data_quality() on the processed CSV file
   This checks for:
   - NaN/Inf values
   - Data type issues
   - Duplicate rows
   - Value range problems
   - Data integrity

STEP 2 - CHECK FOR LEAKAGE:
2. Call check_data_leakage() to ensure no information leakage

STEP 3 - SUGGEST IMPROVEMENTS:
3. Call suggest_improvements() to recommend further optimizations

STEP 4 - PROVIDE FEEDBACK:
4. Call provide_feedback_to_agent_2() with validation results
   - If APPROVED: Preprocessing is good
   - If APPROVED_WITH_WARNINGS: Minor issues but acceptable
   - If REJECTED: Critical issues - Agent 2 needs to fix code

STEP 5 - FINAL REPORT:
5. If approved, call generate_final_report() to create comprehensive summary

YOUR STANDARDS:
- Be thorough but fair
- Distinguish between critical issues (❌) and warnings (⚠️)
- Provide actionable feedback
- If there are critical issues, clearly explain what needs fixing
- Think about downstream ML/DL impact

Start by calling validate_data_quality() on the processed dataset.
""",
        tools=[
            validate_data_quality,
            check_data_leakage,
            suggest_improvements,
            provide_feedback_to_agent_2,
            generate_final_report
        ],
        output_key="agent_3_validation"
    )

    # ========================
    # EXECUTION PIPELINE
    # ========================

    app_name = 'multi_agent_preprocessing'
    user_id = 'user1'

    # Create runner
    runner = InMemoryRunner(
        agent=agent_1,  # Start with Agent 1
        app_name=app_name,
    )

    # Create session
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id
    )

    print("\n🤖 Starting Agent 1 (Data Strategist)...\n")

    # Start Agent 1
    content = types.Content(
        role='user',
        parts=[types.Part.from_text(text="Please analyze the dataset and create a preprocessing strategy.")]
    )

    agent_1_response = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.content.parts and event.content.parts[0].text:
            agent_1_response = event.content.parts[0].text
            print(f"[Agent 1]: {event.content.parts[0].text}\n")

    print("\n" + "="*80)
    print("✅ AGENT 1 COMPLETE")
    print("="*80 + "\n")

    # Now run Agent 2 with same session (to access agent_1_strategy from output_key)
    print("\n" + "="*80)
    print("💻 STARTING AGENT 2: CODE GENERATOR")
    print("="*80 + "\n")

    # Update runner to use Agent 2
    runner_2 = InMemoryRunner(
        agent=agent_2,
        app_name=app_name,
    )

    # Use the same session to access agent_1_strategy
    content_2 = types.Content(
        role='user',
        parts=[types.Part.from_text(text="Please generate and execute preprocessing code based on Agent 1's strategy.")]
    )

    agent_2_response = ""
    async for event in runner_2.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content_2,
    ):
        if event.content.parts and event.content.parts[0].text:
            agent_2_response = event.content.parts[0].text
            print(f"[Agent 2]: {event.content.parts[0].text}\n")

    print("\n" + "="*80)
    print("✅ AGENT 2 COMPLETE")
    print("="*80 + "\n")

    # Now run Agent 3 for validation
    print("\n" + "="*80)
    print("✅ STARTING AGENT 3: QUALITY ASSURANCE")
    print("="*80 + "\n")

    runner_3 = InMemoryRunner(
        agent=agent_3,
        app_name=app_name,
    )

    content_3 = types.Content(
        role='user',
        parts=[types.Part.from_text(text="Please validate the preprocessing results from Agent 2.")]
    )

    agent_3_response = ""
    async for event in runner_3.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content_3,
    ):
        if event.content.parts and event.content.parts[0].text:
            agent_3_response = event.content.parts[0].text
            print(f"[Agent 3]: {event.content.parts[0].text}\n")

    print("\n" + "="*80)
    print("✅ AGENT 3 COMPLETE")
    print("="*80 + "\n")

    print("\n" + "="*80)
    print("🎉 MULTI-AGENT PREPROCESSING PIPELINE COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())