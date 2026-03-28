"""
Master Orchestrator — Single Entry Point for the Prompt2ML Pipeline

Architecture:
    SequentialAgent (master_orchestrator)
    ├── Phase 1: Requirement Gatherer Agent
    │   └── Asks 7 questions, generates comprehensive report, saves to pipeline_state
    ├── Phase 2: Dataset Extractor Agent
    │   └── Searches Kaggle/HuggingFace, downloads relevant datasets
    └── Phase 3: Data Preprocessing Orchestrator
        ├── Agent 1: Dataset Analyzer
        ├── Agent 2: Preprocessing Strategist
        ├── LoopAgent (Agent 3 + Agent 4)
        └── Agent 5: Report Generator

Run:
    python master_orchestrator/agent.py
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "data_preprocessing_agent" / ".env")

from pipeline_state import load_state, save_state

MODEL = "gemini-3.1-flash-lite-preview"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def get_current_pipeline_status() -> str:
    """
    Check if the pipeline already has data from a previous run.
    Returns the current status and what data exists in pipeline_state.json.
    Use this FIRST to decide whether to skip requirement gathering.
    """
    state = load_state()
    status = {
        "status": state.get("status", "empty"),
        "has_user_goal": bool(state.get("user_goal")),
        "has_report": bool(state.get("report")),
        "has_qa_pairs": bool(state.get("qa_pairs")),
        "has_datasets": bool(state.get("downloaded_dataset")),
        "has_preprocessed": bool(state.get("preprocessed_dataset_path")),
        "user_goal_preview": state.get("user_goal", "")[:200],
    }
    return json.dumps(status, indent=2)


def save_requirement_report(
    user_goal: str,
    qa_pairs_json: str,
    report_content: str,
    report_filename: str,
) -> str:
    """
    Save the requirement gathering results to pipeline_state.json and a text file.
    Call this AFTER you have asked all questions and generated the comprehensive report.

    Args:
        user_goal: The user's original stated goal (verbatim, first message).
        qa_pairs_json: JSON array of Q&A pairs. Format: '[{"question":"...","answer":"..."},...]'
        report_content: The full comprehensive 10-section report (minimum 1500 words).
        report_filename: Filename for the report (e.g. 'youtube_trend_analysis_project_report.txt').

    Returns:
        Confirmation message.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / report_filename

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    qa_pairs = json.loads(qa_pairs_json)

    save_state({
        "user_goal": user_goal,
        "qa_pairs": qa_pairs,
        "report": report_content,
        "report_filename": report_filename,
        "status": "report_ready",
    })

    return json.dumps({
        "status": "success",
        "report_path": str(report_path),
        "qa_count": len(qa_pairs),
        "report_length": len(report_content),
    }, indent=2)


# ============================================================
# ========  AGENT DEFINITIONS  ===============================
# ============================================================

from google.adk.agents import Agent
from google.adk.tools import google_search
from data_extractor_agent.agent import dataset_extractor_agent
from data_preprocessing_agent.agent import root_agent as preprocessing_orchestrator


# --- Phase 1: Requirement Gatherer ---
requirement_gatherer_agent = Agent(
    model=MODEL,
    name="requirement_gatherer_agent",
    description="Gathers user requirements through Q&A and generates a comprehensive project report.",
    output_key="requirements_report",
    instruction="""You are an expert Machine Learning, Deep Learning and Data Science assistant with over 20+ years of experience.
You have published groundbreaking research, delivered keynotes at NeurIPS, ICML, and CVPR, and mentored hundreds of students and professionals worldwide.

WORKFLOW:
1. FIRST, call get_current_pipeline_status to check if requirements already exist.
   - If status is 'report_ready' or later AND has_user_goal is true → SKIP to step 5 (just summarize existing data).
   - If status is 'empty' or no user goal → proceed with step 2.

2. The user's FIRST message to you is their project goal. Acknowledge it.

3. Ask EXACTLY 7 follow-up questions, ONE AT A TIME. Wait for the user's response before asking the next.
   Your questions should cover:
   - Q1: Data availability and sources
   - Q2: Expected outcome / success criteria
   - Q3: Technical constraints (compute, budget, timeline)
   - Q4: Prior experience and skill level
   - Q5: Specific techniques or models they want to use
   - Q6: Deployment requirements (API, web app, notebook, etc.)
   - Q7: Any additional context or preferences

   Be concise — one question per response. Tailor each question based on prior answers.

4. After all 7 answers, generate a COMPREHENSIVE report with these 10 sections and use google search to find information and then synthesize it into actionable insights:

   SECTION 1: USER'S GOAL & PROBLEM STATEMENT
   SECTION 2: STATE OF THE ART ANALYSIS
   SECTION 3: ANALYSIS OF USER'S REQUIREMENTS & CONSTRAINTS
   SECTION 4: DATASET & DATA STRATEGY
   SECTION 5: RECOMMENDED TOOLS, FRAMEWORKS & TECHNIQUES
   SECTION 6: STEP-BY-STEP ACTION PLAN (phase-by-phase)
   SECTION 7: CHALLENGES, RISKS & MITIGATION STRATEGIES
   SECTION 8: LEARNING RESOURCES & REFERENCES
   SECTION 9: EXPERT TIPS & ADDITIONAL INSIGHTS
   SECTION 10: FINAL VERDICT & ENCOURAGEMENT

   After all these, give a final conclusion that guides the person for the project and gives him a perspective on how to approach the project, what to focus on, and how to get started. Make it inspiring and motivating.
   Use the user's specific answers to tailor every section. Do not give generic advice.

   The report must be DETAILED, SPECIFIC to the user's answers, and at minimum 1500 words.

5. Call save_requirement_report with:
   - user_goal: the user's original first message
   - qa_pairs_json: all 7 Q&A pairs as JSON array
   - report_content: the full report
   - report_filename: a descriptive filename like '<project>_project_report.txt'

6. Respond with a brief summary and tell the user the pipeline will now find and download datasets.

RULES:
- ALWAYS check pipeline status first — don't re-gather if data exists
- Ask exactly 7 questions, one at a time
- Never generate the report before collecting all 7 answers
- Tailor every section to the user's specific answers — no generic advice
- The report must be comprehensive and actionable
""",
    tools=[get_current_pipeline_status, save_requirement_report, google_search],
)

root_agent = requirement_gatherer_agent

async def run_pipeline():
    """
    Run the requirement gatherer agent with InMemorySessionService.
    Handles the 7-question conversational loop with the user.
    Usage: python master_orchestrator/agent.py
    """
    APP_NAME = "prompt2ml"
    USER_ID = "user1"
    SESSION_ID = "session_master"

    # --- Session Setup ---
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    print("\n" + "=" * 60)
    print("  PROMPT2ML — Requirement Gatherer")
    print("=" * 60)

    user_input = input("\nPlease tell me what you want to build: ")

    # --- Conversational loop: send message → print response → get next input ---
    for _ in range(20):  # Safety cap at 20 turns (7 questions + overhead)
        message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        agent_response_text = ""

        # Retry up to 3 times on Gemini server errors (500)
        for attempt in range(3):
            try:
                async for event in runner.run_async(
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    new_message=message,
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            text = getattr(part, "text", None)
                            if text:
                                author = event.author or "Agent"
                                print(f"\n[{author}]: {text}", flush=True)
                                agent_response_text += text
                break  # Success — exit retry loop
            except Exception as e:
                error_msg = str(e)
                if "500" in error_msg or "INTERNAL" in error_msg:
                    print(f"\n[WARN] Gemini server error (attempt {attempt + 1}/3). Retrying in 5s...", flush=True)
                    await asyncio.sleep(5)
                    if attempt == 2:
                        print("\n[ERROR] Gemini API failed after 3 attempts. Try running again.", flush=True)
                        return
                else:
                    print(f"\n[ERROR] {error_msg}", flush=True)
                    return

        # Check if the agent is done (report saved → status changed)
        current_status = load_state().get("status", "")
        if current_status == "report_ready":
            print("\n[PIPELINE] Report generated and saved!", flush=True)
            break

        # If the agent responded with a question, get user's answer
        if "?" in agent_response_text:
            user_input = input("\nYou: ")
        else:
            # Agent didn't ask a question and status isn't report_ready
            # Nudge it to continue
            user_input = "Please continue with the next question."

    # --- Done ---
    final_state = load_state()
    print("\n" + "=" * 60)
    print("  REQUIREMENT GATHERING COMPLETE")
    print("=" * 60)
    print(f"  Status: {final_state.get('status', 'unknown')}")
    print(f"  User Goal: {final_state.get('user_goal', 'N/A')[:100]}")
    print(f"  Q&A Pairs: {len(final_state.get('qa_pairs', []))}")
    if final_state.get("report"):
        print(f"  Report: {len(final_state['report'])} characters")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline())