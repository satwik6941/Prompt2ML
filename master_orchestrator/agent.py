"""
Master Orchestrator — Single Entry Point for the Prompt2ML Pipeline

Phase 1: Requirement Gatherer Agent (interactive — asks 7 questions)
Phase 2: Dataset Extractor Agent (autonomous — searches & downloads datasets)

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


# ============================================================
# CONSTANTS
# ============================================================

MODEL = "gemini-2.5-flash"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# ========  PHASE 1: REQUIREMENT GATHERER TOOLS  =============
# ============================================================

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

# Import Phase 2 agent
from data_extractor_agent.agent import dataset_extractor_agent
from data_preprocessing_agent.agent import data_preprocessing_agent

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

4. After all 7 answers, generate a COMPREHENSIVE report with these 10 sections:

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

   After all these, give a final conclusion that guides the person for the project and gives him
   a perspective on how to approach the project, what to focus on, and how to get started.
   Make it inspiring and motivating.

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
    tools=[get_current_pipeline_status, save_requirement_report],
)

# root_agent for ADK compatibility (adk web / adk run from this folder)
root_agent = requirement_gatherer_agent


# ============================================================
# ==================  STANDALONE RUNNER  =====================
# ============================================================

async def run_agent_turn(runner, user_id, session_id, user_text):
    """
    Send a message to the agent and collect its full text response.
    Retries up to 3 times on Gemini 500 errors.
    Returns the concatenated agent response text.
    """
    message = types.Content(
        role="user",
        parts=[types.Part(text=user_text)],
    )

    agent_response = ""

    for attempt in range(3):
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            author = event.author or "Agent"
                            print(f"\n[{author}]: {text}", flush=True)
                            agent_response += text
            return agent_response
        except Exception as e:
            error_msg = str(e)
            if "500" in error_msg or "INTERNAL" in error_msg:
                print(f"\n[WARN] Gemini server error (attempt {attempt + 1}/3). Retrying in 5s...", flush=True)
                await asyncio.sleep(5)
                if attempt == 2:
                    print("\n[ERROR] Gemini API failed after 3 attempts.", flush=True)
                    return None
            else:
                print(f"\n[ERROR] {error_msg}", flush=True)
                return None

    return agent_response


async def run_pipeline():
    """
    Run the full pipeline:
      Phase 1: Requirement Gatherer (interactive Q&A with user)
      Phase 2: Dataset Extractor (autonomous — reads report, downloads datasets)
    """
    APP_NAME = "prompt2ml"
    USER_ID = "user1"

    session_service = InMemorySessionService()

    # ==============================================================
    # PHASE 1: Requirement Gatherer (interactive)
    # ==============================================================

    SESSION_PHASE1 = "session_phase1"
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_PHASE1
    )

    phase1_runner = Runner(
        app_name=APP_NAME,
        agent=requirement_gatherer_agent,
        session_service=session_service,
    )

    print("\n" + "=" * 60)
    print("  PROMPT2ML — Phase 1: Requirement Gathering")
    print("=" * 60)

    # Check if Phase 1 already done
    state = load_state()
    if state.get("status") == "report_ready" and state.get("user_goal"):
        print(f"\n[PIPELINE] Report already exists for: {state['user_goal'][:80]}...")
        print("[PIPELINE] Skipping Phase 1, moving to Phase 2.\n")
    else:
        # Get the user's project goal
        user_input = input("\nPlease tell me what you want to build: ")

        # Conversational loop: 7 questions + report generation
        for _ in range(20):  # Safety cap
            response = await run_agent_turn(
                phase1_runner, USER_ID, SESSION_PHASE1, user_input
            )

            if response is None:
                print("\n[ERROR] Agent failed to respond. Exiting.")
                return

            # Check if report was saved
            if load_state().get("status") == "report_ready":
                print("\n[PIPELINE] Phase 1 complete — report saved!")
                break

            # Agent asked a question → get user's answer
            if "?" in response:
                user_input = input("\nYou: ")
            else:
                user_input = "Please continue with the next question."

    # ==============================================================
    # PHASE 2: Dataset Extractor (autonomous)
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PROMPT2ML — Phase 2: Dataset Extraction")
    print("=" * 60)

    # Check if datasets already downloaded
    state = load_state()
    if state.get("status") == "dataset_ready" and state.get("downloaded_dataset"):
        print(f"\n[PIPELINE] Datasets already downloaded. Skipping Phase 2.")
    else:
        SESSION_PHASE2 = "session_phase2"
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_PHASE2
        )

        phase2_runner = Runner(
            app_name=APP_NAME,
            agent=dataset_extractor_agent,
            session_service=session_service,
        )

        # Feed the report to the extractor agent — it reads from pipeline_state
        # but needs a user message to kick off
        state = load_state()
        kickoff_message = (
            f"Find and download relevant datasets for this ML project.\n"
            f"User goal: {state.get('user_goal', '')}\n"
            f"Report summary: {state.get('report', '')}"
        )

        print("\n[PIPELINE] Searching and downloading datasets...\n", flush=True)

        response = await run_agent_turn(
            phase2_runner, USER_ID, SESSION_PHASE2, kickoff_message
        )

        if response is None:
            print("\n[ERROR] Dataset extractor failed. Exiting.")
            return

        print("\n[PIPELINE] Phase 2 complete — datasets downloaded!")

    # ==============================================================
    # DONE
    # ==============================================================

    final_state = load_state()
    print("\n" + "=" * 60)
    print("  PIPELINE STATUS")
    print("=" * 60)
    print(f"  Status: {final_state.get('status', 'unknown')}")
    print(f"  User Goal: {final_state.get('user_goal', 'N/A')[:100]}")
    print(f"  Q&A Pairs: {len(final_state.get('qa_pairs', []))}")
    if final_state.get("report"):
        print(f"  Report: {len(final_state['report'])} characters")
    if final_state.get("downloaded_dataset"):
        ds = final_state["downloaded_dataset"]
        print(f"  Dataset: {ds.get('dataset_name', 'N/A')} ({ds.get('source', 'N/A')})")
        print(f"  Path: {ds.get('path', 'N/A')}")
    print("=" * 60)
    print("\n  Next: Run data_preprocessing_agent to preprocess the downloaded dataset.")

    # ==============================================================
    # PHASE 3: Dataset Preprocessing (autonomous)
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PROMPT2ML — Phase 3: Dataset Preprocessing")
    print("=" * 60)

    state = load_state()
    if state.get("status") == "pipeline_complete" and state.get("report_path"):
        print(f"\n[PIPELINE] Datasets already preprocessed. Skipping Phase 3.")
    else:
        SESSION_PHASE3 = "session_phase3"
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_PHASE3
        )

        phase3_runner = Runner(
            app_name=APP_NAME,
            agent=data_preprocessing_agent,
            session_service=session_service,
        )

        # Feed the report to the preprocessing agent — it reads from pipeline_state
        # but needs a user message to kick off
        state = load_state()
        kickoff_message = (
            "Please do the data preprocessing steps for this ML project "
            "based on the user's goal, requirements, and downloaded datasets."
        )
        print("\n[PIPELINE] Preprocessing datasets...\n", flush=True)

        response = await run_agent_turn(
            phase3_runner, USER_ID, SESSION_PHASE3, kickoff_message
        )

        if response is None:
            print("\n[ERROR] Data preprocessing failed. Exiting.")
            return

        print("\n[PIPELINE] Phase 3 complete — datasets preprocessed!")


if __name__ == "__main__":
    asyncio.run(run_pipeline())