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
import datetime
import traceback
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

from pipeline_state import load_state, save_state, reset_run_dir_cache, backup_state, mark_checkpoint


# ============================================================
# CONSTANTS
# ============================================================

MODEL = "gemini-3-flash-preview"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Per-phase timeouts (seconds). Set to 0 to disable.
PHASE2_TIMEOUT = 900   # 15 min — dataset search + download
PHASE3_TIMEOUT = 1800  # 30 min — multi-agent preprocessing loop
PHASE4_TIMEOUT = 600   # 10 min — report generation


def _write_crash_log(phase: str, error: Exception, context: dict = None) -> None:
    """Write a JSON crash log to outputs/pipeline_error.json on any phase failure."""
    log = {
        "phase": phase,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "context": context or {},
    }
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        crash_file = OUTPUTS_DIR / "pipeline_error.json"
        with open(crash_file, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, default=str)
        print(f"\n[PIPELINE] Crash log written to {crash_file}", flush=True)
    except Exception:
        pass  # never let crash logging itself crash the process


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
    # Guard: catch placeholder strings the LLM sometimes passes instead of the real report
    placeholder_markers = ["(the full report", "(see above)", "(report above)", "(provided above)"]
    if len(report_content) < 500 or any(m in report_content.lower() for m in placeholder_markers):
        return json.dumps({
            "error": "report_content appears to be a placeholder, not the actual report. "
                     "You MUST pass the complete verbatim report text — every section, every word. "
                     "Re-generate the report and call this tool again with the full text.",
            "received_length": len(report_content),
            "received_preview": report_content[:200],
        }, indent=2)

    # Parse qa_pairs_json safely — LLM may pass malformed JSON
    try:
        qa_pairs = json.loads(qa_pairs_json)
        if not isinstance(qa_pairs, list):
            qa_pairs = []
    except (json.JSONDecodeError, TypeError):
        qa_pairs = []

    # Write report to disk
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = OUTPUTS_DIR / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except OSError as e:
        return json.dumps({
            "error": f"Could not write report file: {e}",
            "path": str(OUTPUTS_DIR / report_filename),
            "fix": "Check write permissions on the outputs/ folder.",
        }, indent=2)

    save_state({
        "user_goal": user_goal,
        "qa_pairs": qa_pairs,
        "report": report_content,
        "report_filename": report_filename,
        "status": "report_ready",
    })
    # Clear run-dir cache so downstream agents resolve the slug from the goal we just saved
    reset_run_dir_cache()

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
from data_preprocessing_agent.agent import data_preprocessing_agent, report_generator_agent

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
   - report_content: THE COMPLETE FULL TEXT OF THE REPORT YOU JUST WROTE — every word,
     every section, verbatim. Do NOT pass a placeholder like "(see above)" or
     "(the report provided above)" or a summary. The exact full string must be passed
     or the file will not be saved correctly.
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
            if any(k in error_msg for k in ("500", "503", "INTERNAL", "UNAVAILABLE", "overloaded", "Resource has been exhausted")):
                wait = 10 if "503" in error_msg or "UNAVAILABLE" in error_msg else 5
                print(f"\n[WARN] Gemini transient error (attempt {attempt + 1}/3). Retrying in {wait}s...", flush=True)
                print(f"[WARN] Error: {error_msg[:200]}", flush=True)
                await asyncio.sleep(wait)
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
    report_file_exists = bool(state.get("report_filename")) and (OUTPUTS_DIR / state.get("report_filename", "")).exists()
    if state.get("status") == "report_ready" and state.get("user_goal") and report_file_exists:
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

    # Check if datasets already downloaded (also verify files exist on disk)
    state = load_state()
    ds = state.get("downloaded_dataset", {})
    ds_path = ds.get("path", "") if isinstance(ds, dict) else ""
    if state.get("status") == "dataset_ready" and ds_path and Path(ds_path).exists():
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

        state = load_state()
        kickoff_message = (
            f"Find and download relevant datasets for this ML project.\n"
            f"User goal: {state.get('user_goal', '')}\n"
            f"Report summary: {state.get('report', '')}"
        )

        print("\n[PIPELINE] Searching and downloading datasets...\n", flush=True)

        try:
            response = await asyncio.wait_for(
                run_agent_turn(phase2_runner, USER_ID, SESSION_PHASE2, kickoff_message),
                timeout=PHASE2_TIMEOUT if PHASE2_TIMEOUT > 0 else None,
            )
        except asyncio.TimeoutError as e:
            _write_crash_log("Phase 2", e, {"timeout_seconds": PHASE2_TIMEOUT})
            print(f"\n[ERROR] Phase 2 timed out after {PHASE2_TIMEOUT}s. Exiting.")
            return
        except Exception as e:
            _write_crash_log("Phase 2", e)
            print(f"\n[ERROR] Phase 2 unexpected error: {e}")
            return

        if response is None:
            _write_crash_log("Phase 2", RuntimeError("Agent returned None"), {})
            print("\n[ERROR] Dataset extractor failed. Exiting.")
            return

        mark_checkpoint("phase2_complete")
        backup_state()
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

    _PHASE3_DONE_STATUSES = {"pipeline_complete", "preprocessing_complete", "preprocessing_complete_with_warnings"}
    state = load_state()
    preprocessed_path = state.get("preprocessed_dataset_path", "")
    if state.get("status") in _PHASE3_DONE_STATUSES and preprocessed_path and Path(preprocessed_path).exists():
        print(f"\n[PIPELINE] Datasets already preprocessed (status: {state.get('status')}). Skipping Phase 3.")
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

        kickoff_message = (
            "Please do the data preprocessing steps for this ML project "
            "based on the user's goal, requirements, and downloaded datasets."
        )
        print("\n[PIPELINE] Preprocessing datasets...\n", flush=True)

        response = None
        for phase3_attempt in range(3):
            if phase3_attempt > 0:
                print(f"\n[PIPELINE] Retrying Phase 3 (attempt {phase3_attempt + 1}/3)...", flush=True)
                await asyncio.sleep(15)
                SESSION_PHASE3 = f"session_phase3_retry{phase3_attempt}"
                await session_service.create_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_PHASE3
                )
                phase3_runner = Runner(
                    app_name=APP_NAME,
                    agent=data_preprocessing_agent,
                    session_service=session_service,
                )
            try:
                response = await asyncio.wait_for(
                    run_agent_turn(phase3_runner, USER_ID, SESSION_PHASE3, kickoff_message),
                    timeout=PHASE3_TIMEOUT if PHASE3_TIMEOUT > 0 else None,
                )
            except asyncio.TimeoutError as e:
                _write_crash_log(
                    f"Phase 3 attempt {phase3_attempt + 1}",
                    e,
                    {"timeout_seconds": PHASE3_TIMEOUT},
                )
                print(f"\n[WARN] Phase 3 attempt {phase3_attempt + 1} timed out after {PHASE3_TIMEOUT}s.", flush=True)
                response = None
            except Exception as e:
                _write_crash_log(f"Phase 3 attempt {phase3_attempt + 1}", e)
                print(f"\n[WARN] Phase 3 attempt {phase3_attempt + 1} error: {e}", flush=True)
                response = None

            if response is not None:
                break

        if response is None:
            _write_crash_log("Phase 3", RuntimeError("All 3 attempts failed"), {})
            print("\n[ERROR] Data preprocessing failed after 3 attempts. Exiting.")
            return

        mark_checkpoint("phase3_complete")
        backup_state()
        print("\n[PIPELINE] Phase 3 complete — datasets preprocessed!")

    # ==============================================================
    # PHASE 4: Preprocessing Report Generation (autonomous)
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PROMPT2ML — Phase 4: Preprocessing Report")
    print("=" * 60)

    state = load_state()
    report_path = state.get("report_path", "")
    if state.get("status") == "pipeline_complete" and report_path and Path(report_path).exists():
        print(f"\n[PIPELINE] Report already generated at: {report_path}. Skipping Phase 4.")
    else:
        SESSION_PHASE4 = "session_phase4"
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_PHASE4
        )

        phase4_runner = Runner(
            app_name=APP_NAME,
            agent=report_generator_agent,
            session_service=session_service,
        )

        print("\n[PIPELINE] Generating preprocessing report...\n", flush=True)

        response = None
        for phase4_attempt in range(3):
            if phase4_attempt > 0:
                print(f"\n[PIPELINE] Retrying Phase 4 (attempt {phase4_attempt + 1}/3)...", flush=True)
                await asyncio.sleep(10)
                SESSION_PHASE4 = f"session_phase4_retry{phase4_attempt}"
                await session_service.create_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_PHASE4
                )
                phase4_runner = Runner(
                    app_name=APP_NAME,
                    agent=report_generator_agent,
                    session_service=session_service,
                )
            try:
                response = await asyncio.wait_for(
                    run_agent_turn(
                        phase4_runner, USER_ID, SESSION_PHASE4,
                        "Generate the comprehensive preprocessing report for this ML project.",
                    ),
                    timeout=PHASE4_TIMEOUT if PHASE4_TIMEOUT > 0 else None,
                )
            except asyncio.TimeoutError as e:
                _write_crash_log(
                    f"Phase 4 attempt {phase4_attempt + 1}",
                    e,
                    {"timeout_seconds": PHASE4_TIMEOUT},
                )
                print(f"\n[WARN] Phase 4 attempt {phase4_attempt + 1} timed out.", flush=True)
                response = None
            except Exception as e:
                _write_crash_log(f"Phase 4 attempt {phase4_attempt + 1}", e)
                print(f"\n[WARN] Phase 4 attempt {phase4_attempt + 1} error: {e}", flush=True)
                response = None

            if response is not None:
                break

        if response is None:
            _write_crash_log("Phase 4", RuntimeError("All 3 attempts failed"), {})
            print("\n[ERROR] Report generation failed after 3 attempts.")
        else:
            mark_checkpoint("phase4_complete")
            backup_state()
            print("\n[PIPELINE] Phase 4 complete — preprocessing report saved!")

    # ==============================================================
    # FINAL STATUS
    # ==============================================================

    final_state = load_state()
    print("\n" + "=" * 60)
    print("  FINAL PIPELINE STATUS")
    print("=" * 60)
    print(f"  Status      : {final_state.get('status', 'unknown')}")
    print(f"  Dataset     : {final_state.get('preprocessed_dataset_path', 'N/A')}")
    print(f"  Report      : {final_state.get('report_path', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline())