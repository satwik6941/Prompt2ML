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

from pipeline_state import load_state, save_state, reset_run_dir_cache, reset_run_id, get_outputs_dir, backup_state, mark_checkpoint


# ============================================================
# CONSTANTS
# ============================================================

from model_config import REASONING_MODEL

# Requirement gathering drives everything downstream — use the reasoning tier.
MODEL = REASONING_MODEL

# Per-phase timeouts (seconds). Set to 0 to disable.
PHASE2_TIMEOUT = 900   # 15 min — dataset search + download
PHASE3_TIMEOUT = 1800  # 30 min — multi-agent preprocessing loop
PHASE5_TIMEOUT = 3600  # 60 min — ML strategy planning, model training, and final report


def _write_crash_log(phase: str, error: Exception, context: dict = None) -> None:
    """Write a JSON crash log to outputs/pipeline_error.json on any phase failure."""
    log = {
        "phase": phase,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "context": context or {},
    }
    try:
        crash_file = get_outputs_dir() / "pipeline_error.json"
        with open(crash_file, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, default=str)
        print(f"\n[PIPELINE] Crash log written to {crash_file}", flush=True)
    except Exception:
        pass  # never let crash logging itself crash the process


# ============================================================
# ========  PHASE 1: REQUIREMENT GATHERER TOOLS  =============
# ============================================================

_REQUIREMENTS_DONE_STATUSES = {
    "report_ready", "dataset_ready", "preprocessing_complete",
    "preprocessing_complete_with_warnings", "ml_plan_ready",
    "model_trained", "ml_complete", "pipeline_complete",
}


def get_current_pipeline_status() -> str:
    """
    Check if the pipeline already has data from a previous run.
    Returns the current status and what data exists in pipeline_state.json.
    Use this FIRST to decide whether to skip requirement gathering.

    If requirements_complete is true, skip ALL questions and just summarize.
    """
    state = load_state()
    current_status = state.get("status", "empty")
    status = {
        "status": current_status,
        "requirements_complete": (
            current_status in _REQUIREMENTS_DONE_STATUSES
            and bool(state.get("user_goal"))
            and bool(state.get("report"))
        ),
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
    outputs_dir = get_outputs_dir()
    try:
        report_path = outputs_dir / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except OSError as e:
        return json.dumps({
            "error": f"Could not write report file: {e}",
            "path": str(outputs_dir / report_filename),
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
from data_preprocessing_agent.agent import data_preprocessing_agent
# Import Phase 5 ML pipeline (SequentialAgent: planner → trainer → report writer)
from machine_learning_agent.agent import root_agent as ml_pipeline_agent
# --- Phase 1: Requirement Gatherer ---
requirement_gatherer_agent = Agent(
    model=MODEL,
    name="requirement_gatherer_agent",
    description="Gathers user requirements through Q&A and generates a comprehensive project report.",
    output_key="requirements_report",
    instruction="""You are a world-class Machine Learning and Data Science expert — the kind who has seen hundreds of projects succeed and fail, and knows exactly what questions actually matter versus which ones waste everyone's time.

═══════════════════════════════════════════════════
 WHAT THE PIPELINE HANDLES AUTOMATICALLY
═══════════════════════════════════════════════════
The agents downstream of you handle these automatically — NEVER ask the user about them:
  ✗ Which datasets to use or where to find them       → dataset research + extractor agent
  ✗ How to preprocess or clean the data               → preprocessing research + strategist
  ✗ Which ML libraries or frameworks to use           → ML research agent decides
  ✗ Which algorithms or models to try                 → ML research agent finds SOTA
  ✗ Hyperparameter tuning strategy                    → ML trainer handles it

Asking about these wastes the user's time and adds no value — the pipeline produces
better answers to those questions than any user could give upfront.

═══════════════════════════════════════════════════
 WORKFLOW
═══════════════════════════════════════════════════

STEP 1 — Check pipeline status:
  Call get_current_pipeline_status first.
  If requirements_complete is true → STOP. Summarize the existing goal in one paragraph
  and say "Pipeline will now proceed." Do NOT re-ask or re-generate anything.

STEP 2 — Read their goal carefully:
  Their first message IS their goal. Before responding, think hard:
  • What is GENUINELY unclear that would change my recommendations?
  • What can I already infer from their statement without asking?
  • What would a senior ML engineer actually need to know to help this person?

STEP 3 — Ask adaptive questions, ONE AT A TIME:
  Ask between 3 and 8 questions — no fixed number.
  Ask more only if the goal is genuinely complex or ambiguous.
  STOP asking the moment you have enough to write a precise, specific report.

  ━━━ HOW TO DECIDE WHAT TO ASK ━━━

  Before each question ask yourself: "If I skip this, can I still give the best advice?"
  If yes → skip it. Only ask if the answer would meaningfully change your recommendations.

  Topics worth exploring (only if genuinely unclear from their specific goal):

  → The exact prediction target or model output
    When to ask: goal is vague ("analyse my data", "build something for X")
    Example: "What should the model actually output — a category, a number, a probability,
    a ranking? And what does one prediction correspond to — one row, one image, one user?"
    When to skip: they already stated what's predicted clearly.

  → What success looks like FOR THEM
    When to ask: you don't know their threshold or whether errors are symmetric
    Example: "What result would make this genuinely useful? And between a false positive
    and a false negative — which is more costly in your context?"
    When to skip: task has a universally accepted metric (BLEU, IoU, etc.).

  → Domain constraints or prior knowledge
    When to ask: the domain might have rules that change everything
    Example: "Any constraints on the data or model — privacy regulations, business rules,
    or outputs the model is absolutely not allowed to produce?"
    When to skip: no obvious domain-specific constraints.

  → How the model will be used once trained
    When to ask: deployment context isn't obvious
    Example: "Once it's trained, how will you actually use it — run it yourself,
    expose it as an API, embed it in an app, or just present the results?"
    When to skip: they already said (e.g., "for my notebook", "production API").

  → Timeline and compute constraints
    When to ask: the goal could involve heavy models (NLP, CV, deep learning)
    Example: "What compute do you have — a laptop, a cloud VM, or a GPU server?
    And is there a deadline this needs to meet?"
    When to skip: clearly a lightweight task (small tabular dataset, fast model needed).

  → What they've already tried
    When to ask: they seem experienced or mention prior work
    Example: "Have you tried anything for this already? What worked, what didn't?"
    When to skip: they're clearly starting from scratch.

  → Their background and what this project is for
    When to ask: context would change the depth or style of recommendations
    Example: "Is this for learning, a work project, academic research, or a product?"
    When to skip: obvious from how they described the problem.

  ━━━ NEVER ASK ━━━
  • "What datasets do you have?" / "Where is your data?" → pipeline handles it
  • "Which libraries do you prefer?" → agents decide based on SOTA
  • "Should we use sklearn or PyTorch?" → ML research agent figures it out
  • "How should we preprocess the data?" → preprocessing agent owns this
  • Generic fillers like "Is there anything else you'd like to share?"

STEP 4 — Write the comprehensive report:
  Once you have enough context, write a DETAILED, SPECIFIC report:

  SECTION 1: PROBLEM STATEMENT & PRECISE ML FRAMING
    Translate their goal into exact ML terms:
    "This is a [problem type] where the model predicts [output] given [inputs]."
    Make it unambiguous — future agents will use this framing.

  SECTION 2: STATE OF THE ART
    What approaches exist for this exact task? What wins in competitions and research?
    Be specific to THEIR task and domain, not generic ML.

  SECTION 3: THEIR CONSTRAINTS & CONTEXT
    What the conversation revealed — success criteria, deployment target, timeline,
    skill level, domain constraints. Reference their actual answers, not templates.

  SECTION 4: DATA STRATEGY
    What kinds of data the pipeline will automatically search for and why.
    What to expect from the dataset — typical sizes, quality issues, formats.
    Do NOT ask the user about this — describe what the downstream agents will do.

  SECTION 5: RECOMMENDED APPROACH
    Given THEIR specific constraints and the SOTA: what ML approach suits them best.
    Justify every recommendation against what they actually told you.

  SECTION 6: STEP-BY-STEP ACTION PLAN
    Phase-by-phase: what each pipeline stage will produce for their specific project.

  SECTION 7: CHALLENGES & RISKS
    Real risks for THIS problem — tied to their constraints and domain.
    Not generic overfitting warnings — specific issues they will actually face.

  SECTION 8: SUCCESS METRICS & EVALUATION
    Concrete metrics tied to their stated success definition.
    How to know when the model is good enough for their use case.

  SECTION 9: EXPERT TIPS FOR THIS PROBLEM
    The non-obvious insights a practitioner would share for this exact task and domain.
    Things they won't find in a beginner tutorial.

  SECTION 10: FINAL VERDICT & WHAT TO FOCUS ON
    Direct recommendation: given everything they told you, what matters most and why.
    End with something specific and motivating to their actual project.

  Minimum 1500 words. Every paragraph must reflect what THEY told you.
  No generic ML textbook content — this report should be useless to anyone else.

STEP 5 — Save:
  Call save_requirement_report with:
  - user_goal: their original first message verbatim
  - qa_pairs_json: JSON array of all Q&A pairs (however many were asked)
  - report_content: THE COMPLETE VERBATIM TEXT of the report — every word, every section.
    Never pass a placeholder, summary, or "(see above)".
  - report_filename: '<project_slug>_project_report.txt'

STEP 6 — Close:
  One short paragraph: what was captured, and that the pipeline will now automatically
  find datasets, preprocess them, and train models for their project.

═══════════════════════════════════════════════════
 CORE RULES
═══════════════════════════════════════════════════
• Check pipeline status first — never re-gather if a report already exists
• Every question must emerge from what's actually unclear in THEIR specific goal
• Stop asking when you have enough — no quotas, no padding
• The pipeline owns data, libraries, preprocessing, algorithms — never ask about these
• The report must read like advice from someone who understood their project deeply,
  not a template with their keywords swapped in
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

    # Statuses that mean Phase 1 is already complete (any status after report_ready counts)
    _PHASE1_DONE_STATUSES = {
        "report_ready", "dataset_ready", "preprocessing_complete",
        "preprocessing_complete_with_warnings", "ml_plan_ready",
        "model_trained", "ml_complete", "pipeline_complete",
    }

    # Check if Phase 1 already done
    state = load_state()
    if state.get("status") in _PHASE1_DONE_STATUSES and state.get("user_goal") and state.get("report"):
        print(f"\n[PIPELINE] Report already exists for: {state['user_goal'][:80]}...")
        print("[PIPELINE] Skipping Phase 1, moving to Phase 2.\n")
    else:
        # A brand-new project starts a brand-new run. Without this the run_id
        # persisted from the previous run is reused, so modified_datasets/<run_id>/
        # and outputs/<run_id>/ accumulate across unrelated projects and agents
        # read the last project's reports as if they were this one's.
        reset_run_id()

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

            # Check if report was saved (accept any post-Phase-1 status)
            if load_state().get("status") in _PHASE1_DONE_STATUSES:
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
    # NOTE: preprocessing report generation is NOT a separate phase.
    # report_generator_agent is the last sub-agent of the Phase 3
    # SequentialAgent, so it has already run by this point. Driving the same
    # agent instance from a second Runner also gave it two parents, which ADK
    # does not support (sub_agents assignment sets parent_agent).
    # ==============================================================

    # ==============================================================
    # PHASE 5: Machine Learning (autonomous)
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PROMPT2ML — Phase 5: Machine Learning")
    print("=" * 60)

    state = load_state()
    final_report_path = state.get("final_report_path", "")
    if final_report_path and Path(final_report_path).exists():
        print(f"\n[PIPELINE] ML pipeline already complete. Skipping Phase 5.")
        print(f"  Final report: {final_report_path}")
    else:
        SESSION_PHASE5 = "session_phase5"
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_PHASE5
        )

        phase5_runner = Runner(
            app_name=APP_NAME,
            agent=ml_pipeline_agent,
            session_service=session_service,
        )

        state = load_state()
        user_goal = state.get("user_goal", "Train the best ML model for the available dataset.")
        kickoff_message = (
            f"Run the complete ML pipeline for this project: {user_goal}\n\n"
            "Execute all three phases: planning, training, and reporting."
        )
        # Preprocessing that exhausted its retries proceeds with warnings — the ML
        # phase must know so it can double-check data quality and say so in reports.
        if state.get("status") == "preprocessing_complete_with_warnings":
            last_validation = (state.get("agent4_iterations") or [{}])[-1]
            print("\n[PIPELINE] WARNING: preprocessing finished WITH WARNINGS — "
                  "validation never fully passed. The ML phase will be told to treat "
                  "the data with suspicion.", flush=True)
            kickoff_message += (
                "\n\nIMPORTANT: The preprocessing phase completed WITH WARNINGS — its "
                "validator never gave a full PASS. Unresolved issues: "
                f"{json.dumps(last_validation.get('issues_found', []))}. "
                "Profile the dataset carefully before training, work around remaining "
                "issues where possible, and state these caveats explicitly in the final report."
            )
        print("\n[PIPELINE] Running ML pipeline (strategy → training → report)...\n", flush=True)

        response = None
        for phase5_attempt in range(3):
            if phase5_attempt > 0:
                print(f"\n[PIPELINE] Retrying Phase 5 (attempt {phase5_attempt + 1}/3)...", flush=True)
                await asyncio.sleep(15)
                SESSION_PHASE5 = f"session_phase5_retry{phase5_attempt}"
                await session_service.create_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_PHASE5
                )
                phase5_runner = Runner(
                    app_name=APP_NAME,
                    agent=ml_pipeline_agent,
                    session_service=session_service,
                )
            try:
                response = await asyncio.wait_for(
                    run_agent_turn(phase5_runner, USER_ID, SESSION_PHASE5, kickoff_message),
                    timeout=PHASE5_TIMEOUT if PHASE5_TIMEOUT > 0 else None,
                )
            except asyncio.TimeoutError as e:
                _write_crash_log(
                    f"Phase 5 attempt {phase5_attempt + 1}",
                    e,
                    {"timeout_seconds": PHASE5_TIMEOUT},
                )
                print(f"\n[WARN] Phase 5 attempt {phase5_attempt + 1} timed out after {PHASE5_TIMEOUT}s.", flush=True)
                response = None
            except Exception as e:
                _write_crash_log(f"Phase 5 attempt {phase5_attempt + 1}", e)
                print(f"\n[WARN] Phase 5 attempt {phase5_attempt + 1} error: {e}", flush=True)
                response = None

            if response is not None:
                break

        if response is None:
            _write_crash_log("Phase 5", RuntimeError("All 3 attempts failed"), {})
            print("\n[ERROR] ML pipeline failed after 3 attempts.")
        else:
            mark_checkpoint("phase5_complete")
            backup_state()
            print("\n[PIPELINE] Phase 5 complete — ML pipeline finished!")

    # ==============================================================
    # FINAL STATUS
    # ==============================================================

    final_state = load_state()
    print("\n" + "=" * 60)
    print("  FINAL PIPELINE STATUS")
    print("=" * 60)
    print(f"  Status           : {final_state.get('status', 'unknown')}")
    print(f"  Preprocessed     : {final_state.get('preprocessed_dataset_path', 'N/A')}")
    print(f"  Preprocessing Rpt: {final_state.get('report_path', 'N/A')}")
    print(f"  Best Model       : {final_state.get('best_model_path', 'N/A')}")
    print(f"  Final ML Report  : {final_state.get('final_report_path', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline())