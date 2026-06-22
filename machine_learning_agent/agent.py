"""
Machine Learning Agent System (Google ADK)

Architecture:
    SequentialAgent (root_agent)
    ├── Agent 1: ml_strategy_planner
    │     Reads all pipeline outputs, preprocessed data, and project reports.
    │     Drafts a comprehensive ML training plan and saves it to outputs/.
    ├── Agent 2: ml_model_trainer
    │     Reads the plan, writes training code, executes it in sandbox,
    │     and optionally pushes workloads to Google Colab via colab_mcp.
    │     Saves trained models, metrics, and plots to outputs/ml_outputs/.
    └── Agent 3: ml_report_writer
          Reads everything from outputs/ and writes a detailed final report
          in plain English, saved to outputs/final_ml_report_<slug>.md.

State is shared through pipeline_state.json (see pipeline_state.py).
All output files land in outputs/ or outputs/ml_outputs/.
"""

import os
import sys
import json
import asyncio
import datetime
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data_preprocessing_agent"))

from pipeline_state import (
    load_state, save_state, get_run_dir, get_outputs_dir,
    mark_checkpoint, is_checkpoint_done,
)

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).parent / ".env")

# ============================================================
# CONSTANTS
# ============================================================

MODEL = "gemini-3.1-flash-lite"


# ============================================================
# ==================  AGENT 1 TOOLS  =========================
# ML Strategy Planner
# ============================================================

def get_full_pipeline_context() -> str:
    """
    Load ALL available context from the pipeline — user goal, Q&A pairs,
    project report, selected dataset info, preprocessing plan and results,
    the path to the preprocessed dataset, and pre-phase research findings.
    The ml_training_research field contains SOTA recommendations, hyperparameter
    ranges, and library notes from the research agent that ran before this phase.
    Call this FIRST so you understand everything that has happened so far.
    Returns a JSON string with all relevant fields.
    """
    state = load_state()
    context = {
        "user_goal": state.get("user_goal", ""),
        "qa_pairs": state.get("qa_pairs", []),
        "project_report_summary": (state.get("report") or "")[:5000],
        "selected_dataset": state.get("selected_dataset", {}),
        "preprocessing_plan": state.get("preprocessing_plan", {}),
        "preprocessed_dataset_path": state.get("preprocessed_dataset_path", ""),
        "pipeline_checkpoints": state.get("pipeline_checkpoints", {}),
        "status": state.get("status", ""),
        "ml_plan": state.get("ml_plan", {}),
        "ml_training_research": state.get("ml_training_research", {}),
    }
    return json.dumps(context, indent=2, default=str)


def read_outputs_folder() -> str:
    """
    Read all text/markdown/JSON files from this run's outputs/ folder and run dir.
    Includes preprocessing reports, project reports, ML metrics, and any prior results.
    Use this to understand the full picture of work done so far.
    Returns a JSON object mapping relative path to content (first 8000 chars each).
    """
    outputs_dir = get_outputs_dir()
    run_dir = get_run_dir()
    files = {}
    for search_dir in [outputs_dir, run_dir]:
        for f in sorted(search_dir.rglob("*")):
            if f.is_file() and f.suffix in {".txt", ".md", ".json", ".log", ".csv"}:
                try:
                    key = str(f.relative_to(PROJECT_ROOT))
                    files[key] = f.read_text(encoding="utf-8", errors="ignore")[:8000]
                except Exception as e:
                    files[str(f)] = f"[Read error: {e}]"
    if not files:
        return json.dumps({"message": "No output files found yet."})
    return json.dumps(files, indent=2)


def read_preprocessed_dataset_sample() -> str:
    """
    Load and profile the preprocessed dataset so you can make informed
    decisions about which ML algorithms to recommend.
    Returns shape, column names, dtypes, missing values, summary stats,
    target column info, and the first 10 rows as a sample.
    """
    import pandas as pd

    state = load_state()
    dataset_path = state.get("preprocessed_dataset_path", "")

    if not dataset_path or not Path(dataset_path).exists():
        run_dir = get_run_dir()
        candidates = sorted(run_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return json.dumps({
                "error": "No preprocessed dataset found. Run data_preprocessing_agent first.",
                "searched": str(run_dir),
            })
        dataset_path = str(candidates[-1])

    try:
        df = pd.read_csv(dataset_path)
        sample = df.head(10)

        profile: dict = {
            "file_path": dataset_path,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "sample_rows": sample.to_dict(orient="records"),
        }

        try:
            profile["numeric_stats"] = df.describe(include="number").to_dict()
        except Exception:
            pass

        plan = state.get("preprocessing_plan", {})
        target = plan.get("target_column", "")
        if target and target in df.columns:
            if df[target].dtype == "object" or str(df[target].dtype) == "category":
                profile["target_column"] = target
                profile["target_unique_values"] = int(df[target].nunique())
                profile["target_value_counts"] = (
                    df[target].value_counts().head(20).to_dict()
                )
                profile["problem_type_hint"] = "classification"
            else:
                profile["target_column"] = target
                profile["target_stats"] = df[target].describe().to_dict()
                profile["problem_type_hint"] = "regression"

        return json.dumps(profile, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "dataset_path": dataset_path})


def save_ml_training_plan(
    plan_content: str,
    plan_filename: str,
    problem_type: str,
    target_column: str,
    recommended_models: str,
    evaluation_metrics: str,
    train_test_split_strategy: str,
    feature_selection_plan: str,
    hyperparameter_strategy: str,
    baseline_model: str,
    advanced_models: str,
    training_approach: str,
) -> str:
    """
    Persist the complete ML training plan to disk and pipeline state.

    Args:
        plan_content: Full markdown content of the plan (minimum 1000 words).
        plan_filename: E.g. 'ml_training_plan_walmart_sales.md'.
        problem_type: classification | regression | clustering | time_series | nlp | cv.
        target_column: Target column name, or 'none' for unsupervised learning.
        recommended_models: JSON array of model names ranked by expected performance.
        evaluation_metrics: Primary and secondary evaluation metrics.
        train_test_split_strategy: E.g. 'stratified 80/10/10', 'time-based', '5-fold CV'.
        feature_selection_plan: Feature selection approach and rationale.
        hyperparameter_strategy: grid_search | random_search | bayesian | defaults.
        baseline_model: Simple model to set the performance floor.
        advanced_models: Sophisticated models to try after baseline succeeds.
        training_approach: local_sandbox | colab_gpu | both.

    Returns:
        JSON confirmation or error.
    """
    if len(plan_content) < 500:
        return json.dumps({
            "error": "plan_content is too short — must be at least 500 characters. "
                     "Write the full plan before calling this tool.",
            "received_length": len(plan_content),
        })

    outputs_dir = get_outputs_dir()
    plan_path = outputs_dir / plan_filename
    try:
        plan_path.write_text(plan_content, encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": f"Could not write plan file: {e}"})

    plan_data = {
        "problem_type": problem_type,
        "target_column": target_column,
        "recommended_models": recommended_models,
        "evaluation_metrics": evaluation_metrics,
        "train_test_split_strategy": train_test_split_strategy,
        "feature_selection_plan": feature_selection_plan,
        "hyperparameter_strategy": hyperparameter_strategy,
        "baseline_model": baseline_model,
        "advanced_models": advanced_models,
        "training_approach": training_approach,
        "plan_file": str(plan_path),
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

    save_state({
        "ml_plan": plan_data,
        "ml_plan_content": plan_content,
        "ml_plan_filename": plan_filename,
        "status": "ml_plan_ready",
    })
    mark_checkpoint("ml_planning")

    return json.dumps({
        "status": "success",
        "plan_path": str(plan_path),
        "problem_type": problem_type,
        "models_planned": recommended_models,
    }, indent=2)


# ============================================================
# ==================  AGENT 2 TOOLS  =========================
# ML Model Trainer
# ============================================================

# Reuse the sandbox executor from data_preprocessing_agent
try:
    from sandbox_executor import (
        start_sandbox,
        stop_sandbox,
        run_in_sandbox,
        write_file_to_sandbox,
        read_file_from_sandbox,
    )
except ImportError:
    def start_sandbox() -> str:
        return json.dumps({"error": "sandbox_executor not found. Docker sandbox unavailable."})

    def stop_sandbox() -> str:
        return json.dumps({"status": "no-op"})

    def run_in_sandbox(code: str) -> str:
        return json.dumps({"error": "sandbox_executor not found."})

    def write_file_to_sandbox(filename: str, content: str) -> str:
        return json.dumps({"error": "sandbox_executor not found."})

    def read_file_from_sandbox(filename: str) -> str:
        return json.dumps({"error": "sandbox_executor not found."})


# ── Colab MCP lifecycle ────────────────────────────────────────────────────────
# Set COLAB_MCP_DIR in machine_learning_agent/.env to the cloned colab-mcp repo.
# The McpToolset below is added to the trainer's tool list only when the dir exists.
# _colab_proc tracks the background process so stop_colab_runtime() can kill it.

import subprocess
import threading

_COLAB_MCP_AVAILABLE = False
_colab_proc: "subprocess.Popen | None" = None
colab_mcp = None

_colab_dir = os.getenv("COLAB_MCP_DIR", "")

try:
    from google.adk.tools.mcp_tool import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from mcp import StdioServerParameters

    if _colab_dir and Path(_colab_dir).exists():
        colab_mcp = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="uv",
                    args=["run", "colab-mcp"],
                    cwd=_colab_dir,
                    timeout=30000,
                )
            ),
        )
        _COLAB_MCP_AVAILABLE = True
except Exception:
    pass


def start_colab_runtime() -> str:
    """
    Start the colab-mcp bridge process so the agent can execute code in
    Google Colab. Call this BEFORE sending any GPU-intensive training to Colab.

    First run will open a browser tab for Google OAuth — sign in with the same
    Google account you use for Colab. Credentials are cached locally after that.

    Returns:
        JSON with status and instructions.
    """
    global _colab_proc, _COLAB_MCP_AVAILABLE

    if not _colab_dir or not Path(_colab_dir).exists():
        return json.dumps({
            "status": "skipped",
            "reason": "COLAB_MCP_DIR not set or path does not exist. "
                      "Set it in machine_learning_agent/.env and restart.",
            "fallback": "Use local Docker sandbox for training instead.",
        })

    if _colab_proc is not None and _colab_proc.poll() is None:
        return json.dumps({
            "status": "already_running",
            "pid": _colab_proc.pid,
            "message": "Colab MCP bridge is already running.",
        })

    try:
        _colab_proc = subprocess.Popen(
            ["uv", "run", "colab-mcp"],
            cwd=_colab_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _COLAB_MCP_AVAILABLE = True

        # Stream stderr to console in background so the user sees auth prompts
        def _stream(pipe, prefix):
            for line in iter(pipe.readline, ""):
                print(f"[COLAB-MCP {prefix}] {line.rstrip()}", flush=True)

        threading.Thread(target=_stream, args=(_colab_proc.stdout, "OUT"), daemon=True).start()
        threading.Thread(target=_stream, args=(_colab_proc.stderr, "ERR"), daemon=True).start()

        save_state({"colab_mcp_pid": _colab_proc.pid})

        return json.dumps({
            "status": "started",
            "pid": _colab_proc.pid,
            "message": (
                "Colab MCP bridge started. "
                "If this is the first run, check the console for a Google login URL and authenticate."
            ),
        })
    except FileNotFoundError:
        return json.dumps({
            "status": "error",
            "error": "'uv' command not found. Run: pip install uv",
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def stop_colab_runtime() -> str:
    """
    Shut down the Colab runtime to free GPU hours, then terminate the local
    colab-mcp bridge process. Call this AFTER all GPU training is complete.

    This is important — Colab GPU sessions count against your free quota even
    when idle. Always call this when training is finished.

    Returns:
        JSON confirmation.
    """
    global _colab_proc, _COLAB_MCP_AVAILABLE

    results = {}

    # Step 1: Ask Colab to delete its own runtime via a short Python snippet.
    # colab-mcp exposes the Colab kernel — we can run a runtime-shutdown cell.
    if _COLAB_MCP_AVAILABLE and _colab_proc is not None and _colab_proc.poll() is None:
        try:
            shutdown_script = (
                "from google.colab import runtime\n"
                "runtime.unassign()\n"
            )
            # Write shutdown script to the run dir so the agent can submit it
            run_dir = get_run_dir()
            shutdown_path = run_dir / "_colab_shutdown.py"
            shutdown_path.write_text(shutdown_script, encoding="utf-8")
            results["shutdown_script_written"] = str(shutdown_path)
            results["note"] = (
                "Submit this script as a Colab cell via the colab-mcp execute_code tool "
                "before calling stop_colab_runtime, OR this function will force-terminate "
                "the local bridge process after a 5-second grace period."
            )
        except Exception as e:
            results["shutdown_script_error"] = str(e)

    # Step 2: Terminate the local bridge process
    if _colab_proc is not None:
        try:
            _colab_proc.terminate()
            try:
                _colab_proc.wait(timeout=8)
                results["bridge_process"] = f"terminated (pid {_colab_proc.pid})"
            except subprocess.TimeoutExpired:
                _colab_proc.kill()
                results["bridge_process"] = f"force-killed (pid {_colab_proc.pid})"
        except Exception as e:
            results["bridge_process_error"] = str(e)
        finally:
            _colab_proc = None
            _COLAB_MCP_AVAILABLE = False
    else:
        results["bridge_process"] = "was not running"

    save_state({"colab_mcp_pid": None})
    mark_checkpoint("colab_runtime_stopped")

    results["status"] = "success"
    results["message"] = "Colab GPU runtime released. No further GPU hours will be consumed."
    return json.dumps(results, indent=2)


def get_ml_plan_for_trainer() -> str:
    """
    Load the ML training plan from pipeline state.
    Returns the full plan content, structured metadata, and the preprocessed
    dataset path so the trainer can immediately start writing code.
    Call this FIRST before writing any training code.
    """
    state = load_state()
    return json.dumps({
        "ml_plan": state.get("ml_plan", {}),
        "ml_plan_content": state.get("ml_plan_content", ""),
        "preprocessed_dataset_path": state.get("preprocessed_dataset_path", ""),
        "user_goal": state.get("user_goal", ""),
        "status": state.get("status", ""),
        "colab_mcp_available": _COLAB_MCP_AVAILABLE,
    }, indent=2, default=str)


def get_ml_outputs_dir() -> str:
    """
    Return the path where trained models and plots should be saved.
    Creates the directory if it does not exist.

    Two paths are returned:
    - ml_outputs_dir: HOST path (Windows absolute). Use when calling
      save_training_results(), mark_training_complete(), or any host-side tool.
    - sandbox_ml_outputs_dir: path INSIDE Docker (/workspace/ml_outputs).
      Use as ML_OUTPUTS_DIR in scripts passed to run_in_sandbox().

    The sandbox mounts the run dir as /workspace, so /workspace/ml_outputs
    is the only reliable absolute path inside Linux containers. Passing the
    Windows host path into Docker creates a malformed directory name because
    backslash is not a path separator on Linux.
    """
    run_dir = get_run_dir()
    ml_outputs_dir = run_dir / "ml_outputs"
    ml_outputs_dir.mkdir(parents=True, exist_ok=True)
    return json.dumps({
        "ml_outputs_dir": str(ml_outputs_dir),
        "sandbox_ml_outputs_dir": "/workspace/ml_outputs",
        "note": (
            "In sandbox scripts (run_in_sandbox), set ML_OUTPUTS_DIR = '/workspace/ml_outputs'. "
            "When calling save_training_results() or mark_training_complete(), use ml_outputs_dir "
            "(translate '/workspace/ml_outputs/X' -> ml_outputs_dir + '/X' for the host path)."
        ),
    })


def save_training_results(
    model_name: str,
    metrics_json: str,
    model_file_path: str,
    plots_saved: str,
    training_summary: str,
    best_params: str,
    feature_importance: str,
) -> str:
    """
    Append one model's training results to pipeline state.
    Call this once per model after training completes.

    Args:
        model_name: Model class name, e.g. 'RandomForestClassifier'.
        metrics_json: JSON object with all metric names and values.
        model_file_path: Absolute path to the saved .joblib model file.
        plots_saved: Comma-separated list of saved plot filenames.
        training_summary: 3-5 sentence plain-English summary of the run.
        best_params: JSON object of best hyperparameters (or 'defaults').
        feature_importance: JSON object of top-N feature importances, or 'N/A'.

    Returns:
        JSON confirmation.
    """
    state = load_state()
    results = state.get("training_results", [])
    if not isinstance(results, list):
        results = []

    try:
        metrics = json.loads(metrics_json)
    except Exception:
        metrics = {"raw": metrics_json}

    results.append({
        "model_name": model_name,
        "metrics": metrics,
        "model_file_path": model_file_path,
        "plots_saved": plots_saved,
        "training_summary": training_summary,
        "best_params": best_params,
        "feature_importance": feature_importance,
        "saved_at": datetime.datetime.utcnow().isoformat(),
    })

    save_state({
        "training_results": results,
        "status": "model_trained",
    })
    mark_checkpoint(f"trained_{model_name.lower()[:30].replace(' ', '_')}")

    return json.dumps({
        "status": "success",
        "model": model_name,
        "metrics_saved": metrics,
    }, indent=2)


def mark_training_complete(
    best_model_name: str,
    best_model_path: str,
    best_metrics: str,
    all_models_compared: str,
) -> str:
    """
    Mark the ML training phase as complete and record the winning model.

    Args:
        best_model_name: Name of the best performing model.
        best_model_path: Absolute path to the saved best model file.
        best_metrics: JSON of the best model's metrics.
        all_models_compared: Plain-text summary of all models and their scores.

    Returns:
        JSON confirmation.
    """
    save_state({
        "best_model_name": best_model_name,
        "best_model_path": best_model_path,
        "best_metrics": best_metrics,
        "all_models_compared": all_models_compared,
        "status": "ml_complete",
    })
    mark_checkpoint("ml_training_complete")
    return json.dumps({
        "status": "success",
        "best_model": best_model_name,
        "message": "ML training phase complete. Handoff to report writer.",
    }, indent=2)


# ============================================================
# ==================  AGENT 3 TOOLS  =========================
# ML Report Writer
# ============================================================

def read_all_ml_outputs() -> str:
    """
    Load the complete picture of everything produced by the pipeline:
    pipeline state (goals, checkpoints, training results, best model),
    plus the contents of every readable file in the outputs/ folder.
    Call this FIRST to understand what to write about.
    Returns a nested JSON with 'pipeline_state' and 'output_files' sections.
    """
    state = load_state()

    summary = {
        "pipeline_state": {
            "user_goal": state.get("user_goal", ""),
            "status": state.get("status", ""),
            "pipeline_checkpoints": state.get("pipeline_checkpoints", {}),
            "ml_plan": state.get("ml_plan", {}),
            "training_results": state.get("training_results", []),
            "best_model_name": state.get("best_model_name", ""),
            "best_model_path": state.get("best_model_path", ""),
            "best_metrics": state.get("best_metrics", ""),
            "all_models_compared": state.get("all_models_compared", ""),
            "preprocessed_dataset_path": state.get("preprocessed_dataset_path", ""),
        },
        "output_files": {},
    }

    outputs_dir = get_outputs_dir()
    run_dir = get_run_dir()
    for search_dir in [outputs_dir, run_dir]:
        for f in sorted(search_dir.rglob("*")):
            if not f.is_file():
                continue
            try:
                rel = str(f.relative_to(PROJECT_ROOT))
            except ValueError:
                rel = str(f)
            if f.suffix in {".txt", ".md", ".json", ".log"}:
                try:
                    summary["output_files"][rel] = f.read_text(
                        encoding="utf-8", errors="ignore"
                    )[:6000]
                except Exception as e:
                    summary["output_files"][rel] = f"[Read error: {e}]"
            elif f.suffix in {".png", ".jpg", ".csv", ".joblib", ".pkl"}:
                summary["output_files"][rel] = (
                    f"[Binary/data file — {f.suffix} — {f.stat().st_size} bytes]"
                )

    return json.dumps(summary, indent=2, default=str)


def save_final_report(
    report_content: str,
    report_filename: str,
) -> str:
    """
    Save the final comprehensive Markdown report to outputs/.

    Args:
        report_content: Full Markdown report (minimum 2000 words).
        report_filename: E.g. 'final_ml_report_walmart_sales.md'.

    Returns:
        JSON confirmation or validation error.
    """
    placeholder_markers = [
        "(see above)", "(report above)", "(provided above)", "(the full report"
    ]
    if len(report_content) < 1000 or any(
        m in report_content.lower() for m in placeholder_markers
    ):
        return json.dumps({
            "error": "report_content is too short or contains a placeholder. "
                     "You MUST pass the full verbatim Markdown report — every section, every word.",
            "received_length": len(report_content),
            "hint": "Write the entire report inline, then pass it to this function.",
        })

    outputs_dir = get_outputs_dir()
    report_path = outputs_dir / report_filename
    try:
        report_path.write_text(report_content, encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": f"Could not write report: {e}"})

    save_state({
        "final_report_path": str(report_path),
        "final_report_filename": report_filename,
        "status": "pipeline_complete",
    })
    mark_checkpoint("final_report_complete")

    return json.dumps({
        "status": "success",
        "report_path": str(report_path),
        "report_length_chars": len(report_content),
        "message": "Pipeline complete! Final report saved.",
    }, indent=2)


# ============================================================
# ========  AGENT DEFINITIONS  ===============================
# ============================================================

from google.adk.agents import Agent, SequentialAgent

# ---- Agent 1: ML Strategy Planner ----
ml_strategy_planner = Agent(
    model=MODEL,
    name="ml_strategy_planner",
    description="Reads all prior pipeline outputs and drafts a comprehensive ML training plan.",
    output_key="ml_training_plan",
    instruction="""You are a world-class ML strategist with 20+ years of experience across every domain:
tabular ML, NLP, computer vision, time series, and deep learning.

YOUR TASK: Produce a comprehensive, data-specific ML training plan from all available pipeline context.

WORKFLOW:
1. Call get_full_pipeline_context() to see the pipeline state and prior work.
   - If pipeline_checkpoints contains 'ml_planning' AND status is 'ml_plan_ready':
     The plan already exists — output a short summary of the existing plan and STOP (do not re-plan).

2. Call read_outputs_folder() to read all output files from prior pipeline stages.

3. Call read_preprocessed_dataset_sample() to profile the data structure.

4. Using EVERYTHING gathered, write a DETAILED ML Training Plan in Markdown:

   # ML Training Plan: [Project Title derived from user_goal]

   ## 1. Problem Analysis
   - Problem type (classification / regression / clustering / time_series / NLP / CV)
   - Target variable analysis (distribution, class balance, cardinality)
   - Dataset size and dimensionality assessment
   - Key challenges specific to THIS dataset (reference actual column names and values)

   ## 2. Recommended Algorithms (ranked)
   - Baseline model — always start simple (LogisticRegression, LinearRegression, DecisionTree)
   - 3-5 main algorithms ranked by expected performance with clear reasoning
   - Ensemble / stacking strategy if warranted
   - Why each algorithm fits this specific problem and dataset

   ## 3. Evaluation Strategy
   - Primary metric with justification (accuracy, F1-macro, RMSE, MAE, AUC-ROC)
   - Secondary metrics
   - Cross-validation approach (k-fold, stratified k-fold, time-series split)
   - Data leakage prevention for this specific dataset

   ## 4. Feature Engineering & Selection
   - Additional features to derive from existing columns
   - Feature selection approach (correlation, importance-based, PCA if needed)
   - Dimensionality concerns

   ## 5. Hyperparameter Optimization
   - Search strategy and justification
   - Top 3-5 hyperparameters to tune per model
   - Search budget (iterations / time limit)

   ## 6. Training Strategy
   - Train / validation / test split ratios and rationale
   - Class imbalance handling strategy (if applicable)
   - Whether Colab GPU training is recommended and for which models
   - Estimated training time for each model

   ## 7. Model Persistence & Outputs
   - Save format (joblib for sklearn-compatible models)
   - Required output artifacts: model .joblib, metrics .json, comparison .csv, plots .png

   ## 8. Risks & Fallbacks
   - Top 3 risks (overfitting, data leakage, class imbalance) with concrete mitigations
   - Fallback plan if top-ranked models underperform

5. Call save_ml_training_plan() with:
   - plan_content: THE FULL VERBATIM MARKDOWN plan (every word of what you just wrote)
   - All other structured metadata fields populated from your plan

RULES:
- Every recommendation must reference specific columns, values, or statistics from the data profile
- Do NOT write generic ML advice — be concrete about THIS exact dataset
- plan_content must be comprehensive (minimum 1000 words)
- Never pass a placeholder or summary to save_ml_training_plan
""",
    tools=[
        get_full_pipeline_context,
        read_outputs_folder,
        read_preprocessed_dataset_sample,
        save_ml_training_plan,
    ],
)


# ---- Agent 2: ML Model Trainer ----
_trainer_tools = [
    get_ml_plan_for_trainer,
    get_ml_outputs_dir,
    start_colab_runtime,
    stop_colab_runtime,
    start_sandbox,
    stop_sandbox,
    run_in_sandbox,
    write_file_to_sandbox,
    read_file_from_sandbox,
    save_training_results,
    mark_training_complete,
]
if colab_mcp is not None:
    _trainer_tools.append(colab_mcp)

ml_model_trainer = Agent(
    model=MODEL,
    name="ml_model_trainer",
    description="Writes and executes ML training code; saves trained models and results.",
    output_key="ml_training_results",
    instruction="""You are a senior ML engineer who writes clean, reproducible training code.
You are expert in scikit-learn, XGBoost, LightGBM, CatBoost, and standard Python ML tooling.

YOUR TASK: Execute the ML training plan — train every planned model, evaluate rigorously, save all outputs.

WORKFLOW:
1. Call get_ml_plan_for_trainer() — load the plan and get the preprocessed dataset path.
2. Call get_ml_outputs_dir() — get ML_OUTPUTS_DIR where all models/plots will be saved.
3. Call start_sandbox() — initialize the Docker code execution environment.
   If Docker is unavailable (error in response), write scripts to disk and note the limitation.

4. FOR EACH model in the plan (baseline first, then advanced models):

   a. Write a complete self-contained Python training script containing:

      REQUIRED imports at the top:
        import pandas as pd, numpy as np, joblib, json, os, warnings
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pathlib import Path
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
        from sklearn.metrics import (classification_report, confusion_matrix,
                                      mean_squared_error, r2_score, roc_auc_score)
        warnings.filterwarnings('ignore')

      PATH CRITICAL — read carefully:
        get_ml_outputs_dir() returns TWO paths:
          "sandbox_ml_outputs_dir": "/workspace/ml_outputs"   ← USE THIS in sandbox scripts
          "ml_outputs_dir": "C:\\...\\modified_datasets\\...\\ml_outputs"  ← USE THIS for host tools

        In every training script that runs via run_in_sandbox(), set:
            ML_OUTPUTS_DIR = '/workspace/ml_outputs'
        NEVER use the Windows path (ml_outputs_dir) inside sandbox scripts.
        On Linux the backslash-separated Windows path is treated as a single
        filename component, creating a directory named 'C:\\...' inside /workspace.

      REQUIRED logic:
        - At the top of every training script:
            ML_OUTPUTS_DIR = '/workspace/ml_outputs'
            import os; os.makedirs(ML_OUTPUTS_DIR, exist_ok=True)
        - Load the preprocessed CSV — use /workspace/<filename>.csv for files in the run dir
        - Split 80/10/10 with random_state=42
        - Train the model
        - 5-fold cross-validation and print all scores
        - For classification: confusion matrix PNG, ROC curve PNG (binary only)
        - For regression: actual-vs-predicted PNG, residual plot PNG
        - Feature importance PNG if model supports it (tree-based models)
        - Save model:
            # joblib is safe here — models are saved by this pipeline and loaded
            # only within the same run from our own ml_outputs/ directory.
            # Never load joblib files from untrusted / external sources.
            joblib.dump(model, os.path.join(ML_OUTPUTS_DIR, 'ModelName.joblib'))
        - Save metrics: json.dump(metrics, open(os.path.join(ML_OUTPUTS_DIR, 'metrics_ModelName.json'),'w'))
        - Print a clear metrics summary at the end

   b. Call write_file_to_sandbox('train_ModelName.py', script_code)
   c. Call run_in_sandbox('python train_ModelName.py')
   d. If execution fails, examine the error, fix it, and retry ONCE before moving on.
   e. Call save_training_results() — for model_file_path use the HOST path:
        host_ml_outputs_dir + '/ModelName.joblib'
      where host_ml_outputs_dir is the 'ml_outputs_dir' field from get_ml_outputs_dir().

5. Write and run a comparison script that:
   - Sets ML_OUTPUTS_DIR = '/workspace/ml_outputs'
   - Reads all metrics_*.json files from ML_OUTPUTS_DIR
   - Ranks models by the primary metric from the plan
   - Saves: os.path.join(ML_OUTPUTS_DIR, 'model_comparison.csv')

6. Call mark_training_complete() with the best model details.
   For best_model_path use the HOST path: ml_outputs_dir + '/ModelName.joblib'
7. Call stop_sandbox() to free resources.

CODE REQUIREMENTS:
- Always random_state=42 for reproducibility
- Always matplotlib.use('Agg') before any other matplotlib operation
- Save models with joblib.dump, never pickle
- Wrap each model block in try/except so one failure does not stop others
- In sandbox scripts always use sandbox_ml_outputs_dir ('/workspace/ml_outputs'), never the Windows ml_outputs_dir

COLAB LIFECYCLE (GPU sessions cost free quota — manage carefully):
- For standard ML (scikit-learn, XGBoost, LightGBM): use local Docker sandbox only. Do NOT start Colab.
- For GPU-needed models (neural networks, large deep learning): follow this exact sequence:
    1. Call start_colab_runtime() — starts the bridge. First run opens a browser for Google login.
    2. Wait for "started" status, then proceed with colab_mcp execute_code tools for training.
    3. When ALL GPU training is done, call stop_colab_runtime() IMMEDIATELY to free GPU hours.
       This is critical — idle Colab GPU sessions still drain the free quota.
    4. Then call mark_training_complete() to finish the phase.
- If start_colab_runtime() returns "skipped" (COLAB_MCP_DIR not configured), fall back to sandbox.
""",
    tools=_trainer_tools,
)


# ---- Agent 3: ML Report Writer ----
ml_report_writer = Agent(
    model=MODEL,
    name="ml_report_writer",
    description="Reads all pipeline outputs and writes the comprehensive final project report.",
    output_key="final_report",
    instruction="""You are an expert technical writer who bridges ML engineering and business communication.
Your reports are readable by both data scientists (technical depth) and stakeholders (plain English).

YOUR TASK: Write the definitive final report covering the entire ML project end-to-end.

WORKFLOW:
1. Call read_all_ml_outputs() — load every output file and the full pipeline state.

2. Write a COMPREHENSIVE FINAL REPORT in Markdown using this exact structure:

---
# ML Pipeline Final Report: [Derive title from user_goal]
*Report Generated: [Today's date]*

## Executive Summary
3-5 sentences a non-technical manager can understand: what was built, what was found,
the main result, and its value. Include the top metric in plain English
(e.g., "The model correctly identifies 87 out of every 100 cases, which means...").

## 1. Project Overview
- What the user wanted to accomplish (use exact user_goal language)
- Why machine learning is the right approach
- Success criteria and how they were met

## 2. Data Journey
- Dataset used (name, source, original size, key columns)
- Quality issues found in the raw data
- How preprocessing transformed it (summarise from the preprocessing reports)
- Final dataset shape and characteristics going into ML training

## 3. ML Strategy Chosen
- Problem type identified and evidence from the data that confirmed it
- Why each algorithm was selected (connect to the specific data profile)
- Evaluation approach chosen and justification

## 4. Model Training Results
For EACH model trained, create a subsection:
### [Model Name]
- Plain-English explanation of what this algorithm does (2 sentences)
- Hyperparameters used and why
- Cross-validation scores (mean ± std)
- Test set metrics — explain EACH metric in plain English
- Model's strengths and weaknesses on this specific dataset

## 5. Model Comparison & Winner
| Model | Primary Metric | Notes |
|-------|---------------|-------|
| ...   | ...            | ...   |

- Which model won, with specific reasoning backed by actual numbers
- What the feature importances reveal about the data and the problem
- Any surprising or unexpected findings

## 6. Complete File Inventory
| File Path | Description |
|-----------|-------------|
| outputs/ml_outputs/ModelName.joblib | Trained model file |
| outputs/ml_outputs/metrics_ModelName.json | Evaluation metrics |
| outputs/ml_outputs/confusion_matrix_ModelName.png | Performance visualization |
| ... | ... |

How to load and use the best model:
```python
import joblib, pandas as pd

# Safe to load: this file was saved by this pipeline from our own training code.
# Do NOT load joblib files from untrusted/external sources (equivalent to pickle).
model = joblib.load('outputs/ml_outputs/<BestModelName>.joblib')
# new_data must have the same columns as the training data, preprocessed the same way
predictions = model.predict(new_data)
```

## 7. Limitations & Next Steps
- Honest limitations of the current model
- Top 3 improvements ranked by expected impact
- Deployment checklist (what is needed before production use)

## 8. Conclusion
- End-to-end summary: from raw data to working model
- Value delivered in plain language
- Concrete next action items for the user
---

3. Call save_final_report() with:
   - report_content: THE FULL VERBATIM MARKDOWN REPORT you just wrote (every section, every word)
   - report_filename: 'final_ml_report_<project_slug>.md'

RULES:
- Minimum 2000 words — comprehensive means comprehensive
- Explain every metric in plain English; never just print a number without context
- If training results are missing, say so honestly and describe what would be there
- Never pass a placeholder or "(see above)" reference to save_final_report
""",
    tools=[
        read_all_ml_outputs,
        save_final_report,
    ],
)


# ---- ML Research Agent (runs first, scoped to this pipeline) ----------------

from tavily import TavilyClient as _TavilyClient
_tavily_ml = _TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))



def _ml_search(query: str, domain_filter: str = "all") -> str:
    """
    Search for SOTA models, hyperparameters, benchmarks, and evaluation best practices.

    Args:
        query: e.g. 'XGBoost hyperparameter tuning tabular classification 2025'
        domain_filter: comma-separated domains or 'all'.
            e.g. 'paperswithcode.com,arxiv.org' or 'xgboost.readthedocs.io'
    Returns:
        JSON list with title, url, snippet per result.
    """
    try:
        kwargs: dict = {"query": query, "search_depth": "advanced", "max_results": 5}
        if domain_filter and domain_filter.lower() != "all":
            kwargs["include_domains"] = [d.strip() for d in domain_filter.split(",")]
        resp = _tavily_ml.search(**kwargs)
        return json.dumps([
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:600]}
            for r in resp.get("results", [])
        ], indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _ml_save_research(
    problem_type: str,
    ranked_models: str,
    hyperparameter_ranges: str,
    evaluation_strategy: str,
    sota_benchmarks: str,
    training_tips: str,
    warnings: str,
) -> str:
    """
    Save ML research so the Strategy Planner (next agent) uses it immediately.

    Args:
        problem_type: Confirmed type with evidence (e.g. 'binary classification, imbalanced 80/20').
        ranked_models: Models ranked by expected fit, each with one-line justification.
        hyperparameter_ranges: Specific numeric ranges per model
            (e.g. 'XGBoost: n_estimators=100-1000, max_depth=3-8, lr=0.01-0.3').
        evaluation_strategy: Metric, CV method, split ratios
            (e.g. 'AUC-ROC primary; StratifiedKFold(5); 80/10/10 split').
        sota_benchmarks: Real performance targets from papers or competitions.
        training_tips: Code-level tips for the trainer
            (e.g. 'set early_stopping_rounds=50; use scale_pos_weight for imbalance').
        warnings: Overfitting risks, metric pitfalls, leakage risks.
    Returns:
        Confirmation JSON.
    """
    save_state({
        "ml_training_research": {
            "problem_type": problem_type,
            "ranked_models": ranked_models,
            "hyperparameter_ranges": hyperparameter_ranges,
            "evaluation_strategy": evaluation_strategy,
            "sota_benchmarks": sota_benchmarks,
            "training_tips": training_tips,
            "warnings": warnings,
        }
    })
    return json.dumps({"status": "saved"}, indent=2)


_ml_research_tools = [get_full_pipeline_context, _ml_search, _ml_save_research]
try:
    from mcp_servers.mcp_servers import tavily_mcp as _ml_tavily_mcp
    _ml_research_tools.append(_ml_tavily_mcp)
except Exception:
    pass

_ml_research_agent = Agent(
    model="gemini-3.1-flash-lite",
    name="ml_research_agent",
    description="Researches SOTA models, hyperparameters, and evaluation strategies before the strategy planner runs.",
    output_key="ml_research_output",
    instruction="""You are the ML Research Agent — the FIRST agent in the ML training pipeline.

PIPELINE CONTEXT
You live inside a SequentialAgent:
  [You] → ML Strategy Planner → ML Model Trainer → ML Report Writer

The ML Strategy Planner reads your findings from pipeline_state.json immediately after you.
Give it SPECIFIC numbers and ranked recommendations — it will build the training plan from them.

YOUR TASK

1. Call get_full_pipeline_context() to load the full project state, preprocessed data profile,
   and user constraints (compute, timeline, skill level from qa_pairs).

2. Confirm the exact problem type from the preprocessed dataset:
   • Binary / multi-class classification vs regression vs clustering vs time-series
   • Dataset size and dimensionality (affects model capacity)
   • Class balance ratio (critical for metric and strategy choice)

3. Research SOTA for this exact task — use ALL available search tools:
   _ml_search('state of the art <problem_type> <domain> 2025 benchmark', domain_filter='paperswithcode.com,arxiv.org')
   _ml_search('best model tabular <task> kaggle winning solution 2024 2025')
   Google Search: 'site:paperswithcode.com <task> <domain> leaderboard'

4. Research hyperparameters for the top 3-5 candidate models:
   _ml_search('XGBoostClassifier best hyperparameters <task> 2025 optuna')
   _ml_search('LightGBM num_leaves learning_rate tuning tabular 2025')
   _ml_search('sklearn RandomForestClassifier hyperparameter grid 2025')
   Find ACTUAL NUMERIC RANGES — not vague guidance.

5. Research evaluation best practices:
   _ml_search('evaluation metrics <problem_type> sklearn pitfalls 2025')
   • Which metric is correct for this task and class distribution?
   • StratifiedKFold vs KFold vs TimeSeriesSplit?
   • Common mistakes (e.g. accuracy on imbalanced data, AUC vs F1 trade-off)

6. Call _ml_save_research() with ALL fields:
   • ranked_models: ordered list with one-line evidence per model
   • hyperparameter_ranges: NUMBERS (e.g. 'max_depth: 3-10, lr: 0.01-0.3')
   • evaluation_strategy: exact metric name, CV method, n_splits
   • training_tips: code-level instructions the trainer agent can use directly

RULES
- Hyperparameter ranges must be numeric — never 'tune appropriately'
- Every model ranking must cite a reason tied to THIS dataset's characteristics
- Use search to get 2024/2025 info — your training data may be outdated
- sota_benchmarks must be real numbers from actual papers or Kaggle leaderboards
""",
    tools=_ml_research_tools,
)


# ---- Root Sequential Agent ----
root_agent = SequentialAgent(
    name="ml_pipeline_agent",
    description=(
        "End-to-end ML pipeline: "
        "research → strategy planning → model training → final report generation."
    ),
    sub_agents=[
        _ml_research_agent,
        ml_strategy_planner,
        ml_model_trainer,
        ml_report_writer,
    ],
)


# ============================================================
# ========  STANDALONE RUNNER  ================================
# ============================================================

async def run_ml_pipeline() -> None:
    """Run the complete ML agent pipeline as a standalone script."""
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types

    state = load_state()
    status = state.get("status", "empty")
    print(f"\n[ML PIPELINE] Current pipeline status: {status}", flush=True)

    if status not in {
        "preprocessing_complete", "ml_plan_ready",
        "model_trained", "ml_complete",
    }:
        print(
            "[ML PIPELINE] WARNING: Data preprocessing may not be complete.\n"
            "Run master_orchestrator/agent.py first for best results.\n"
            "Proceeding with whatever data is available...\n",
            flush=True,
        )

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="ml_pipeline_agent",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="ml_pipeline_agent",
        user_id="user",
    )

    user_goal = state.get("user_goal", "Train the best ML model for the available dataset.")
    content = types.Content(
        role="user",
        parts=[types.Part(
            text=(
                f"Run the complete ML pipeline for this project: {user_goal}\n\n"
                "Execute all three phases: planning, training, and reporting."
            )
        )],
    )

    print("[ML PIPELINE] Starting...\n" + "=" * 60, flush=True)

    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=content,
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text, flush=True)

    final_state = load_state()
    print("\n[ML PIPELINE] Done!", flush=True)
    if final_state.get("final_report_path"):
        print(f"[ML PIPELINE] Final report → {final_state['final_report_path']}", flush=True)
    if final_state.get("best_model_path"):
        print(f"[ML PIPELINE] Best model  → {final_state['best_model_path']}", flush=True)


if __name__ == "__main__":
    asyncio.run(run_ml_pipeline())
