# Prompt2ML

**From a one-sentence ML project idea to a preprocessed, ML-ready dataset — automatically.**

Prompt2ML is a multi-agent pipeline built on [Google ADK](https://github.com/google/adk-python) and Gemini. You tell it what you want to build (e.g. *"I want to build a stock market prediction system"*) and it walks you through a structured interview, finds and downloads relevant datasets, profiles them, builds a preprocessing plan, executes that plan inside a sandboxed environment, validates the result in a retry loop, and finally writes a comprehensive report of every transformation it applied.

You bring the idea. Prompt2ML brings it to the point where you can start training a model.

---

## Why this exists

Most ML projects die in the first 48 hours — not because the model is hard, but because the *data preparation* is tedious:

- You don't know which dataset to use
- You download three and none of them match your problem
- You spend an afternoon writing pandas code to handle nulls, encode categoricals, fix dtypes
- You realize halfway through that you made the wrong decision in step 2
- You give up

Prompt2ML automates the part between "I have an idea" and "I have clean data in a CSV". It's not a black box — every decision is logged, every transformation is justified in a final markdown report, and the intermediate state is persisted so you can inspect what happened.

---

## Architecture at a glance

```text
┌─────────────────────────────────────────────────────────────┐
│                    MASTER ORCHESTRATOR                      │
│               (single entry point, 3 phases)                │
└─────────────────────────────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   PHASE 1    │  │   PHASE 2    │  │   PHASE 3    │
    │ Requirement  │  │   Dataset    │  │     Data     │
    │  Gatherer    │  │  Extractor   │  │ Preprocessing│
    │              │  │              │  │              │
    │ 7 questions  │  │ Kaggle + HF  │  │  5 agents in │
    │ → 10-section │  │ search &     │  │  a sequential│
    │   report     │  │ download     │  │  pipeline    │
    └──────────────┘  └──────────────┘  └──────────────┘
                                               │
                 ┌─────────────────────────────┤
                 │                             │
                 ▼                             ▼
        ┌────────────────┐           ┌──────────────────┐
        │    Agent 1     │           │     Agent 5      │
        │    Dataset     │           │  Report Writer   │
        │    Analyzer    │           │  (A-to-Z .md)    │
        └────────────────┘           └──────────────────┘
                 │                             ▲
                 ▼                             │
        ┌────────────────┐           ┌──────────────────┐
        │    Agent 2     │           │    LoopAgent     │
        │ Preprocessing  │──────────▶│ (max 5 retries)  │
        │  Strategist    │           └──────────────────┘
        └────────────────┘             │             ▲
                                       ▼             │
                              ┌────────────┐  ┌────────────┐
                              │  Agent 3   │  │  Agent 4   │
                              │ Executor   │──│ Validator  │
                              │ (sandbox)  │  │ PASS/FAIL  │
                              └────────────┘  └────────────┘
```

### The three phases

| Phase | Agent | Mode | What it does |
|---|---|---|---|
| **1** | `requirement_gatherer_agent` | Interactive Q&A | Asks 7 follow-up questions about your goal, then generates a 10-section project report (~1500+ words) |
| **2** | `dataset_extractor_agent` | Autonomous | Searches Kaggle and HuggingFace via Tavily, picks the most relevant datasets, downloads them |
| **3** | `data_preprocessing_agent` | Autonomous (5 sub-agents) | Profiles the dataset, plans the preprocessing, executes it in a sandbox, validates with retry loop, writes a final markdown report |

### The 5 preprocessing sub-agents

| # | Agent | Role |
|---|---|---|
| 1 | `dataset_analyzer_agent` | Scans the `datasets/` folder, picks the best file, extracts schema + sample rows |
| 2 | `preprocessing_strategist_agent` | Builds a step-by-step preprocessing plan (what to drop, encode, scale, impute, engineer) |
| 3 | `preprocessing_executor_agent` | Executes the plan using 13 built-in preprocessing tools + 7 OpenSandbox tools for custom code |
| 4 | `validation_agent` | Runs quality checks, compares before/after, issues a PASS or FAIL verdict with specific feedback |
| 5 | `report_generator_agent` | Writes an exhaustive markdown report documenting every transformation |

**The Agent 3 ↔ Agent 4 retry loop** is the clever part. If validation fails, Agent 4's feedback goes back to Agent 3, which fixes the specific issues and re-runs. The loop caps at 5 iterations via a custom `LoopEscalationChecker`.

---

## How the pipeline stays coordinated

Agents in Prompt2ML don't talk to each other through ADK's in-memory session — they communicate through a single **shared JSON file**: `pipeline_state.json` at the project root. Each tool that finishes successfully writes its output under a unique key plus a `status` marker:

```python
save_state({
    "agent1_output": {...},
    "selected_dataset_path": "...",
    "status": "agent1_done",
})
```

**Why a file instead of in-memory state?**

- **Resumable.** Kill the pipeline mid-run, restart it an hour later, and it picks up exactly where it left off. Phase 1 and Phase 2 skip if their artifacts already exist.
- **Inspectable.** You can open `pipeline_state.json` in any editor to see exactly what each agent produced. No hidden state, no debugger required.
- **Cross-process.** Each phase can be run from its own process without sharing Python memory.

Every tool write is a `dict.update()` merge — unique keys accumulate, the shared `status` key overwrites with the current phase marker. See [pipeline_state.py](pipeline_state.py) for the full 28-line implementation.

---

## Project layout

```text
Prompt2ML/
├── master_orchestrator/
│   ├── agent.py              # Single entry point — runs all 3 phases
│   └── __init__.py
├── data_extractor_agent/
│   ├── agent.py              # Phase 2 — Kaggle + HuggingFace dataset search/download
│   └── __init__.py
├── data_preprocessing_agent/
│   ├── agent.py              # Phase 3 — 5 sub-agents, 13 preprocessing tools, loop logic
│   ├── sandbox_executor.py   # OpenSandbox async wrapper (start/stop, upload/download, run)
│   └── __init__.py
├── mcp_servers/
│   └── mcp_servers.py        # Tavily + HuggingFace MCP toolset definitions (currently commented out)
├── pipeline_state.py         # Shared JSON state bus (load_state / save_state)
├── pipeline_state.json       # Live state file — written by every agent
├── datasets/                 # Downloaded raw datasets (created by Phase 2)
├── modified_datasets/        # Preprocessed output files (created by Phase 3)
├── outputs/                  # Phase 1 requirement reports + Phase 3 final reports
├── requirements.txt
└── .env                      # API keys (see below)
```

---

## Setup

### 1. Clone and install Python dependencies

```bash
git clone <your-repo-url>
cd Prompt2ML
pip install -r requirements.txt
pip install opensandbox opensandbox-server opensandbox-code-interpreter
```

### 2. Create a `.env` file at the project root

```env
GOOGLE_API_KEY=your_gemini_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
HF_TOKEN=your_huggingface_token
TAVILY_API_KEY=your_tavily_api_key
```

- **Gemini API key:** <https://aistudio.google.com/apikey>
- **Kaggle API key:** <https://www.kaggle.com/settings> → "Create New Token" (downloads `kaggle.json`)
- **HuggingFace token:** <https://huggingface.co/settings/tokens>
- **Tavily API key:** <https://tavily.com/> (used by the dataset extractor for search)

### 3. Start Docker and pull the sandbox image

Phase 3's executor agent runs preprocessing code inside an isolated Docker container managed by OpenSandbox. Make sure Docker Desktop is running, then:

```bash
docker pull opensandbox/code-interpreter:v1.0.2
```

### 4. Initialize and start the OpenSandbox server

```bash
opensandbox-server init-config ~/.sandbox.toml --example docker
opensandbox-server
```

Leave this terminal running. The preprocessing agent will talk to it at `localhost:8080`.

### 5. (Windows only) Fix the docker-py bug

On Windows you may hit a `heb_timeout` error caused by a `docker-py` + `urllib3` version mismatch. Fix it once:

```bash
pip install --force-reinstall "docker==7.1.0" "urllib3<2" "requests<2.32"
```

---

## Running the pipeline

With OpenSandbox running in one terminal, open a second terminal and run:

```bash
python master_orchestrator/agent.py
```

### What you'll see

```text
============================================================
  PROMPT2ML — Phase 1: Requirement Gathering
============================================================

Please tell me what you want to build: I want to build a stock market prediction system

[Agent]: Great! I'd love to help you build this. First question:
         What kind of data do you have available — historical prices,
         trading volumes, news articles, economic indicators?

You: historical stock prices, I'm not sure of the source
... (6 more questions)

[PIPELINE] Phase 1 complete — report saved!

============================================================
  PROMPT2ML — Phase 2: Dataset Extraction
============================================================

[PIPELINE] Searching and downloading datasets...
[SUCCESS] Downloaded Kaggle dataset 'hammadansari7/google-stock-market-dataset-20252026'

[PIPELINE] Phase 2 complete — datasets downloaded!

============================================================
  PROMPT2ML — Phase 3: Dataset Preprocessing
============================================================

[PIPELINE] Preprocessing datasets...
[Agent 1] Selected: GOOG.csv (2517 rows × 7 columns)
[Agent 2] Plan: drop_columns → handle_missing → parse_dates → feature_engineering → scale
[SANDBOX] Started sandbox container
[Agent 3] Executed 8 preprocessing steps, downloaded result
[Agent 4] VERDICT: PASS (quality score: 9/10)
[Agent 5] Report saved to outputs/preprocessing_report.md

[PIPELINE] Phase 3 complete — datasets preprocessed!
```

### Resuming a killed run

Because the pipeline state lives on disk, you can kill the process at any point (Ctrl+C) and restart it — completed phases get skipped automatically based on what's already in `pipeline_state.json`. This is controlled by the status-check blocks in [master_orchestrator/agent.py](master_orchestrator/agent.py).

### Starting fresh

To force a completely fresh run, delete these:

```bash
rm pipeline_state.json
rm -rf datasets/ modified_datasets/ outputs/
```

---

## Where things end up

| Folder | Contents |
|---|---|
| `datasets/` | Raw datasets downloaded by Phase 2 (Kaggle/HF) |
| `modified_datasets/` | Intermediate + final preprocessed CSVs from Phase 3 |
| `outputs/` | Phase 1 requirement reports (`.txt`) and Phase 3 preprocessing reports (`.md`) |
| `pipeline_state.json` | The live state bus — inspect this to see what each agent produced |

---

## The 13 built-in preprocessing tools (Agent 3)

These are Python functions the executor agent can call directly — no LLM-generated code, no sandbox needed for standard operations:

| Tool | Purpose |
|---|---|
| `handle_missing_values` | Fill/drop nulls with column-specific strategies (mean, median, mode, ffill, interpolate) |
| `remove_duplicates` | De-duplicate rows |
| `drop_columns` | Remove unused columns |
| `encode_categorical_columns` | One-hot, label, target, or binary encoding |
| `scale_numeric_columns` | StandardScaler, MinMaxScaler, RobustScaler |
| `handle_outliers` | IQR or z-score based capping/removal |
| `parse_datetime_columns` | Extract year, month, day, weekday features |
| `engineer_features` | Arithmetic, ratio, interaction features |
| `process_text_columns` | Length, word count, TF-IDF, keyword flags, hashing |
| `detect_and_fix_data_types` | Auto-coerce strings to numeric/datetime |
| `validate_dataset` | Assert post-conditions (no nulls, no dups, shape, types) |
| `save_preprocessed_output` | Copy final file to `modified_datasets/` and update state |
| `get_preprocessing_context` | Read the plan + retry feedback from state |

### OpenSandbox tools (for custom Python)

When a preprocessing step is too complex for the built-in tools, Agent 3 writes a Python script into the sandbox and runs it there. Seven async helpers manage this:

`start_sandbox`, `stop_sandbox`, `upload_dataset_to_sandbox`, `run_in_sandbox`, `write_file_to_sandbox`, `read_file_from_sandbox`, `download_from_sandbox`

See [data_preprocessing_agent/sandbox_executor.py](data_preprocessing_agent/sandbox_executor.py).

---

## Design decisions worth knowing

### Why file-based state instead of ADK's in-memory session?

ADK's `InMemorySessionService` lives in Python process memory. Kill the process, lose the state. Prompt2ML's pipeline has:

- A conversational Phase 1 that takes 5–15 minutes of user thinking
- A download Phase 2 that can hit network errors
- A preprocessing loop that may iterate 5 times

Surviving a mid-run crash is essential, so state lives on disk.

### Why OpenSandbox instead of ADK's `BuiltInCodeExecutor`?

Tried it — ADK's `BuiltInCodeExecutor` is a **server-side Gemini tool** that can't be mixed with regular function-calling tools in the same agent. That causes a `400 INVALID_ARGUMENT` error. OpenSandbox tools are just async Python functions, so they mix freely with the 13 built-in preprocessing tools. Bonus: actual Docker isolation instead of Gemini's opaque server-side executor.

### Why is Agent 3's tool set so big (20 tools)?

Deliberate. Standard preprocessing operations (null handling, encoding, scaling) are deterministic and don't need LLM-generated code — they need a well-named Python function. Reserve code generation for the 20% of cases that are genuinely custom. This makes the pipeline cheaper (fewer LLM tokens), more reliable (less hallucination surface area), and more auditable (you can read the exact function that ran).

### Why does the requirement gatherer ask exactly 7 questions?

Fewer than 5 and the report is too shallow to guide dataset selection. More than 10 and users abandon mid-interview. Seven is the sweet spot from iteration.

---

## Current status

- ✅ Phase 1 — Requirement gatherer (interactive, 7 questions, 10-section report)
- ✅ Phase 2 — Kaggle + HuggingFace dataset search/download via Tavily
- ✅ Phase 3 — 5-agent preprocessing pipeline with sandbox execution and retry loop
- ⏳ Phase 4 — Model training & evaluation (planned)
- ⏳ Phase 5 — Model deployment / serving (planned)

MCP toolsets for Tavily and HuggingFace (`mcp_servers/mcp_servers.py`) are implemented but currently commented out in the extractor agent — the direct API calls are more reliable for now.

---

## Tech stack

- **Agents:** Google Agent Development Kit (ADK) with `SequentialAgent`, `LoopAgent`, custom `BaseAgent`
- **LLMs:** Gemini 2.5 Flash (Phase 1) and Gemini 3.1 Flash Lite Preview (Phases 2 and 3)
- **Dataset sources:** Kaggle (`kagglehub`), HuggingFace (`datasets`), Tavily search API
- **Sandbox:** [Alibaba OpenSandbox](https://github.com/alibaba/OpenSandbox) with `opensandbox/code-interpreter:v1.0.2`
- **Data processing:** pandas, numpy, scikit-learn, scipy (inside the sandbox)
- **State bus:** plain JSON file + merge-update semantics (no database)

---

## Troubleshooting

**`heb_timeout` error on Windows** → `pip install --force-reinstall "docker==7.1.0" "urllib3<2" "requests<2.32"` (see Setup step 5).

**`503: DOCKER::INITIALIZATION_ERROR`** → Docker Desktop isn't running, or `opensandbox-server` was started before Docker was ready. Start Docker first, then restart the server.

**`400 INVALID_ARGUMENT` from Gemini** → You probably mixed `google_search` or `BuiltInCodeExecutor` with function-calling tools. Remove the server-side tool — they can't coexist in the same agent.

**Pipeline re-asks the 7 questions after a completed run** → Status-based resume checks in `master_orchestrator/agent.py` are strict. Either delete `pipeline_state.json` to start fresh, or update Phase 1/2 checks to use artifact-based resume (check for `user_goal` and `downloaded_dataset` presence instead of exact status strings).

**`[Errno 2] No such file or directory: '/workspace/output/...'`** → The executor agent handed a sandbox path to a host-side built-in tool. The `_resolve_local_path()` helper in [data_preprocessing_agent/agent.py](data_preprocessing_agent/agent.py) should catch this with a clear error message directing the fix.

---

## License

Add your license of choice here (MIT, Apache 2.0, etc.).

## Acknowledgments

Built on [Google ADK](https://github.com/google/adk-python), [Alibaba OpenSandbox](https://github.com/alibaba/OpenSandbox), [Gemini](https://ai.google.dev/), [Kaggle](https://www.kaggle.com/), [HuggingFace Datasets](https://huggingface.co/datasets), and [Tavily](https://tavily.com/).
