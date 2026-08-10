# Prompt2ML — Implementation Plan

Target state: a Windows-first CLI that takes a plain-English project idea and produces a
trained model (classical ML **or** deep learning, across tabular / text / vision), its
evaluation plots, the exact dataset it was trained on, and a full set of reports — with all
data and compute local, and only the LLM behind an API.

This document is the working plan. It is ordered by dependency, not by excitement:
foundations first, because DL on top of a leaky preprocessing stage produces confident
garbage.

---

## 0. Design decisions that drive everything else

Six decisions shape the rest of the plan. They're stated up front so the milestones read as
consequences rather than a list of chores.

### D1 — Modality is a first-class routing key, not a prompt hint

Today `datasets/` is scanned for CSVs and everything downstream assumes a rectangular
DataFrame. CV/NLP support is not a feature you can bolt onto that. Instead:

`DatasetManifest` (produced by extraction) declares a **modality** — `tabular`, `text`,
`vision`, `audio`, `timeseries` — plus on-disk layout, split info, and where labels live.
The orchestrator uses it to select a *pipeline*, and each pipeline owns its own preprocessing
tools and trainer recipe. Shared plumbing (run state, research, codegen gate, reporting) is
modality-agnostic.

### D2 — A typed run contract replaces the global state blob

`pipeline_state.json` is one file, one `status` string with two meanings, and a
list-or-dict ambiguity handled by a `_latest()` helper. It also makes concurrent runs
impossible and grows without bound (it stores every generated script inline).

Replace it with a per-run directory of small, typed, validated documents:

```
runs/<run_id>/
  run.json               # phases, statuses, timings, checkpoints
  requirements.json      # structured facts from the interview
  dataset_manifest.json  # modality, files, splits, label location
  data_contract.json     # features, target, dtypes, splits, deferred transforms
  plan.json              # training plan
  hardware.json          # capability profile + enforced budgets
  logs/                  # per-agent transcripts, generated scripts
  artifacts/             # model, plots, final dataset, reports
```

Each document is a Pydantic model. Writes are atomic + file-locked. Phase status lives in
`run.json` per phase, so `"pipeline_complete"` can never mean two different things.

### D3 — Research becomes a tool, not a phase

Right now each phase has a research agent that runs once, up front, and dumps prose into
state. The main-flow agents can't ask follow-up questions when they actually hit the
uncertainty.

Wrap the researcher as an **ADK `AgentTool`** — `research(question, context)` — and give it
to the interviewer, the strategist, the planner, and the trainer. Back it with a disk cache
keyed by normalized query so repeated runs don't re-pay Tavily latency or cost.

**On A2A:** not recommended yet. A2A buys you cross-process/cross-machine agent boundaries,
which this system doesn't have — everything runs in one CLI process on one PC. It would add
a transport, a server lifecycle, and a failure mode for zero capability gain today. Revisit
only if research moves to a persistent shared service across runs. `AgentTool` gets you the
delegation semantics you're describing with none of that.

### D4 — Generated code passes a gate before it runs

"Code should be high quality like a human's" becomes four mechanical steps, because prompt
instructions alone don't hold:

1. **Grounding** — the generator gets pinned sandbox library versions (already exists) plus
   real fetched docs for the specific APIs it intends to use, via a cached `ask_docs` tool.
2. **Static gate (host, pre-execution)** — `ast.parse` → `ruff` → a project-specific
   **leakage linter** (see D5).
3. **Smoke run** — execute against a sampled subset (200 rows, 1 epoch, `fast_dev_run`)
   before committing to the full run. Catches ~most crashes at ~1% of the cost.
4. **Repair loop** — failures return structured diagnostics; capped retries; every attempt
   logged to `logs/`.

### D5 — Leakage is enforced by a linter, not by instructions

The current system correctly defers scaling and target encoding out of preprocessing — but
imputation, frequency encoding, and outlier bounds are still fit on the full dataset, and the
ML validator cannot detect that class of leakage. Instructions didn't hold; a check will.

`leakage_lint` is an AST pass over every generated training script asserting:

- no `.fit(` / `.fit_transform(` on a frame before the split call
- every transformer appears inside a `Pipeline` / `ColumnTransformer`
- `cross_val_score` / `*SearchCV` receives the pipeline, never a bare estimator
- `random_state=` present on every split and every stochastic estimator
- time-series runs never call `shuffle=True`

Same idea applied host-side: preprocessing tools that currently fit statistics on the whole
frame either move into the training pipeline or record themselves as `deferred` in
`data_contract.json`.

### D6 — Hardware profile is a budget the planner must obey

A scan that only *informs* the prompt gets ignored. `hardware.json` carries both the raw
profile and a derived, **enforced** budget:

```json
{
  "gpu": {"name": "RTX 4060", "vram_gb": 8, "cuda": "12.4", "docker_passthrough": true},
  "cpu_cores": 16, "ram_gb": 32, "disk_free_gb": 210,
  "budget": {
    "max_params_m": 200, "max_batch_tokens": 8192, "precision": "fp16",
    "max_train_minutes": 45, "allow_full_finetune": true, "prefer_lora": false
  }
}
```

The planner reads it; the codegen gate **rejects** scripts that exceed it (batch × seq_len,
model size, epoch count). That turns "select the GPU" into a constraint the system can't
silently violate.

**Windows GPU reality:** GPU-in-Docker on Windows needs the WSL2 backend + NVIDIA Container
Toolkit. `doctor` probes it explicitly by running a throwaway `--gpus all` container. If it
fails but a GPU is present, **host venv** training is offered with explicit consent (fast,
not isolated — the tradeoff is stated plainly, never assumed). If there's no usable local
GPU at all, the run falls through to the remote tier described in D7.

### D7 — Execution is a swappable backend, and remote GPU is a first-class tier

The current sandbox contract is "a host directory volume-mounted as `/workspace` in a local
container." Remote GPU breaks that assumption — the workspace is no longer local — so
execution has to move behind an interface:

```python
class ExecutionBackend(Protocol):
    def capabilities(self) -> Capabilities   # gpu, vram_gb, $/hr, isolation, egress
    def prepare(self, workspace: Path) -> Handle
    def put(self, files: list[Path]) -> None
    def run(self, cmd: str, timeout_s: int) -> ExecResult
    def get(self, remote_paths: list[str], dest: Path) -> None
    def teardown(self, disposition: Disposition) -> None   # PAUSE | DESTROY
```

Five implementations:

| Backend | Mechanism | Isolation | Data leaves PC | Notes |
|---|---|---|---|---|
| `LocalDocker` | existing volume mount | container | no | default; GPU via `--gpus all` when probed OK |
| `LocalVenv` | host Python + native CUDA | **none** | no | consented fallback when passthrough fails |
| `JarvisLabs` | `jl` CLI | instance | **yes** | managed-job model fits our flow closely |
| `RunPod` | `runpod` Python SDK | pod | **yes** | runs *our* image → best version parity |
| `Colab` | existing bridge | notebook | **yes** | last resort; currently half-wired |

Everything upstream is unchanged: the codegen gate, leakage lint, and smoke run all operate
on the script *before* it reaches a backend, so they apply identically whether the job runs
in a local container or on an H100.

**Provider notes (APIs verified, not assumed):**

*JarvisLabs* exposes a managed-job abstraction that maps almost 1:1 onto our needs:
`jl create --gpu <type> --storage <gb> --yes --json` → `jl run . --script train.py --on <id>
--requirements requirements.txt --json --yes` → poll `jl run logs <run_id> --tail 50` →
`jl download <id> /home/results ./results -r` → `jl pause|destroy <id> --yes --json`.
Three constraints shape the adapter: always pass `--json` (otherwise the CLI blocks
streaming logs), never `--follow`, and `machine_id` **can change after `resume`** — so the
lease record stores whatever ID the last call returned, never a cached one.

*RunPod* is lower-level and therefore the better parity story: `runpod.create_pod(name,
image_name, gpu_type_id, ..., network_volume_id=...)` launches **our own published sandbox
image**, so the pinned versions the codegen grounding promises are the versions that
actually run. Lifecycle is `stop_pod` (keeps the volume, stops compute billing) vs
`terminate_pod` (deletes everything) — mapped onto the shared `Disposition` enum along with
JarvisLabs' pause/destroy.

**Image parity is a correctness requirement, not an optimization.** D4 grounds generated
code against pinned versions from `docker/requirements.txt`. If a remote box runs different
versions, that grounding is a lie. So the sandbox image gets published in two variants —
`prompt2ml-sandbox:cpu` and `prompt2ml-sandbox:cuda12` — and remote backends launch the
same image. Where a provider imposes its own template (JarvisLabs), the adapter reconciles
via `--requirements` and **verifies** the resulting versions on the instance before training
starts, failing loudly on mismatch.

**The broker picks the backend, not the LLM.** `ComputeBroker.select()` is deterministic and
unit-testable. The planner only declares a *need* (`vram_gb`, `est_minutes`,
`needs_network`); the broker ranks feasible backends by policy:

```
local GPU (fits budget)  →  local CPU (if est_minutes under threshold)
                         →  remote GPU (cheapest feasible within cost ceiling)
                         →  fail with a clear explanation
```

Overridable with `--backend`, `--max-cost-usd`, `--local-only`.

**Transfer strategy.** Never ship raw datasets by default. The broker's provisioning plan
chooses between *upload* (preprocessed bundle, compressed) and *remote-fetch* (the instance
pulls from Kaggle/HF directly using forwarded credentials, so only scripts and the data
contract cross the wire). For large vision sets remote-fetch is dramatically cheaper and
faster.

### D7a — Money and data safety (non-negotiable guardrails)

Remote GPU introduces two failure modes the current codebase has no defense against. Both
get harness-level enforcement, never agent discretion.

**Runaway cost.** Today's `stop_colab_runtime()` can only *write a script asking* the model
to release the GPU — the exact anti-pattern to avoid. Instead:

- Every remote instance is created through a **lease**: `{provider, instance_id, created_at,
  ttl_s, max_cost_usd, disposition_on_expiry}` persisted to `runs/<id>/leases.json` **before**
  the instance exists (so a crash between create and record can't orphan it).
- A supervisor thread enforces the TTL and terminates on expiry **regardless of agent state**.
- `atexit` + `SIGINT`/`SIGTERM` handlers tear down on any exit path.
- `prompt2ml doctor` and every startup reconcile persisted leases against the provider's live
  instance list and offer to kill orphans. `prompt2ml clean --leases` forces it.
- Cost is estimated before launch ($/hr × budget minutes) and shown for confirmation; runs
  exceeding `--max-cost-usd` are refused, not truncated mid-training.

**Data egress.** You said you want data and compute local — cloud GPU directly contradicts
that for whatever dataset gets uploaded. So it is never implicit:

- Remote backends require per-run opt-in (`--allow-remote`, or an interactive confirmation
  naming the provider and the exact bytes to be transferred).
- `--local-only` hard-fails rather than silently falling back to a remote tier.
- Credentials forwarded for remote-fetch are scoped and short-lived where the provider
  supports it, and never written into generated scripts or logs.
- The final report records where every training step physically ran.

---

## 1. Package & CLI shape

A "good CLI" needs an installable package with an entry point. Current flat top-level
packages can't provide that.

```
pyproject.toml                     # entry point: prompt2ml = prompt2ml.cli:app
src/prompt2ml/
  cli/            app.py, commands/{init,doctor,run,resume,status,report,clean}.py
  core/           run.py, contracts.py, state.py, hardware.py, credentials.py, errors.py
  research/       tool.py, cache.py, sources.py
  data/
    extract/      kaggle.py, huggingface.py, local.py      -> DatasetManifest
    adapters/     arrow_to_parquet.py, imagefolder.py, textcorpus.py
  pipelines/
    tabular/      profile.py, plan.py, execute.py, validate.py
    text/         ...
    vision/       ...
  codegen/        grounding.py, lint.py, smoke.py, repair.py
  compute/
    broker.py     # deterministic backend selection
    leases.py     # lease record, TTL supervisor, orphan reconciliation
    backends/     local_docker.py, local_venv.py, jarvislabs.py, runpod.py, colab.py
  sandbox/        docker.py, gpu.py, limits.py, images.py
  reporting/      bundle.py, plots.py, templates/
  agents/         (ADK agent definitions, thin — logic lives in the modules above)
```

Commands:

| Command | Behaviour |
|---|---|
| `prompt2ml init` | One-time setup wizard: Kaggle / HF / Gemini / Tavily / (optional) JarvisLabs, RunPod, Colab. Validates each credential with a live ping. Stores via OS keyring, `.env` fallback. |
| `prompt2ml doctor` | Hardware + environment scan → `hardware.json`. Probes Docker, image presence, GPU passthrough, disk, remote-provider reachability, and **orphaned leases**. Green/yellow/red readiness table with the exact fix for each red row. |
| `prompt2ml run "<idea>"` | Full pipeline. `--modality`, `--budget`, `--backend`, `--max-cost-usd`, `--allow-remote`, `--local-only`, `--dry-run`. |
| `prompt2ml resume [run_id]` | Resumes from the last completed phase in `run.json`. |
| `prompt2ml status` / `list` | Table of runs, phase, elapsed, artifact counts, **live remote instances + accrued cost**. |
| `prompt2ml report <run_id>` | Opens/exports the bundle. |
| `prompt2ml clean` | Prunes intermediates, keeps artifacts. `--leases` force-terminates orphaned remote instances. |

Rich for tables/progress, Typer for parsing. Live phase progress replaces the current wall
of `[PIPELINE]` prints; full detail goes to `logs/`.

**Migration approach — strangler, not big-bang.** Existing agents keep running while their
I/O moves behind the new contracts one phase at a time. `pipeline_state.py` becomes a
shim over `core/state.py` during the transition so nothing breaks mid-refactor.

---

## 2. The adaptive interview

Fixes the "hardcoded, not conditioned on the previous answer" problem. The mechanism is
**slot coverage, not a question count.**

Required slots: `task_intent`, `model_output`, `success_definition`, `error_asymmetry`,
`deployment_context`, `data_domain`, `constraints`. Optional slots carry information value
weights.

Interviewer tools:

- `recall_facts()` — structured facts so far, with confidence
- `record_fact(slot, value, source, confidence)` — every answer becomes structured data
- `research(question)` — the D3 AgentTool, callable **mid-interview** (user says "predict
  churn for my SaaS" → look up how churn is normally framed and what data exists → ask a
  sharper second question)
- `finalize_spec()` — **refuses** with the list of empty required slots if coverage is
  incomplete

That refusal is what makes questions adaptive: the model can't hit a quota and stop, and it
can't ask filler either, because every question must close a named slot. `record_fact` takes
a `follow_up_to` reference so the transcript shows the chain.

Also: "I don't know" is a valid answer — the agent decides on the user's behalf and records
it as an assumption with `confidence: low`. Assumptions surface in the final report so the
user sees what was chosen for them.

Downstream agents read `requirements.json` (structured slots), not a 1500-word essay.

---

## 3. Modality pipelines

### 3.1 Extraction produces a manifest (fixes the broken NLP path today)

`download_huggingface_dataset` currently calls `save_to_disk()`, writing Arrow shards that
`SUPPORTED_EXTENSIONS` cannot read — while the extractor routes *all NLP work to HF first*.
That path is broken end to end and is the first thing to fix in M4.

New extraction contract: inspect `dataset.features`, classify modality, then normalize:

| Detected | Normalized to |
|---|---|
| tabular features | `data/train.parquet` (+ splits) |
| text + label | `data/train.parquet` with `text` / `label` columns |
| image features | `data/images/<split>/<class>/*.jpg` + `index.parquet` |
| audio | `data/audio/` + `index.parquet` |

Plus size guards (`--max-dataset-gb`, default warn at 5 GB) — currently a 20 GB download is
unbounded.

### 3.2 Per-modality preprocessing

Each pipeline emits the same `data_contract.json`, so the trainer and reporter stay generic.

- **tabular** — existing tools, with fitted-statistic steps moved to `deferred`
- **text** — normalization, language/length stats, dedup/near-dedup, label balance,
  tokenizer selection (recorded as deferred; tokenization happens in the training pipeline),
  stratified split
- **vision** — corrupt-file scan, size/aspect distribution, class balance, dedup by
  perceptual hash, transform *plan* (recorded; augmentation applied in the DataLoader,
  train-split only)
- **timeseries** — chronological split enforced at contract level; row-dropping steps
  (dedup, outlier removal) rejected because they punch holes in the index

### 3.3 Training recipes

`train_sklearn` (existing, tabular) and `train_torch` (new). The torch recipe is a
**templated** loop, not free-form generation — the LLM fills in a config, not the training
loop, which is where most generated-code bugs live:

- deterministic seeding, AMP, grad accumulation, early stopping, checkpoint-best
- `fast_dev_run` mode for the D4 smoke step
- per-epoch metrics streamed to `metrics_<model>.json` in the schema the existing validator
  already audits, so `validate_ml_results` extends rather than forks
- transfer learning by default for vision/text (timm / HF `transformers`), full fine-tune vs
  LoRA chosen by the D6 budget

---

## 4. Milestones

Sequenced by dependency. Sizes are rough for one person.

### M0 — Correctness & hygiene *(~1 week)*

Everything here is a bug that silently corrupts results or blocks a new user. No new features.

- Leakage: move imputation / frequency encoding / outlier bounds into the deferred set
  (`data_preprocessing_agent/agent.py:707-730, 924, 1124`)
- `hash()` → stable hashing (`agent.py:1534`)
- `astype(bool)` string coercion (`agent.py:1606`)
- `pd.to_numeric(errors="ignore")` — removed in pandas ≥2.2 (`agent.py:1612`)
- `json.dumps()` every value injected into the generated validation script
  (`agent.py:2006-2149`)
- Call `reset_run_id()` at run start; scope `datasets/` per run (`pipeline_state.py:164`)
- Remove the double-parented `report_generator_agent` / redundant Phase 4
  (`agent.py:3231`, `master_orchestrator/agent.py:183,646`)
- Single import path for `sandbox_executor` (`machine_learning_agent/agent.py:31,271`)
- Sandbox hardening: exec timeout, `mem_limit`, `cpu_quota`, `network_disabled` for
  untrusted steps, recreate container on workspace-mount mismatch, `pip install --user`
  under the non-root user (`sandbox_executor.py:201,253,356,371`)
- `requirements.txt`: drop `safeexecute`, `tavily` → `tavily-python`, pin the stack, scope
  the urllib3/requests workaround
- `.gitignore`: un-ignore `.env.example`, ignore `runs/`, `datasets/`, `outputs/`
- Add `docker/build.ps1` (referenced at `sandbox_executor.py:346`, doesn't exist)
- Verify `model_config.py` IDs resolve against the live API
- **pytest suite over the preprocessing tools** — the regression net for everything after this

*Done when:* full run on a known CSV is byte-identical across two runs, and every M0 bug has
a failing-then-passing test.

### M1 — Package, run contract, CLI, doctor *(~2 weeks)*

`pyproject.toml` + `src/` layout; `core/contracts.py` + `core/state.py`; `pipeline_state.py`
shim; `cli/` with `init`, `doctor`, `run`, `resume`, `status`; `core/hardware.py` + the
Docker GPU passthrough probe.

*Done when:* `pip install -e .` → `prompt2ml doctor` prints an accurate readiness table on a
clean Windows box, and `prompt2ml run` completes the existing tabular flow through the new
run contract.

### M2 — Research as a tool + adaptive interview *(~1 week)*

`research/tool.py` (AgentTool + cache), wired into interviewer / strategist / planner /
trainer. Slot-based interview with `finalize_spec()` gating. Retire the three per-phase
research agents.

*Done when:* two different project ideas produce visibly different question sequences, and
the same research query across two runs hits cache on the second.

### M3 — Codegen quality gate *(~1.5 weeks)*

`codegen/grounding.py` (`ask_docs`, cached — Context7 MCP is a good backend here),
`codegen/lint.py` (ruff + `leakage_lint` + budget check), `codegen/smoke.py`,
`codegen/repair.py`. Every generated script routed through gate → smoke → full.

*Done when:* a deliberately leaky generated script is rejected pre-execution, and smoke-run
catches an injected crash without paying for the full training run.

### M4 — Text/NLP pipeline *(~2 weeks)*

Manifest-producing extraction with HF normalization (fixes the broken Arrow path),
`pipelines/text/`, `train_torch` for text classification with transfer learning.

*Done when:* `prompt2ml run "classify support tickets by urgency"` produces a fine-tuned
model + metrics + plots with no manual intervention.

### M5 — Compute backends: local GPU + JarvisLabs + RunPod *(~2 weeks)*

`compute/broker.py`, `compute/leases.py` (TTL supervisor, atexit/signal teardown, orphan
reconciliation), `backends/local_docker.py` (GPU), `backends/local_venv.py`,
`backends/jarvislabs.py`, `backends/runpod.py`. Published `prompt2ml-sandbox:{cpu,cuda12}`
images + remote version-parity verification. Cost estimation, `--max-cost-usd`,
`--allow-remote` / `--local-only` consent flow.

*Done when:* the same training job runs unmodified on all three of local-GPU, JarvisLabs, and
RunPod; a `SIGINT` mid-training terminates the remote instance within the TTL; and a
deliberately orphaned instance is detected and offered for cleanup by `doctor`.

**This milestone is the one to build with a hard cost ceiling on a throwaway account.** Every
other milestone fails for free; this one fails with a bill.

### M6 — Vision pipeline + DL training *(~2.5 weeks)*

`pipelines/vision/`, imagefolder adapter, `train_torch` recipes for vision, budget-driven
model selection, LoRA path for tight VRAM, checkpoint/resume for interruptible instances.

*Done when:* an image-classification idea trains end to end on whichever backend the broker
selects, and the same run degrades gracefully to a smaller model + reduced budget on CPU.

### M7 — Deliverable bundle & polish *(~1 week)*

Your definition of done, made explicit and verified. Every run ends with:

```
runs/<run_id>/artifacts/
  model/         best_model.{joblib|pt} + inference.py + input schema
  data/          final training dataset + data_contract.json
  plots/         confusion matrix / ROC / residuals / training curves / feature importance
  reports/       requirements.md, preprocessing.md, training.md, final_report.md
  RUN_SUMMARY.md
```

A `verify_bundle()` check fails the run if any required artifact is missing — so "done"
is asserted, not hoped for. Plus a generated `inference.py` that loads the model and scores
a new file, which is what makes the output usable rather than just present.

---

## 5. Sequencing rationale

- **M0 before everything.** Adding DL to a pipeline whose metrics can't be trusted multiplies
  the debugging surface without adding value.
- **M1 before M4/M5.** Multi-modality needs the manifest and the run contract; retrofitting
  them later means touching every pipeline twice.
- **M2/M3 before M4-M6.** The quality gate and research tool are exactly what make generated
  *DL* code survivable — DL scripts are longer, slower to fail, and more expensive per
  failure than sklearn scripts. On rented GPUs, a bad script now costs money per minute.
  Build the net before the high wire.
- **Text before vision.** HF is already an integrated source, the data stays rectangular, and
  the failure modes are cheaper to debug than image pipelines.
- **Compute backends (M5) before vision (M6).** Vision is what actually needs the GPU, and
  debugging a new pipeline and a new execution backend simultaneously means never knowing
  which one broke. Land the backends against the already-working text trainer first.

---

## 6. Open items to decide before M4

1. **Vision dataset ceiling** — cap by rows, GB, or wall-clock? Affects whether subsampling
   is automatic or prompted.
2. **Host-venv training consent** — one-time global opt-in during `init`, or per-run
   confirmation? (Recommend: per-run, since it's the only path that drops isolation.)
3. **Colab's role now that JarvisLabs and RunPod are in** — both are strictly better
   behaved (real APIs, real teardown, predictable billing). Recommend demoting Colab to
   "supported if configured, never auto-selected", or dropping it entirely rather than
   maintaining a third remote path.
4. **Default disposition on run completion** — PAUSE (fast resume, storage still bills) or
   DESTROY (zero residual cost, cold start next time)? Recommend DESTROY by default with
   `--keep-warm` to opt into PAUSE.
5. **Cost ceiling default** — should `--max-cost-usd` have a non-zero default (e.g. $5) so
   an unattended run can never surprise you, or default to unlimited with explicit opt-in?
6. **Model registry** — do repeated runs on the same dataset accumulate a comparable history,
   or is each run standalone?
