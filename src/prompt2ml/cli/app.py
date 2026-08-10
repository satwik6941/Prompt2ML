"""
Prompt2ML command-line interface.

    prompt2ml init      one-time credential setup
    prompt2ml doctor    hardware and environment readiness
    prompt2ml run       start a run
    prompt2ml status    inspect runs
    prompt2ml clean     prune intermediates, keep artifacts

Built on argparse so the diagnostic commands keep working when a dependency
install is the thing that has gone wrong.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ..core import credentials as creds
from ..core.contracts import Phase, PhaseStatus
from ..core.runstore import DEFAULT_RUNS_ROOT, RunStore, RunStoreError
from .ui import FAIL, OK, WARN, Console

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    from ..core import hardware

    con = Console()
    creds.load_env_file(PROJECT_ROOT / ".env")

    con.title("Credentials")
    for cred in creds.CREDENTIALS:
        if cred.is_set:
            con.row(OK, cred.label, cred.masked())
        else:
            con.row(
                FAIL if cred.required else WARN,
                cred.label,
                "not set",
                f"get one at {cred.obtain_url}, then add {cred.env_var} to .env",
            )

    con.title("Machine")
    hw = hardware.profile(probe_gpu_passthrough=not args.fast)
    con.row(OK, "Platform", hw.platform)
    con.row(OK, "CPU cores", str(hw.cpu_cores))
    con.row(
        OK if hw.ram_gb >= 8 else WARN,
        "RAM",
        f"{hw.ram_gb} GB",
        "8 GB or more is recommended for local training",
    )
    con.row(
        OK if hw.disk_free_gb >= 20 else WARN,
        "Disk free",
        f"{hw.disk_free_gb} GB",
        "datasets and checkpoints need headroom — 20 GB or more",
    )

    con.title("GPU")
    if not hw.gpus:
        con.row(WARN, "NVIDIA GPU", "none detected",
                "training will run on CPU, or on a rented GPU via --backend")
    else:
        for i, gpu in enumerate(hw.gpus):
            marker = " (selected)" if i == hw.selected_gpu else ""
            con.row(OK, f"GPU {i}", f"{gpu.name} — {gpu.vram_gb} GB VRAM{marker}")
        if hw.docker_gpu_passthrough is True:
            con.row(OK, "Docker passthrough", "working — sandboxed GPU training available")
        elif hw.docker_gpu_passthrough is False:
            con.row(WARN, "Docker passthrough", "not working",
                    "install the NVIDIA Container Toolkit and enable the WSL2 backend; "
                    "until then GPU training needs host-venv consent")
        else:
            con.row(WARN, "Docker passthrough", "not probed",
                    "re-run without --fast to test it")

    con.title("Sandbox")
    con.row(
        OK if hw.docker_available else FAIL,
        "Docker daemon",
        "running" if hw.docker_available else "unreachable",
        "start Docker Desktop and wait for it to finish initialising",
    )
    con.row(
        OK if hw.sandbox_image_present else FAIL,
        "Sandbox image",
        hardware.SANDBOX_IMAGE_CPU if hw.sandbox_image_present else "not built",
        "run docker/build.ps1 from the project root",
    )

    if hw.budget:
        con.title("Training budget")
        b = hw.budget
        con.dim("  Derived from the hardware above. The planner must stay inside these,")
        con.dim("  and the codegen gate rejects scripts that exceed them.")
        con.print()
        con.table(
            ["limit", "value"],
            [
                ["max model size", f"{b.max_params_m} M params"],
                ["max batch tokens", str(b.max_batch_tokens)],
                ["precision", b.precision],
                ["max train time", f"{b.max_train_minutes} min"],
                ["full fine-tune", "allowed" if b.allow_full_finetune else "no — use LoRA"],
                ["max image size", f"{b.max_image_px}px"],
            ],
        )

    con.title("Verdict")
    con.row(OK, "Recommended backend", hardware.recommended_backend(hw))

    blockers = len(creds.missing_required()) + (0 if hw.docker_available else 1)
    if blockers:
        con.print()
        con.dim(f"  {blockers} blocking issue(s) above must be fixed before a run.")
    else:
        con.print()
        con.dim("  Ready to run:  prompt2ml run \"your project idea\"")

    if args.save:
        store = RunStore.latest(args.runs_root)
        if store:
            store.write(hw)
            con.print()
            con.dim(f"  Profile saved to {store.path_for(type(hw))}")
    con.print()
    return 1 if blockers else 0


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    con = Console()
    env_path = PROJECT_ROOT / ".env"
    creds.load_env_file(env_path)

    con.title("Prompt2ML setup")
    con.dim("  Credentials are collected once, up front, so a run never dies")
    con.dim("  three phases in because a token was missing.")
    con.print()

    if not env_path.exists():
        template = creds.write_env_template(PROJECT_ROOT / ".env.example")
        con.row(WARN, ".env", "not found",
                f"copy {template.name} to .env and fill it in")
    else:
        con.row(OK, ".env", str(env_path))

    con.print()
    for cred in creds.CREDENTIALS:
        if cred.is_set:
            con.row(OK, cred.label, cred.masked())
        else:
            con.row(
                FAIL if cred.required else WARN,
                cred.label,
                "required" if cred.required else "optional",
                f"{cred.obtain_url} — needed for {cred.needed_for}",
            )

    missing = creds.missing_required()
    con.print()
    if missing:
        con.dim(f"  Add {len(missing)} required credential(s) to .env, then run:  prompt2ml doctor")
        return 1
    con.dim("  All required credentials present. Next:  prompt2ml doctor")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    from ..core import hardware

    con = Console()
    creds.load_env_file(PROJECT_ROOT / ".env")

    missing = creds.missing_required()
    if missing and not args.force:
        con.error(
            "missing required credentials: "
            + ", ".join(c.env_var for c in missing)
            + " — run 'prompt2ml init', or pass --force to continue anyway"
        )
        return 1

    store = RunStore.create(args.runs_root, goal=args.goal)
    con.title(f"Run {store.run_id}")
    con.row(OK, "Goal", args.goal)
    con.row(OK, "Run directory", str(store.dir))

    hw = hardware.profile(probe_gpu_passthrough=not args.fast)
    store.write(hw)
    con.row(OK, "Compute", hardware.recommended_backend(hw))

    meta = store.meta()
    con.print()
    con.table(
        ["phase", "status"],
        [[p.value, meta.phase(p).status.value] for p in Phase],
    )
    con.print()
    con.dim("  Run scaffolded. Phase execution is being migrated onto this contract")
    con.dim("  milestone by milestone — see docs/IMPLEMENTATION_PLAN.md.")
    con.dim(f"  Inspect with:  prompt2ml status {store.run_id}")
    con.print()
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    con = Console()

    if args.run_id:
        try:
            store = RunStore.open(args.runs_root, args.run_id)
        except RunStoreError as exc:
            con.error(str(exc))
            return 1
        return _show_run(con, store)

    runs = RunStore.list_runs(args.runs_root)
    if not runs:
        con.title("Runs")
        con.dim("  No runs yet. Start one:  prompt2ml run \"your project idea\"")
        con.print()
        return 0

    con.title("Runs")
    rows = []
    for store in runs[: args.limit]:
        try:
            meta = store.meta()
        except RunStoreError:
            rows.append([store.run_id, "unreadable", "—", "—"])
            continue
        nxt = meta.next_phase()
        done = sum(1 for p in Phase if meta.phase(p).is_done)
        rows.append([
            meta.run_id,
            f"{done}/{len(Phase)}",
            nxt.value if nxt else "complete",
            (meta.goal[:44] + "…") if len(meta.goal) > 45 else (meta.goal or "—"),
        ])
    con.table(["run", "phases", "next", "goal"], rows)
    con.print()
    return 0


def _show_run(con: Console, store: RunStore) -> int:
    from ..core.contracts import (
        DataContract, DatasetManifest, HardwareProfile, MLPlan, Requirements,
    )

    meta = store.meta()
    con.title(f"Run {meta.run_id}")
    con.row(OK, "Goal", meta.goal or "—")
    con.row(OK, "Created", meta.created_at.strftime("%Y-%m-%d %H:%M UTC"))
    if meta.cost_usd:
        con.row(OK, "Cloud spend", f"${meta.cost_usd:.2f}")

    con.title("Phases")
    status_marks = {
        PhaseStatus.COMPLETE: OK,
        PhaseStatus.SKIPPED: OK,
        PhaseStatus.RUNNING: WARN,
        PhaseStatus.PENDING: WARN,
        PhaseStatus.FAILED: FAIL,
    }
    for p in Phase:
        rec = meta.phase(p)
        detail = rec.status.value
        if rec.attempts > 1:
            detail += f" ({rec.attempts} attempts)"
        con.row(status_marks.get(rec.status, WARN), p.value, detail, rec.error or "")

    con.title("Documents")
    for doc in (Requirements, DatasetManifest, DataContract, MLPlan, HardwareProfile):
        present = store.exists(doc)
        con.row(OK if present else WARN, doc.filename,
                "written" if present else "not yet produced")

    artifacts = sorted(store.artifacts_dir.rglob("*")) if store.artifacts_dir.exists() else []
    files = [a for a in artifacts if a.is_file()]
    con.title("Artifacts")
    if not files:
        con.dim("  (none yet)")
    else:
        con.table(
            ["file", "size"],
            [[str(f.relative_to(store.artifacts_dir)), f"{f.stat().st_size:,} B"]
             for f in files[:25]],
        )
    con.print()
    return 0


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

def cmd_clean(args: argparse.Namespace) -> int:
    con = Console()
    runs = RunStore.list_runs(args.runs_root)
    if not runs:
        con.dim("  Nothing to clean.")
        return 0

    freed = 0
    removed = 0

    if args.orphans:
        root = Path(args.runs_root)
        orphans = [
            d for d in root.iterdir()
            if d.is_dir() and not (d / "run.json").exists()
        ] if root.exists() else []
        for d in orphans:
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            removed += 1
            if not args.dry_run:
                shutil.rmtree(d)
        verb = "would remove" if args.dry_run else "removed"
        con.dim(f"  {verb} {len(orphans)} incomplete run director(ies).")

    for store in runs:
        for path in store.dir.glob("step_*"):
            if path.is_file():
                freed += path.stat().st_size
                removed += 1
                if not args.dry_run:
                    path.unlink()
        logs = store.logs_dir
        if logs.exists() and not args.keep_logs:
            for path in logs.rglob("*"):
                if path.is_file():
                    freed += path.stat().st_size
                    removed += 1
            if not args.dry_run:
                shutil.rmtree(logs)
                logs.mkdir(exist_ok=True)

    verb = "would free" if args.dry_run else "freed"
    con.dim(f"  {verb} {freed / 1024 / 1024:.1f} MB across {removed} file(s); artifacts kept.")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt2ml",
        description="Turn a project idea into a trained model, locally.",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
        help="directory holding run folders (default: runs/)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="one-time credential setup")
    p_init.set_defaults(func=cmd_init)

    p_doctor = sub.add_parser("doctor", help="check hardware and environment readiness")
    p_doctor.add_argument("--fast", action="store_true",
                          help="skip the Docker GPU passthrough probe")
    p_doctor.add_argument("--save", action="store_true",
                          help="write the profile into the most recent run")
    p_doctor.set_defaults(func=cmd_doctor)

    p_run = sub.add_parser("run", help="start a run")
    p_run.add_argument("goal", help="what you want to build, in plain English")
    p_run.add_argument("--fast", action="store_true", help="skip the GPU passthrough probe")
    p_run.add_argument("--force", action="store_true",
                       help="start even with credentials missing")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="list runs, or show one in detail")
    p_status.add_argument("run_id", nargs="?", help="run to inspect")
    p_status.add_argument("--limit", type=int, default=20)
    p_status.set_defaults(func=cmd_status)

    p_clean = sub.add_parser("clean", help="prune intermediates, keep artifacts")
    p_clean.add_argument("--dry-run", action="store_true")
    p_clean.add_argument("--keep-logs", action="store_true")
    p_clean.add_argument("--orphans", action="store_true",
                         help="also delete run directories with no run.json")
    p_clean.set_defaults(func=cmd_clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        Console().error("interrupted")
        return 130
    except RunStoreError as exc:
        Console().error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
