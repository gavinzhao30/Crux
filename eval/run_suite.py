"""Run paired Task 9 or Boxing evaluations with persistent Modal jobs.

Examples:
    python -m eval.run_suite task9 --arms single_agent,crux --cases dev --repeats 1 \
        --suite-dir /private/task9-suite --references /private/task9 --grader-repo /private/BixBench3 \
        --grader-python /private/grader-venv/bin/python
    python -m eval.run_suite boxing --arms single_agent,crux_random_selection \
        --cases dev --repeats 1

Use --dry-run to validate and write the job manifest without launching paid work.
The Task 9 truth key and references are read only by the coordinator, after jobs finish.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import modal
from modal.types import FileEntryType

from agents.llm import CHEAP_MODEL, STRONG_MODEL
from core.config import load_config
from execution.staging import COUNTS, SAMPLES, Task9Inputs, stage

ARMS = ("single_agent", "crux", "crux_random_selection", "crux_no_critic", "crux_role_only_memory")
BOXING_ARMS = ("single_agent", "crux", "crux_random_selection")
BOXING_SEEDS = {"dev": tuple(range(1000, 1005)), "test": tuple(range(2000, 2020))}
VOLUME_NAME = "crux-eval-checkpoints"
ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _load_local_credentials() -> None:
    """Load only required local keys; only the model key enters the remote secret."""
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition("=")
            if separator and key in {"OPENAI_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}:
                value = raw.strip().strip('"\'')
                if value:
                    os.environ.setdefault(key, value)


_load_local_credentials()

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_pyproject(str(ROOT / "pyproject.toml"))
         .add_local_python_source("agents", "core", "eval", "execution", "labs")
         .add_local_file(str(ROOT / "labs" / "bixbench3" / "task9_contract.json"),
                         remote_path="/root/labs/bixbench3/task9_contract.json"))
app = modal.App("crux-eval-suite")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = (modal.Secret.from_name(os.environ["CRUX_EVAL_SECRET"])
          if os.environ.get("CRUX_EVAL_SECRET")
          else modal.Secret.from_local_environ(["OPENAI_API_KEY"]))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def resolved_config(arm: str, seed: int, runs_dir: str, max_seconds: int) -> dict:
    preset = "default" if arm == "crux" else arm
    cfg = load_config(preset, seed=seed, runs_dir=runs_dir, max_seconds=max_seconds)
    cfg.llm.cache = False  # independent repeats must make independent model calls
    return cfg.model_dump(mode="json")


def plan(args: argparse.Namespace) -> dict:
    if args.model not in {CHEAP_MODEL, STRONG_MODEL} or (
            args.suite == "boxing" and args.model != CHEAP_MODEL):
        raise ValueError("Boxing requires the configured cheap model; Task 9 supports both configured models")
    arms = tuple(dict.fromkeys(args.arms.split(",")))
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError(f"Choose arms from {', '.join(ARMS)}")
    if args.suite == "boxing" and any(arm not in BOXING_ARMS for arm in arms):
        raise ValueError("Boxing's current numeric adapter cannot exercise critic or role-only memory; "
                         f"choose from {', '.join(BOXING_ARMS)}")
    if not 1 <= args.repeats <= 100 or not 1 <= args.concurrency <= 100:
        raise ValueError("repeats and concurrency must be between 1 and 100")
    if not 1 <= args.timeout <= 3600 or args.seed < 0:
        raise ValueError("timeout must be 1..3600 seconds and seed must be nonnegative")
    if args.suite == "task9":
        if args.suite_dir is None:
            raise ValueError("Task 9 requires --suite-dir")
        truth_path = args.suite_dir / "evaluator" / "truth.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))["cases"]
        case_ids = sorted(k for k, v in truth.items() if args.cases == "all" or v["split"] == args.cases)
        for case_id in case_ids:
            folder = args.suite_dir / "cases" / case_id
            if not (folder / COUNTS).is_file() or not (folder / SAMPLES).is_file():
                raise ValueError(f"Missing Task 9 inputs for {case_id}")
        if not args.dry_run:
            for path in (args.references, args.grader_repo, args.grader_python):
                if path is None or not path.exists():
                    raise ValueError("Task 9 grading requires existing --references, --grader-repo, and --grader-python")
    else:
        if args.cases == "all":
            case_ids = [str(s) for split in ("dev", "test") for s in BOXING_SEEDS[split]]
        else:
            case_ids = [str(s) for s in BOXING_SEEDS[args.cases]]
    if args.case_ids:
        selected = set(args.case_ids.split(","))
        if not selected <= set(case_ids):
            raise ValueError("--case-ids must be a subset of the chosen case split")
        case_ids = [case_id for case_id in case_ids if case_id in selected]
    if not case_ids:
        raise ValueError("Selected case set is empty")
    batch_id = hashlib.sha256(str(getattr(args, "output", args.suite_dir)).encode()).hexdigest()[:8]
    jobs = []
    for case_id in case_ids:
        for repeat in range(args.repeats):
            seed = args.seed + repeat if args.suite == "task9" else int(case_id) + repeat * 100_000
            for arm in arms:
                job_id = f"{batch_id}-{args.suite}-{case_id}-{arm}-r{repeat:02d}"
                jobs.append({"job_id": job_id, "suite": args.suite, "case_id": case_id,
                             "arm": arm, "repeat": repeat, "seed": seed,
                             "timeout": args.timeout, "model": args.model,
                             "config": resolved_config(arm, seed, "/checkpoints/jobs", args.timeout)
                             if arm != "single_agent" and args.suite == "task9" else None})
    return {"suite": args.suite, "cases": args.cases, "arms": arms, "batch_id": batch_id,
            "repeats": args.repeats, "seed": args.seed, "jobs": jobs,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "boxing_seed_splits": BOXING_SEEDS if args.suite == "boxing" else None}


def _checkpoint_loop(stop: threading.Event) -> None:
    while not stop.wait(10):
        try:
            volume.commit()
        except Exception as exc:  # noqa: BLE001 - final commit retries after transient errors
            logger.warning("Periodic checkpoint failed: %s", exc)


async def _task9(job: dict, manifest: str, directory: Path) -> dict:
    from decimal import Decimal

    from openai import AsyncOpenAI

    from agents.llm import LLM, LLMConfig
    from core.config import EngineConfig
    from core.orchestrator import run
    from eval.baseline_agent import run_baseline
    from execution.sandbox import ModalExecutor
    from labs.bixbench3.swarm_lab import TASK9_PRICES, build_task9_run
    from labs.bixbench3.task9_contract import load_contract, load_prompt

    run_id = job["job_id"]
    inputs = Task9Inputs(**job["input_manifest_data"])
    Path(manifest).write_text(inputs.json(), encoding="utf-8")
    if job["arm"] == "single_agent":
        model = job["model"]
        rate_in, rate_out = TASK9_PRICES[model]
        executor = ModalExecutor(inputs, Path("/checkpoints/jobs"))
        prompt, source = load_prompt()
        async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0) as client:
            report = await run_baseline(
                executor, client, model=model, run_id=run_id, prompt=prompt,
                prompt_source=source, input_usd_per_million=Decimal(str(rate_in)),
                output_usd_per_million=Decimal(str(rate_out)), contract=load_contract(),
                execution_seconds=job["timeout"],
            )
        return {"status": report["status"], "cost_usd": report.get("usd_unadjusted_for_cache"),
                "model_calls": report.get("calls"), "input_tokens": report.get("input_tokens"),
                "output_tokens": report.get("output_tokens")}
    cfg = EngineConfig.model_validate(job["config"])
    cfg.llm = LLMConfig(role_models={"default": job["model"]},
                        prices_per_mtok=TASK9_PRICES, cache=False)
    llm = LLM(cfg.llm)
    lab, deps = build_task9_run(cfg, manifest=Path(manifest), llm=llm)
    await run(lab, cfg, deps, run_id=run_id)
    usage = llm.usage_totals()
    return {"status": "finished", "cost_usd": usage.get("cost_usd"),
            "model_calls": usage.get("calls"), "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens")}


async def _boxing(job: dict, directory: Path) -> dict:
    from agents.llm import LLM, LLMConfig
    from eval.boxing_study import FamilyProposal, run_episode

    arm = {"single_agent": "single_agent", "crux": "crux_eig",
           "crux_random_selection": "crux_random"}[job["arm"]]
    llm = LLM(LLMConfig(role_models={"default": job["model"]}, cache=False,
                        prices_per_mtok={job["model"]: (.4, 1.6)},
                        timeout_s=60, max_retries=1, max_output_tokens=2048))
    llm.study_cap = 2.0
    proposals = None
    if arm.startswith("crux_"):
        if job.get("proposal_error"):
            raise ValueError(f"Paired proposal failed: {job['proposal_error']}")
        proposals = [FamilyProposal.model_validate(p) for p in job["proposals"]]
    record = await run_episode(arm, job["seed"], [5, 10, 20], directory, llm, proposals,
                               defer_grading=True)
    final = record["rows"][-1]
    return {"status": "finished" if final["status"] == "pending_grade" else final["status"],
            "mse": final["mse"],
            "cost_usd": record["usage"].get("cost_usd"),
            "model_calls": record["usage"].get("calls"),
            "input_tokens": record["usage"].get("input_tokens"),
            "output_tokens": record["usage"].get("output_tokens"),
            "experiments": len(record.get("history", [])), "failure": final.get("failure"),
            "checkpoints": record["rows"],
            "heldout_predictions": record["heldout_predictions"]}


@app.function(image=image, secrets=[secret], timeout=300, max_containers=8)
def propose_boxing(seed: int, model: str) -> dict:
    """Generate one hypothesis set shared by every Crux arm on a paired seed."""
    import numpy as np

    from agents.llm import LLM, LLMConfig
    from core.model_sandbox import SandboxedFamily
    from eval.boxing_study import GRID, Families, ask

    llm = LLM(LLMConfig(role_models={"default": model}, cache=False,
                        prices_per_mtok={model: (.4, 1.6)}, timeout_s=60,
                        max_retries=1, max_output_tokens=2048))

    async def make() -> dict:
        batch = await ask(llm, Families, {"task": "Propose two distinct executable numpy models. "
            "Each code must define predict(designs, params), returning (n_particles,n_designs), "
            "and sample_prior(rng,n), returning (n,p). Only numpy is allowed; no I/O."})
        for i, proposal in enumerate(batch.families):
            family = SandboxedFamily.from_source(str(i), proposal.code)
            params = family.sample_prior(np.random.default_rng(seed), 4)
            family.predict(GRID[:2], params)
        return {"proposals": [p.model_dump(mode="json") for p in batch.families],
                "usage": llm.usage_totals()}

    try:
        return asyncio.run(make())
    except Exception as exc:  # noqa: BLE001 - proposal failure is retained for all paired arms
        return {"proposal_error": f"{type(exc).__name__}: {exc}"[:1000],
                "usage": llm.usage_totals()}


@app.function(image=image, secrets=[secret], volumes={"/checkpoints": volume},
              timeout=4000, max_containers=8)
def worker(job: dict) -> dict:
    """One isolated Modal job. No evaluator truth or references enter this function."""
    directory = Path("/checkpoints/jobs") / job["job_id"]
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "status.json", {"state": "running", "job": job})
    volume.commit()
    stop = threading.Event()
    thread = threading.Thread(target=_checkpoint_loop, args=(stop,), daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        if job["suite"] == "task9":
            result = asyncio.run(asyncio.wait_for(
                _task9(job, str(directory / "source_input_manifest.json"), directory),
                timeout=job["timeout"] + 60))
        else:
            result = asyncio.run(asyncio.wait_for(_boxing(job, directory), timeout=job["timeout"] + 60))
        result["state"] = "complete"
    except Exception as exc:  # noqa: BLE001 - preserve scheduled failures as data
        result = {"state": "failed", "status": "failed", "failure": type(exc).__name__,
                  "failure_detail": str(exc)[:1000]}
    finally:
        stop.set()
        thread.join(timeout=2)
    result.update(job_id=job["job_id"], seconds=time.monotonic() - started)
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "status.json", {"state": result["state"], "job": job})
    volume.commit()
    return result


def _download_job(job: dict, output: Path) -> None:
    """Copy persisted worker evidence into the local result directory."""
    prefix = f"/jobs/{job['job_id']}"
    for entry in volume.iterdir(prefix, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        relative = Path(entry.path.lstrip("/")).relative_to(prefix.lstrip("/"))
        target = output / "evidence" / job["job_id"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            for chunk in volume.read_file(entry.path):
                stream.write(chunk)


def _grade_task9(job: dict, args: argparse.Namespace, evidence: Path) -> dict:
    from eval.grade_bixbench3 import grade_task9

    artifacts = evidence / "artifacts"
    if not artifacts.is_dir():
        return {"score": 0.0, "grade_status": "missing_artifacts"}
    try:
        grade = grade_task9(artifacts_dir=artifacts, references=args.references,
                            output_dir=args.output / "grades" / job["job_id"],
                            grader_repo=args.grader_repo, python=args.grader_python)
        return {"score": grade["mean"], "grade_status": "graded",
                "fraction_at_least_0_80": grade["fraction_at_least_0_80"]}
    except Exception as exc:  # noqa: BLE001 - grade infrastructure failure is distinct
        return {"score": None, "grade_status": "grader_error", "grade_error": str(exc)[:1000]}


def _grade_boxing(job: dict, result: dict) -> dict:
    from labs.boxing.location_finding import LocationFinding

    predictions = result.get("heldout_predictions", {})
    env = LocationFinding(job["seed"])
    checkpoints = []
    for budget in (5, 10, 20):
        values = predictions.get(str(budget), predictions.get(budget))
        try:
            mse = float(env.score(values)) if values is not None else None
        except (TypeError, ValueError, OverflowError):
            mse = None
        valid = mse is not None and math.isfinite(mse) and mse >= 0
        checkpoints.append({"budget": budget, "mse": mse, "score": 1 / (1 + mse) if valid else 0.0,
                            "status": "graded" if valid else "failed"})
    final = checkpoints[-1]
    return {"score": final["score"], "raw_mse": final["mse"],
            "grade_status": final["status"], "checkpoints": checkpoints}


def _summary(rows: list[dict], output: Path) -> None:
    fields = ["job_id", "suite", "case_id", "arm", "repeat", "seed", "state", "status",
              "grade_status", "score", "raw_mse", "cost_usd", "model_calls",
              "input_tokens", "output_tokens", "shared_proposal_cost_usd",
              "shared_proposal_input_tokens", "shared_proposal_output_tokens",
              "seconds", "experiments", "fault",
              "fault_detected", "failure", "failure_detail", "artifact_ref"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def recover(args: argparse.Namespace) -> None:
    """Reconcile every planned job after a coordinator interruption, without new model calls."""
    manifest = json.loads((args.output / "jobs.json").read_text(encoding="utf-8"))
    if manifest["suite"] != args.suite:
        raise ValueError("Output manifest belongs to a different suite")
    truth = (json.loads((args.suite_dir / "evaluator" / "truth.json").read_text())["cases"]
             if args.suite == "task9" and args.suite_dir else {})
    rows = []
    for job in manifest["jobs"]:
        saved = args.output / "jobs" / f"{job['job_id']}.json"
        if saved.is_file():
            row = json.loads(saved.read_text(encoding="utf-8"))
        else:
            try:
                raw = b"".join(volume.read_file(f"/jobs/{job['job_id']}/result.json"))
                result = json.loads(raw)
                _download_job(job, args.output)
            except (OSError, ValueError, modal.exception.Error) as exc:
                result = {"state": "missing", "status": "interrupted", "failure": "MissingCheckpoint",
                          "failure_detail": str(exc)[:1000]}
            evidence = args.output / "evidence" / job["job_id"]
            if args.suite == "boxing":
                graded = _grade_boxing(job, result)
            elif result.get("state") == "complete" and result.get("status") == "finished" and all(
                    (args.references, args.grader_repo, args.grader_python)):
                graded = _grade_task9(job, args, evidence)
            else:
                graded = {"score": 0.0 if result.get("state") != "complete" else None,
                          "grade_status": "failed" if result.get("state") != "complete" else "needs_evaluator"}
            row = {**job, **result, **graded, "artifact_ref": str(evidence)}
            if job.get("usage"):
                row["shared_proposal_cost_usd"] = job["usage"].get("cost_usd")
            if job["case_id"] in truth:
                row["fault"] = truth[job["case_id"]]["fault"]
            atomic_json(saved, row)
        rows.append(row)
    _summary(rows, args.output)
    print(f"Reconciled {len(rows)} planned jobs; summary: {args.output / 'summary.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suite", choices=["task9", "boxing"])
    parser.add_argument("--arms")
    parser.add_argument("--cases", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--case-ids", help="comma-separated subset of the chosen case split")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model", default=CHEAP_MODEL)
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--grader-repo", type=Path)
    parser.add_argument("--grader-python", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recover", action="store_true", help="reconcile saved jobs without launching new work")
    args = parser.parse_args(argv)
    if args.recover:
        if args.output is None:
            parser.error("--recover requires --output")
        args.output = args.output.resolve()
        recover(args)
        return
    if not args.arms:
        parser.error("--arms is required except with --recover")
    args.output = (args.output or ROOT / "eval" / "results" / args.suite /
                   datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")).resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty")
    try:
        manifest = plan(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "jobs.json", manifest)
    if args.dry_run:
        print(f"Planned {len(manifest['jobs'])} jobs in {args.output}")
        return
    if args.suite == "task9":
        input_volume = modal.Volume.from_name("crux-task9-inputs", create_if_missing=True)
        for case_id in sorted({j["case_id"] for j in manifest["jobs"]}):
            folder = args.suite_dir / "cases" / case_id
            inputs = stage(input_volume, folder / COUNTS, folder / SAMPLES, None,
                           temp_dir=args.output)
            case_manifest = args.output / "inputs" / f"{case_id}.json"
            case_manifest.parent.mkdir(parents=True, exist_ok=True)
            case_manifest.write_text(inputs.json(), encoding="utf-8")
            for job in manifest["jobs"]:
                if job["case_id"] == case_id:
                    job["input_manifest"] = str(case_manifest)
    # The worker image has no access to local coordinator paths; include input metadata in each job.
    for job in manifest["jobs"]:
        if job["suite"] == "task9":
            job["input_manifest_data"] = json.loads(Path(job.pop("input_manifest")).read_text())
    atomic_json(args.output / "jobs.json", manifest)
    results = []
    with app.run():
        if args.suite == "boxing":
            paired = sorted({(j["case_id"], j["repeat"], j["seed"])
                             for j in manifest["jobs"] if j["arm"].startswith("crux")})
            proposals = list(propose_boxing.map((seed for _, _, seed in paired),
                                                 (args.model for _ in paired),
                                                 return_exceptions=True))
            for key, proposal in zip(paired, proposals, strict=True):
                if isinstance(proposal, Exception):
                    proposal = {"proposal_error": str(proposal)[:1000]}
                for job in manifest["jobs"]:
                    if (job["case_id"], job["repeat"], job["seed"]) == key and job["arm"].startswith("crux"):
                        job.update(proposal)
            atomic_json(args.output / "jobs.json", manifest)
        remote = worker.with_options(max_containers=args.concurrency)
        for start in range(0, len(manifest["jobs"]), args.concurrency):
            batch = manifest["jobs"][start:start + args.concurrency]
            results.extend(remote.map(batch, order_outputs=True, return_exceptions=True))
    rows = []
    truth = (json.loads((args.suite_dir / "evaluator" / "truth.json").read_text())["cases"]
             if args.suite == "task9" else {})
    for job, result in zip(manifest["jobs"], results, strict=True):
        if isinstance(result, Exception):
            result = {"state": "missing", "status": "failed", "failure": type(result).__name__,
                      "failure_detail": str(result)[:1000]}
        try:
            _download_job(job, args.output)
        except Exception as exc:  # noqa: BLE001 - remote evidence failures stay visible
            result.setdefault("evidence_error", str(exc)[:1000])
        evidence = args.output / "evidence" / job["job_id"]
        graded = (_grade_task9(job, args, evidence) if args.suite == "task9" and result.get("state") == "complete" and result.get("status") == "finished"
                  else _grade_boxing(job, result) if args.suite == "boxing"
                  else {"score": 0.0, "grade_status": "failed"})
        row = {**job, **result, **graded, "artifact_ref": str(evidence)}
        if job.get("usage"):
            row["shared_proposal_cost_usd"] = job["usage"].get("cost_usd")
            row["shared_proposal_input_tokens"] = job["usage"].get("input_tokens")
            row["shared_proposal_output_tokens"] = job["usage"].get("output_tokens")
        if args.suite == "task9":
            row["fault"] = truth[job["case_id"]]["fault"]
            row["fault_detected"] = None  # reviews are not uniformly machine-readable
        atomic_json(args.output / "jobs" / f"{job['job_id']}.json", row)
        rows.append(row)
        _summary(rows, args.output)
    print(f"Completed {len(rows)} jobs; summary: {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
