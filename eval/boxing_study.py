"""Seed-paired Location Finding study. No hidden truth is given to agents.

Run the free pilot before selecting model arms. Checkpoint JSON is written per
episode; DU-23 can call run_episode without depending on this CLI.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from agents.llm import CHEAP_MODEL, LLM, LLMConfig
from core.config import load_config
from core.eig_exec import ModelFamily, ParticleBelief
from core.events import jsonable
from core.orchestrator import Deps, run
from core.schemas import Hypothesis, Question, Report
from core.tracing import TraceSession, current_run
from eval.boxing_metrics import FAILURE_ERROR, TARGET_ERROR, paired_contrasts, summarize
from labs.boxing.location_finding import LocationFinding, predict_sources, source_prior

ARMS = ("random", "grid", "single_agent", "crux_random", "crux_eig", "oracle")
GRID = np.array([(x, y) for x in np.linspace(-2, 2, 7) for y in np.linspace(-2, 2, 7)])
BRIEF = ("Predict scalar signal intensity on a 2-D plane from noisy measurements. "
         "There are three unknown equal-strength sources. Measurement noise SD is 0.5; "
         "observations are rounded to two decimals. Choose measurements inside [-2,2]^2. "
         "Do not assume source locations. Return only requested structured fields.")


class Point(BaseModel):
    x: float = Field(ge=-2, le=2)
    y: float = Field(ge=-2, le=2)


class Predictions(BaseModel):
    values: list[float] = Field(min_length=10, max_length=10)


class FamilyProposal(BaseModel):
    statement: str
    code: str


class Families(BaseModel):
    families: list[FamilyProposal] = Field(min_length=2, max_length=3)


async def ask(llm, schema, task):
    bound = (len(json.dumps(task).encode()) + len(json.dumps(schema.model_json_schema()).encode()) + 4096) * .4 / 1e6 + 2048 * 1.6 / 1e6
    if llm.usage_totals()["cost_usd"] + bound > getattr(llm, "study_cap", float("inf")):
        raise RuntimeError("StudySpendLimit")
    return (await llm.structured("proposer", schema,
            [{"role": "system", "content": BRIEF},
             {"role": "user", "content": json.dumps(task)}]))[0]


class BoxingLab:
    name = "boxing"
    sigma = .5
    max_measurements = 20

    def __init__(self, seed, budgets, defer_grading=False):
        self.env = LocationFinding(seed)
        self.budgets = budgets
        self.max_measurements = max(budgets)
        self.scores = {}
        self.predictions = {}
        self.defer_grading = defer_grading
        self.diagnostics = {}

    def brief(self):
        return {"goal": BRIEF, "contract": {}}

    def candidate_designs(self):
        return [{"x": float(p[0]), "y": float(p[1])} for p in GRID]

    async def profile(self):
        return []

    async def run(self, experiment):
        from core.schemas import Observation

        p = experiment.design
        return Observation(experiment_id=experiment.id, valid=True,
                           value=self.env.observe([p["x"], p["y"]]))

    def evaluate_belief(self, belief, step):
        self.diagnostics[step] = belief_diagnostics(belief)
        if step in self.budgets:
            predictions = belief.predict_means(self.env.evaluation_points())
            if self.defer_grading:
                self.predictions[step] = np.asarray(predictions).tolist()
            else:
                self.scores[step] = self.env.score(predictions)

    async def deliver(self, state):
        return []


class BoxingRoles:
    def __init__(self, proposals):
        self.proposals = proposals

    async def propose(self, brief, evidence):
        q = Question(id="signal", text="Which executable model predicts intensity?", kind="model", raised_by="proposer")
        hs = [Hypothesis(id=f"family-{i}", question_id=q.id, statement=p.statement,
                         author="proposer", rationale="Model-generated before observations",
                         model_code=p.code, prior=1, posterior=1)
              for i, p in enumerate(self.proposals)]
        return [q], hs

    async def critic(self, artifacts, contract):
        return []

    async def report(self, state, artifacts):
        return Report(run_id=state.run_id, artifacts=[], methods_md="Numeric study; see episode.json",
                      rejected=[], unresolved=[], cost_usd=0, seconds=0)


def fingerprint(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def belief_diagnostics(belief):
    return {"moves": {name: dict(values) for name, values in belief.move_diagnostics.items()},
            "effective_sample_sizes": belief.effective_sample_sizes(),
            "parameter_std": {name: belief.particles(name)[0].std(axis=0).tolist()
                              for name in belief.family_names}}


async def run_episode(arm, seed, budgets, directory, llm, proposals=None, particles=256, simulations=16,
                      defer_grading=False, move_steps=4):
    started = time.monotonic()
    env = LocationFinding(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 410]))
    scores, failure, target, history = {}, None, None, []
    lab = None
    diagnostics = {}
    heldout_predictions = {}
    initial = llm.usage_totals()
    session = TraceSession(directory.name, lambda: arm)
    token = current_run.set(session)
    try:
        if arm in {"crux_random", "crux_eig"}:
            lab = BoxingLab(seed, budgets, defer_grading=defer_grading)
            cfg = load_config(runs_dir=str(directory.parent), seed=seed, critic=False,
                              null_hypothesis=False, numeric_eig_goal="total",
                              numeric_stop_on_settle=False, numeric_particles=particles,
                              numeric_simulations=simulations, numeric_move_steps=move_steps,
                              selection="random" if arm == "crux_random" else "eig")
            cfg.llm = llm.config
            await run(lab, cfg, Deps(agents=BoxingRoles(proposals), predictor=None, llm=llm),
                      run_id=directory.name)
            scores, history = lab.scores, lab.env.history
        elif arm == "single_agent":
            snapshots = {}
            for step in range(1, max(budgets) + 1):
                point = await ask(llm, Point, {"task": "Choose next measurement", "observations": history})
                p = [point.x, point.y]
                history.append((p, env.observe(p)))
                if step in budgets:
                    snapshots[step] = list(history)
            # Evaluation is after *all* designs; held-out points cannot affect later choices.
            for step, observations in snapshots.items():
                predictions = await ask(llm, Predictions, {"task": "Predict intensities at these points",
                    "observations": observations, "points": env.evaluation_points().tolist()})
                if defer_grading:
                    heldout_predictions[step] = predictions.values
                else:
                    scores[step] = env.score(predictions.values)
        else:
            family = ModelFamily("sources", predict_sources, source_prior, log_prior=lambda p: -.5*(p*p).sum(axis=1))
            belief = ParticleBelief([family], .5, n_particles=particles, n_sim=simulations,
                                    seed=seed, null_weight=0, move_steps=move_steps)
            sequence = GRID[np.linspace(0, len(GRID) - 1, max(budgets), dtype=int)]
            for step in range(1, max(budgets) + 1):
                if arm == "random":
                    point = rng.uniform(-2, 2, 2)
                elif arm == "grid":
                    point = sequence[step - 1]
                else:
                    gains = belief.expected_information_gain(GRID, goal="total")
                    point = GRID[int(np.argmax(gains))]
                observation = env.observe(point)
                history.append((point.tolist(), observation))
                belief.update(point, observation)
                diagnostics[step] = belief_diagnostics(belief)
                if step in budgets:
                    prediction = belief.predict_means(env.evaluation_points())
                    if defer_grading:
                        heldout_predictions[step] = np.asarray(prediction).tolist()
                    else:
                        scores[step] = env.score(prediction)
        for step in sorted(scores):
            if scores[step] <= TARGET_ERROR and target is None:
                target = step
    except Exception as exc:  # noqa: BLE001 - failed episodes must remain in the results
        failure = type(exc).__name__
    finally:
        if lab is not None:
            scores, history = lab.scores, lab.env.history
            diagnostics = lab.diagnostics
            heldout_predictions = lab.predictions
        await session.flush()
        current_run.reset(token)
    target = next((step for step in sorted(scores) if scores[step] <= TARGET_ERROR), None)
    totals = llm.usage_totals()
    usage = {k: totals[k] - initial[k] for k in totals}
    rows = [{"arm": arm, "seed": seed, "budget": b,
             "status": "pending_grade" if defer_grading and b in heldout_predictions else
                       "ok" if b in scores else "failed",
             "mse": scores.get(b, None if defer_grading else FAILURE_ERROR),
             "failure": (failure or "NoPrediction") if b not in scores and b not in heldout_predictions else None}
            for b in budgets]
    record = {"rows": rows, "seed": seed, "arm": arm, "history": history, "usage": usage,
              "seconds": time.monotonic() - started, "target_reached_at_checkpoint": target,
              "test_points_sha256": fingerprint(env.evaluation_points()),
              "sources_sha256": fingerprint(env._sources), "traces": session.refs,
              "particles": particles, "simulations": simulations, "move_steps": move_steps}
    record["belief_diagnostics"] = diagnostics
    if defer_grading:
        record["heldout_predictions"] = heldout_predictions
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory / "episode.tmp"
    temp.write_text(json.dumps(jsonable(record), indent=2) + "\n")
    temp.replace(directory / "episode.json")
    return record


async def study(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    revision = await asyncio.to_thread(subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True)
    (output / "protocol.json").write_text(json.dumps({
        "source_commit": revision.strip(),
        "arms": args.arms, "seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
        "budgets": args.budgets, "particles": args.particles, "simulations": args.simulations,
        "candidate_grid_sha256": fingerprint(GRID), "model": args.model,
        "accounted_spend_cap_usd": args.max_usd, "target_error": TARGET_ERROR,
        "failure_error": FAILURE_ERROR, "bootstrap_resamples": 2000,
        "oracle": "known structure; numerical total-information estimate; no hidden coordinates",
        "likelihood": "continuous Normal approximation to rounded observations",
        "move_steps": args.move_steps, "repair_limit_per_family": 2,
        "inference": "resample-move MH, full observation history; prior independence moves; reference local Normal-prior moves",
    }, indent=2))
    llm = LLM(LLMConfig(role_models={"default": args.model}, cache=False,
                        prices_per_mtok={args.model: (.4, 1.6)}, timeout_s=60, max_retries=1,
                        max_output_tokens=2048))
    llm.study_cap = args.max_usd
    records = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        proposals = None
        proposal_usage = None
        if any(a.startswith("crux_") for a in args.arms):
            before = llm.usage_totals()
            session = TraceSession(f"{output.name}-proposals-{seed}", lambda: "propose")
            token = current_run.set(session)
            from eval.boxing_proposals import generate_proposals

            try:
                proposal_record = await generate_proposals(llm, seed)
                proposals = [FamilyProposal.model_validate(p) for p in proposal_record["families"]]
            finally:
                await session.flush()
                current_run.reset(token)
            proposal_usage = {k: llm.usage_totals()[k] - before[k] for k in before}
            proposal_record.update(usage=proposal_usage, traces=session.refs)
            (output / f"proposals-{seed}.json").write_text(json.dumps(proposal_record, indent=2))
        for arm in args.arms:
            if arm in {"single_agent", "crux_random", "crux_eig"} and llm.usage_totals()["cost_usd"] >= args.max_usd:
                # No new paid calls after the study cap; still account for all jobs.
                failed = {"rows": [{"arm": arm, "seed": seed, "budget": b, "status": "failed",
                          "mse": FAILURE_ERROR, "failure": "StudySpendLimit"} for b in args.budgets]}
                records.append(failed)
                target = output / f"{output.name}-{arm}-{seed:02d}"
                target.mkdir(exist_ok=True)
                (target / "episode.json").write_text(json.dumps(failed, indent=2))
                continue
            result = await run_episode(arm, seed, args.budgets, output / f"{output.name}-{arm}-{seed:02d}",
                                       llm, proposals, args.particles, args.simulations, move_steps=args.move_steps)
            if arm.startswith("crux_"):
                result["paired_proposal_usage"] = proposal_usage
                (output / f"{output.name}-{arm}-{seed:02d}" / "episode.json").write_text(json.dumps(jsonable(result), indent=2))
            records.append(result)
            print(f"seed={seed} arm={arm} statuses={[r['status'] for r in result['rows']]}", flush=True)
    export_results(records, llm.usage_totals(), output)


def export_results(records, usage, output):
    """Rebuild reports from preserved episodes without making model calls."""
    rows = [row for record in records for row in record["rows"]]
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "paired_comparisons.json").write_text(json.dumps(paired_contrasts(rows), indent=2))
    with (output / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "costs.json").write_text(json.dumps({"actual_wrapper_usage": usage,
        "episodes": [{k: v for k, v in r.items() if k != "history"} for r in records]}, indent=2))
    with (output / "costs.csv").open("w", newline="") as stream:
        shared_keys = ["calls", "input_tokens", "output_tokens", "cost_usd"]
        writer = csv.DictWriter(stream, fieldnames=["arm", "seed", *shared_keys,
            *[f"shared_proposal_{key}" for key in shared_keys], "seconds", "measurements",
            "failed_checkpoints", "target_reached_at_checkpoint", "target_censored"])
        writer.writeheader()
        for record in records:
            row = record["rows"][0]
            writer.writerow({"arm": row["arm"], "seed": row["seed"],
                **{k: record.get("usage", {}).get(k, 0) for k in ["calls", "input_tokens", "output_tokens", "cost_usd"]},
                **{f"shared_proposal_{key}": (record.get("paired_proposal_usage") or {}).get(key, 0) for key in shared_keys},
                "seconds": record.get("seconds"), "measurements": len(record.get("history", [])),
                "failed_checkpoints": sum(r["status"] != "ok" for r in record["rows"]),
                "target_reached_at_checkpoint": record.get("target_reached_at_checkpoint"),
                "target_censored": record.get("target_reached_at_checkpoint") is None})
    plot(summary, output)


def plot(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for metric in ["median", "mean_log10"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for arm in sorted({r["arm"] for r in summary}):
            selected = sorted([r for r in summary if r["arm"] == arm], key=lambda r: r["budget"])
            ax.plot([r["budget"] for r in selected], [r[metric]["estimate"] for r in selected], label="oracle (numerical)" if arm == "oracle" else arm, marker="o")
            ax.fill_between([r["budget"] for r in selected], [r[metric]["low"] for r in selected],
                            [r[metric]["high"] for r in selected], alpha=.12)
        if metric == "median":
            ax.set_yscale("log")
        ax.set(xlabel="Measurements", ylabel="Median held-out MSE" if metric == "median" else "Mean log10 held-out MSE",
               title="Location Finding: paired seeds, bootstrap 95% CIs")
        ax.set_xticks(sorted({r["budget"] for r in summary}))
        ax.legend()
        fig.text(.5, .015, "Failed checkpoints retained with MSE penalty 1e12; shaded bands are 95% confidence intervals.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=lambda s: s.split(","), default=["random", "grid", "oracle"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budgets", type=lambda s: [int(x) for x in s.split(",")], default=[5, 10, 20])
    parser.add_argument("--move-steps", type=int, default=4)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--model", choices=[CHEAP_MODEL], default=CHEAP_MODEL)
    parser.add_argument("--max-usd", type=float, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not set(args.arms) <= set(ARMS) or not args.budgets or min(args.budgets) < 1 or args.seeds < 1:
        parser.error("Invalid arm, budget or seed count")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a fresh study directory")
    asyncio.run(study(args))


if __name__ == "__main__":
    main()
