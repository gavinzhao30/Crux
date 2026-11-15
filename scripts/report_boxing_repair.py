"""Write the repair study report from audited primary results; makes no model calls."""
import json
import statistics
from pathlib import Path

root = Path("eval/results/boxing")
main = root / "repair-main-20261115"
summary = json.loads((main / "summary.json").read_text())
paired = json.loads((main / "paired_comparisons.json").read_text())
audit = json.loads((main / "audit.json").read_text())
records = [json.loads(p.read_text()) for p in main.glob("*/episode.json")]
proposals = [json.loads(p.read_text()) for p in main.glob("proposals-*.json")]
assert len(records) == 120 and len(proposals) == 20
fixed = sum(not p["validation_error"] and any(a["error"] for a in p["attempts"]) for p in proposals)
exhausted = sum(bool(p["validation_error"]) for p in proposals)
repairs = sum(a["repair"] > 0 for p in proposals for a in p["attempts"])
last = paired[-1]
conclusion = ("EIG has lower paired log-error at 20 measurements in this run." if last["high"] < 0 else
              "Random selection has lower paired log-error at 20 measurements in this run." if last["low"] > 0 else
              "This run does not establish an EIG advantage; the paired 95% confidence interval includes zero.")
usage = audit["actual_wrapper_usage"]
findings = {"conclusion": conclusion, "paired_at_20": last,
            "repaired_proposal_sets": fixed, "unusable_proposal_sets": exhausted,
            "repair_versions_validated": repairs, "usage": usage, "diagnostics": {}}
for arm in ["random", "grid", "oracle", "crux_random", "crux_eig"]:
    moves = [m for r in records if r["arm"] == arm for d in r.get("belief_diagnostics", {}).values()
             for m in d.get("moves", {}).values() if m["proposed"]]
    findings["diagnostics"][arm] = {
        "resampling_events": len(moves),
        "pooled_move_acceptance": sum(m["accepted"] for m in moves)/sum(m["proposed"] for m in moves) if moves else None,
        "median_unique_particles_after_move": statistics.median(m["unique_particles"] for m in moves) if moves else None}
(main / "findings.json").write_text(json.dumps(findings, indent=2))
lines = ["# DU-41 repaired model and EIG study — November 15, 2026", "", "**" + conclusion + "**", "",
    "The new study retains all 120 episodes and 360 checkpoints on fresh paired seeds 4000–4019.",
    "Failed checkpoints remain in every primary aggregate with MSE penalty 1e12. Lower error is better.", "",
    "## Changes", "",
    "- Shared strict shape, finite-output, determinism and reproducibility checks for local and Modal runners; sandbox prediction inputs are read-only.",
    "- At most two AI repair attempts per hypothesis, retaining original and repaired code, errors and usage. Both Crux arms share the same final models per seed.",
    "- Four resample-and-move sweeps using the full observation history, replacing arbitrary jitter in this study. Generated models use prior-independence MH; the known-structure reference also uses local moves with its Normal prior.",
    "- 512 particles and 64 EIG simulations, frozen before the main run. Other labs retain legacy settings unless moves are enabled explicitly.",
    "- Modal source packaging excludes result/evaluator files and disables automatic source mounting.", "",
    "[Frozen repair protocol](../../../docs/DU-41-REPAIR-PROTOCOL.md) and [execution protocol/revision](repair-main-20261115/protocol.json).", "",
    "## Primary results", "",
    "| Arm | Median MSE at 20 [95% CI] | Mean log10 MSE [95% CI] | Failed episodes / 20 | Reached MSE ≤ 1 |",
    "|---|---:|---:|---:|---:|"]
for row in summary:
    if row["budget"] != 20:
        continue
    def fmt(x):
        return f"{x['estimate']:.4g} [{x['low']:.4g}, {x['high']:.4g}]"
    hit = sum(r["target_reached_at_checkpoint"] is not None for r in records if r["arm"] == row["arm"])
    lines.append(f"| {row['arm']} | {fmt(row['median'])} | {fmt(row['mean_log10'])} | {row['failures']} | {hit} |")
lines += ["", "CIs use 2,000 whole-seed bootstrap resamples. Target times are checkpoint-resolved and censored when not reached.", "",
    "| Measurements | Paired mean log10 MSE: EIG minus random | 95% CI |", "|---:|---:|---:|"]
for p in paired:
    lines.append(f"| {p['budget']} | {p['mean_log10_error_difference']:.4f} | [{p['low']:.4f}, {p['high']:.4f}] |")
lines += ["", "Negative favors EIG. These results compare the two current Crux arms, not EIG in general.", "",
    "![Median error](repair-main-20261115/median.png)", "", "![Mean log error](repair-main-20261115/mean_log10.png)", "",
    "## Reliability, cost and limitations", "",
    f"{fixed}/20 proposal sets were accepted after repair; {exhausted}/20 remained unusable. {repairs} repaired versions were validated. All terminal failures remain in the dataset.", "",
    f"Primary wrapper-accounted spend: **${usage['cost_usd']:.7f}**, {usage['calls']} calls, {usage['input_tokens']:,} input and {usage['output_tokens']:,} output tokens. Shared proposals/repairs count once in actual spend and are shown with each Crux arm. These are configured list-price estimates, not an invoice; unreturned billed usage cannot be fully reconciled. Development and Modal checks are separate.", "",
    "The additional EIG failure was a Windows NumPy DLL import failure at seed 4018, not an array-shape failure. It is retained with the declared penalty and materially affects the paired contrast. A narrowly scoped, two-retry startup guard was added and tested after the run; it does not change this archived outcome. There were no production array-contract failures among the accepted model sets.", "",
    "The reference knows the true formula and prior, never hidden coordinates. It remains numerical, with a Gaussian approximation to rounded observations. Moves are posterior-preserving kernels, but four sweeps do not guarantee mixing or posterior accuracy. Low acceptance and particle duplication persist: pooled move acceptance is about 6.5% for Crux random and 4.9% for Crux EIG; median post-move distinct particles are 51 and 11 out of 512, respectively. This remaining loss of diversity limits claims that inference is fully fixed. See per-episode diagnostics and [aggregated move diagnostics](repair-main-20261115/findings.json).", "",
    "[Ranking sensitivity](repair-numerics-20261115.json) on development seeds compared 512/64 with 2048/256 particles/simulations before data and after five fixed measurements. Rank correlations were about 0.92–0.98; the top design agreed in 3/6 comparisons. This does not establish convergence of the selected design.", "",
    "The scalar Crux study disables artifact criticism. The single-agent comparison uses a different inference representation. Several fixes and numerical settings changed together, and cases differ from the original study, so before/after differences do not identify the effect of one fix. This is simulator evidence, not scientific discovery.", "",
    "## Evidence", "",
    "- [Per-seed scores](repair-main-20261115/per_seed.csv), [summary](repair-main-20261115/summary.json), [paired comparisons](repair-main-20261115/paired_comparisons.json), [cost table](repair-main-20261115/costs.csv), [audit](repair-main-20261115/audit.json).",
    "- Proposal JSON files retain validation and repair attempts. Episode directories retain histories, diagnostics, and Crux events/manifests. The audit checks pairing, every saved observation, shared hypothesis sources, cost reconciliation and representative event replays.",
    "- **367 repository tests passed; Ruff passed.** Tests cover malformed arrays, repair exhaustion, analytic Gaussian posterior moments with both move types, EIG against analytic and independent quadrature references, noise sensitivity, simple-model resolution checks, reproducibility, scoring-query purity, source exclusions and narrowly bounded NumPy startup retries.",
    "- [Two-job Modal integration check](repair-modal-03-20261115/summary.csv): single-agent and Crux both executed, persisted evidence and graded. Two earlier packaging attempts failed before job dispatch; their manifests remain archived.",
    "- [Original study report](ORIGINAL_STUDY.md), [original results](main-20261114/summary.json), [archived original issue](../../../docs/DU-41-original-linear.md). Original failures and the missed historical deadline are not rewritten. The optional second environment remains deferred."]
(root / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print(json.dumps(findings, indent=2))
