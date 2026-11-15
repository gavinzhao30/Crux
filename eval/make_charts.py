"""Build the four pitch charts (PNG at 200 dpi plus SVG) into docs/figures/.

Usage:
    uv run python -m eval.make_charts [--task9 RESULTS.json ...] [--calibration FILE] [--out DIR]

Each chart whose input is missing is skipped with a printed note.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, colourblind safe. OURS is the swarm / our method, OTHER the comparison.
OURS = "#0072B2"
OTHER = "#E69F00"
GREY = "#8A8A8A"
INK = "#222222"
BG = "#FAFAF7"

DEFAULT_TASK9 = [
    Path("../evaluator/results/batch-02-suite/results.json"),
    Path("../evaluator/results/batch-02-clean/results.json"),
]
DEFAULT_CALIBRATION = Path("eval/results/predictor_calibration.json")
DEFAULT_OUT = Path("docs/figures")

# Source: eval/results/real_data_diagnostics.md, "Detection on real counts with
# planted faults (DU-49)". (caught, planted); None means the design cannot carry it.
FAULT_DATASETS = ["COVID lung/colon", "Pasilla", "Airway"]
FAULT_DETECTION = {
    "Sample swap": [(2, 2), (2, 2), (2, 2)],
    "Label inversion": [(2, 2), (0, 2), (0, 2)],
    "Duplicate library": [(2, 2), (2, 2), (2, 2)],
    "Low depth": [(2, 2), (2, 2), (2, 2)],
    "Cell-line swap": [(2, 2), None, None],
    "Gene version": [(2, 2), (2, 2), None],
}

# Source: eval/results/real_data_diagnostics.md, "Airway: the albuterol labels look
# swapped in two of four donors". Dex-signature score vs the donor's untreated sample.
AIRWAY = {
    # donor: (labelled albuterol, labelled albuterol + dex, dex for scale)
    "N052611": (2.86, 0.36, 3.39),
    "N061011": (0.00, 3.47, 3.72),
    "N080611": (3.11, 0.55, 3.21),
    "N61311": (-0.14, 3.16, 3.31),
}
AIRWAY_SWAPPED = {"N052611", "N080611"}

PREDICTORS = [
    ("library", "Library\nstated rows"),
    ("sampled", "gpt-4.1-mini\nsampled"),
    ("base", "Qwen3-8B\nbase"),
    ("finetuned", "Qwen3-8B\nLoRA fine-tune"),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.955, title, fontsize=20, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.885, "\n".join(_wrap(subtitle, 105)), fontsize=13, color="#555555",
             ha="left", va="top", linespacing=1.3)


def _save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.svg"]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _short(name: str, limit: int = 34) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "..."


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width)


# ---------------------------------------------------------------- task 9


def load_task9(paths: list[Path]) -> tuple[list[dict], list[Path]]:
    """Runs from every readable results.json, each tagged with a case name."""
    runs, used = [], []
    for p in paths:
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("runs", []):
            runs.append({**r, "case": r.get("case") or p.parent.name})
        used.append(p)
    return runs, used


def _run_score(r: dict) -> float:
    return r["mean"] if r.get("mean") is not None else 0.0  # failed counts as 0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def chart_task9_swarm_vs_baseline(runs: list[dict], out: Path) -> list[Path]:
    arms = [a for a in ("swarm", "baseline") if any(r["arm"] == a for r in runs)]
    arms += sorted({r["arm"] for r in runs} - set(arms))
    colour = {"swarm": OURS, "baseline": OTHER}
    artifacts: list[str] = []
    for r in runs:
        for n in (r.get("artifacts") or {}):
            if n not in artifacts:
                artifacts.append(n)
    # per artifact and arm: scores over runs (a failed run scores 0)
    scores: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        got = r.get("artifacts") or {}
        for n in artifacts:
            scores.setdefault((n, r["arm"]), []).append(got.get(n, 0.0))
    if not artifacts:
        for r in runs:
            scores.setdefault(("overall", r["arm"]), []).append(_run_score(r))
        artifacts = ["overall"]

    overall = {a: _mean([_run_score(r) for r in runs if r["arm"] == a]) for a in arms}
    n_runs = {a: sum(r["arm"] == a for r in runs) for a in arms}
    failed = sum(
        1 for r in runs if r.get("status") not in (None, "ok") or r.get("mean") is None
    )
    if "baseline" in overall and "swarm" in overall:
        title = (
            f"On Task 9 the swarm scores {overall['swarm']:.2f}; "
            f"the single agent scores {overall['baseline']:.2f}"
        )
    else:
        title = "Task 9 official score by arm: " + ", ".join(
            f"{a} {v:.2f}" for a, v in overall.items()
        )
    sub = (
        "Pilot: 1 task, counts as input. Bars = mean per artifact, whiskers = min to max. "
        "Runs per arm: " + ", ".join(f"{a} {n_runs[a]}" for a in arms) + "."
    )
    if failed:
        sub += f" {failed} failed or ungraded run(s) count as 0."

    height = max(7.0, 1.6 + 0.5 * len(artifacts) * len(arms))
    fig, ax = plt.subplots(figsize=(13, height))
    fig.subplots_adjust(left=0.30, right=0.95, top=1 - 1.4 / height, bottom=0.9 / height)
    _title(fig, title, sub)
    bar_h = 0.8 / len(arms)
    for j, arm in enumerate(arms):
        for i, name in enumerate(artifacts):
            vals = scores.get((name, arm), [])
            if not vals:
                continue
            y = i + (j - (len(arms) - 1) / 2) * bar_h
            m = _mean(vals)
            ax.barh(y, m, height=bar_h * 0.92, color=colour.get(arm, GREY),
                    label=arm if i == 0 else None)
            if len(vals) > 1:
                ax.errorbar(m, y, xerr=[[m - min(vals)], [max(vals) - m]], color=INK,
                            capsize=4, lw=1.6)
            ax.text(max(vals) + 0.015, y, f"{m:.2f}", va="center", fontsize=12)
    ax.set_yticks(range(len(artifacts)))
    ax.set_yticklabels([_short(a) for a in artifacts])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Official score (0 to 1)")
    ax.legend(loc="lower right", fontsize=14, frameon=False)
    return _save(fig, out, "task9_swarm_vs_baseline")


def chart_task9_per_case(runs: list[dict], out: Path) -> list[Path]:
    cases = list(dict.fromkeys(r["case"] for r in runs))
    arms = [a for a in ("baseline", "swarm") if any(r["arm"] == a for r in runs)]
    colour = {"swarm": OURS, "baseline": OTHER}
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.80, bottom=0.18)
    means: dict[tuple[str, str], float] = {}
    for x, case in enumerate(cases):
        for arm in arms:
            vals = [_run_score(r) for r in runs if r["case"] == case and r["arm"] == arm]
            if not vals:
                continue
            m = _mean(vals)
            means[(case, arm)] = m
            jitter = [(i - (len(vals) - 1) / 2) * 0.04 for i in range(len(vals))]
            ax.scatter([x + j for j in jitter], vals, s=60, color=colour.get(arm, GREY),
                       alpha=0.35, zorder=2)
            ax.scatter([x], [m], s=260, color=colour.get(arm, GREY), edgecolor=INK,
                       zorder=3, label=arm if x == 0 else None)
            ax.text(x + 0.12, m, f"{m:.2f}", va="center", fontsize=14, fontweight="bold")
        if len(arms) == 2 and all((case, a) in means for a in arms):
            ax.plot([x, x], [means[(case, arms[0])], means[(case, arms[1])]], color=GREY,
                    lw=2, zorder=1)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([_short(c, 22) for c in cases])
    ax.set_xlim(-0.6, len(cases) - 0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Mean official score")
    ax.legend(loc="upper right", fontsize=14, frameon=False)
    n = {a: sum(r["arm"] == a for r in runs) for a in arms}
    paired = [c for c in cases if (c, "swarm") in means and (c, "baseline") in means]
    if paired:
        wins = sum(means[(c, "swarm")] > means[(c, "baseline")] for c in paired)
        title = f"The swarm beats the single agent in {wins} of {len(paired)} cases"
    else:
        title = "Task 9 official score per case"
    _title(fig, title, "Pilot: counts as input. Big dot = mean over runs, faint dots = single "
           "runs; runs per arm: " + ", ".join(f"{a} {n[a]}" for a in arms) + ".")
    return _save(fig, out, "task9_swarm_vs_baseline_per_case")


# ---------------------------------------------------------------- faults


def chart_fault_detection(out: Path) -> list[Path]:
    rows = list(FAULT_DETECTION)
    cells = [e for v in FAULT_DETECTION.values() for e in v if e]
    caught = sum(c for c, _ in cells)
    planted = sum(p for _, p in cells)
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.78, bottom=0.04)
    ax.set_xlim(0, len(FAULT_DATASETS) + 3.0)
    ax.set_ylim(len(rows) + 1.2, -1.2)
    ax.axis("off")
    for j, d in enumerate(FAULT_DATASETS):
        ax.text(2.0 + j + 0.5, -0.65, d, ha="center", va="center", fontsize=15,
                fontweight="bold")
    for i, name in enumerate(rows):
        ax.text(1.85, i + 0.5, name, ha="right", va="center", fontsize=15)
        for j, e in enumerate(FAULT_DETECTION[name]):
            x = 2.0 + j
            if e is None:
                ax.add_patch(plt.Rectangle((x + 0.04, i + 0.06), 0.92, 0.88, color="#E4E4E0"))
                ax.text(x + 0.5, i + 0.5, "n/a", ha="center", va="center", fontsize=14,
                        color="#777777")
                continue
            c, p = e
            full = c == p
            ax.add_patch(plt.Rectangle((x + 0.04, i + 0.06), 0.92, 0.88,
                                       color=OURS if full else OTHER))
            ax.text(x + 0.5, i + 0.5, f"{c}/{p}", ha="center", va="center", fontsize=20,
                    fontweight="bold", color="white" if full else INK)
    y = len(rows)
    ax.text(1.85, y + 0.5, "Caught", ha="right", va="center", fontsize=15, fontweight="bold")
    for j in range(len(FAULT_DATASETS)):
        es = [FAULT_DETECTION[r][j] for r in rows if FAULT_DETECTION[r][j]]
        ax.text(2.0 + j + 0.5, y + 0.5, f"{sum(e[0] for e in es)}/{sum(e[1] for e in es)}",
                ha="center", va="center", fontsize=20, fontweight="bold")
    xa = 2.0 + len(FAULT_DATASETS) + 0.2
    ax.text(xa, -0.65, "All", ha="left", va="center", fontsize=15, fontweight="bold")
    ax.text(xa, y + 0.5, f"{caught}/{planted}", ha="left", va="center", fontsize=24,
            fontweight="bold", color=OURS)
    _title(fig, f"The swarm catches {caught} of {planted} planted faults on real data",
           "Every miss is a label inversion on a single-cell-line design, where the check "
           "abstains. 2 plants per fault per dataset: a sanity check, not a benchmark.")
    return _save(fig, out, "fault_detection_real_data")


# ---------------------------------------------------------------- airway


def chart_airway_label_swap(out: Path) -> list[Path]:
    donors = list(AIRWAY)
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.78, bottom=0.2)
    w = 0.34
    for i, d in enumerate(donors):
        alb, albdex, dex = AIRWAY[d]
        for off, v, col, lab in (
            (-w / 2, alb, OTHER, 'Labelled "albuterol"'),
            (w / 2, albdex, OURS, 'Labelled "albuterol + dex"'),
        ):
            ax.bar(i + off, v, width=w, color=col, label=lab if i == 0 else None)
            ax.text(i + off, max(v, 0) + 0.12, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=15, fontweight="bold", zorder=5,
                    bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.5})
        ax.plot([i - w - 0.04, i + w + 0.04], [dex, dex], color=INK, lw=2, ls=(0, (4, 2)),
                zorder=1,
                label="Dex-only sample, for scale" if i == 0 else None)
        if d in AIRWAY_SWAPPED:
            ax.text(i, 4.3, "SWAPPED", ha="center", fontsize=16, fontweight="bold",
                    color="#B33A00")
    ax.set_xticks(range(len(donors)))
    ax.set_xticklabels(donors)
    ax.set_ylim(-0.4, 4.7)
    ax.set_ylabel("Dexamethasone-signature score")
    ax.set_xlabel("Donor")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=13,
              frameon=False)
    _title(fig, 'In 2 of 4 donors the "albuterol" sample carries the full dex response',
           "Airway dataset (GSE52778). Inferred from the data alone; not confirmed by the "
           "dataset authors. Score: dex signature vs the donor's untreated sample.")
    return _save(fig, out, "airway_label_swap")


# ---------------------------------------------------------------- calibration


def chart_predictor_calibration(path: Path, out: Path) -> list[Path]:
    methods = json.loads(path.read_text(encoding="utf-8"))["methods"]
    ll = {
        (m, s): methods[m][s]["stated"]["log_loss"]
        for m, _ in PREDICTORS
        if m in methods
        for s in ("test", "shift")
        if s in methods[m] and "stated" in methods[m][s]
    }
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.74, bottom=0.14)
    w = 0.36
    shown = [(m, lab) for m, lab in PREDICTORS if (m, "test") in ll]
    for i, (m, _) in enumerate(shown):
        col = OURS if m == "finetuned" else OTHER
        for off, s, hatch in ((-w / 2, "test", None), (w / 2, "shift", "///")):
            v = ll.get((m, s))
            if v is None:
                continue
            ax.bar(i + off, v, width=w, color=col, hatch=hatch,
                   edgecolor="white" if hatch else col, lw=0)
            ax.text(i + off, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=14,
                    fontweight="bold")
            ax.text(i + off, 0.02, s, ha="center", va="bottom", fontsize=12,
                    color="white" if col == OURS else INK)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([lab for _, lab in shown])
    ax.set_ylabel("Log loss (nats, lower is better)")
    ax.set_ylim(0, max(ll.values()) * 1.15)
    t, a = ll.get(("finetuned", "test")), ll.get(("library", "test"))
    ts, as_ = ll.get(("finetuned", "shift")), ll.get(("library", "shift"))
    if None not in (t, a, ts, as_):
        title = (f"Fine-tuning beats the stated rows in distribution ({t:.2f} vs {a:.2f}), "
                 f"but only ties them under shift ({ts:.2f} vs {as_:.2f})")
    else:
        title = "Predictor log loss on the stated-row forks"
    fig.text(0.04, 0.955, "\n".join(_wrap(title, 62)), fontsize=20, fontweight="bold",
             ha="left", va="top", linespacing=1.15)
    fig.text(0.04, 0.84, "Synthetic cases, stated-row forks only (sample_labels, "
             "group_direction). test = training distribution; shift = weaker effects or 3 "
             "replicates.", fontsize=13, color="#555555", ha="left", va="top")
    return _save(fig, out, "predictor_calibration")


# ---------------------------------------------------------------- main


def build(task9: list[Path], calibration: Path, out: Path) -> list[Path]:
    _style()
    made: list[Path] = []
    runs, used = load_task9(task9)
    if runs:
        print(f"task9: using {', '.join(str(p) for p in used)}")
        made += chart_task9_swarm_vs_baseline(runs, out)
        made += chart_task9_per_case(runs, out)
    else:
        print("skipped task9_swarm_vs_baseline: no readable results.json "
              f"({', '.join(str(p) for p in task9)})")
    made += chart_fault_detection(out)
    made += chart_airway_label_swap(out)
    if calibration.is_file():
        made += chart_predictor_calibration(calibration, out)
    else:
        print(f"skipped predictor_calibration: {calibration} not found")
    for p in made:
        print(f"wrote {p}")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task9", nargs="*", type=Path, default=DEFAULT_TASK9,
                    help="results.json file(s) from eval/run_task9_arms.py; each file is one case")
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build(args.task9, args.calibration, args.out)


if __name__ == "__main__":
    main()
