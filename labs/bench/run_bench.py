"""Standalone bench loop: pick a grey level, measure it on the node, update, repeat.

    python -m labs.bench.run_bench --node http://<jetson>:8765 --strategy eig --max 10 \\
        --seed 0 --events runs/<run_id>/events.jsonl [--tint-after 6 --tint 0.5]
    python -m labs.bench.run_bench --node http://<jetson>:8765 --compare --max 12

Writes events in the format of docs/EVENTS.md (plain dicts, no dependency on core.schemas).
"""

import argparse
import json
import random
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.eig_exec import NULL_NAME, ParticleBelief
from labs.bench.lab import DEFAULT_SIGMA, BenchClient, BenchLab, exposure_families

THRESHOLD = 0.95
QUESTION_ID = "q1"
CURVE_X = [float(v) for v in np.linspace(0.0, 255.0, 32)]


def hyp_id(name: str) -> str:
    return NULL_NAME if name == NULL_NAME else f"h-{name}"


def level_id(level: float) -> str:
    return f"level={round(level)}"


class EventWriter:
    """Envelope writer: contiguous seq from 0, one JSON object per line, flushed per event."""

    def __init__(self, run_id: str, path=None) -> None:
        self.run_id = run_id
        self.seq = 0
        self.fh = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(path, "w", encoding="utf-8")  # noqa: SIM115

    def emit(self, type_: str, payload: dict) -> None:
        if self.fh is None:
            return
        event = {
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": time.time(),
            "type": type_,
            "payload": payload,
        }
        self.seq += 1
        self.fh.write(json.dumps(event) + "\n")
        self.fh.flush()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def _hypothesis(name, belief, families, prior, mode="screen") -> dict:
    is_null = name == NULL_NAME
    if is_null:
        statement = (
            "None of these: the proposed response laws do not explain the measurements"
        )
    else:
        response = (
            "exposure-to-brightness" if mode == "exposure" else "screen-to-camera"
        )
        statement = f"The {response} response follows a {name} law"
    return {
        "id": hyp_id(name),
        "question_id": QUESTION_ID,
        "statement": statement,
        "author": "bench",
        "rationale": "built-in null"
        if is_null
        else f"{name} family from labs/bench/lab.py",
        "answer_key": None,
        "model_code": None,
        "is_null": is_null,
        "prior": prior,
        "posterior": prior,
        "status": "active",
        "rejected_by": [],
    }


def _curves(belief, families, points) -> dict:
    """Per-family weighted mean and sd of the predicted curve over CURVE_X."""
    xs = np.array(CURVE_X).reshape(-1, 1)
    post = belief.family_posterior()
    out = {}
    for fam in families:
        params, w = belief.particles(fam.name)
        pred = np.asarray(fam.predict(xs, params), dtype=float)
        mean = w @ pred
        sd = np.sqrt(np.maximum(w @ (pred - mean) ** 2, 0.0))
        out[hyp_id(fam.name)] = {
            "mean": [float(v) for v in mean],
            "sd": [float(v) for v in sd],
            "weight": float(post[fam.name]),
        }
    if belief.null_name is not None:
        params, w = belief.particles(NULL_NAME)
        c = float(w @ params[:, 0])
        sd = float(np.sqrt(max(w @ (params[:, 0] - c) ** 2, 0.0)))
        out[NULL_NAME] = {
            "mean": [c] * len(CURVE_X),
            "sd": [sd] * len(CURVE_X),
            "weight": float(post[NULL_NAME]),
        }
    return {
        "question_id": QUESTION_ID,
        "x": CURVE_X,
        "families": out,
        "points": [{"x": float(x), "y": float(y)} for x, y in points],
    }


def _by_id(post: dict) -> dict:
    return {hyp_id(k): float(v) for k, v in post.items()}


def _best_proposed(post: dict, null_name) -> tuple:
    items = [(k, v) for k, v in post.items() if k != null_name]
    return max(items, key=lambda kv: kv[1])


def run_loop(
    node: str,
    strategy: str = "eig",
    max_n: int = 10,
    seed: int = 0,
    events_path=None,
    tint_after=None,
    tint: float = 0.5,
    sigma: float = DEFAULT_SIGMA,
    run_id=None,
    verbose: bool = True,
    n_particles: int = 500,
    timeout: float = 30.0,
    mode: str = "screen",
) -> dict:
    """Run one experiment-selection loop against ``node``. Returns a summary dict."""
    if strategy not in ("eig", "random"):
        raise ValueError("strategy must be 'eig' or 'random'")
    if mode not in ("screen", "exposure"):
        raise ValueError("mode must be screen or exposure")
    started = time.time()
    run_id = (
        run_id or "r-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    )
    client = BenchClient(node, timeout=timeout)
    lab = BenchLab(client, max_measurements=max_n)
    health = client.health()
    node_mode = health.get("camera_settings", {}).get("experiment", "screen")
    if node_mode != mode:
        raise ValueError(f"requested {mode} experiment but node reports {node_mode}")
    if health.get("fake"):
        client.seed(seed)
        client.tint(1.0)
    families = exposure_families() if mode == "exposure" else lab.families()
    belief = ParticleBelief(families, sigma=sigma, n_particles=n_particles, seed=seed)
    ev = EventWriter(run_id, events_path)
    rng = random.Random(seed)
    null_name = belief.null_name
    prior = belief.family_posterior()

    ev.emit(
        "run.started",
        {
            "lab": "bench",
            "goal": (
                "Which law maps normalized manual exposure (0-255) to camera brightness?"
                if mode == "exposure"
                else "Which law maps screen grey level to camera value?"
            ),
            "config": {
                "strategy": strategy,
                "mode": mode,
                "camera_settings": health.get("camera_settings", {}),
                "max": max_n,
                "seed": seed,
                "sigma": sigma,
                "threshold": THRESHOLD,
                "tint_after": tint_after,
                "tint": tint if tint_after is not None else None,
                "fake": bool(health.get("fake")),
            },
            "started_at": datetime.now(UTC).isoformat(),
        },
    )
    ev.emit(
        "question.opened",
        {
            "question": {
                "id": QUESTION_ID,
                "text": (
                    "How does camera brightness respond to normalized exposure (0-255)?"
                    if mode == "exposure"
                    else "What is the screen-to-camera response law?"
                ),
                "kind": "model",
                "impact": 1.0,
                "status": "open",
                "raised_by": "bench",
            }
        },
    )
    for name in belief.family_names:
        ev.emit(
            "hypothesis.proposed",
            {"hypothesis": _hypothesis(name, belief, families, prior[name], mode)},
        )
    kept = _best_proposed(prior, null_name)[0]
    ev.emit(
        "hypothesis.merged",
        {
            "question_id": QUESTION_ID,
            "kept": hyp_id(kept),
            "merged": [],
            "priors": _by_id(prior),
        },
    )

    points: list = []
    measured: set = set()
    settled_at = None
    n_to_threshold = None
    n_surprises = 0
    last_post = prior
    measurements = []
    for step in range(max_n):
        if tint_after is not None and step == tint_after and health.get("fake"):
            client.tint(tint)
            if verbose:
                print(f"-- tint {tint} applied after {step} measurements --")
        remaining = [d for d in lab.candidate_designs() if d not in measured]
        if not remaining:
            break
        if strategy == "eig" or events_path is not None:
            ranked = belief.rank_designs(remaining, goal="family")
        else:
            ranked = []
        if strategy == "eig":
            chosen = float(ranked[0][0])
        else:
            chosen = float(rng.choice(remaining))
        eid = level_id(chosen)
        ev.emit(
            "experiment.scored",
            {
                "question_id": QUESTION_ID,
                "ranking": [
                    {
                        "experiment_id": level_id(d),
                        "kind": "measure",
                        "goal": "family",
                        "eig_bits": float(e),
                        "est_cost_s": 1.0,
                        "est_cost_usd": 0.0,
                        "score": float(s),
                    }
                    for d, e, s in ranked
                ],
                "chosen": eid,
            },
        )
        ev.emit(
            "experiment.started",
            {"experiment_id": eid, "question_ids": [QUESTION_ID], "kind": "measure"},
        )
        t0 = time.time()
        client.patch(chosen)
        measurement = client.measure()
        value = float(measurement["mean"])
        if (
            mode == "exposure"
            and not health.get("fake")
            and measurement.get("controls_verified") is not True
        ):
            raise ValueError("real exposure measurement lacks verified camera controls")
        measurements.append({"setting": chosen, **measurement})
        seconds = time.time() - t0
        measured.add(chosen)
        points.append((chosen, value))
        before = belief.family_posterior()
        res = belief.update(chosen, value)
        last_post = res.family_posterior
        ev.emit(
            "experiment.finished",
            {
                "observation": {
                    "experiment_id": eid,
                    "valid": True,
                    "value": value,
                    "bin_id": None,
                    "surprise": float(res.neg_log_pred),
                    "referee_note": (
                        f"Exposure {measurement.get('exposure_ms')} ms; controls verified: "
                        f"{measurement.get('controls_verified')}"
                        if mode == "exposure"
                        else ""
                    ),
                    "log_ref": "",
                    "seconds": seconds,
                    "cost_usd": 0.0,
                }
            },
        )
        ev.emit(
            "belief.updated",
            {
                "question_id": QUESTION_ID,
                "experiment_id": eid,
                "observed": value,
                "prior": _by_id(before),
                "posterior": _by_id(res.family_posterior),
            },
        )
        ev.emit("model.curves", _curves(belief, families, points))
        if res.surprise:
            n_surprises += 1
            null_mass = res.family_posterior.get(NULL_NAME)
            reason = (
                f"observed {value:.1f} at {eid}; model predicted {res.pred_mean:.1f} "
                f"+/- {res.pred_sd:.1f} (z={res.z_score:.1f})"
            )
            ev.emit(
                "surprise.raised",
                {
                    "question_id": QUESTION_ID,
                    "experiment_id": eid,
                    "reason": reason,
                    "z_score": float(res.z_score),
                    "null_mass": None if null_mass is None else float(null_mass),
                },
            )
        top, top_p = _best_proposed(res.family_posterior, null_name)
        if verbose:
            post_s = " ".join(f"{k}={v:.2f}" for k, v in res.family_posterior.items())
            flag = " SURPRISE" if res.surprise else ""
            print(
                f"[{step + 1:2d}] {'exposure_setting' if mode == 'exposure' else 'level'}="
                f"{int(chosen):3d} value={value:6.1f} "
                f"| {post_s} | z={res.z_score:+.1f}{flag}"
            )
        if top_p >= THRESHOLD and settled_at is None:
            settled_at = step + 1
            n_to_threshold = step + 1
            ev.emit(
                "question.settled",
                {
                    "question_id": QUESTION_ID,
                    "hypothesis_id": hyp_id(top),
                    "posterior": top_p,
                },
            )
            if tint_after is None:
                break

    top, top_p = _best_proposed(last_post, null_name)
    if settled_at is None:
        ev.emit(
            "question.unresolved",
            {
                "question_id": QUESTION_ID,
                "reason": f"no family reached {THRESHOLD} within {len(points)} measurements",
                "top": [hyp_id(top)],
            },
        )
    ev.emit(
        "run.finished",
        {
            "report": {
                "run_id": run_id,
                "artifacts": [],
                "methods_md": (
                    f"Strategy {strategy}, {len(points)} measurements, "
                    f"top family {top} at {top_p:.3f}."
                ),
                "rejected": [],
                "unresolved": [] if settled_at is not None else [QUESTION_ID],
                "cost_usd": 0.0,
                "seconds": time.time() - started,
            },
            "status": "ok",
        },
    )
    ev.close()
    return {
        "run_id": run_id,
        "strategy": strategy,
        "seed": seed,
        "n_measured": len(points),
        "reached": settled_at is not None,
        "n_to_threshold": n_to_threshold,
        "top": top,
        "top_posterior": top_p,
        "surprises": n_surprises,
        "posterior": last_post,
        "mode": mode,
        "measurements": measurements,
        "fake": bool(health.get("fake")),
    }


def compare(
    node: str, seeds=range(10), max=12, quiet: bool = False, mode: str = "screen"
) -> dict:
    """Mean measurements to reach 0.95 for EIG and random (runs that never get there count as max).

    Every seed runs both strategies on a freshly seeded node, so the noise stream is the same
    kind of draw for both. Returns ``{"eig": mean, "random": mean, "runs": {...}}``.
    """
    seeds = list(seeds)
    runs = {}
    reached = {}
    details = {}
    for strategy in ("eig", "random"):
        counts = []
        converged = []
        trials = []
        for s in seeds:
            r = run_loop(
                node, strategy=strategy, max_n=max, seed=s, verbose=False, mode=mode
            )
            counts.append(r["n_to_threshold"] if r["reached"] else max)
            converged.append(r["reached"])
            trials.append(r)
        runs[strategy] = counts
        reached[strategy] = converged
        details[strategy] = trials
    means = {k: sum(v) / len(v) for k, v in runs.items()}
    if not quiet:
        print(
            f"measurements to reach {THRESHOLD} posterior (max {max}, {len(seeds)} seeds)"
        )
        print(f"{'strategy':<10}{'mean':>8}{'never reached':>16}")
        for k in ("eig", "random"):
            missed = sum(not value for value in reached[k])
            print(f"{k:<10}{means[k]:>8.2f}{missed:>16d}")
    return {
        "eig": means["eig"],
        "random": means["random"],
        "runs": runs,
        "reached": reached,
        "mode": mode,
        "details": details,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Crux bench loop")
    p.add_argument("--node", default="http://127.0.0.1:8765")
    p.add_argument("--mode", choices=["screen", "exposure"], default="screen")
    p.add_argument("--strategy", choices=["eig", "random"], default="eig")
    p.add_argument("--max", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--events", default=None, help="path of the events.jsonl to write")
    p.add_argument(
        "--summary",
        type=Path,
        help="save JSON run/comparison details and simulation flags",
    )
    p.add_argument("--tint-after", type=int, default=None)
    p.add_argument("--tint", type=float, default=0.5)
    p.add_argument(
        "--compare", action="store_true", help="EIG vs random over seeds 0-9"
    )
    args = p.parse_args(argv)
    if args.compare:
        summary = compare(args.node, seeds=range(10), max=args.max, mode=args.mode)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
        return
    r = run_loop(
        args.node,
        strategy=args.strategy,
        max_n=args.max,
        seed=args.seed,
        events_path=args.events,
        tint_after=args.tint_after,
        tint=args.tint,
        mode=args.mode,
    )
    verdict = "reached" if r["reached"] else "did not reach"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(
        f"{r['top']} {r['top_posterior']:.3f}: {verdict} {THRESHOLD} "
        f"after {r['n_measured']} measurements"
    )


if __name__ == "__main__":
    main()
