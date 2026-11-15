"""Laptop-side camera checks and recorded rehearsals for DU-44.

Run with ``python -m labs.bench.preflight --node http://JETSON:8765
--camera-locked --rehearse`` after manually locking exposure, gain and white balance.
Fake-node reports can exercise the workflow but never qualify as hardware evidence.
"""

import argparse
import json
import math
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from labs.bench.lab import DEFAULT_SIGMA, BenchClient
from labs.bench.run_bench import THRESHOLD, run_loop


def _measurement(client: BenchClient) -> dict:
    raw = client.measure()
    mean, sd, n = float(raw["mean"]), float(raw["std"]), raw["n"]
    if not math.isfinite(mean) or not 0 <= mean <= 255:
        raise ValueError("camera mean must be finite and in 0..255")
    if not math.isfinite(sd) or sd < 0:
        raise ValueError("camera standard deviation must be finite and nonnegative")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("camera frame count must be a positive integer")
    result = {"mean": mean, "std": sd, "n": n}
    for key in ("exposure_units", "exposure_ms", "controls_verified"):
        if key in raw:
            result[key] = raw[key]
    return result


def _stability(measurements: list[dict]) -> dict:
    """Trend is fitted brightness change over the actual measurement timestamps."""
    ys = [m["mean"] for m in measurements]
    xs = [m["elapsed_s"] for m in measurements]
    xmean, ymean = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - xmean) ** 2 for x in xs)
    slope = (
        sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys, strict=True))
        / denominator
        if denominator > 0
        else 0.0
    )
    return {
        "count": len(ys),
        "mean": ymean,
        "sample_sd": statistics.stdev(ys),
        "peak_to_peak": max(ys) - min(ys),
        "fitted_change": slope * (xs[-1] - xs[0]),
        "duration_s": xs[-1] - xs[0],
    }


def run_preflight(
    node: str,
    *,
    camera_locked: bool = False,
    rehearse: bool = False,
    output: str | Path = "runs/bench-preflight/report.json",
    max_spread: float = 1.5,
    max_drift: float = 1.5,
    max_black: float = 15.0,
    max_white: float = 245.0,
    min_range: float | None = None,
    sigma: float = DEFAULT_SIGMA,
    max_n: int = 10,
    seconds_limit: float = 60.0,
    timeout: float = 5.0,
    mode: str = "screen",
) -> dict:
    """Collect evidence; do not infer exposure lock from OpenCV property numbers.

    Twenty *measurement batches* are sampled, each using the node's configured frame
    averaging. Screen mode needs operator confirmation of locked camera controls;
    exposure mode uses the node's V4L2 readback checks instead.
    Rehearsals are skipped if any preflight check fails.
    """
    if mode not in ("screen", "exposure"):
        raise ValueError("mode must be screen or exposure")
    min_range = (
        (30.0 if mode == "exposure" else 100.0) if min_range is None else min_range
    )
    positive = (
        max_spread,
        max_drift,
        max_black,
        max_white,
        min_range,
        sigma,
        seconds_limit,
        timeout,
    )
    if any(not math.isfinite(v) or v <= 0 for v in positive):
        raise ValueError(
            "thresholds, sigma, timeout and time limit must be finite and positive"
        )
    if not 1 <= max_n <= 16:
        raise ValueError("max_n must be between 1 and 16")

    output = Path(output)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "node": node,
        "mode": mode,
        "health": None,
        "fake": None,
        "jetson_detected": False,
        "camera_lock_operator_confirmed": camera_locked,
        "thresholds": {
            "max_peak_to_peak": max_spread,
            "max_abs_fitted_change": max_drift,
            "max_black": max_black,
            "max_white_exclusive": max_white,
            "min_dynamic_range": min_range,
            "run_seconds_exclusive": seconds_limit,
            "posterior": THRESHOLD,
            "max_measurements": max_n,
            "sigma": sigma,
        },
        "checks": {},
        "measurements": [],
        "rehearsals_requested": rehearse,
        "rehearsals": [],
        "errors": [],
    }
    checks = report["checks"]

    def check(name, passed, detail):
        checks[name] = {"passed": bool(passed), "detail": detail}

    client = BenchClient(node, timeout=timeout)
    started = time.perf_counter()
    patched = False
    try:
        health = client.health()
        report["health"] = health
        if health.get("ok") is not True or not isinstance(health.get("fake"), bool):
            raise ValueError(
                "node health must report ok=true and an explicit fake boolean"
            )
        fake = report["fake"] = health["fake"]
        settings = health.get("camera_settings", {})
        node_mode = settings.get("experiment", "screen")
        if node_mode != mode:
            raise ValueError(
                f"requested {mode} experiment but node reports {node_mode}"
            )
        report["jetson_detected"] = bool(health.get("board"))
        ready = fake or (
            health.get("display") == "headless-exposure"
            if mode == "exposure"
            else health.get("display") in ("cv2", "tk")
        )
        check("display_ready", ready, health.get("display"))
        check(
            "camera_lock_confirmed",
            fake
            or (
                settings.get("controls_verified") is True
                if mode == "exposure"
                else camera_locked
            ),
            "not applicable to simulation"
            if fake
            else "V4L2 manual controls verified by node"
            if mode == "exposure"
            else "operator confirmation; not auto-detected",
        )
        if not ready:
            raise ValueError(
                "real node has no active display; no brightness measurements taken"
            )

        # Each call samples the same patch, so changes cannot be explained by the design.
        patched = True
        client.patch(128)
        for _ in range(20):
            value = _measurement(client)
            value["elapsed_s"] = time.perf_counter() - started
            report["measurements"].append(value)
        stability = report["stability"] = _stability(report["measurements"])
        check(
            "stable_spread",
            stability["peak_to_peak"] <= max_spread,
            stability["peak_to_peak"],
        )
        check(
            "no_drift",
            abs(stability["fitted_change"]) <= max_drift,
            stability["fitted_change"],
        )

        client.patch(0)
        black = report["black"] = _measurement(client)
        client.patch(255)
        white = report["white"] = _measurement(client)
        if mode == "exposure":
            # End-point saturation is a rival response law, not automatically a bad sensor.
            check(
                "mid_exposure_not_clipped",
                5.0 < stability["mean"] < 245.0,
                stability["mean"],
            )
        else:
            check("black_level", black["mean"] <= max_black, black["mean"])
            check("white_not_clipped", white["mean"] < max_white, white["mean"])
        dynamic_range = white["mean"] - black["mean"]
        check("dynamic_range", dynamic_range >= min_range, dynamic_range)
    except Exception as exc:  # noqa: BLE001 - persist failed camera/HTTP qualification
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if patched:
            try:
                client.patch(128)
            except Exception as exc:  # noqa: BLE001 - record cleanup failures too
                report["errors"].append(f"could not restore mid-grey: {exc}")

    report["preflight_passed"] = (
        bool(checks)
        and all(c["passed"] for c in checks.values())
        and not report["errors"]
    )
    if rehearse and report["preflight_passed"]:
        session = uuid.uuid4().hex[:10]
        for index in range(3):
            run_id = f"bench-check-{session}-{index + 1}"
            events = output.parent / (output.stem + "-runs") / run_id / "events.jsonl"
            run_started = time.perf_counter()
            try:
                result = run_loop(
                    node,
                    max_n=max_n,
                    seed=index,
                    events_path=events,
                    run_id=run_id,
                    sigma=sigma,
                    verbose=False,
                    timeout=timeout,
                    mode=mode,
                )
                elapsed = time.perf_counter() - run_started
                passed = (
                    result["reached"]
                    and result["top_posterior"] >= THRESHOLD
                    and result["surprises"] == 0
                    and elapsed < seconds_limit
                )
                report["rehearsals"].append(
                    {
                        "passed": bool(passed),
                        "seconds": elapsed,
                        "events": str(events),
                        **result,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - save partial rehearsal evidence
                report["errors"].append(
                    f"rehearsal {index + 1}: {type(exc).__name__}: {exc}"
                )
                break

    report["three_runs_passed"] = len(report["rehearsals"]) == 3 and all(
        r["passed"] for r in report["rehearsals"]
    )
    passed = (
        report["preflight_passed"]
        and not report["errors"]
        and (not rehearse or report["three_runs_passed"])
    )
    report["jetson_rehearsal_passed"] = bool(
        passed
        and report["three_runs_passed"]
        and report["jetson_detected"]
        and report["fake"] is False
    )
    report["status"] = (
        "failed" if not passed else "simulation_only" if report["fake"] else "passed"
    )
    report["seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="http://127.0.0.1:8765")
    parser.add_argument("--mode", choices=["screen", "exposure"], default="screen")
    parser.add_argument("--output", default="runs/bench-preflight/report.json")
    parser.add_argument(
        "--camera-locked",
        action="store_true",
        help="confirm exposure, gain and white balance were locked",
    )
    parser.add_argument(
        "--rehearse",
        action="store_true",
        help="record three consecutive timed EIG runs",
    )
    parser.add_argument(
        "--max-spread", type=float, default=1.5, help="mid-grey peak-to-peak limit"
    )
    parser.add_argument(
        "--max-drift",
        type=float,
        default=1.5,
        help="absolute fitted brightness change limit",
    )
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--max", type=int, default=10, dest="max_n")
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="HTTP timeout in seconds"
    )
    args = parser.parse_args(argv)
    try:
        report = run_preflight(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{report['status']}: report saved to {args.output}")
    print(f"Jetson three-run qualification: {report['jetson_rehearsal_passed']}")
    for name, check in report["checks"].items():
        if not check["passed"]:
            print(f"FAIL {name}: {check['detail']}")
    for error in report["errors"]:
        print(f"ERROR {error}")
    return {"passed": 0, "failed": 1, "simulation_only": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
