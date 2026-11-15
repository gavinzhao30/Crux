"""Command line for the One Letter engine.

    python -m labs.oneletter.run --reader http://<jetson>:8770 [--events PATH] [--timeout S]
        [--watch | --strips N] [--lessons PATH]

    # no hardware: a fake reader over HTTP on a free port; HbS, then the judge loads HbC
    python -m labs.oneletter.run --fake hbs,hbc --events runs/demo/oneletter_fake.jsonl

``--reader`` defaults to ``$CRUX_ONELETTER_READER``. Events go to
``<runs-dir>/<run_id>/events.jsonl`` unless ``--events`` names a new file. ``--watch`` keeps
reading the settled window after each settle and reopens the question when its letters
change (the judge's key press on the strip page); stop it with Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from core.events import new_run_id
from core.schemas import RunEvent
from labs.oneletter.lab import (
    DEFAULT_LESSONS,
    READER_ENV,
    OneLetterConfig,
    run_oneletter,
)


def _printer(event: RunEvent) -> None:
    p, t = event.payload, event.type
    if t == "experiment.scored":
        bits = ", ".join(f"{r['experiment_id']}={r['eig_bits']:.2f}" for r in p["ranking"])
        line = f"scored  {bits} -> {p['chosen']}"
    elif t == "dna.request":
        line = (
            f"request window {p['window']}{' (replicate)' if p['replicate'] else ''}: {p['reason']}"
        )
    elif t == "dna.read":
        line = f"read    window {p['window']} = {p['bases']} ({p['bin_id']})"
    elif t == "belief.updated":
        line = "belief  " + " ".join(f"{k}={v:.3f}" for k, v in p["posterior"].items())
        line += " (replay)" if p.get("replay") else ""
    elif t == "surprise.raised":
        line = f"SURPRISE {p['reason']}"
    elif t == "hypothesis.proposed":
        line = f"hypothesis {p['hypothesis']['id']}: {p['hypothesis']['statement']}"
    elif t == "question.settled":
        line = f"SETTLED {p['hypothesis_id']} at {p['posterior']:.3f}"
    elif t == "question.unresolved":
        line = f"unresolved: {p['reason']}"
    elif t == "question.opened" and p.get("reopened"):
        line = f"REOPENED strip {p['strip']}: {p['reason']}"
    elif t == "dna.verdict":
        line = f"verdict [{p['author']}] {p['text']}"
    elif t == "dna.lesson":
        line = f"lesson  added {p['added']} ({p['lessons_total']} in total)"
    elif t in ("run.finished", "run.error"):
        line = f"{t} {p.get('status', p.get('message', ''))}"
    else:
        return
    print(f"[{event.seq:3d}] {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One Letter: which HBB variant is on the strip?")
    ap.add_argument("--reader", default=os.environ.get(READER_ENV), help="reader service URL")
    ap.add_argument("--events", default=None, help="write events here (a new file)")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait per window")
    ap.add_argument("--watch", action="store_true", help="keep watching for new samples")
    ap.add_argument("--strips", type=int, default=None, help="samples to settle, then stop")
    ap.add_argument("--watch-timeout", type=float, default=None)
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--lessons", default=DEFAULT_LESSONS.as_posix())
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fake", default=None, help="fake reader samples, e.g. hbs,hbc")
    ap.add_argument(
        "--fake-misread",
        action="append",
        default=[],
        help="sample:window:showing:bases",
    )
    ap.add_argument("--fake-show-delay", type=float, default=1.5)
    ap.add_argument("--fake-swap-delay", type=float, default=3.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.events and Path(a.events).exists():
        ap.error(f"{a.events} exists; delete it or choose another path")
    server = None
    if a.fake:
        from labs.oneletter.fake_reader import FakeReader, Misread, serve

        samples = [s.strip() for s in a.fake.split(",") if s.strip()]
        fake = FakeReader(
            samples[0],
            swaps=samples[1:],
            show_delay_s=a.fake_show_delay,
            swap_delay_s=a.fake_swap_delay,
            misreads=[Misread.parse(m) for m in a.fake_misread],
        )
        server = serve(fake)
        reader_url = server.url
        strips = a.strips or len(samples)
    else:
        if not a.reader:
            ap.error(f"--reader is required (or set {READER_ENV})")
        reader_url, strips = a.reader, a.strips or 1
    cfg = OneLetterConfig(
        reader_url=reader_url,
        timeout_s=a.timeout,
        poll_s=a.poll,
        strips=strips,
        watch=a.watch,
        watch_timeout_s=a.watch_timeout,
        lessons_path=a.lessons,
        fake_reader=server is not None,
    )
    run_id = a.run_id or new_run_id()
    where = a.events or Path(a.runs_dir, run_id, "events.jsonl").as_posix()
    if not a.quiet:
        print(f"run {run_id}: reader {reader_url}; events {where}", flush=True)
    try:
        results = asyncio.run(
            run_oneletter(
                cfg,
                run_id=run_id,
                runs_dir=a.runs_dir,
                events_path=a.events,
                on_event=None if a.quiet else _printer,
            )
        )
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130
    finally:
        if server is not None:
            server.close()
    return 0 if results and all(r.outcome == "settled" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
