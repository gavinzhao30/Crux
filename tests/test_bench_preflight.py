"""Hardware qualification must reject unstable, missing and simulated instruments."""

import json
import threading
from pathlib import Path

import pytest

from labs.bench import node, preflight


class Instrument:
    def __init__(self):
        self.level = 0
        self.mid_reads = 0
        self.white = 225.0
        self.drift = 0.0
        self.invalid = None
        self.health_data = {
            "ok": True,
            "fake": False,
            "display": "cv2",
            "board": "# R36 (release)",
        }

    def health(self):
        return self.health_data

    def patch(self, level):
        self.level = level

    def measure(self):
        if self.invalid is not None:
            return self.invalid
        if self.level == 128:
            self.mid_reads += 1
            value = 115.0 + self.drift * self.mid_reads + 0.02 * (self.mid_reads % 2)
        else:
            value = 4.0 if self.level == 0 else self.white
        return {"mean": value, "std": 0.1, "n": 8}


@pytest.fixture
def instrument(monkeypatch):
    camera = Instrument()
    monkeypatch.setattr(preflight, "BenchClient", lambda *a, **kw: camera)
    return camera


def test_real_preflight_records_measurements_but_does_not_claim_three_runs(
    instrument, tmp_path
):
    output = tmp_path / "report.json"
    report = preflight.run_preflight("http://rig", camera_locked=True, output=output)
    assert report["status"] == "passed"
    assert len(report["measurements"]) == 20
    assert report["stability"]["peak_to_peak"] < 1.5
    assert not report["three_runs_passed"] and not report["jetson_rehearsal_passed"]
    assert instrument.level == 128
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize(
    "fault", ["display", "lock", "drift", "clipping", "range", "health"]
)
def test_failed_preflight_never_starts_rehearsals(
    instrument, tmp_path, monkeypatch, fault
):
    if fault == "display":
        instrument.health_data["display"] = None
    elif fault == "drift":
        instrument.drift = 0.2
    elif fault == "clipping":
        instrument.white = 255.0
    elif fault == "range":
        instrument.white = 50.0
    elif fault == "health":
        instrument.health_data.pop("fake")

    def unexpected_run(*args, **kwargs):
        pytest.fail("rehearsal started on an unqualified instrument")

    monkeypatch.setattr(preflight, "run_loop", unexpected_run)
    report = preflight.run_preflight(
        "http://rig",
        camera_locked=fault != "lock",
        rehearse=True,
        output=tmp_path / "report.json",
    )
    assert report["status"] == "failed"
    assert not report["preflight_passed"]
    assert not report["jetson_rehearsal_passed"]
    assert report["rehearsals"] == []
    if fault == "display":
        assert instrument.mid_reads == 0


@pytest.mark.parametrize(
    "measurement",
    [
        {"mean": float("nan"), "std": 1, "n": 8},
        {"mean": 128, "std": float("inf"), "n": 8},
        {"mean": 256, "std": 1, "n": 8},
        {"mean": 128, "std": -1, "n": 8},
        {"mean": 128, "std": 1, "n": 0},
        {"mean": 128, "std": 1, "n": True},
    ],
)
def test_bad_camera_readings_produce_saved_failure(instrument, tmp_path, measurement):
    instrument.invalid = measurement
    output = tmp_path / "report.json"
    report = preflight.run_preflight("http://rig", camera_locked=True, output=output)
    assert report["status"] == "failed" and report["errors"]
    assert json.loads(output.read_text())["status"] == "failed"
    assert instrument.level == 128


def test_unreachable_node_saves_failure_report(monkeypatch, tmp_path):
    class Offline(Instrument):
        def health(self):
            raise TimeoutError("camera node offline")

    monkeypatch.setattr(preflight, "BenchClient", lambda *a, **kw: Offline())
    report = preflight.run_preflight("http://offline", output=tmp_path / "report.json")
    assert report["status"] == "failed"
    assert "offline" in report["errors"][0]


@pytest.mark.parametrize(
    "fault", [None, "deadline", "unresolved", "surprise", "exception", "not_jetson"]
)
def test_three_run_qualification(instrument, tmp_path, monkeypatch, fault):
    calls = []
    if fault == "not_jetson":
        instrument.health_data["board"] = None

    def rehearsal(node_url, **kwargs):
        calls.append(kwargs)
        if fault == "exception":
            raise TimeoutError("camera stopped responding")
        return {
            "run_id": kwargs["run_id"],
            "reached": fault != "unresolved",
            "top_posterior": 0.99,
            "surprises": int(fault == "surprise"),
            "n_measured": 6,
        }

    monkeypatch.setattr(preflight, "run_loop", rehearsal)
    # Every timed section takes one second; the deadline comparison is strict.
    counter = iter(float(i) for i in range(200))
    monkeypatch.setattr(preflight.time, "perf_counter", lambda: next(counter))
    report = preflight.run_preflight(
        "http://rig",
        camera_locked=True,
        rehearse=True,
        output=tmp_path / "report.json",
        seconds_limit=1.0 if fault == "deadline" else 60.0,
    )
    expected_pass = fault in (None, "not_jetson")
    assert report["three_runs_passed"] is expected_pass
    assert report["jetson_rehearsal_passed"] is (fault is None)
    assert report["status"] == ("passed" if expected_pass else "failed")
    assert len(calls) == (1 if fault == "exception" else 3)
    assert len({c["run_id"] for c in calls}) == len(calls)
    assert all(c["events_path"].name == "events.jsonl" for c in calls)


def test_fake_instrument_cannot_qualify_as_real_even_with_board_and_lock(
    instrument, tmp_path
):
    instrument.health_data["fake"] = True
    instrument.health_data["display"] = None
    report = preflight.run_preflight(
        "http://fake", camera_locked=True, output=tmp_path / "report.json"
    )
    assert report["status"] == "simulation_only"
    assert report["preflight_passed"] and not report["jetson_rehearsal_passed"]


def test_real_http_simulation_and_three_recordings(tmp_path):
    server = node.build_server(
        "127.0.0.1", 0, backend=node.FakeBackend(gain=0.9, noise_sd=0.1), settle=0.0
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = preflight.run_preflight(
            f"http://127.0.0.1:{server.server_address[1]}",
            rehearse=True,
            output=tmp_path / "report.json",
        )
        assert report["status"] == "simulation_only", report
        assert report["three_runs_passed"]
        assert not report["jetson_rehearsal_passed"]
        for run in report["rehearsals"]:
            events = [
                json.loads(line)
                for line in Path(run["events"]).read_text().splitlines()
            ]
            assert events[0]["type"] == "run.started"
            assert events[-1]["type"] == "run.finished"
            assert events[0]["run_id"] == run["run_id"]
            assert events[0]["payload"]["config"]["fake"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_cli_simulation_has_distinct_exit_code(instrument, tmp_path):
    instrument.health_data["fake"] = True
    assert (
        preflight.main(["--node", "http://fake", "--output", str(tmp_path / "r.json")])
        == 2
    )


@pytest.mark.parametrize(
    "options", [{"sigma": 0}, {"max_n": 17}, {"max_spread": float("nan")}]
)
def test_invalid_thresholds_fail_before_hardware_access(tmp_path, options):
    with pytest.raises(ValueError):
        preflight.run_preflight("http://rig", output=tmp_path / "r.json", **options)
