"""Headless camera control contracts, physical-mode guards and local HTTP integration."""

import ast
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from labs.bench import exposure_node, node, preflight
from labs.bench.lab import BenchClient
from labs.bench.run_bench import compare, run_loop


class CameraControls:
    def __init__(self):
        self.values = {
            "exposure_time_absolute": 166,
            "auto_exposure": 3,
            "white_balance_automatic": 1,
            "white_balance_temperature": 4600,
            "gain": 5,
            "focus_automatic_continuous": 1,
            "focus_absolute": 100,
            "exposure_dynamic_framerate": 1,
        }
        self.available = {name: {"value": value} for name, value in self.values.items()}
        self.available["exposure_time_absolute"].update(min=3, max=2047, step=1)
        self.writes = []

    def find(self, *names):
        return next((name for name in names if name in self.available), None)

    def get(self, names):
        return {name: self.values[name] for name in names}

    def set(self, name, value):
        self.writes.append((name, value))
        self.values[name] = value


@pytest.fixture
def camera(monkeypatch):
    controls = CameraControls()
    cap = SimpleNamespace(
        isOpened=lambda: True,
        read=lambda: (
            True,
            np.full((10, 10), min(255, controls.values["exposure_time_absolute"])),
        ),
        release=lambda: None,
    )
    opened = []

    def open_camera(device, backend):
        opened.append((device, backend))
        return cap

    monkeypatch.setattr(
        node, "cv2", SimpleNamespace(VideoCapture=open_camera, CAP_V4L2=200)
    )
    return controls, opened


def test_manual_controls_exposure_readback_and_restore(camera):
    controls, opened = camera
    before = dict(controls.values)
    backend = exposure_node.ExposureCamera("/dev/video2", controls=controls)
    try:
        assert opened == [("/dev/video2", 200)]
        assert controls.values["auto_exposure"] == 1
        assert controls.values["white_balance_automatic"] == 0
        assert controls.values["focus_automatic_continuous"] == 0
        assert controls.values["gain"] == before["gain"]
        backend.show(255)
        measured = backend.measure(3, 2, 8)
        assert measured["exposure_units"] == 300
        assert measured["exposure_ms"] == 30.0
        assert measured["controls_verified"] is True
        assert backend.info()["experiment"] == "exposure"
        controls.values["gain"] += 1
        with pytest.raises(RuntimeError, match="changed"):
            backend.measure(3, 2, 8)
    finally:
        backend.close()
    assert controls.values == before
    # Restore manual values before automatic modes, so auto modes cannot block them.
    restored = [name for name, _ in controls.writes[-len(before) :]]
    assert restored.index("exposure_time_absolute") < restored.index("auto_exposure")


def test_partial_startup_restores_controls(camera):
    controls, _ = camera
    before = dict(controls.values)
    original_set = controls.set
    failed = False

    def reject_once(name, value):
        nonlocal failed
        if name == "white_balance_automatic" and not failed:
            failed = True
            raise RuntimeError("manual white balance rejected")
        original_set(name, value)

    controls.set = reject_once
    with pytest.raises(RuntimeError, match="rejected"):
        exposure_node.ExposureCamera("/dev/video0", controls=controls)
    assert controls.values == before


def test_unsupported_controls_fail_before_camera_open(camera):
    controls, opened = camera
    del controls.available["auto_exposure"]
    with pytest.raises(RuntimeError, match="must expose"):
        exposure_node.ExposureCamera("/dev/video0", controls=controls)
    assert not opened


def test_exposure_quantization_respects_step_and_limits():
    mapping = exposure_node.exposure_map(11, 301, 5, 3)
    assert len(mapping) == 256
    assert min(mapping) >= 11 and max(mapping) <= 301
    assert all((value - 3) % 5 == 0 for value in mapping)
    assert len({mapping[i] for i in range(0, 256, 17)}) == 16
    with pytest.raises(ValueError, match="16 distinct"):
        exposure_node.exposure_map(10, 20, 1, 1)
    fake = exposure_node.FakeExposure(low=20, high=200)
    assert fake.mapping[0] == 20 and fake.mapping[-1] == 200


def test_v4l2_parser_aliases_and_rejected_readback():
    values = {"exposure_absolute": 100, "exposure_auto": 3}
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        option = args[-1]
        if option == "--list-ctrls":
            text = "  exposure_absolute 0x009a0902 (int) : min=3 max=2047 step=1 default=166 value=100\n  exposure_auto 0x009a0901 (menu) : min=0 max=3 default=3 value=3\n"
        elif option.startswith("--get-ctrl="):
            names = option.split("=", 1)[1].split(",")
            text = "\n".join(f"{name}: {values[name]}" for name in names)
        else:
            text = ""  # emulate a driver that silently ignores the write
        assert kwargs["check"] and kwargs["timeout"] == 5
        return SimpleNamespace(stdout=text)

    controls = exposure_node.Controls("/dev/video2", command=command)
    assert (
        controls.find("exposure_time_absolute", "exposure_absolute")
        == "exposure_absolute"
    )
    assert controls.available["exposure_absolute"]["min"] == 3
    with pytest.raises(RuntimeError, match="rejected or clamped"):
        controls.set("exposure_absolute", 200)
    assert all(
        command[:3] == ["v4l2-ctl", "--device", "/dev/video2"] for command in commands
    )


def test_missing_v4l2_utility_reports_actionable_error():
    def missing(*args, **kwargs):
        raise FileNotFoundError("v4l2-ctl")

    with pytest.raises(RuntimeError, match="v4l-utils"):
        exposure_node.Controls("/dev/video0", command=missing)


@pytest.fixture(scope="module")
def exposure_url():
    server = node.build_server(
        "127.0.0.1",
        0,
        backend=exposure_node.FakeExposure(),
        display=exposure_node.HeadlessDisplay(),
        settle=0.0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_headless_http_measurement(exposure_url):
    client = BenchClient(exposure_url)
    health = client.health()
    assert health["display"] == "headless-exposure"
    assert health["camera_settings"]["experiment"] == "exposure"
    client.patch(255)
    result = client.measure()
    assert result["exposure_ms"] == 30.0 and result["controls_verified"] is False
    assert 190 < result["mean"] < 220


def test_wrong_mode_rejected_before_running(exposure_url, tmp_path):
    with pytest.raises(ValueError, match="node reports exposure"):
        run_loop(exposure_url)
    report = preflight.run_preflight(exposure_url, output=tmp_path / "wrong.json")
    assert (
        report["status"] == "failed" and "node reports exposure" in report["errors"][0]
    )


def test_exposure_loop_and_events_are_labeled_and_keep_readbacks(
    exposure_url, tmp_path
):
    path = tmp_path / "events.jsonl"
    result = run_loop(
        exposure_url, mode="exposure", max_n=16, seed=0, verbose=False, events_path=path
    )
    assert result["mode"] == "exposure"
    assert result["measurements"] and all(
        "exposure_ms" in m for m in result["measurements"]
    )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert "exposure" in events[0]["payload"]["goal"]
    assert events[0]["payload"]["config"]["mode"] == "exposure"
    assert (
        len(events[0]["payload"]["config"]["camera_settings"]["exposure_map_units"])
        == 256
    )
    assert events[-1]["type"] == "run.finished"
    assert all(
        "exposure" in e["payload"]["hypothesis"]["statement"]
        for e in events
        if e["type"] == "hypothesis.proposed"
        and not e["payload"]["hypothesis"]["is_null"]
    )
    assert all(
        "Exposure" in e["payload"]["observation"]["referee_note"]
        for e in events
        if e["type"] == "experiment.finished"
    )


def test_exposure_preflight_accepts_no_display_but_remains_simulated(
    exposure_url, tmp_path
):
    report = preflight.run_preflight(
        exposure_url, mode="exposure", output=tmp_path / "report.json"
    )
    assert report["status"] == "simulation_only", report
    assert report["checks"]["display_ready"]["passed"]
    assert "white_not_clipped" not in report["checks"]
    assert not report["jetson_rehearsal_passed"]


@pytest.mark.parametrize("verified", [True, False])
def test_real_exposure_preflight_uses_control_readback_and_allows_saturated_endpoint(
    monkeypatch, tmp_path, verified
):
    class Instrument:
        level = 128

        def health(self):
            return {
                "ok": True,
                "fake": False,
                "board": "R36",
                "display": "headless-exposure",
                "camera_settings": {
                    "experiment": "exposure",
                    "controls_verified": verified,
                },
            }

        def patch(self, level):
            self.level = level

        def measure(self):
            return {
                "mean": {0: 32.0, 128: 140.0, 255: 255.0}[self.level],
                "std": 0.1,
                "n": 8,
            }

    monkeypatch.setattr(preflight, "BenchClient", lambda *a, **kw: Instrument())
    report = preflight.run_preflight(
        "http://rig", mode="exposure", output=tmp_path / "r.json"
    )
    assert report["preflight_passed"] is verified
    assert report["status"] == ("passed" if verified else "failed")
    assert "black_level" not in report["checks"]
    assert "white_not_clipped" not in report["checks"]
    assert not report[
        "jetson_rehearsal_passed"
    ]  # preflight alone is not three rehearsals


def test_compare_does_not_count_last_measurement_convergence_as_failure(
    monkeypatch, capsys
):
    import labs.bench.run_bench as runner

    monkeypatch.setattr(
        runner, "run_loop", lambda *a, **kw: {"n_to_threshold": 12, "reached": True}
    )
    result = compare("unused", seeds=[0], max=12, mode="exposure")
    assert result["reached"] == {"eig": [True], "random": [True]}
    assert "0" in capsys.readouterr().out.splitlines()[-1]


def test_thin_node_stays_python36_and_never_creates_windows():
    source = Path(exposure_node.__file__).read_text()
    ast.parse(source, feature_version=(3, 6))
    assert ".imshow(" not in source and ".namedWindow(" not in source
    assert "from __future__" not in source
