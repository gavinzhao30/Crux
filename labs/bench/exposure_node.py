"""Headless USB webcam exposure experiment (Python 3.6 compatible).

python3 -m labs.bench.exposure_node --device /dev/video0 --port 8765
python3 -m labs.bench.exposure_node --fake --port 8765

The existing /patch endpoint selects a NORMALIZED EXPOSURE setting (0..255), not
a screen patch. Health records the mapping to V4L2 exposure units (100 us each).
Only Linux USB/V4L2 cameras with verifiable manual controls are supported.
"""

# ruff: noqa: UP004
import argparse
import math
import re
import subprocess
import sys

if __package__:
    from . import node
else:  # also works with node.py and this file copied to the Jetson together
    import node


class HeadlessDisplay(object):
    kind = "headless-exposure"

    def set_level(self, level):
        pass


class Controls(object):
    """v4l2-ctl adapter with an explicit device path and no shell commands."""

    def __init__(self, device, command=subprocess.run):
        self.device = device
        self.command = command
        text = self._run("--list-ctrls")
        self.available = {}
        for line in text.splitlines():
            match = re.match(r"\s*(\w+)\s+0x[\da-fA-F]+\s+\([^)]+\)\s*:\s*(.*)", line)
            if match:
                name, details = match.groups()
                self.available[name] = {
                    key: int(value)
                    for key, value in re.findall(
                        r"\b(min|max|step|value)=(-?\d+)", details
                    )
                }

    def _run(self, argument):
        try:
            result = self.command(
                ["v4l2-ctl", "--device", self.device, argument],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=5,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"v4l2-ctl failed; check v4l-utils, device and camera permissions: {exc}"
            )
        return result.stdout

    def find(self, *names):
        return next((name for name in names if name in self.available), None)

    def get(self, names):
        text = self._run("--get-ctrl=" + ",".join(names))
        values = {
            name: int(value)
            for name, value in re.findall(r"(?m)^\s*(\w+):\s*(-?\d+)", text)
        }
        if set(values) != set(names):
            raise RuntimeError("camera did not return every requested control")
        return values

    def set(self, name, value):
        self._run(f"--set-ctrl={name}={value}")
        if self.get([name])[name] != value:
            raise RuntimeError(f"camera rejected or clamped {name}={value}")


def exposure_map(low, high, step, origin):
    """Quantize all 256 settings; ensure the 16 experiment candidates are distinct."""
    if step < 1 or low < 1 or high <= low:
        raise ValueError("invalid exposure range or step")
    first = origin + math.ceil((low - origin) / float(step)) * step
    last = origin + math.floor((high - origin) / float(step)) * step
    if last <= first:
        raise ValueError("requested exposure range contains too few legal values")
    units = [
        first + round((last - first) * level / 255.0 / step) * step
        for level in range(256)
    ]
    if len({units[level] for level in range(0, 256, 17)}) != 16:
        raise ValueError(
            "need at least 16 distinct exposure candidates; widen the range"
        )
    return units


class ExposureCamera(node.CameraBackend):
    def __init__(self, device, low=10, high=300, controls=None):
        self.controls = controls or Controls(device)
        c = self.controls
        self.exposure = c.find("exposure_time_absolute", "exposure_absolute")
        self.auto = c.find("auto_exposure", "exposure_auto")
        self.wb = c.find("white_balance_automatic", "white_balance_temperature_auto")
        self.gain = c.find("gain")
        if not all((self.exposure, self.auto, self.wb, self.gain)):
            raise RuntimeError(
                "camera must expose absolute exposure, manual exposure mode, white-balance auto and gain controls"
            )
        spec = c.available[self.exposure]
        self.mapping = exposure_map(
            max(low, spec["min"]),
            min(high, spec["max"]),
            spec.get("step", 1),
            spec["min"],
        )
        self.fixed = {self.auto: 1, self.wb: 0}
        for name in (
            c.find("autogain", "gain_automatic"),
            c.find("focus_automatic_continuous", "focus_auto"),
            c.find("exposure_dynamic_framerate", "exposure_auto_priority"),
        ):
            if name:
                self.fixed[name] = 0
        self.auto_names = set(self.fixed)
        manual = [self.gain]
        for name in (c.find("white_balance_temperature"), c.find("focus_absolute")):
            if name:
                manual.append(name)
        names = list(self.fixed) + [self.exposure] + manual
        self.saved = c.get(names)
        for name in manual:
            self.fixed[name] = self.saved[name]
        self.applied = self.saved[self.exposure]
        self.cap = None
        try:
            # Open by device path through V4L2, never as a GStreamer string or GUI.
            if node.cv2 is None:
                raise RuntimeError("OpenCV is required for the real webcam")
            self.cap = node.cv2.VideoCapture(device, node.cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f"could not open USB webcam {device}")
            self.source = device
            self.level = 0
            for name, value in self.fixed.items():
                c.set(name, value)
            self.applied = c.get([self.exposure])[self.exposure]
            self.show(128)
        except Exception:
            self.close()
            raise

    def verify(self):
        expected = dict(self.fixed)
        expected[self.exposure] = self.applied
        if self.controls.get(list(expected)) != expected:
            raise RuntimeError("camera controls changed during the experiment")
        return True

    def show(self, level):
        self.verify()
        value = self.mapping[int(level)]
        self.controls.set(self.exposure, value)
        self.applied = value
        self.level = int(level)

    def measure(self, frames, discard, roi):
        self.verify()
        result = super().measure(frames, discard, roi)
        self.verify()
        result.update(
            exposure_units=self.applied,
            exposure_ms=self.applied / 10.0,
            controls_verified=True,
        )
        return result

    def info(self):
        return {
            "experiment": "exposure",
            "controls_verified": self.verify(),
            "fixed_controls": dict(self.fixed),
            "exposure_control": self.exposure,
            "exposure_units": self.applied,
            "exposure_ms": self.applied / 10.0,
            "exposure_map_units": self.mapping,
            "units_per_ms": 10,
        }

    def close(self):
        errors = []
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # Restore dependent manual settings before restoring their automatic modes.
        ordered = [name for name in self.saved if name not in self.auto_names]
        ordered += [name for name in self.saved if name in self.auto_names]
        for name in ordered:
            try:
                self.controls.set(name, self.saved[name])
            except Exception as exc:  # noqa: BLE001 - attempt all restorations
                errors.append(f"{name}: {exc}")
        if errors:
            sys.stderr.write(
                "camera settings restoration incomplete: {}\n".format("; ".join(errors))
            )


class FakeExposure(node.FakeBackend):
    def __init__(self, seed=0, low=10, high=300):
        super().__init__(gamma=0.65, gain=0.8, noise_sd=0.1, seed=seed)
        self.source = "fake-exposure"
        self.mapping = exposure_map(low, high, 1, 1)

    def measure(self, frames, discard, roi):
        result = super().measure(frames, discard, roi)
        units = self.mapping[self.level]
        result.update(
            exposure_units=units, exposure_ms=units / 10.0, controls_verified=False
        )
        return result

    def info(self):
        result = super().info()
        result.update(
            experiment="exposure",
            controls_verified=False,
            exposure_map_units=self.mapping,
            units_per_ms=10,
        )
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--min-exposure", type=int, default=10, help="100-us units")
    parser.add_argument("--max-exposure", type=int, default=300, help="100-us units")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle", type=float, default=0.4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--discard", type=int, default=5)
    parser.add_argument("--roi", type=int, default=80)
    args = parser.parse_args(argv)
    if args.frames < 1 or args.discard < 0 or args.roi < 2 or args.settle < 0:
        parser.error("frames/ROI must be positive; discard/settle must be nonnegative")
    backend = (
        FakeExposure(args.seed, args.min_exposure, args.max_exposure)
        if args.fake
        else ExposureCamera(args.device, args.min_exposure, args.max_exposure)
    )
    server = None
    try:
        server = node.build_server(
            args.host,
            args.port,
            backend=backend,
            display=HeadlessDisplay(),
            settle=args.settle,
            frames=args.frames,
            discard=args.discard,
            roi=args.roi,
        )
        print(
            f"crux exposure node on {args.host}:{args.port} (fake={args.fake}); no display required"
        )
        sys.stdout.flush()
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        backend.close()


if __name__ == "__main__":
    main()
