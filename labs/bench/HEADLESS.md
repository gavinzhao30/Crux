# Headless exposure experiment (DU-51)

No screen is needed on the Jetson. Point a Logitech USB webcam at a stationary matte
target (for example, plain paper) under steady room light. Crux changes **camera exposure
time** and observes brightness, using the same information-gain engine as the screen demo.
The laptop displays the beliefs and curves; the Jetson runs only the camera service.

Question: does the current camera/ISP/scene response look linear, power/gamma, logarithmic
or saturating? The model family is an approximation of this setup, not a universal physical
law. Four built-in models and a null option run without an LLM in the measurement loop.

## 1. Start on the Jetson over SSH

From the repository root, use the Jetson's system Python and OpenCV; do not install the
project's Python-3.12 laptop environment on an older JetPack just to run this thin node.
`node.py` and `exposure_node.py` can also be copied together and run directly.

```bash
cat /etc/nv_tegra_release
python3 --version
ls /dev/video*
v4l2-ctl --list-devices
v4l2-ctl --device /dev/video0 --list-ctrls-menus
python3 -c 'import cv2; print(cv2.__version__)'
```

Use the webcam's capture device, not its metadata device. If tools are missing, install
`v4l-utils` and the JetPack-compatible Python OpenCV package on the Jetson. The account
needs access to the camera device; the node does not invoke sudo or change permissions.

```bash
python3 -m labs.bench.exposure_node --device /dev/video0 --port 8765
```

No desktop session, HDMI monitor, Tk or OpenCV window is created. The node uses V4L2
directly through OpenCV. A Logitech name alone does not guarantee supported controls:
startup fails explicitly if the required manual controls or exact readback are unavailable.

It detects common old/new aliases for absolute exposure, automatic exposure, automatic
white balance and gain. It disables automatic exposure/white balance, freezes gain and
available white-balance temperature/focus controls, and disables automatic focus/gain and
dynamic frame rate when exposed. Each exposure write is read back; fixed settings are
verified before and after capture. A change or a clamped write fails the measurement.
Saved settings are restored on orderly shutdown/Ctrl-C, with restoration errors printed
to stderr. A forced kill/power loss cannot guarantee restoration.

## 2. Choose an exposure range and target

Default requested range: **10–300 V4L2 absolute units**, approximately **1–30 ms** under
the standard [Linux V4L2 definition](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/ext-ctrls-camera.html).
It is intersected with the device's actual limits and quantized to its legal steps. Startup
requires 16 distinct candidate exposures. Adjust with `--min-exposure` / `--max-exposure`.
Do not change driver controls from another app while the experiment is running.

The existing `/patch` API is reused as a normalized **exposure setting** from 0 to 255.
In exposure mode it never paints a patch. Sixteen candidate settings map to real exposure
times; the full mapping is saved in health/events. Plot x values are normalized settings,
not milliseconds. Every observation and JSON summary retains exposure readback in ms.

Keep target/camera positions and lighting fixed. Avoid flickering illumination. Mid-range
exposure must give measurable, unsaturated brightness. If the scene is too dark, improve
lighting or increase the range; if most settings saturate, reduce the range or illumination.
Long-exposure saturation is itself a candidate response, so the screen demo's requirement
that maximum brightness never clips does not apply here.

## 3. Preflight and three rehearsals (on the laptop)

```bash
uv run python -m labs.bench.preflight --mode exposure --node http://JETSON:8765 --rehearse --max 16 --output runs/exposure-check/report.json
```

This runs 20 averaged capture batches at fixed exposure, checks <=1.5 peak-to-peak brightness
spread and <=1.5 fitted drift, requires mid-exposure brightness between 5 and 245, and requires
at least 30 brightness units of response across the range. Fixed camera controls are verified
by the node, so `--camera-locked` is not needed for this mode. Three trials must each converge
at posterior >=0.95 with no surprises and finish in strictly less than 60 seconds. Time includes
measurement and inference. The command records replay events and reports failures explicitly.

Fake runs never qualify as Jetson hardware evidence. Real qualification also requires a
reported Jetson board release. These checks certify readiness/convergence, not model accuracy.

## 4. Run and compare

```bash
uv run python -m labs.bench.run_bench --mode exposure --node http://JETSON:8765 --max 16 --events runs/exposure-01/events.jsonl --summary runs/exposure-01/summary.json
uv run python -m labs.bench.run_bench --mode exposure --node http://JETSON:8765 --compare --max 16 --summary runs/exposure-comparison.json
```

The comparison uses seeds 0–9 and retains individual results, convergence flags, simulated
status and exposure readbacks. On real hardware, seeds control the inference/selection
randomness, not the camera noise; the trials are sequential, not identical physical noise
replays. Keep room conditions fixed, alternate strategy order in follow-up measurements,
and keep failures in the results. No claim that EIG beats random is justified before testing.

Events use the normal Bench format, with exposure-specific goal/hypothesis text and a full
exposure mapping. The API can replay `runs/<run_id>/events.jsonl`; the live launcher still
does not start a Bench run itself. For a lens-cover surprise, run with `--tint-after 6` to keep
measuring after early convergence and cover the real lens by hand. Only fake mode applies
`--tint` programmatically. Automatic hypothesis generation/recovery is separate work.

## 5. Test without hardware

```bash
python -m labs.bench.exposure_node --fake --settle 0 --port 8765
```

Run the laptop commands against `http://127.0.0.1:8765`. Successful fake preflight exits 2
(`simulation_only`), never 0/hardware-qualified. The synthetic response has known parameters
and small measurement noise; it exercises the workflow, not the Logitech hardware.

**Inference limit:** the finite-particle approximation and overlapping model families can
give high posterior confidence to a different family even on the synthetic power response.
Convergence and low experiment count are not correctness measurements. Preserve the selected
family and validate held-out predictions / calibrate priors before presenting response-law
identification as an accuracy result. Do not quote the simulated EIG/random comparison as
a Jetson performance or scientific discovery result.

## Hardware acceptance still needed

- Exact Logitech model, control listing and verified camera capture over the network.
- A stable target/range that passes real preflight.
- Three real timed runs with reports and replay logs, plus a recorded presentation fallback.
- Prediction/model calibration and an honest real EIG/random comparison.

DU-51 is separate from the original screen-based DU-44. Both preserve the existing biology
benchmark/evaluator separation and require no changes to benchmark reference data.
