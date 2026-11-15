# Bench: screen grey level -> Jetson camera

**No Jetson display available?** Use the separate [headless USB-webcam exposure experiment
(DU-51)](HEADLESS.md). It varies camera exposure against a stationary target, with no HDMI
screen or GUI, and reuses the EIG loop via `--mode exposure`.

A screen shows a grey level chosen by the EIG engine, a camera on the Jetson measures it, and four rival response laws (linear, power, log, saturating) are discriminated live. Files:

* `node.py`: the thin node that runs on the Jetson (Python 3.6, standard library + optional OpenCV/numpy). The commands for locking the camera are in its docstring.
* `lab.py`: laptop side, the four `ModelFamily` hypotheses (priors documented at the top) and `BenchClient` / `BenchLab`.
* `run_bench.py`: standalone loop that works without the orchestrator and writes `events.jsonl`.
* `preflight.py`: laptop-side instrument checks and three recorded, timed rehearsals.

## 1. Hardware check (on the Jetson)

```
cat /etc/nv_tegra_release
python3 --version
ls /dev/video*
```

Point the camera at the screen so the patch fills the central 80x80 pixels, in a dark or evenly lit room. Do not let the camera see its own screen glare or ceiling lights.

## 2. Start the node

Real camera (USB webcam index 0, or a CSI GStreamer pipeline, see the `node.py` docstring):

```
python3 node.py --camera 0 --port 8765
```

No hardware (laptop development), simulating `y = 255*(x/255)**0.85 + noise`:

```
python3 labs/bench/node.py --fake --port 8765
```

Check from the laptop: `curl http://<jetson>:8765/health`.

## Phone as the screen (no monitor on the Jetson)

The node serves its own stimulus page, so any phone or laptop on the same network can be the screen:

```
python3 node.py --camera 0 --display none --exposure 300 --wb 4600 --gain 0 --port 8765   # on the Jetson
```

Open `http://<jetson>:8765/screen` on the phone and tap once: it goes full screen and stays awake. Turn off auto-brightness and Night Shift/True Tone, set brightness to a fixed level, and prop the phone so its screen fills the centre of the camera's view. `POST /patch` takes `{"rgb": [r, g, b]}` or `{"level": v}`. It returns only after the page has painted the new colour (`"screen_ack": true`), then waits `--settle`.

The node owns the camera on one reader thread. `/measure` returns the grey mean, per-channel `r`/`g`/`b` and `spatial_sd` (an occluded lens shows up here), and `/stream` and `/snapshot.jpg` show what the camera sees, so `cam_stream.py` isn't needed while the node runs. `--exposure`, `--wb` and `--gain` pin the camera through OpenCV's V4L2 controls; `/health` reports the values the camera actually took.

## 3. Lock the camera and check drift

Turn off auto exposure, auto white balance and gain with `v4l2-ctl` (or the `nvarguscamerasrc` locks), as in the `node.py` docstring. Then run a 20-frame drift check on a mid-grey patch:

```
curl -X POST -d '{"level":128}' http://<jetson>:8765/patch
for i in $(seq 20); do curl -s http://<jetson>:8765/measure; echo; done
```

The means must agree within about 1.5 grey levels with no trend. Drift can indicate an unlocked camera or changing illumination: fix that before going on. Also check that level 255 reads below 245 (no clipping) and level 0 reads close to 0.

### Automated preflight and rehearsal evidence

After locking exposure, gain and white balance, run this **on the laptop** from the repo root
(Python 3.12; the thin node remains compatible with Python 3.6):

```
uv run python -m labs.bench.preflight --node http://<jetson>:8765 --camera-locked --rehearse --output runs/bench-check/report.json
```

`--camera-locked` records **operator confirmation**, not automatic verification. The command
does not change camera controls: check them on the Jetson first. OpenCV property values are
included in the health snapshot for diagnosis but do not prove exposure lock across drivers.
Run without `--rehearse` to check the instrument before starting the experiment loop.

The report records:

* Node health, board release, Python version, display and camera settings. A real node without
  an active OpenCV/Tk display fails before any patches are requested.
* **20 measurement batches** at mid-grey, each averaging the node's configured frame count
  (8 by default). This is more than 20 individual camera frames. The report keeps every batch,
  timestamps, sample SD, peak-to-peak spread and fitted brightness change over time.
* Default limits: peak-to-peak spread <= 1.5, absolute fitted change <= 1.5, black mean <= 15,
  white mean < 245 and white-minus-black range >= 100, all on the 0–255 brightness scale.
  `--max-spread` and `--max-drift` override stability limits and are recorded in the report.
* After a successful preflight, `--rehearse` runs **three consecutive EIG trials** with seeds
  0, 1 and 2, up to 10 measurements each. Every trial must finish in **strictly less than 60 s**,
  end at posterior >= 0.95 and have no surprise events. Each gets a distinct replayable
  `events.jsonl` under `report-runs/<run_id>/`, next to the report. Time includes event writing
  and model scoring. A failed check skips rehearsals; an HTTP failure preserves the partial
  report. `--timeout` sets the per-request HTTP timeout (default 5 s), not a whole-run deadline.
* `jetson_rehearsal_passed` is true only for a real node reporting a board release, successful
  preflight and three passing trials. It does not sign off the live UI, tint recovery or video.

Exit codes are **0** for passed physical checks (including rehearsals when requested), **1**
for failure, and **2** for otherwise successful simulation-only checks. Fake nodes can never
set `jetson_rehearsal_passed`, even with `--camera-locked`. The stock fake node clips at full
white and has independent measurement noise, so it may correctly fail the strict checks;
the integration test uses an unclipped, stable simulated instrument.

`--sigma` sets observation noise in the EIG model (default 2.0). The measured mid-grey SD is
diagnostic evidence, not automatically a noise estimate valid at every brightness. Camera
exposure or screen-brightness changes can alter the response enough that the built-in priors
do not fit: keep an unresolved or surprised trial as a failure and inspect it rather than
loosening thresholds merely to pass. Runs on a laptop camera can pass physical checks but
do not qualify as Jetson rehearsals when the board release is absent.

These files are local under ignored `runs/` by default. Share the report and recordings with
the team and post the real-hardware results on DU-44; do not commit large recording files.

## 4. Run the loop

```
python -m labs.bench.run_bench --node http://<jetson>:8765 --strategy eig --max 10 --seed 0 \
    --events runs/bench-01/events.jsonl
python -m labs.bench.run_bench --node http://<jetson>:8765 --strategy random --max 10 --seed 0
python -m labs.bench.run_bench --node http://<jetson>:8765 --compare --max 12    # real node: slow
```

It prints one line per measurement (level, value, family posteriors, z-score) and stops when one family reaches 0.95 or at `--max`. On a fake node `--compare` is the number to quote: mean measurements to reach 0.95 for EIG against random over seeds 0-9.

## Live view (aim the camera, show the demo)

`cam_stream.py` serves the Jetson's camera to any device on the same network, for aiming the camera at the screen and for showing the rig during the demo:

```
python3 cam_stream.py --camera 0 --port 8080      # on the Jetson
```

Open `http://<jetson>:8080/` for the live view and board status (frame rate, SoC temperatures). `/snapshot.jpg` gives a single frame. Stop it before running `node.py` on the real camera, since both open the same device.

With `--screen :1` it mirrors an X desktop instead of the camera. That needs a logged-in desktop session and its `XAUTHORITY`; the login screen belongs to the display manager and can't be read without root.

## 5. The tint moment

Run with `--tint-after 6 --tint 0.5`. On the fake node the script scales the gain after 6 measurements. On the real rig, cover the lens with a film (or sunglasses lens) at that moment instead; the observed values drop, the loop prints `SURPRISE`, and a `surprise.raised` event is written.

## 6. Kill rules

* No exposure lock in 45 minutes: switch to a USB webcam, or the laptop webcam.
* Not working by 11 PM: stop and use a recording (below).

## 7. Replay a recorded run in the UI

Keep the file from a good run, for example `runs/demo/events.jsonl`. The API server replays `runs/<run_id>/events.jsonl` and then follows new lines, so copy the recording to `runs/demo/events.jsonl` and open that run in the UI. Event format: `docs/EVENTS.md`.
