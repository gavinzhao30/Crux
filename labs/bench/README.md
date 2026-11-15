# Bench: screen grey level -> Jetson camera

A screen shows a grey level chosen by the EIG engine, a camera on the Jetson measures it, and four rival response laws (linear, power, log, saturating) are discriminated live. Files:

* `node.py`: the thin node that runs on the Jetson (Python 3.6, standard library + optional OpenCV/numpy). The commands for locking the camera are in its docstring.
* `lab.py`: laptop side, the four `ModelFamily` hypotheses (priors documented at the top) and `BenchClient` / `BenchLab`.
* `run_bench.py`: standalone loop that works without the orchestrator and writes `events.jsonl`.

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

## 3. Lock the camera and check drift

Turn off auto exposure, auto white balance and gain with `v4l2-ctl` (or the `nvarguscamerasrc` locks), as in the `node.py` docstring. Then run a 20-frame drift check on a mid-grey patch:

```
curl -X POST -d '{"level":128}' http://<jetson>:8765/patch
for i in $(seq 20); do curl -s http://<jetson>:8765/measure; echo; done
```

The means must agree within about 1.5 grey levels with no trend. A trend means exposure or white balance is still auto: fix that before going on. Also check that level 255 reads below 245 (no clipping) and level 0 reads close to 0.

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
