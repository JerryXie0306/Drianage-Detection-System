# DrainGuard

Software layer of the **Adaptive Reserve Drainage System** (ENGR90051, team 4B, Southbank).
It answers the question *"how does the cover open during a flood?"*:

```
camera frame ──► Detection ──► Controller ──► Actuator (hinged cover)
 (360° camera)   clear/partial/  decide: debounce,   open / close
                 blocked + conf. threshold, mode,        │
                      ▲          flood-risk signal       │
                      └──────── Control app (phone UI) ◄─┘  cover-state feedback + delay
```

| Module | Purpose |
|---|---|
| `drainguard/detector.py` | Blockage detection. `heuristic` backend (colour-based debris coverage in the grate ROI, no training needed) or `model` backend (scikit-learn, trained on your photos) |
| `drainguard/controller.py` | Cover state machine, AUTO/MANUAL, debounce, 80 % confidence threshold, flood-risk input, timeouts → FAULT |
| `drainguard/actuator.py` | `SimulatedActuator` (default) and `SerialActuator` (Arduino-style, untested on hardware) |
| `drainguard/server.py` + `web/index.html` | Stdlib HTTP server + mobile control app matching `Figure2_app_wireframe.png` |
| `drainguard/evaluate.py` | **Test 1**: accuracy on labelled photos, incl. simulated dark / blur / noise conditions |
| `drainguard/latency.py` | **Test 2**: command → cover-confirmed delay |
| `drainguard/train.py` | Optional classifier training (colour-feature backend) |
| `drainguard/segment.py` | **Segmentation backend**: tiled CLIPSeg + optional trained head → % of grate covered, split into vegetation / mud / litter, plus chamber (sump) fill |
| `drainguard/label.py` | Browser tool: calibrate grate/sump regions, label masks, model pre-labelling |
| `drainguard/train_head.py`, `evaluate_seg.py` | Train the head (leave-one-image-out CV vs zero-shot) and evaluate on held-out photos |

## Quick start (Mac, no experience needed)

1. On GitHub click **Code → Download ZIP**, then double-click the ZIP to unzip it.
2. In the unzipped folder, **right-click `setup_and_run.command` → Open → Open**
   (macOS asks once because the file came from the internet).
3. Wait 1-3 minutes the first time while it installs. Your browser opens the control app by itself.
   Next time just double-click `setup_and_run.command` again; it starts in seconds.

If Python is missing it opens the download page: install it, then double-click the file again.
Optional AI backend (debris % per type): double-click `install_ai.command` once (large download).
If you see "permission denied", run `chmod +x *.command` in Terminal inside the folder.


## Quick start (Windows)

1. On GitHub click **Code → Download ZIP**, then right-click the ZIP → **Extract All**.
2. Open the extracted folder and **double-click `setup_and_run.bat`**.
   If Windows shows "Windows protected your PC", click **More info → Run anyway**.
3. Wait 1-3 minutes the first time while it installs. Your browser opens the control app by itself.
   Next time just double-click it again.

If Python is missing it opens the download page: install it and **tick "Add python.exe to PATH"**
on the first installer screen, then double-click the file again.
Optional AI backend: double-click `install_ai.bat` once (large download).


## ESP32-CAM (AI-Thinker) camera

1. Flash the Arduino example **File → Examples → ESP32 → Camera → CameraWebServer**: uncomment
   `#define CAMERA_MODEL_AI_THINKER`, fill in your Wi-Fi name and password, board **AI Thinker ESP32-CAM**,
   partition **Huge APP (3MB No OTA)**. Open the Serial Monitor (115200) and read the IP it prints.
2. Put the laptop and the camera on the **same 2.4 GHz Wi-Fi**. Campus Wi-Fi (eduroam) often blocks device-to-device
   traffic: use a **phone hotspot** (iPhone: turn on "Maximise Compatibility").
3. Put this in `camera.txt` (first line, then the seconds between analyses):

   ```
   snapshot:http://192.168.x.x/capture
   5
   ```
4. Double-click **`check_camera`** (`.command` on Mac, `.bat` on Windows). It must say **PASS** and saves `camera_test.jpg`.
   On FAIL it lists what to check. Close any browser tab showing the camera first: it serves one viewer at a time.
5. Double-click **`start_live`**. The app now analyses the camera by itself and, in AUTO mode, opens the simulated cover.

`http://IP:81/stream` also works (needs `install_ai` for OpenCV) but `snapshot:` is more robust on ESP32-CAMs.
No hardware yet? Run `python fake_camera.py --state blocked` and use `snapshot:http://127.0.0.1:8081/capture`.

## Real motor (ESP32 over Wi-Fi)

DrainGuard drives the cover through the motor board's URLs: **open** = `GET /timed?dir=forward&ms=N`,
**close** = the opposite direction, then it waits for the motor to finish. It only uses `/timed` (stops by itself)
and `/stop` (sent when the app exits). It never calls `/forward`, `/reverse` or `/auto`.

1. Edit `motor.txt` (address, open ms, close ms, which direction opens). With no active line the cover stays **simulated**.
2. Double-click `check_camera`: it also asks the motor for `/status` (read-only, it never moves it).
3. Double-click `start_live`. The first line it prints says `Cover: REAL MOTOR` or `Cover: SIMULATED`.

**Read before turning it on:** there is **no position sensor**. "Open" means "the motor ran that long", so the cover must
start **physically closed**, and a jam or slip is not detected. Test the direction and run time with short pulses on the motor's
own web page first, and agree with the hardware teammate what cuts power if the cover jams. Dry-run first with
`python fake_motor.py` and `http://127.0.0.1:8082` in `motor.txt`.

## Fully automatic recognition (live camera)

no clicking. Every `interval` seconds it takes the newest frame, measures the blocked % per debris type, and in AUTO mode the
controller opens the cover after `open-frames` (default 3) consecutive confident BLOCKED frames, and closes it again after
10 clear frames (and at least 30 s open). Flood-risk input and manual override still work.

Sources (`camera.txt`): webcam index `0`, an RTSP/HTTP stream URL, a video file (loops), or `folder:/path` (newest image).
CLI equivalents: `--camera SRC`, `--watch DIR`, `--interval S`, `--open-frames N`, `--close-frames N`. The stream reader keeps only the
newest frame and reconnects if the camera drops; the app shows the camera error if it cannot open the source.
Tested with a real video file through OpenCV and with simulated cameras; **not yet tested on a real webcam / RTSP camera**
(macOS will ask Terminal for camera access on first use). Analysis takes ~3-4 s/frame, so the effective period is
max(interval, analysis time).

## One-click start

Double-click one of these files in the project folder (Mac `.command`, Windows `.bat`):

- `setup_and_run` - first run installs what is needed, then starts the control app and opens your browser. Next runs start in seconds.
- `start_live` - fully automatic: reads the camera in `camera.txt` (and the motor in `motor.txt`), analyses it and shows the result.
- `check_camera` - tests that the camera and the motor board can be reached, without moving the motor.
- `install_ai` - optional, adds the AI segmentation backend.
- `start_label_tool.command` (Mac) - labelling and region-calibration tool with model pre-labelling.

The first time on a Mac, right-click the file and choose Open (macOS may block files downloaded from the internet).
Close the Terminal window, or press Ctrl+C, to stop.

## Quick start (no installs needed beyond numpy, Pillow, scipy, scikit-learn, pytest)

```bash
cd ~/Desktop/DrainGuard
./run_demo.sh            # generates synthetic demo frames, starts http://127.0.0.1:8000
python3 -m pytest -q     # 27 tests
```

In the app: frames replay every 2 s. After 3 consecutive confident BLOCKED frames the controller sends OPEN
(AUTO mode). Pressing OPEN/CLOSE yourself switches to MANUAL (operator override). "Simulate heavy rain"
sends a flood-risk signal that opens the cover immediately. "Upload photo" classifies your own image.

Phone on the same Wi-Fi: `python3 -m drainguard.server --host 0.0.0.0 ...` (no authentication - trusted network only).

## Producing REAL evidence for the project log

1. Put real photos in `data/test/clear|partial|blocked/` (e.g. your Southbank grate photos, more at different
   times of day / wet). Use different photos for `data/train/` if you train the model backend.
2. `python3 -m drainguard.evaluate --data data/test --conditions normal,dark,blur,noise`
   → accuracy table + confusion matrix + per-image CSV in `results/`. Note the *blocked-called-clear* rate:
   that is the dangerous error.
3. `python3 -m drainguard.latency --cycles 20` → software path only (simulated actuator). With the team's
   hardware: add `--serial /dev/tty.usbmodemXXXX` for a real command → open measurement.
4. Screenshot the app (`http://127.0.0.1:8000`) for the log, and annotate which parts you built.

## Segmentation backend: how blocked, and with what

For the in-drain upward fisheye camera. One-time setup (installs torch + transformers into `.venv`, ~1-2 GB,
CLIPSeg model downloads on first use):

```bash
.venv/bin/pip install torch transformers
```

1. **Calibrate regions** (once per camera position): `.venv/bin/python -m drainguard.label --images data/seg/images --masks data/seg/masks --regions regions/drain.json`
   → *Regions* tab: click the **grate** polygon (where debris blocks flow) and the **sump** polygon (chamber floor).
   `regions/example_fisheye.json` is only an example of the file format; the image it was drawn on is not included, so draw your own regions on your own photos.
2. **Label real photos** from your own camera (same tool, add `--predict` to pre-label with the model, then correct).
   Classes: vegetation, mud, litter, grate_bars, open_slot, wall. Aim for 15-20+ photos in varied conditions
   (wet / dry, day / dusk, different debris); keep 5-8 in a separate folder for testing only.
3. **Train the head**: `.venv/bin/python -m drainguard.train_head` → prints leave-one-image-out IoU for zero-shot vs head.
4. **Evaluate on held-out photos**: `.venv/bin/python -m drainguard.evaluate_seg --images data/seg/test/images --masks data/seg/test/masks --regions regions/drain.json --head models/seg_head.joblib`
5. **Run the app**: `.venv/bin/python -m drainguard.server --backend segment --regions regions/drain.json --head models/seg_head.joblib`

Definitions: `blocked_pct` = debris pixels / (open_ratio x grate-polygon pixels), capped at 100, independent of lighting.
`_open_ratio` (in the regions file) is the share of the polygon that is open when the grate is clear, so a clean
grate reads ~0 % and a fully covered one 100 %. `obstruction_pct` (debris / (debris + light through slots)) is a
daylight-only cross-check and is hidden when too little light is visible. Label: clear < 10 %, partial 10-40 %, blocked >= 40 %.

What to expect (measured on one example image, no ground truth yet): zero-shot finds grate bars, plastic/cans and
leaves reasonably, but reads wet mud as vegetation and cannot separate bars from gaps without daylight. The head
is meant to fix exactly that, but its accuracy is **unknown until you label and evaluate real photos**.
Inference takes ~3-4 s per frame on an Apple-silicon laptop, so this is a still-image monitor, not video.

## Honest limitations

- **Synthetic images are for smoke tests and the demo only.** They follow the detector's own colour
  assumptions, so accuracy on them proves nothing. Never report it as a result.
- The heuristic detector flags green and brown/orange pixels as debris. It will miss grey rubbish and
  can false-alarm on rust, brown paving or coloured litter. The `model` backend, trained on real photos,
  is the way to improve this; compare both on the same test set.
- The "confidence" of the heuristic backend is a margin score, not a calibrated probability.
- "dark / blur / noise" conditions are simulated by editing photos; photograph real wet and low-light grates
  for the critical-aspect claim (waterproof camera, condensation, mud on the lens are not covered).
- The simulated actuator only validates the software path. The serial protocol is a placeholder: agree on
  the real one with the hardware teammates (`OPEN\n`/`CLOSE\n` → `OPENED\n`/`CLOSED\n` by default).
- Segmentation: percentages are pixel shares in a squashed fisheye image, not physical areas; fine for a fixed camera,
  not comparable across different cameras. A head trained on one drain will not generalise to another without new labels.
  Lens condensation, mud on the lens and night operation are untested.
- No authentication, no persistence, one drain (`SB-01`). Prototype scope.
- Camera input is a folder replay or manual upload. A 360° camera needs the relevant view cropped out first
  (use `--roi x0 y0 x1 y1` for the grate area); live capture is not implemented.

## API

`GET /api/status` · `GET /api/frame.jpg` · `POST /api/frame` (raw image body) · `POST /api/command {"action":"open|close"}` ·
`POST /api/mode {"mode":"auto|manual"}` · `POST /api/flood {"active":true}` · `POST /api/replay {"playing":false}`
