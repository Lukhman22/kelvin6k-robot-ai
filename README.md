# kelvin6k — Vision + Voice Monitoring Agent for a 3D Printing Robot

A real-time monitoring agent for a layer-based 3D printing robot. One camera
watches the print; classical computer vision measures the layers, wall, and
visible quality; a rule engine raises spoken alerts; and the operator can ask
questions out loud and get spoken answers, Alexa-style.

> **Monitoring and advisory only.** This system never sends commands to the
> printer, pump, accelerator, or robot. It can *recommend* an action
> ("increase feed rate"), but it will not change any machine setting.

> **Honesty rule.** Nothing in this system invents a value. Anything that
> cannot actually be measured from the camera, or read from a connected
> machine, is reported as *unavailable* with the reason — on the dashboard and
> in the voice answers. Simulated values are labelled `SIMULATED`.

---

## Architecture

```
                 ┌──────────────────────────────────────────┐
  CAMERA ──────> │ MAIN THREAD: vision_loop.py                │
                 │   camera.py    -> frame + FPS               │
                 │   segmentation.py -> isolate the part       │
                 │   layer_detector.py -> layers, width, trend │
                 │   wall_detector.py  -> wall, continuity     │
                 │   quality.py        -> quality score        │
                 │   dashboard.py      -> live window          │
                 └───────────────────┬──────────────────────┘
                                      │ writes
                          ┌───────────▼────────────┐
                          │  state.py PrinterState   │  single source of truth
                          └───────────▲────────────┘
             writes                    │ reads
  ┌──────────────────────┐            │        ┌────────────────────────────┐
  │ THREAD: agent_loop.py  │────────────┘        │ THREAD: voice_assistant.py   │
  │  sensors.py (SIMULATED)│                     │  mic -> listener.py (Whisper)│
  │  machine link (N/C)     │                     │  -> qa.py (Gemini)           │
  │  rules.py -> alerts     │                     │  -> voice.py (TTS) -> speaker│
  └──────────────────────┘                     └────────────────────────────┘
```

If the camera can't be opened, the app prints the reason and keeps running
headless — monitoring, alerts, and voice Q&A all still work.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt
copy .env.example .env          # then fill in the values below
```

---

## Web dashboard (React + TypeScript)

A browser dashboard with the live camera, production data, machine/material
status and the voice assistant. It talks to a FastAPI backend that wraps the
existing Python modules — no machine logic is duplicated.

**Two terminals:**

```powershell
# Terminal 1 — backend
cd "C:\Users\Anand Sharma\OneDrive\Desktop\kelvin6k"
python run_api.py                       # http://127.0.0.1:8000  (docs at /docs)

# Terminal 2 — frontend
cd "C:\Users\Anand Sharma\OneDrive\Desktop\kelvin6k\ui"
npm install                             # first time only
npm run dev                             # http://localhost:5173
```

Open <http://localhost:5173> in **Chrome or Edge** (the wake word and browser
speech recognition need the Web Speech API).

### Voice flow

```
"Hey Robot" -> Listening -> Speech-to-Text (browser Web Speech API)
            -> POST /api/ask  -> assistant.py (tools + RAG + SQLite)
            -> answer -> UI transcript -> TTS (browser speechSynthesis)
```

The **text box and voice use the same `/api/ask` endpoint**, so a typed and a
spoken question follow an identical path. `POST /api/stt` also exposes the
project's faster-whisper for browsers without the Web Speech API.

### Data provenance — REAL / MOCK / UNAVAILABLE

Every value on the dashboard carries a badge, so a simulated or missing number
can never look measured:

| Badge | Meaning |
|---|---|
| `REAL` | Measured by the camera/AI pipeline, or entered for the tracked job |
| `MOCK` | From the built-in simulator (`SimulatedSensors`) — **not** the real machine |
| `UNAVAILABLE` | Not connected / not measured — the reason is shown, never a guess |

Today that means: camera + vision + production data are **REAL**, process
sensors (flow, temperatures) are **MOCK**, and pump/accelerator/print-speed are
**UNAVAILABLE** until a `MachineInterface` is implemented.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Backend, LLM, STT, RAG and calibration status |
| GET | `/api/status` | All dashboard values, each with provenance |
| POST | `/api/ask` | **Shared query service** for text and voice |
| POST | `/api/stt` | Server-side speech-to-text (faster-whisper) |
| GET | `/api/camera/stream` | MJPEG live video with detection overlay |
| GET | `/api/camera/snapshot` | Single annotated JPEG |
| GET | `/api/camera/status` | Camera connection details |

**READ-ONLY:** there is no endpoint that starts, stops or otherwise commands the
robot, pump or accelerator.

---

## Voice assistant (tool calling + database + RAG)

An Alexa-style assistant answers questions about the current job. It works
**without a camera or microphone** — type your questions.

```powershell
python main.py --assistant --demo-job     # seeds a demo job, then ask away
python assistant_cli.py --ask "kitni wall print ho gayi hai?"
python assistant_cli.py --voice           # speak instead of typing
python assistant_cli.py --tools           # show which tool each question routes to
```

```
You: kitni wall print ho gayi hai?
Assistant: 650 millimeter printed, 350 millimeter remaining out of 1000 millimeter (65.0 percent).
You: current layer width kitni hai?
Assistant: Current layer width is 18.5 millimeter.
```

### How it answers (the model never invents values)

```
Voice/text -> LLM (Gemini) -> TOOL CALL -> real data -> spoken answer
                               │
    get_current_print_status() │ get_print_progress()
    get_current_layer()        │ get_current_layer_width()
    get_machine_status()       │ search_docs()  -> RAG
```

**Structured data for live numbers, RAG for documentation** — never the reverse.
`agent/services/machine_state.py` is the only place machine facts come from; it
reads the live `PrinterState` when the camera loop is running in-process, and
otherwise the SQLite `machine_state` table. If a value isn't available it says
so; if confidence is low it says the measurement is uncertain.

Without `GEMINI_API_KEY` the assistant falls back to a deterministic keyword
router, so the whole stack still runs offline (this is what the tests use).

### Database

SQLite at `DATABASE_PATH` (default `kelvin6k.db`), tables `print_jobs`,
`machine_state`, `layer_measurements`, `machine_events`. `remaining_length_mm`
is a **generated column** — `MAX(total - completed, 0)` — so it can never drift
from the values it derives from. The vision loop writes `machine_state` rows
every `STATE_PERSIST_INTERVAL_SECONDS`, which is how the camera AI feeds the
assistant.

```powershell
python -c "from agent.services import machine_state as m; print(m.start_print_job('Wall 1', 1000.0, 'concrete', 60))"
python -c "from agent.services import machine_state as m; print(m.update_progress(650.0, 42))"
```

### RAG — adding documentation

```powershell
python ingest_docs.py docs\                     # ingest a folder (recursive)
python ingest_docs.py manual.pdf                # one file
python ingest_docs.py --list                    # what's indexed
python ingest_docs.py --search "nozzle clog"    # test retrieval
```

Supports `.txt .md .rst .log .pdf`. Chunks keep **filename, doc type, page,
section** so answers cite their source. With `GEMINI_API_KEY` set it uses Gemini
embeddings (semantic); without one it uses an offline BM25 lexical search — and
tells you which mode is active rather than pretending. **If nothing relevant is
found it says so instead of inventing an answer.**

### Tests

```powershell
python tests\test_assistant.py        # 22 tests, no API key or camera needed
python -m pytest tests -v             # same, via pytest
```

### Gemini API setup (required for voice answers)

1. Get a free API key at <https://aistudio.google.com/apikey>.
2. Put it in `.env`:
   ```
   GEMINI_API_KEY=your-key-here
   GEMINI_MODEL=gemini-3-flash-preview
   ```
3. Verify it:
   ```bash
   python check_gemini_key.py
   ```
   The diagnostic reports key presence, key format, model, and a live test
   request — and never prints the key. It distinguishes an invalid key, an
   invalid model, a rate/quota limit, a server error, and a network problem.
   If your key can't use the configured model, set a model it can, e.g.
   `GEMINI_MODEL=gemini-2.5-flash`.

`.env` is gitignored; no key or secret is ever printed or logged.

---

## Camera setup

### Positioning

The vision pipeline measures **stacked horizontal layer bands**, so:

- Mount the camera **perpendicular to the wall being printed**, looking at the
  side of the part (not down at the nozzle).
- Keep the camera **fixed** — any movement changes the calibration.
- Frame the part so the **layer lines run horizontally** across the image.
- Use even, diffuse lighting. Strong shadows and specular highlights break the
  foreground segmentation (the system will say "cannot separate the printed
  part from the background" rather than report bad numbers).
- Maximise contrast between the material and whatever is behind it.

### Configuration

```
CAMERA_INDEX=0        # 0 = first camera; try 1, 2… if the wrong one opens
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30
SHOW_CAMERA_WINDOW=true
```

### EZVIZ / IP camera over RTSP

Set `CAMERA_SOURCE=rtsp` and either the pieces or a full URL:

```
CAMERA_SOURCE=rtsp
CAMERA_IP=192.168.1.64
CAMERA_USERNAME=admin
CAMERA_PASSWORD=ABCDEF          # EZVIZ: 6-char verification code on the label
CAMERA_RTSP_PORT=554
CAMERA_RTSP_PATH=/H.264/ch1/main/av_stream
CAMERA_RTSP_TRANSPORT=tcp
```

**Finding the camera IP:** open the EZVIZ app → your device → Settings → Device
Information → IP address. Or check your router's DHCP client list, or run
`arp -a` on the PC after the camera is online. Give it a DHCP reservation so the
IP doesn't change.

**Enabling RTSP:** in the EZVIZ app, the local RTSP service must be on and the
device must be on the same LAN as the PC. The **verification code** (label on
the camera body) is the RTSP password, not your EZVIZ account password.

**Stream paths — auto-probed.** EZVIZ models differ, so if you leave
`CAMERA_RTSP_PATH` at its default the app tries each known EZVIZ/Hikvision path
until one actually delivers frames, then logs the winner so you can pin it:

```
/H.264/ch1/main/av_stream   /H.264/ch1/sub/av_stream
/Streaming/Channels/101     /Streaming/Channels/102
/h264/ch1/main/av_stream    /ch1/main            /live/ch0
```

Use the sub-stream path if the main stream is laggy.

**Voice in Hindi/Hinglish:** set `WHISPER_LANGUAGE=auto` (default) and
`WHISPER_MODEL_SIZE=small` or `medium` — `tiny`/`base` transcribe Hindi poorly.
The assistant replies in whichever language you asked in, so "Print chal raha
hai kya?" gets a Hindi answer built from the same live state.

Credentials are read from `.env` only, never hard-coded, and are **masked as
`rtsp://admin:***@...` everywhere they're printed or logged.**

The RTSP path also: forces TCP transport, keeps a 1-frame buffer for low
latency, counts consecutive read failures, and **auto-reconnects** after
`CAMERA_READ_FAILURE_LIMIT` failures without taking the application down.

### Test it

```bash
python find_cameras.py             # scan every index/backend, report what works
python find_cameras.py --preview   # live preview of each working camera
python find_cameras.py --modes 0   # list resolutions index 0 actually supports
python camera_test.py              # live window, shows FPS, Q/ESC exits
python camera_test.py --no-window  # headless check (no GUI required)
python test_print_monitor.py       # full pipeline: frame -> layer -> width -> status
python test_print_monitor.py --synthetic   # pipeline test with no camera needed
python voice_test.py --text "Is printing running?"   # voice answer from live state
```

### Print status, layer detection and the query API

`agent/print_monitor.py` turns per-frame analysis into a **stable** machine
state and is the single source of truth for printing questions:

| Status | Meaning |
|---|---|
| `PRINTING` | Activity detected in the ROI (or reported by a machine link) |
| `NO_PRINT` | No activity |
| `LAYER_DETECTED` | A layer is being resolved, above the confidence threshold |
| `LAYER_MISSING` | Machine looks active but no layer is resolvable |
| `ABNORMAL_LAYER` | Width outside `LAYER_WIDTH_MIN_MM`..`LAYER_WIDTH_MAX_MM` |
| `CAMERA_ERROR` | No usable frame |

It never flips on a single frame: a reading must clear
`LAYER_CONFIDENCE_THRESHOLD` and then hold for `LAYER_DETECT_FRAMES` /
`LAYER_MISSING_FRAMES` consecutive analysed frames.

Pipeline output (`PrintObservation.to_dict()`):

```json
{ "printing": true, "layer_detected": true, "layer_width": 12.4,
  "layer_width_px": 167.0, "width_unit": "mm", "width_valid": true,
  "confidence": 0.94, "status": "PRINTING", "layer_status": "LAYER_DETECTED",
  "timestamp": 1750000000.0 }
```

**Raw pixels and calibrated millimetres are kept strictly separate.** With
`PIXELS_PER_MM` unset, `layer_width` is `null` and only `layer_width_px` is
filled — a pixel count is never reported as a millimetre.

Query API used by the voice assistant and any future feed-control logic:

```python
from agent.print_monitor import (
    get_current_printing_status, get_current_layer_status,
    get_current_layer_width, get_camera_status,
)
get_current_printing_status(state)   # {'printing': True, 'status': 'PRINTING', ...}
get_current_layer_width(state)       # {'width': 12.4, 'unit': 'mm', 'width_px': 167.0, ...}
```

The language model **queries these** — it never decides them itself.

### External USB camera on Windows (important)

OpenCV's default Windows backend is **Media Foundation (MSMF)**, which commonly
opens an external USB camera and then fails on read with
`cap_msmf.cpp ... OnReadSample() ... error status: -2147024890`. This project
therefore tries **DirectShow first** on Windows, then MSMF, then ANY.

The opener also:

- scans indexes `0..CAMERA_SCAN_MAX_INDEX` if the configured one gives no frames,
- reads the camera's **advertised formats** and uses those (no assuming 720p/MJPG),
- only requests MJPG when the device actually advertises it,
- verifies real frames arrive before accepting a camera,
- selects by name via `CAMERA_NAME_HINT` so the external camera wins over the
  built-in one,
- prints index, backend, requested/actual resolution, FPS, and frame status,
- refuses to open the same device twice from different parts of the app.

**OpenCV version matters:** the `opencv-python` **5.0** wheel reports
`DSHOW: backend is generally available but can't be used to capture by index`,
which disables DirectShow capture entirely. `requirements.txt` therefore pins
`opencv-python>=4.10,<5`. If DirectShow stops working, check your version:

```bash
python -c "import cv2; print(cv2.__version__)"
pip install "opencv-python>=4.10,<5"
```

Install `pygrabber` (Windows, in `requirements.txt`) to get device names and
format lists:

```bash
pip install pygrabber
python -c "from agent.camera import list_camera_names, list_device_formats; print(list_camera_names()); print(list_device_formats(0))"
```

---

## Calibration (required for millimetre values)

**A pixel is not a millimetre.** Until you calibrate, all sizes are reported in
pixels and every mm value reads "requires calibration" — the system will not
guess a conversion.

To calibrate:

1. Place an object of **known width** (e.g. a 20 mm gauge block) at the same
   distance from the camera as the printed wall.
2. Capture a frame and count how many pixels it spans (e.g. 200 px).
3. `PIXELS_PER_MM = pixels ÷ millimetres` → `200 ÷ 20 = 10.0`
4. Put it in `.env`:
   ```
   PIXELS_PER_MM=10.0
   ```
5. Re-calibrate whenever the camera, lens, zoom, or distance changes.

---

## What the system measures

### Layer detection (`agent/layer_detector.py`)

| Measured from the camera | Notes |
|---|---|
| Layer bands visible in frame | The count **in view**, not the total printed |
| Layer pitch / height (px → mm) | Median band centre-to-centre spacing |
| Layer (bead) width (px → mm) | Median horizontal material run |
| Layer consistency % | How uniform the layer pitch is |
| Layer width trend | increasing / decreasing / stable, from recent samples |

**Not available from a camera** (reported as unavailable, with the reason):
absolute **current layer number**, **total layer count**, and **print
progress %**. These need the machine's layer counter or the sliced G-code.

### Wall detection (`agent/wall_detector.py`)

| Measured | Notes |
|---|---|
| Outer wall detected | Yes/no |
| Wall thickness (px → mm) | Median material run per row |
| Wall continuity % | Horizontal completeness of the deposited material |
| Interior gaps | Holes *inside* the wall span = missing sections |
| Inner wall | Only when clearly separated vertical bands are visible |

**Wall count cannot be reliably determined from a single camera view** — the
outer surface occludes the inner walls. The system says exactly that instead of
guessing a number.

### Print quality (`agent/quality.py`)

A weighted score built **only** from metrics actually obtained this frame:

| Component | Weight | Detects |
|---|---|---|
| layer consistency | 0.4 | uneven layers, layer shifting |
| wall continuity | 0.4 | gaps, missing material, under-extrusion |
| width stability | 0.2 | over-/under-extrusion drift |

If none are available the score is **not scored** — never a random or
flattering number. Status is GOOD (≥90), FAIR (≥75), POOR (<75).

### Machine / sensor data (`agent/sensors.py`)

| Channel | Current status |
|---|---|
| nozzle/bed temp, print height, material flow, machine state | **SIMULATED** — labelled as such everywhere |
| pump running / pump speed | **NOT CONNECTED** |
| accelerator running / speed | **NOT CONNECTED** |
| print speed, feed speed, pressure, motor status, error state | **NOT CONNECTED** |
| current layer number, total layers | **NOT CONNECTED** |

To connect the real machine, subclass `MachineInterface` in `agent/sensors.py`,
implement `read()` against your controller, and pass it to `MonitoringAgent`.
Leave any channel your hardware doesn't expose as `None` — the agent will
report it as unavailable rather than fake it.

---

## Alerts (`agent/rules.py`)

Alerts fire only when a **real** measurement crosses a threshold, and reuse the
existing cooldown so the same warning never repeats:

- layer width too high / too low (needs calibration to fire)
- print quality below `PRINT_QUALITY_MIN_PERCENT`
- observed visual defects (missing material, layer shift)
- material flow low, temperature out of range, machine error (simulated source)
- pump stopped, accelerator stopped, machine error state (silent until the
  machine link is connected — it will not warn about equipment it can't see)
- pipe-cleaning and maintenance reminders

All alerts are spoken through the same TTS path.

---

## Running

```bash
python main.py
```

- A window opens with the live camera, layer overlay, and the dashboard panel:
  live camera readouts, wall analysis, print quality, machine status, alerts,
  and the last voice question/answer.
- **Q or ESC** closes the window and stops the app; `Ctrl+C` also works.
- To ask a question: **press Enter in the terminal**, then speak.

## Voice commands to test

```
"How many layers have been printed?"
"What is the current layer width?"
"Is the layer width increasing or decreasing?"
"What is the print quality?"
"Is there any problem with the print?"
"How many walls have been printed?"
"Is material flowing?"
"Is the pump running?"
"What is the pump speed?"
"Is the accelerator running?"
"What is the print speed?"
"Is the printer currently printing?"
```

Expected honest behaviour: questions about layer bands, width, consistency,
wall continuity, and quality are answered from live camera measurements.
Questions about pump/accelerator/print speed answer "not connected". Questions
about total layers, current layer number, and progress answer that the camera
cannot determine them.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `CAMERA_INDEX` / `CAMERA_WIDTH` / `CAMERA_HEIGHT` / `CAMERA_FPS` | Camera selection and capture format |
| `SHOW_CAMERA_WINDOW` | `false` runs vision without a GUI window |
| `PIXELS_PER_MM` | Calibration; blank ⇒ pixel-only measurements |
| `MIN_LAYER_BOUNDARIES` | Minimum bands before layer data is trusted |
| `LAYER_WIDTH_HISTORY` | Samples used for the width trend |
| `LAYER_WIDTH_MIN_MM` / `LAYER_WIDTH_MAX_MM` | Layer-width alert thresholds |
| `PRINT_QUALITY_MIN_PERCENT` | Quality alert threshold |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Voice Q&A backend |
| `WHISPER_MODEL_SIZE` / `MAX_RECORD_SECONDS` | Speech-to-text |
| `ELEVENLABS_API_KEY` / `ELEVENLABS_VOICE_ID` | Optional higher-quality TTS |
| `POLL_INTERVAL_SECONDS`, temp/flow thresholds, reminder timers | Monitoring loop |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` / `ANTHROPIC_WORKSPACE_ID` | Optional alternative AI backend |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not open any working camera` | Run `python find_cameras.py` — it lists every index/backend tried. Then check the rows below. |
| MSMF `OnReadSample() ... error status: -2147024890` | MSMF can't drive this camera. Force DirectShow: `CAMERA_BACKEND=dshow` in `.env`. Make sure OpenCV is 4.x (`pip install "opencv-python>=4.10,<5"`). |
| `DSHOW: backend is generally available but can't be used to capture by index` | You're on `opencv-python` 5.x, which can't do DirectShow capture. Downgrade: `pip install "opencv-python>=4.10,<5"`. |
| Camera enumerates but never returns frames | Close every other camera user (Camera app, Teams, Zoom, OBS, **browser tabs** — Chrome/Edge hold cameras silently), unplug/replug the camera, try another USB port (prefer a direct port over a hub), then re-run `python find_cameras.py`. Verify status with `Get-PnpDevice -Class Camera`. |
| Wrong camera opens (built-in instead of USB) | Set `CAMERA_NAME_HINT=Sonix` (or part of your camera's name) in `.env`, and/or pin `CAMERA_INDEX`. |
| Requested resolution fails | Check what the camera really supports: `python find_cameras.py --modes 0`. Set `CAMERA_WIDTH`/`CAMERA_HEIGHT` to a listed mode (many USB cameras only do 640x480 YUY2). |
| App runs but no window | `SHOW_CAMERA_WINDOW=false` in `.env`, or the camera failed to open (the reason is printed). |
| "LAYER STATUS: NOT DETECTED" | Low contrast or bad angle. Improve lighting, view the wall from the side, ensure layer lines run horizontally. |
| Layer width shows px, not mm | `PIXELS_PER_MM` isn't set — see Calibration. |
| "Cannot separate the printed part from the background" | Too little contrast or the part fills the whole frame. Adjust framing/lighting. |
| Quality shows "cannot be scored" | No layer or wall measurements available yet — usually the same framing/lighting cause. |
| Voice answers "no Gemini API key" | Set `GEMINI_API_KEY` in `.env`, restart. |
| Voice answers "model was not found" | Your key can't use `GEMINI_MODEL`; try `gemini-2.5-flash`. |
| Voice answers "hit the rate limit" | Free-tier quota; wait and retry. |
| Whisper is slow | Set `WHISPER_MODEL_SIZE=tiny` in `.env`. |
| Pump/accelerator always "not connected" | Expected — implement `MachineInterface` (see above). |
| RTSP: `Could not connect to the RTSP camera` | `ping <CAMERA_IP>` first. Check same LAN, RTSP enabled, port 554 open, and that `CAMERA_PASSWORD` is the camera's **verification code** (not your EZVIZ account password). |
| RTSP connects but no frames | Wrong stream path — try `CAMERA_RTSP_PATH=/H.264/ch1/sub/av_stream`. |
| RTSP laggy / high latency | Use the sub-stream path, keep `CAMERA_RTSP_TRANSPORT=tcp`, and raise `FRAME_SKIP`. |
| RTSP drops repeatedly | Wi-Fi signal. The app auto-reconnects (`CAMERA_RECONNECT_DELAY_SECONDS`); prefer wired/stronger signal. |
| Status stuck on `CAMERA_ERROR` | No recent analysis — check the camera test first. |
| Status flip-flops | Raise `LAYER_DETECT_FRAMES` / `LAYER_MISSING_FRAMES` or `LAYER_CONFIDENCE_THRESHOLD`. |
| `printing` always false | Lower `PRINT_MOTION_THRESHOLD` (the ROI may show little movement) or set an ROI over the deposition area. |
| Width reported in px, never mm | `PIXELS_PER_MM` not calibrated — see Calibration. |

---

## Roadmap / what still needs work

1. **Machine link** — implement `MachineInterface` for pump, accelerator, drive
   speeds, pressure, and job info (this also unlocks true current-layer number
   and print progress).
2. **Trained CV model** — the current detectors are classical (segmentation +
   projection profiles). A model trained on your actual prints would improve
   defect classification (spaghetti, warping, delamination) and could make
   inner-wall separation possible. Swap the body of `LayerDetector.analyze()` /
   `WallDetector.analyze()` and keep the return types.
3. **Multi-camera** — a second viewpoint is what would make wall counting and
   inner-wall measurement reliable.
4. **Control module** — deliberately not implemented; monitoring only for now.
