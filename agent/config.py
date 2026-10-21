import os

from dotenv import load_dotenv

load_dotenv()


def _env(*names: str, default: str = "") -> str:
    """
    First non-empty value among several accepted variable names.

    Lets the same setting be written either way in .env, e.g. RTSP_URL or
    CAMERA_RTSP_URL, LAYER_MIN_WIDTH_MM or LAYER_WIDTH_MIN_MM.
    """
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default

# --- Camera System ---
CAMERA_ENABLED = os.getenv("KELVIN6K_CAMERA_ENABLED", "false").strip().lower() in ("1", "true", "yes")

# --- Voice / TTS ---
# Leave ELEVENLABS_API_KEY blank to fall back to free offline TTS (pyttsx3).
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")

# --- Monitoring loop ---
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "5"))

# --- Reminder timers ---
# NOTE: shortened for demo purposes so you can see them fire quickly.
# For real deployment, set these much higher, e.g. 14400 for every 4 hours.
PIPE_CLEAN_INTERVAL_SECONDS = float(os.getenv("PIPE_CLEAN_INTERVAL_SECONDS", "120"))
MAINTENANCE_INTERVAL_SECONDS = float(os.getenv("MAINTENANCE_INTERVAL_SECONDS", "300"))

# --- Safety thresholds ---
NOZZLE_TEMP_MIN_C = float(os.getenv("NOZZLE_TEMP_MIN_C", "180"))
NOZZLE_TEMP_MAX_C = float(os.getenv("NOZZLE_TEMP_MAX_C", "260"))
BED_TEMP_MAX_C = float(os.getenv("BED_TEMP_MAX_C", "110"))
HEIGHT_TOLERANCE_MM = float(os.getenv("HEIGHT_TOLERANCE_MM", "0.5"))
MATERIAL_FLOW_MIN_PERCENT = float(os.getenv("MATERIAL_FLOW_MIN_PERCENT", "85"))

# Minimum time between repeats of the same non-timer alert, so the agent
# doesn't spam the operator with the same warning every poll cycle.
ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))

# --- Camera source: "usb" (local UVC camera) or "rtsp" (EZVIZ / IP camera) ---
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "usb").strip().lower()

# --- EZVIZ / RTSP IP camera ---
# Either set CAMERA_RTSP_URL directly, or set IP + credentials and let the URL
# be built. Credentials are NEVER hard-coded and never logged.
CAMERA_IP = os.getenv("CAMERA_IP", "").strip()
CAMERA_USERNAME = os.getenv("CAMERA_USERNAME", "admin").strip()
# EZVIZ cameras use the 6-character VERIFICATION CODE printed on the device
# label as the RTSP password.
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", os.getenv("CAMERA_VERIFICATION_CODE", ""))
CAMERA_RTSP_PORT = int(os.getenv("CAMERA_RTSP_PORT", "554"))
# EZVIZ: /H.264/ch1/main/av_stream (HD) or .../sub/av_stream (lower latency)
CAMERA_RTSP_PATH = os.getenv("CAMERA_RTSP_PATH", "/H.264/ch1/main/av_stream")
# Accepts either CAMERA_RTSP_URL or the shorter RTSP_URL.
CAMERA_RTSP_URL = _env("CAMERA_RTSP_URL", "RTSP_URL")
# "tcp" is far more reliable than udp for RTSP over Wi-Fi/LAN.
CAMERA_RTSP_TRANSPORT = os.getenv("CAMERA_RTSP_TRANSPORT", "tcp").strip().lower()
# Connection timeout for the RTSP handshake.
CAMERA_RTSP_TIMEOUT_SECONDS = float(_env("CAMERA_RTSP_TIMEOUT_SECONDS", default="8"))

# Reconnect behaviour for network cameras.
CAMERA_RECONNECT_DELAY_SECONDS = float(os.getenv("CAMERA_RECONNECT_DELAY_SECONDS", "3"))
# 0 = retry forever.
CAMERA_MAX_RECONNECT_ATTEMPTS = int(os.getenv("CAMERA_MAX_RECONNECT_ATTEMPTS", "0"))
# Consecutive failed reads before the stream is considered dead.
CAMERA_READ_FAILURE_LIMIT = int(os.getenv("CAMERA_READ_FAILURE_LIMIT", "30"))

# How long to wait for the FIRST frames when opening. RTSP streams routinely
# need 1-3 seconds to start; a couple of instant reads would wrongly declare a
# working camera dead.
CAMERA_WARMUP_SECONDS = float(os.getenv("CAMERA_WARMUP_SECONDS", "3.0"))

# --- Camera (USB) ---
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
# 640x480 @ 30 is the mode virtually every USB/UVC camera supports. Higher modes
# are attempted first only if you raise these; the opener falls back safely.
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
SHOW_CAMERA_WINDOW = os.getenv("SHOW_CAMERA_WINDOW", "true").strip().lower() != "false"

# The live window is video + dashboard panel side by side, which is wider than
# many laptop screens. It is scaled to fit inside these bounds.
WINDOW_MAX_WIDTH = int(os.getenv("WINDOW_MAX_WIDTH", "1600"))
WINDOW_MAX_HEIGHT = int(os.getenv("WINDOW_MAX_HEIGHT", "900"))

# Capture backend: auto | dshow | msmf | any.
# "auto" prefers DirectShow on Windows, which is far more reliable with
# external USB/UVC cameras than Media Foundation (MSMF).
CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "auto").strip().lower()

# Substring of the camera's name to prefer, e.g. "Sonix" for an external USB
# camera. Used to avoid grabbing the laptop's built-in webcam by accident.
# Requires the optional `pygrabber` package on Windows for name lookup.
CAMERA_NAME_HINT = os.getenv("CAMERA_NAME_HINT", "").strip()

# If the configured index fails, scan indexes 0..CAMERA_SCAN_MAX_INDEX.
CAMERA_AUTO_SCAN = os.getenv("CAMERA_AUTO_SCAN", "true").strip().lower() != "false"
CAMERA_SCAN_MAX_INDEX = int(os.getenv("CAMERA_SCAN_MAX_INDEX", "5"))

# --- Camera 2: Dedicated Site People-Counting Camera ---
SITE_CAMERA_ENABLED = os.getenv("SITE_CAMERA_ENABLED", "true").strip().lower() != "false"
SITE_CAMERA_SOURCE = os.getenv("SITE_CAMERA_SOURCE", "usb").strip().lower()  # "usb" | "rtsp"
SITE_CAMERA_INDEX = int(os.getenv("SITE_CAMERA_INDEX", "1"))
SITE_CAMERA_WIDTH = int(os.getenv("SITE_CAMERA_WIDTH", "1280"))
SITE_CAMERA_HEIGHT = int(os.getenv("SITE_CAMERA_HEIGHT", "720"))
SITE_CAMERA_FPS = int(os.getenv("SITE_CAMERA_FPS", "15"))
SITE_CAMERA_NAME_HINT = os.getenv("SITE_CAMERA_NAME_HINT", "").strip()
SITE_CAMERA_RTSP_URL = os.getenv("SITE_CAMERA_RTSP_URL", "").strip()
SITE_CAMERA_CACHE_SECONDS = float(os.getenv("SITE_CAMERA_CACHE_SECONDS", "2.0"))
PEOPLE_DETECTOR_CONFIDENCE_THRESHOLD = float(os.getenv("PEOPLE_DETECTOR_CONFIDENCE_THRESHOLD", "0.65"))
SITE_CAMERA_OBJECT_CONFIDENCE_THRESHOLD = float(os.getenv("SITE_CAMERA_OBJECT_CONFIDENCE_THRESHOLD", "0.30"))
SITE_CAMERA_MODEL_NAME = os.getenv("SITE_CAMERA_MODEL_NAME", "yolov8n-seg.pt").strip()
SITE_CAMERA_EYEWEAR_MODEL_NAME = os.getenv("SITE_CAMERA_EYEWEAR_MODEL_NAME", "yolov8s-world.pt").strip()
SITE_CAMERA_EYEWEAR_CONFIDENCE_THRESHOLD = float(os.getenv("SITE_CAMERA_EYEWEAR_CONFIDENCE_THRESHOLD", "0.50"))
SITE_CAMERA_EYEWEAR_CONSECUTIVE_FRAMES = int(os.getenv("SITE_CAMERA_EYEWEAR_CONSECUTIVE_FRAMES", "2"))
SITE_CAMERA_EYEWEAR_MAX_MISSED_FRAMES = int(os.getenv("SITE_CAMERA_EYEWEAR_MAX_MISSED_FRAMES", "1"))
SITE_CAMERA_FIRE_CONFIDENCE_THRESHOLD = float(os.getenv("SITE_CAMERA_FIRE_CONFIDENCE_THRESHOLD", "0.25"))
SITE_CAMERA_LIGHTER_CONFIDENCE_THRESHOLD = float(os.getenv("SITE_CAMERA_LIGHTER_CONFIDENCE_THRESHOLD", "0.25"))

# --- Vision calibration ---
# MUST be calibrated for your actual camera position, lens, and distance to the
# part. Place an object of known width in the frame, measure how many pixels it
# spans, and divide by its width in mm. Leave blank/0 to disable millimeter
# conversion entirely -- measurements are then reported in pixels only, and
# anything requiring mm is reported as "requires calibration" rather than guessed.
_pixels_per_mm_raw = os.getenv("PIXELS_PER_MM", "").strip()
PIXELS_PER_MM = float(_pixels_per_mm_raw) if _pixels_per_mm_raw else 0.0

# Minimum number of detected layer boundaries before layer measurements are
# considered meaningful rather than noise.
MIN_LAYER_BOUNDARIES = int(os.getenv("MIN_LAYER_BOUNDARIES", "3"))

# How many recent layer-width samples to keep for trend analysis.
LAYER_WIDTH_HISTORY = int(os.getenv("LAYER_WIDTH_HISTORY", "30"))

# --- Vision-derived alert thresholds (only evaluated when a real measurement exists) ---
# Accepts LAYER_WIDTH_MIN_MM or LAYER_MIN_WIDTH_MM (either spelling).
# NOTE: these defaults suit a LARGE-FORMAT machine like Kelvin6K, where a bead
# is on the order of 10 mm -- not a desktop FDM printer (~0.4 mm). Set them to
# your machine's real acceptable range or every layer reads as ABNORMAL.
LAYER_WIDTH_MIN_MM = float(_env("LAYER_WIDTH_MIN_MM", "LAYER_MIN_WIDTH_MM", default="5.0"))
LAYER_WIDTH_MAX_MM = float(_env("LAYER_WIDTH_MAX_MM", "LAYER_MAX_WIDTH_MM", default="25.0"))
PRINT_QUALITY_MIN_PERCENT = float(os.getenv("PRINT_QUALITY_MIN_PERCENT", "70"))

# --- Region of interest -------------------------------------------------
# Restrict analysis to the area where material is actually deposited. All zeros
# (the default) analyses the whole frame. Values are pixels in the raw frame.
ROI_X = int(os.getenv("ROI_X", "0"))
ROI_Y = int(os.getenv("ROI_Y", "0"))
ROI_WIDTH = int(os.getenv("ROI_WIDTH", "0"))
ROI_HEIGHT = int(os.getenv("ROI_HEIGHT", "0"))

# --- Print-status detection --------------------------------------------
# Analyse every Nth frame (display stays smooth, CV cost stays affordable).
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))

# Minimum detection confidence before a layer reading is acted on.
LAYER_CONFIDENCE_THRESHOLD = float(os.getenv("LAYER_CONFIDENCE_THRESHOLD", "0.6"))

# Temporal debouncing: how many CONSECUTIVE analysed frames must agree before
# the machine state flips. Stops one bad frame from changing the state.
LAYER_DETECT_FRAMES = int(os.getenv("LAYER_DETECT_FRAMES", "5"))
LAYER_MISSING_FRAMES = int(os.getenv("LAYER_MISSING_FRAMES", "8"))

# "Printing" is inferred from frame-to-frame change in the ROI (material being
# deposited / machine moving). Mean absolute pixel difference above this counts
# as activity; below it for PRINT_IDLE_FRAMES analysed frames means idle.
# Measured empirically: deposition inside a typical ROI produces a mean
# absolute frame difference well under 1.0, so a threshold of 2.0 never fires.
# Tune it by watching the "Motion" value on the live dashboard: note the value
# while idle and while printing, then set this between the two.
PRINT_MOTION_THRESHOLD = float(os.getenv("PRINT_MOTION_THRESHOLD", "0.3"))
PRINT_ACTIVE_FRAMES = int(os.getenv("PRINT_ACTIVE_FRAMES", "3"))
PRINT_IDLE_FRAMES = int(os.getenv("PRINT_IDLE_FRAMES", "10"))

# --- Database (persistent machine state / print jobs) ---
DATABASE_PATH = os.getenv("DATABASE_PATH", "kelvin6k.db")
# Machine state older than this is treated as stale, never reported as current.
MACHINE_STATE_MAX_AGE_SECONDS = float(os.getenv("MACHINE_STATE_MAX_AGE_SECONDS", "30"))
# Below this confidence the assistant says the measurement is uncertain.
MIN_REPORT_CONFIDENCE = float(os.getenv("MIN_REPORT_CONFIDENCE", "0.5"))
# How often the vision loop persists a machine_state row (seconds).
STATE_PERSIST_INTERVAL_SECONDS = float(os.getenv("STATE_PERSIST_INTERVAL_SECONDS", "2.0"))

# --- Shop-floor context (clock/config facts, not machine telemetry) ---
# Shift windows as "Name:startHour-endHour", comma separated.
def _parse_shifts(raw: str):
    windows = []
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part or "-" not in part:
            continue
        name, _, span = part.partition(":")
        start, _, end = span.partition("-")
        try:
            windows.append((name.strip(), int(start), int(end)))
        except ValueError:
            continue
    return windows


SHIFT_WINDOWS = _parse_shifts(
    os.getenv("SHIFT_WINDOWS", "Morning:6-14,Afternoon:14-22,Night:22-6")
)
# Shown as the current operator when the job doesn't record one.
OPERATOR_NAME = os.getenv("OPERATOR_NAME", "").strip()

# --- RAG (documentation retrieval) ---
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "rag_index")
RAG_DOCS_DIR = os.getenv("RAG_DOCS_DIR", "docs")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# Retrieval below this score is treated as "not found in the documentation".
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.15"))
# Threshold to consider historical RAG match sufficiently authoritative for troubleshooting
RAG_TROUBLESHOOTING_THRESHOLD = float(os.getenv("RAG_TROUBLESHOOTING_THRESHOLD", "0.35"))
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

# --- Attachment upload configuration ---
MAX_ATTACHMENT_SIZE_BYTES = int(os.getenv("MAX_ATTACHMENT_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB limit
MAX_ATTACHMENT_COUNT = int(os.getenv("MAX_ATTACHMENT_COUNT", "4"))
INCIDENT_IMAGES_DIR = os.getenv(
    "INCIDENT_IMAGES_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "incident_images"),
)
REFERENCE_PARTS_DIR = os.getenv(
    "REFERENCE_PARTS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reference_parts"),
)
INCIDENT_RETENTION_DAYS = int(os.getenv("INCIDENT_RETENTION_DAYS", "365"))

# --- Voice Q&A / Multimodal AI (two-way interaction) ---
# Google Gemini powers the conversational answers and vision analysis.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))

# --- Anthropic (kept for the optional Anthropic backend / check_api_key.py) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

# Required only for identity-linked API keys (created via SSO/user identity
# in the Console rather than a classic organization key). Leave blank if
# your key doesn't need it -- the API will say so via a 400 error if it does.
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "")

# Set false to run monitoring without the voice assistant.
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").strip().lower() != "false"

# Local speech-to-text model size (faster-whisper): tiny | base | small | medium
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Speech-to-text language. "auto" lets Whisper detect it, which is what you want
# for mixed Hindi/English ("Print chal raha hai kya?"). Set "en" to force English.
# NOTE: Hindi/Hinglish needs a larger model -- use WHISPER_MODEL_SIZE=small or
# medium; "tiny"/"base" transcribe non-English poorly.
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto").strip().lower()

# How long to record after the operator presses Enter to ask a question.
MAX_RECORD_SECONDS = float(os.getenv("MAX_RECORD_SECONDS", "6"))


# ---------------------------------------------------------------------------
# Perpendicular layer-width metrology (agent/metrology.py)
#
# Deliberately its own block. The existing PIXELS_PER_MM above feeds the layer
# detector, the wall detector and the dashboard's width card, so it is left
# exactly as it is; METROLOGY_PIXELS_PER_MM overrides it only for metrology,
# and falls back to it when blank. That way a metrology calibration taken at
# the 30-35 cm working height cannot disturb the existing measurements.
# ---------------------------------------------------------------------------

METROLOGY_ENABLED = os.getenv("METROLOGY_ENABLED", "true").strip().lower() in ("1", "true", "yes")

# Engineering target and tolerance for the printed bead.
METROLOGY_TARGET_WIDTH_MM = float(os.getenv("METROLOGY_TARGET_WIDTH_MM", "45.0"))
METROLOGY_TOLERANCE_MM = float(os.getenv("METROLOGY_TOLERANCE_MM", "2.0"))

# Number of measurement stations placed along the centreline (W1..Wn).
METROLOGY_STATIONS = int(os.getenv("METROLOGY_STATIONS", "10"))

# Below this many valid stations the average is not trustworthy and the status
# is MEASUREMENT INVALID -- never PASS or a width warning.
METROLOGY_MIN_VALID_STATIONS = int(os.getenv("METROLOGY_MIN_VALID_STATIONS", "5"))

# Blank -> use PIXELS_PER_MM. Zero on both -> no millimetre values at all.
_metrology_ppm_raw = os.getenv("METROLOGY_PIXELS_PER_MM", "").strip()
METROLOGY_PIXELS_PER_MM = float(_metrology_ppm_raw) if _metrology_ppm_raw else 0.0

# The calibrated working distance this scale is valid for. Recorded so the
# overlay can state the reference setup; changing the height invalidates the
# calibration and it must be measured again.
METROLOGY_CAMERA_HEIGHT_MIN_CM = float(os.getenv("METROLOGY_CAMERA_HEIGHT_MIN_CM", "30"))
METROLOGY_CAMERA_HEIGHT_MAX_CM = float(os.getenv("METROLOGY_CAMERA_HEIGHT_MAX_CM", "35"))

# Voice announcement pacing. The camera runs continuously, so an announcement
# fires on a status CHANGE, or after the status has been stable this long,
# and never more often than the cooldown.
METROLOGY_VOICE_ALERT_COOLDOWN_SEC = float(os.getenv("METROLOGY_VOICE_ALERT_COOLDOWN_SEC", "10"))
METROLOGY_VOICE_STABLE_SEC = float(os.getenv("METROLOGY_VOICE_STABLE_SEC", "3"))
# "Unable to measure" gets its own, longer cooldown: an uncovered lens or an
# empty bed would otherwise repeat it indefinitely.
METROLOGY_VOICE_INVALID_COOLDOWN_SEC = float(
    os.getenv("METROLOGY_VOICE_INVALID_COOLDOWN_SEC", "60")
)

# Segmentation guards, mirroring agent/segmentation.py's refusal to measure a
# frame with nothing in it.
METROLOGY_MIN_CONTRAST = float(os.getenv("METROLOGY_MIN_CONTRAST", "5.0"))
METROLOGY_MIN_AREA_RATIO = float(os.getenv("METROLOGY_MIN_AREA_RATIO", "0.005"))
METROLOGY_MAX_AREA_RATIO = float(os.getenv("METROLOGY_MAX_AREA_RATIO", "0.90"))
METROLOGY_MIN_CONTOUR_AREA = float(os.getenv("METROLOGY_MIN_CONTOUR_AREA", "500"))

# Centreline tracing.
METROLOGY_MIN_SKELETON_PX = int(os.getenv("METROLOGY_MIN_SKELETON_PX", "20"))
METROLOGY_SMOOTH_WINDOW = int(os.getenv("METROLOGY_SMOOTH_WINDOW", "9"))
METROLOGY_SKELETON_MAX_GAP_PX = float(os.getenv("METROLOGY_SKELETON_MAX_GAP_PX", "6"))


def metrology_pixels_per_mm() -> float:
    """The scale metrology should use: its own if set, otherwise the shared one."""
    return METROLOGY_PIXELS_PER_MM if METROLOGY_PIXELS_PER_MM > 0 else PIXELS_PER_MM


def metrology_range_min() -> float:
    return METROLOGY_TARGET_WIDTH_MM - METROLOGY_TOLERANCE_MM


def metrology_range_max() -> float:
    return METROLOGY_TARGET_WIDTH_MM + METROLOGY_TOLERANCE_MM

# Centreline sampling: slices taken across the bead's principal axis. Each
# slice yields one ordered centreline point.
METROLOGY_CENTERLINE_SAMPLES = int(os.getenv("METROLOGY_CENTERLINE_SAMPLES", "60"))
METROLOGY_MIN_AXIS_LENGTH_PX = float(os.getenv("METROLOGY_MIN_AXIS_LENGTH_PX", "40"))

# Metrology is far heavier than the per-frame vision analysis (~84 ms on a
# 640x480 frame), so it runs on its own clock rather than every analysed
# frame. The overlay redraws the cached result in between, which keeps the
# MJPEG stream at full rate. A bead does not change shape in 0.5 s.
METROLOGY_MIN_INTERVAL_SEC = float(os.getenv("METROLOGY_MIN_INTERVAL_SEC", "0.5"))
