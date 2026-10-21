import type {
  AskResponse,
  AttendanceRecord,
  HealthResponse,
  HistoryResponse,
  StatusResponse,
  WallProductionRecord,
  WallProductionTotal,
} from "./types";

export const API_BASE =
  import.meta.env.VITE_API_BASE?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return getJson<HealthResponse>("/api/health", signal);
}

export function getStatus(signal?: AbortSignal): Promise<StatusResponse> {
  return getJson<StatusResponse>("/api/status", signal);
}

/** Shared by text input AND voice. Can send text, attachments, or both. `speak` triggers server-side TTS. */
export async function ask(question: string, files?: File[], speak = false, signal?: AbortSignal): Promise<AskResponse> {
  let response: Response;
  if (files && files.length > 0) {
    const form = new FormData();
    form.append("question", question);
    form.append("speak", String(speak));
    for (const f of files) {
      form.append("files", f, f.name);
    }
    response = await fetch(`${API_BASE}/api/assistant/query`, {
      method: "POST",
      body: form,
      signal,
    });
  } else {
    response = await fetch(`${API_BASE}/api/assistant/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, speak }),
      signal,
    });
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(`Ask failed: ${detail}`);
  }
  return (await response.json()) as AskResponse;
}

/** Server-side speech-to-text (faster-whisper) -- fallback when the browser has no Web Speech API. */
/** Maps a MediaRecorder MIME type to the matching file extension. */
function audioExtension(mime: string): string {
  if (mime.includes("webm")) return "webm";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp4") || mime.includes("m4a")) return "mp4";
  if (mime.includes("wav")) return "wav";
  return "webm";
}

export async function transcribeAudio(blob: Blob): Promise<string> {
  // The filename must match the ACTUAL format. It was hard-coded to
  // "question.wav" while the browser produces WebM/Opus, so the server wrote
  // WebM bytes into a file named .wav and the decoder had to guess.
  const filename = `question.${audioExtension(blob.type)}`;
  const url = `${API_BASE}/api/stt`;

  console.info("[STT] POST", url);
  console.info(`[STT] Blob: ${blob.size} bytes, type "${blob.type}", sent as ${filename}`);

  const form = new FormData();
  form.append("audio", blob, filename);

  const response = await fetch(url, { method: "POST", body: form });
  console.info("[STT] HTTP", response.status, response.statusText);

  if (!response.ok) {
    const raw = await response.text().catch(() => response.statusText);
    console.error("[STT] Error body:", raw);
    // FastAPI wraps errors as {"detail": "..."}; show the sentence, not JSON.
    let message = raw;
    try {
      const parsed = JSON.parse(raw) as { detail?: unknown };
      if (typeof parsed.detail === "string") message = parsed.detail;
    } catch {
      /* not JSON -- use the raw text */
    }
    throw new Error(message);
  }

  const data = (await response.json()) as { transcript: string };
  console.info("[STT] Transcript:", data.transcript);
  return data.transcript;
}

export const cameraStreamUrl = () => `${API_BASE}/api/camera/stream`;
export const cameraSnapshotUrl = () => `${API_BASE}/api/camera/snapshot?t=${Date.now()}`;
export const camera2StreamUrl = () => `${API_BASE}/api/camera2/stream`;
export const camera2SnapshotUrl = () => `${API_BASE}/api/camera2/snapshot?t=${Date.now()}`;

export function getHistory(signal?: AbortSignal): Promise<HistoryResponse> {
  return getJson<HistoryResponse>("/api/history", signal);
}

/** Speak text through the SERVER's speaker (existing pyttsx3/ElevenLabs TTS). */
export async function speakOnServer(text: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(`Server TTS failed: ${await response.text().catch(() => response.statusText)}`);
  }
}

/**
 * Fetches synthesised speech from the backend as an audio Blob.
 * The BROWSER plays it, so the sound comes out of the operator's speakers
 * rather than the server's.
 */
export async function fetchSpeechAudio(text: string): Promise<{ blob: Blob; engine: string }> {
  const response = await fetch(`${API_BASE}/api/tts/audio`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(`TTS failed: ${await response.text().catch(() => response.statusText)}`);
  }
  return {
    blob: await response.blob(),
    engine: response.headers.get("X-TTS-Engine") ?? "unknown",
  };
}

export interface TtsDiagnostics {
  provider: string;
  elevenlabs_key_configured: boolean;
  voice_count: number;
  pyttsx3_voices: string[];
  volume?: number;
  current_voice?: string | null;
  engine_error?: string;
}

export function getTtsDiagnostics(): Promise<TtsDiagnostics> {
  return getJson<TtsDiagnostics>("/api/tts/diagnostics");
}

// --- Attendance ---


export async function checkInAttendance(employee_id?: string, employee_name?: string): Promise<AttendanceRecord> {
  const response = await fetch(`${API_BASE}/api/v1/attendance/check-in`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ employee_id, employee_name }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || "Check-in failed");
  }
  return response.json();
}

export async function checkOutAttendance(employee_id?: string, employee_name?: string): Promise<AttendanceRecord> {
  const response = await fetch(`${API_BASE}/api/v1/attendance/check-out`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ employee_id, employee_name }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || "Check-out failed");
  }
  return response.json();
}

export async function getAttendanceList(date?: string): Promise<AttendanceRecord[]> {
  const path = date ? `/api/v1/attendance?date=${date}` : `/api/v1/attendance`;
  return getJson<AttendanceRecord[]>(path);
}

export async function getOperators(): Promise<{ operators: any[] }> {
  return getJson<{ operators: any[] }>("/api/operators");
}

export async function createOperator(data: { name: string; employee_code?: string; role: string; active: number }): Promise<any> {
  const res = await fetch(`${API_BASE}/api/operators`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Failed to create operator.");
  }
  return res.json();
}

/**
 * Resolves an API path or image URL against API_BASE.
 * Handles absolute URLs, data/blob URLs, and relative /api/... paths.
 */
export function resolveApiUrl(path: string): string {
  if (!path) return "";
  if (
    path.startsWith("http://") ||
    path.startsWith("https://") ||
    path.startsWith("blob:") ||
    path.startsWith("data:")
  ) {
    return path;
  }
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${cleanPath}`;
}

/**
 * Returns a direct download URL with download=1 query parameter.
 */
export function getDownloadUrl(imageUrl: string): string {
  const resolved = resolveApiUrl(imageUrl);
  const separator = resolved.includes("?") ? "&" : "?";
  return `${resolved}${separator}download=1`;
}

/**
 * Safely downloads an incident image (original, annotated, or reference),
 * verifying that genuine image bytes were received (rejecting HTML error pages),
 * and triggers a client-side file save with the server's meaningful filename.
 */
export async function downloadIncidentImage(url: string, fallbackName?: string): Promise<void> {
  const downloadUrl = getDownloadUrl(url);
  try {
    const res = await fetch(downloadUrl);
    if (!res.ok) {
      throw new Error(`Download failed with status ${res.status}`);
    }
    const contentType = res.headers.get("Content-Type") || "";
    // Verify it is actually an image and not HTML
    if (contentType.includes("text/html") || (!contentType.startsWith("image/") && !contentType.includes("octet-stream"))) {
      throw new Error(`Invalid server response: expected image bytes but received ${contentType}`);
    }

    // Extract filename from Content-Disposition header if present
    let filename = fallbackName;
    const disposition = res.headers.get("Content-Disposition");
    if (disposition) {
      const match = disposition.match(/filename=["']?([^"';]+)["']?/i);
      if (match && match[1]) {
        filename = match[1].trim();
      }
    }
    if (!filename) {
      const ext = contentType.includes("png") ? ".png" : contentType.includes("webp") ? ".webp" : ".jpg";
      filename = `kelvin6k_image_${Date.now()}${ext}`;
    }

    const blob = await res.blob();
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 2000);
  } catch (err) {
    console.error("[DOWNLOAD] Blob download failed, falling back to direct navigation:", err);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = fallbackName || "kelvin6k_image";
    link.target = "_blank";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
}

// --- Employee Wall Production Tracking ---

export interface RecordWallProductionParams {
  operator_id: string; // name, code, or numeric id
  quantity: number;
  job_id?: string;
  shift_id?: number;
  notes?: string;
  request_id?: string;
  recorded_by?: string;
}

export async function recordWallProduction(data: RecordWallProductionParams): Promise<{ success: boolean; record: WallProductionRecord }> {
  const response = await fetch(`${API_BASE}/api/v1/production/walls`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to record wall production");
  }
  return response.json();
}

export interface GetWallProductionParams {
  operator_id?: string;
  scope?: "today" | "this_week" | "job" | "active_shift" | "all_time" | "date_range" | string;
  job_id?: string;
  date_from?: string;
  date_to?: string;
  shift_id?: number;
  include_records?: boolean;
}

export async function getWallProduction(params?: GetWallProductionParams): Promise<WallProductionTotal> {
  const q = new URLSearchParams();
  if (params?.operator_id) q.set("operator_id", params.operator_id);
  if (params?.scope) q.set("scope", params.scope);
  if (params?.job_id) q.set("job_id", params.job_id);
  if (params?.date_from) q.set("date_from", params.date_from);
  if (params?.date_to) q.set("date_to", params.date_to);
  if (params?.shift_id !== undefined) q.set("shift_id", String(params.shift_id));
  if (params?.include_records !== undefined) q.set("include_records", String(params.include_records));
  const qs = q.toString();
  return getJson<WallProductionTotal>(`/api/v1/production/walls${qs ? `?${qs}` : ""}`);
}

export interface CorrectWallProductionParams {
  record_id: string;
  corrected_quantity: number;
  reason?: string;
  corrected_by?: string;
}

export async function correctWallProduction(data: CorrectWallProductionParams): Promise<{
  success: boolean;
  status: string;
  original_record_id: string;
  original_quantity: number;
  replacement_record: WallProductionRecord;
}> {
  const response = await fetch(`${API_BASE}/api/v1/production/walls/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to correct wall production");
  }
  return response.json();
}

