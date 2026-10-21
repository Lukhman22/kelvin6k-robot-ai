/**
 * Types mirroring the Kelvin6K Python API (api/server.py).
 *
 * Every machine value arrives wrapped in a Field<T> carrying its provenance, so
 * the UI can never render a simulated or missing number as if it were measured.
 */

export type Provenance = "REAL" | "MOCK" | "UNAVAILABLE";

export interface Field<T> {
  value: T | null;
  provenance: Provenance;
  unit: string;
  reason: string;
}

export interface DetectedEntity {
  class_id: number;
  class_name: string;
  confidence: number;
  box: [number, number, number, number]; // [x, y, width, height]
  width_px: number;
  height_px: number;
  mask?: [number, number][]; // [[x, y], ...]
  track_id?: number | null;
}

export interface CameraInfo {
  status: string; // not started | connecting | live | reconnecting | unavailable | stopped
  source: string; // usb | rtsp
  fps: number;
  resolution: [number, number] | number[];
  error: string | null;
  provenance: Provenance;
  people_count?: number | null;
  fire_count?: number | null;
  detections?: DetectedEntity[];
}

export interface Production {
  job_name: Field<string>;
  total_mm: Field<number>;
  completed_mm: Field<number>;
  remaining_mm: Field<number>;
  percentage: Field<number>;
  current_layer: Field<number>;
  total_layers: Field<number>;
}

export interface VisionStatus {
  printing: Field<boolean>;
  print_status: Field<string>;
  layer_number: Field<number>;
  layer_width_mm: Field<number>;
  layer_width_px: Field<number>;
  confidence: Field<number>;
  motion_score: Field<number>;
  note: string | null;
}

export interface MaterialStatus {
  material_flow_percent: Field<number>;
  nozzle_temp_c: Field<number>;
  bed_temp_c: Field<number>;
  machine_state: Field<string>;
  source_note: string;
}

export interface MachineTelemetry {
  pump_running: Field<boolean>;
  pump_speed_rpm: Field<number>;
  accelerator_running: Field<boolean>;
  accelerator_speed_rpm: Field<number>;
  print_speed_mm_s: Field<number>;
  pressure_bar: Field<number>;
}

export interface AlertItem {
  level: string;
  message: string;
  seconds_ago: number;
}

export interface StatusResponse {
  timestamp: number;
  camera: CameraInfo;
  camera2?: CameraInfo;
  people_count?: number | null;
  production: ProductionFull;
  vision: VisionStatus;
  geometry: Geometry;
  material: MaterialFull;
  machine_telemetry: MachineTelemetry;
  robot: RobotInfo;
  problems: Problems;
  alerts: AlertItem[];
  calibrated: boolean;
}

export interface HealthResponse {
  ok: boolean;
  camera_source: string;
  camera_status: string;
  llm: string;
  llm_model: string | null;
  stt: string;
  tts?: string;
  rag_chunks: number;
  rag_mode: string;
  calibrated: boolean;
  read_only: boolean;
}

export interface AttachmentInfo {
  filename: string;
  content_type: string;
  size_bytes: number;
  kind: "image" | "document" | "text";
  observations?: string | null;
  width?: number | null;
  height?: number | null;
  previewUrl?: string;
}

export interface AskResponse {
  question: string;
  answer: string;
  tool: string | null;
  intent: string | null;
  facts: Record<string, unknown> | null;
  knowledge_sources: string[] | null;
  data_available: boolean;
  llm: string;
  timestamp: number;
  attendance_updated?: boolean;
  attachments?: AttachmentInfo[] | null;
  fallback_used?: boolean;
  rag_score?: number | null;
  primary_image_id?: string | null;
  annotated_image_id?: string | null;
  reference_image_id?: string | null;
  reference_component?: Record<string, unknown> | null;
}

/* --- extended dashboard sections --- */

export interface Timing {
  elapsed: Field<string>;
  elapsed_seconds: Field<number>;
  print_rate_mm_per_min: Field<number>;
  eta: Field<string>;
  eta_timestamp: Field<number>;
  deadline: Field<number>;
  deadline_status: Field<string>;
  time_to_deadline?: Field<string>;
}

export type ProductionFull = Production & Timing;

export interface Geometry {
  wall_height_mm: Field<number>;
  wall_width_mm: Field<number>;
  layer_width_mm: Field<number>;
  layer_height_mm: Field<number>;
  layer_width_px: Field<number>;
  layer_height_px: Field<number>;
  current_layer: Field<number>;
  total_layers: Field<number>;
}

export interface MaterialFull extends MaterialStatus {
  material_type: Field<string>;
  material_used_kg: Field<number>;
  material_remaining_kg: Field<number>;
  material_ratio_percent: Field<number>;
  consumption_rate_kg_per_hour: Field<number>;
}

export interface Communication {
  api: Field<string>;
  camera: Field<string>;
  machine_link: Field<string>;
}

export interface RobotInfo {
  robot_online: Field<boolean>;
  robot_status: Field<string>;
  machine_state: Field<string>;
  current_operation: Field<string>;
  current_shift: Field<string>;
  current_operator: Field<string>;
  communication: Communication;
}

export interface Problems {
  warnings: AlertItem[];
  critical: AlertItem[];
  machine: AlertItem[];
  camera: AlertItem[];
  counts: { warnings: number; critical: number; machine: number; camera: number };
}

export interface HistoryPoint {
  t: number;
  completed_mm: number | null;
  layer_width_mm: number | null;
  printing: boolean | null;
}

export interface HistoryResponse {
  windows: Record<string, Field<number>>;
  series: HistoryPoint[];
  series_provenance: Provenance;
  series_note: string;
}

/** One perpendicular measurement station along the bead centreline (W1..Wn). */
export interface MetrologyStation {
  label: string;
  x: number;
  y: number;
  width_px: number | null;
  width_mm: number | null;
  valid: boolean;
  reason: string;
}

/**
 * Perpendicular layer-width measurement, published on /api/status.
 *
 * The overlay itself is drawn server-side and arrives as pixels in the MJPEG
 * stream; this is the same result as numbers, for panels and for the voice
 * announcement, so the two can never disagree.
 */
export interface MetrologyResult {
  mask_method: string;
  contour_ok: boolean;
  centerline_ok: boolean;
  stations_total: number;
  stations_valid: number;
  stations: MetrologyStation[];
  average_width_mm: number | null;
  deviation_mm: number | null;
  target_width_mm: number;
  tolerance_mm: number;
  range_min_mm: number;
  range_max_mm: number;
  status: string;
  reason: string;
  pixels_per_mm: number;
  calibrated: boolean;
  timestamp: number;
}

/** A layer-width result the assistant should speak. `id` increments per
 *  announcement, so the UI can tell a new one from a repeat. */
export interface MetrologyAnnouncement {
  id: number;
  text: string;
  status: string;
  average_width_mm: number | null;
  timestamp: number;
}

export interface MetrologyAnnouncementResponse {
  announcement: MetrologyAnnouncement | null;
  enabled: boolean;
}

// --- Attendance ---

export interface AttendanceRecord {
  id: number;
  operator_id?: string;
  employee_name: string;
  attendance_date: string;
  shift_start: number;
  shift_end: number | null;
  status: "ACTIVE" | "COMPLETED" | "CANCELLED";
  created_at: number;
  updated_at: number;
  worked_seconds?: number;
  worked_hours?: string;
  worked_display?: string;
  overtime_seconds?: number;
  overtime_display?: string;
  overtime_spoken?: string;
  server_now?: number;
}

// --- Employee Wall Production Tracking ---

export interface WallProductionRecord {
  record_id: string;
  operator_id: number;
  operator_name: string;
  employee_code?: string;
  quantity: number;
  job_id?: string;
  shift_id?: number;
  notes?: string;
  is_corrected: number;
  created_at: number;
  updated_at: number;
  recorded_by?: string;
}

export interface WallProductionOperatorSummary {
  operator_id: number;
  operator_name: string;
  employee_code?: string;
  total_quantity: number;
  record_count: number;
}

export interface WallProductionTotal {
  success: boolean;
  scope: string;
  total_quantity: number;
  record_count: number;
  by_operator: WallProductionOperatorSummary[];
  records?: WallProductionRecord[];
}
