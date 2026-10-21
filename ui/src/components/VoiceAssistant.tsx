import "../audio/sharedContext";
/**
 * Kelvin6K Voice Assistant -- hands-free, Alexa-style.
 *
 * The operator presses nothing. The loop runs forever:
 *
 *   IDLE (waiting for "Okay Kelvin")
 *     -> WAKE      robot says "Yes, what would you like to know?"
 *     -> LISTENING records until ~1.8 s of silence follows speech
 *     -> THINKING  faster-whisper -> query router -> database/RAG -> Claude
 *     -> SPEAKING  answer through the backend's speaker
 *     -> IDLE
 *
 * Every failure path also returns to IDLE, out loud, so the assistant can
 * never end up permanently deaf.
 *
 * The manual buttons still exist, under Developer / Diagnostics, because they
 * are how the microphone gets debugged when something is wrong.
 *
 * READ-ONLY: this panel only asks questions; it never commands the machine.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  API_BASE,
  ask,
  getHealth,
  speakOnServer,
  resolveApiUrl,
  downloadIncidentImage,
} from "../api";
import type { HealthResponse, MetrologyAnnouncementResponse } from "../types";
import { KelvinCore3D, type CoreState } from "./KelvinCore3D";
import { IndependentMicTest } from "./IndependentMicTest";
import { MicDiagnostic } from "./MicDiagnostic";
import { useAudioPlayer } from "../hooks/useAudioPlayer";
import { useRecorder } from "../hooks/useRecorder";
import { VoiceLevel } from "./VoiceLevel";
import { useVoice, WAKE_WORDS } from "../hooks/useVoice";
import "./VoiceAssistant.css";

interface ExchangeAttachment {
  filename: string;
  kind: "image" | "document" | "text";
  size_bytes: number;
  previewUrl?: string;
}

interface Exchange {
  question: string;
  answer: string;
  intent: string | null;
  dataAvailable: boolean;
  at: number;
  attachments?: ExchangeAttachment[];
  llm?: string;
  fallbackUsed?: boolean;
  primaryImageId?: string | null;
  annotatedImageId?: string | null;
  referenceImageId?: string | null;
  referenceComponent?: Record<string, unknown> | null;
}

export interface PreviewModalData {
  imageUrl: string;
  title: string;
  subtitle?: string;
  isAnnotation?: boolean;
  isReference?: boolean;
  primaryImageId?: string | null;
  annotatedImageId?: string | null;
  alternativeUrl?: string;
  alternativeLabel?: string;
}

function IncidentImageCard({
  src,
  alt,
  primaryImageId,
  annotatedImageId,
  onPreviewImage,
}: {
  src: string;
  alt: string;
  primaryImageId?: string | null;
  annotatedImageId?: string | null;
  onPreviewImage?: (data: PreviewModalData) => void;
}) {
  const lowerAlt = alt.toLowerCase();
  const lowerSrc = src.toLowerCase();

  const isRobotRef =
    lowerAlt.includes("robot reference") ||
    lowerAlt.includes("k6k-ref") ||
    lowerSrc.includes("k6k-ref") ||
    lowerAlt.includes("rover configuration") ||
    lowerAlt.includes("feed tube routing") ||
    lowerAlt.includes("printhead mounted") ||
    lowerAlt.includes("linear axis setup");

  const isSpec =
    lowerAlt.includes("specification") ||
    lowerAlt.includes("technical spec") ||
    lowerAlt.includes("atom-spec") ||
    lowerSrc.includes("atom-spec") ||
    lowerAlt.includes("atom table") ||
    lowerAlt.includes("safety table");

  const isComponentRef =
    !isRobotRef &&
    !isSpec &&
    (lowerAlt.includes("component reference") ||
      lowerAlt.includes("reference image") ||
      lowerAlt.includes("reference photograph") ||
      lowerAlt.includes("kelvin6k reference") ||
      lowerSrc.includes("ref-") ||
      lowerSrc.includes("spec-") ||
      lowerSrc.includes("k6k-ref") ||
      lowerAlt.includes("reference"));

  const isReference = isRobotRef || isSpec || isComponentRef;

  // STRICT annotation check:
  // 1. Must NOT be a reference image of any kind.
  // 2. Must match whole word 'ai' or 'annotat' or 'observation overlay'.
  const isAnnotation =
    !isReference &&
    (/\b(ai|annotated|annotation|observations?)\b/i.test(alt) || lowerAlt.includes("observation overlay"));

  // Only allow toggling between annotated overlay and original raw photo when:
  // 1. This is NOT a reference image.
  // 2. Both annotatedImageId and primaryImageId exist.
  // 3. This card corresponds to the primary/annotated incident image (or src matches either ID).
  const matchesIncidentDefect =
    !isReference &&
    Boolean(
      (primaryImageId && src.includes(primaryImageId)) ||
      (annotatedImageId && src.includes(annotatedImageId)) ||
      (!lowerSrc.includes("ref") && !lowerSrc.includes("spec") && (primaryImageId || annotatedImageId))
    );

  const hasBoth = matchesIncidentDefect && Boolean(annotatedImageId && primaryImageId);
  const [showOriginal, setShowOriginal] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  const activeSrc = showOriginal && primaryImageId ? `/api/incidents/images/${primaryImageId}` : src;
  const activeAlt = showOriginal ? "Original Operator Upload" : alt;
  const activeIsAnnotation = !showOriginal && isAnnotation;

  const resolvedActiveSrc = resolveApiUrl(activeSrc);

  useEffect(() => {
    setHasError(false);
  }, [showOriginal, activeSrc]);

  const handleOpenPreview = () => {
    if (!onPreviewImage) return;
    onPreviewImage({
      imageUrl: resolvedActiveSrc,
      title: activeIsAnnotation
        ? "AI Defect Annotation Overlay"
        : isRobotRef
        ? "Robot Reference Photograph"
        : isSpec
        ? "Technical Specification Source"
        : isComponentRef
        ? "Component Technical Reference"
        : "Site Incident Image",
      subtitle: activeIsAnnotation
        ? "AI-assisted visual observation • Not a confirmed root-cause diagnosis"
        : isRobotRef
        ? "Reference photograph of approved robot configuration • For visual comparison only"
        : isSpec
        ? "Verified manufacturer specification record"
        : isComponentRef
        ? "Verified component reference • For inspection comparison"
        : activeAlt,
      isAnnotation: activeIsAnnotation,
      isReference,
      primaryImageId: hasBoth ? primaryImageId : undefined,
      annotatedImageId: hasBoth ? annotatedImageId : undefined,
      alternativeUrl: hasBoth
        ? resolveApiUrl(
            showOriginal
              ? `/api/incidents/images/${annotatedImageId}`
              : `/api/incidents/images/${primaryImageId}`
          )
        : undefined,
      alternativeLabel: hasBoth
        ? showOriginal
          ? "Switch to AI Annotation"
          : "Switch to Original Photo"
        : undefined,
    });
  };

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation();
    const fallbackName = activeIsAnnotation
      ? `kelvin6k_annotation_${annotatedImageId || "defect"}.jpg`
      : isRobotRef
      ? `kelvin6k_robot_ref_${activeSrc.split("/").pop()}.jpg`
      : isSpec
      ? `kelvin6k_spec_${activeSrc.split("/").pop()}.png`
      : isComponentRef
      ? `kelvin6k_component_${activeSrc.split("/").pop()}.jpg`
      : `kelvin6k_defect_${primaryImageId || "upload"}.jpg`;
    downloadIncidentImage(activeSrc, fallbackName);
  };

  let badgeLabel = "📷 SITE INCIDENT IMAGE";
  let badgeColor = "#6aa8ff";
  let badgeBg = "rgba(106, 168, 255, 0.15)";
  let badgeBorder = "rgba(106, 168, 255, 0.3)";

  if (activeIsAnnotation) {
    badgeLabel = "👁️ AI DEFECT ANNOTATION";
    badgeColor = "#ff9800";
    badgeBg = "rgba(255, 140, 0, 0.15)";
    badgeBorder = "rgba(255, 140, 0, 0.3)";
  } else if (isRobotRef) {
    badgeLabel = "🤖 ROBOT REFERENCE PHOTOGRAPH";
    badgeColor = "#38bdf8";
    badgeBg = "rgba(56, 189, 248, 0.15)";
    badgeBorder = "rgba(56, 189, 248, 0.3)";
  } else if (isSpec) {
    badgeLabel = "📐 TECHNICAL SPECIFICATION";
    badgeColor = "#a78bfa";
    badgeBg = "rgba(167, 139, 250, 0.15)";
    badgeBorder = "rgba(167, 139, 250, 0.3)";
  } else if (isComponentRef) {
    badgeLabel = "⚙️ COMPONENT TECHNICAL REFERENCE";
    badgeColor = "#4ade80";
    badgeBg = "rgba(74, 222, 128, 0.15)";
    badgeBorder = "rgba(74, 222, 128, 0.3)";
  } else if (showOriginal || lowerAlt.includes("original") || lowerAlt.includes("upload")) {
    badgeLabel = "📷 ORIGINAL PHOTOGRAPH";
    badgeColor = "#6aa8ff";
    badgeBg = "rgba(106, 168, 255, 0.15)";
    badgeBorder = "rgba(106, 168, 255, 0.3)";
  }

  return (
    <div
      style={{
        margin: "12px 0",
        padding: "10px",
        background: "#161b22",
        borderRadius: "8px",
        border: activeIsAnnotation
          ? "1px solid rgba(255, 140, 0, 0.45)"
          : isRobotRef
          ? "1px solid rgba(56, 189, 248, 0.45)"
          : isSpec
          ? "1px solid rgba(167, 139, 250, 0.45)"
          : isComponentRef
          ? "1px solid rgba(74, 222, 128, 0.35)"
          : "1px solid #2e3846",
        boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "8px",
          flexWrap: "wrap",
          gap: "8px",
        }}
      >
        <span
          style={{
            fontSize: "10px",
            fontWeight: 700,
            letterSpacing: "0.5px",
            padding: "3px 8px",
            borderRadius: "4px",
            background: badgeBg,
            color: badgeColor,
            border: `1px solid ${badgeBorder}`,
          }}
        >
          {badgeLabel}
        </span>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          {hasBoth && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setShowOriginal(!showOriginal);
              }}
              style={{
                fontSize: "11px",
                padding: "3px 8px",
                borderRadius: "4px",
                border: "1px solid #3b4758",
                background: "#212836",
                color: "#e6ecf3",
                cursor: "pointer",
                fontWeight: 500,
              }}
              title={showOriginal ? "Switch to AI annotation view" : "Switch to original uploaded photo"}
            >
              {showOriginal ? "👁️ View Annotation" : "📷 View Original"}
            </button>
          )}

          <button
            type="button"
            onClick={handleOpenPreview}
            style={{
              fontSize: "11px",
              padding: "3px 8px",
              borderRadius: "4px",
              border: "1px solid #3b4758",
              background: "#1c2433",
              color: "#6aa8ff",
              cursor: "pointer",
              fontWeight: 500,
            }}
            title="Open full resolution in preview modal"
          >
            🔍 Preview
          </button>

          <button
            type="button"
            onClick={handleDownload}
            style={{
              fontSize: "11px",
              padding: "3px 8px",
              borderRadius: "4px",
              border: "1px solid rgba(74, 222, 128, 0.3)",
              background: "rgba(74, 222, 128, 0.12)",
              color: "#4ade80",
              cursor: "pointer",
              fontWeight: 600,
            }}
            title="Download actual image file"
          >
            📥 Download
          </button>

          <a
            href={resolvedActiveSrc}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: "11px",
              color: "#8a99ad",
              textDecoration: "none",
              padding: "3px 6px",
            }}
            title="Open raw image in new browser tab"
          >
            ↗
          </a>
        </div>
      </div>

      {hasError ? (
        <div
          style={{
            textAlign: "center",
            padding: "20px 14px",
            background: "rgba(248, 113, 113, 0.08)",
            borderRadius: "6px",
            border: "1px dashed rgba(248, 113, 113, 0.35)",
            color: "#e6ecf3",
          }}
        >
          <div style={{ fontSize: "22px", marginBottom: "4px" }}>⚠️</div>
          <div style={{ fontSize: "12px", fontWeight: 600, color: "#f87171" }}>
            Image Unavailable or Failed to Load
          </div>
          <div style={{ fontSize: "11px", color: "#8a99ad", margin: "4px 0 10px", wordBreak: "break-all" }}>
            {activeSrc}
          </div>
          <div style={{ display: "flex", gap: "8px", justifyContent: "center" }}>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setHasError(false);
                setRetryKey((k) => k + 1);
              }}
              style={{
                fontSize: "11px",
                padding: "4px 10px",
                background: "#212836",
                border: "1px solid #3b4758",
                borderRadius: "4px",
                color: "#6aa8ff",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              🔄 Retry Load
            </button>
            <a
              href={resolvedActiveSrc}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              style={{
                fontSize: "11px",
                padding: "4px 10px",
                background: "#212836",
                border: "1px solid #3b4758",
                borderRadius: "4px",
                color: "#9aa1af",
                textDecoration: "none",
              }}
            >
              Direct Link ↗
            </a>
          </div>
        </div>
      ) : (
        <div
          style={{
            textAlign: "center",
            cursor: "pointer",
            background: "#0d1117",
            padding: "6px",
            borderRadius: "6px",
            border: "1px solid #21262d",
          }}
          onClick={handleOpenPreview}
          title="Click to enlarge image in preview modal"
        >
          <img
            key={retryKey}
            src={resolvedActiveSrc}
            alt={activeAlt}
            onError={() => setHasError(true)}
            style={{
              maxWidth: "100%",
              maxHeight: "340px",
              borderRadius: "4px",
              objectFit: "contain",
              display: "inline-block",
              verticalAlign: "middle",
            }}
          />
        </div>
      )}

      <div
        style={{
          fontSize: "11px",
          color: "#8a99ad",
          marginTop: "6px",
          textAlign: "center",
          fontStyle: "italic",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "0 4px",
        }}
      >
        <span>{activeAlt}</span>
        <span style={{ fontSize: "10px", color: "#54657a" }}>Click image to enlarge • 📥 Download</span>
      </div>
    </div>
  );
}

function renderInlineFormatted(
  text: string,
  onPreviewImage?: (data: PreviewModalData) => void,
  primaryImageId?: string | null,
  annotatedImageId?: string | null,
) {
  const regex = /(\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g;
  const parts = text.split(regex);
  return parts.map((part, pIdx) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={pIdx} style={{ color: "#e6ecf3" }}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      const linkText = linkMatch[1];
      const linkUrl = linkMatch[2];
      const isIncidentImage = linkUrl.includes("/api/incidents/images/");
      const resolvedUrl = resolveApiUrl(linkUrl);

      if (isIncidentImage && onPreviewImage) {
        return (
          <button
            key={pIdx}
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onPreviewImage({
                imageUrl: resolvedUrl,
                title: linkText.includes("Original") ? "Original Uploaded Defect Image" : linkText,
                subtitle: "Direct image view from incident store",
                primaryImageId,
                annotatedImageId,
              });
            }}
            style={{
              background: "transparent",
              border: "none",
              padding: 0,
              color: "#6aa8ff",
              textDecoration: "underline",
              cursor: "pointer",
              fontWeight: 500,
              fontSize: "inherit",
              display: "inline",
            }}
            title="Click to preview image in modal"
          >
            {linkText}
          </button>
        );
      }

      return (
        <a
          key={pIdx}
          href={resolvedUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: "#6aa8ff",
            textDecoration: "underline",
            cursor: "pointer",
            fontWeight: 500,
          }}
        >
          {linkText}
        </a>
      );
    }
    return part;
  });
}

function FormattedMessage({
  text,
  primaryImageId,
  annotatedImageId,
  referenceImageId: _referenceImageId,
  onPreviewImage,
}: {
  text: string;
  primaryImageId?: string | null;
  annotatedImageId?: string | null;
  referenceImageId?: string | null;
  onPreviewImage?: (data: PreviewModalData) => void;
}) {
  const lines = text.split("\n");
  return (
    <div className="va-chat-formatted">
      {lines.map((line, idx) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return <div key={idx} className="va-chat-spacer" />;
        }

        const imgMatch = trimmed.match(/^!\[(.*?)\]\((.*?)\)$/);
        if (imgMatch) {
          const alt = imgMatch[1] || "Incident Image";
          const src = imgMatch[2];

          return (
            <IncidentImageCard
              key={idx}
              src={src}
              alt={alt}
              primaryImageId={primaryImageId}
              annotatedImageId={annotatedImageId}
              onPreviewImage={onPreviewImage}
            />
          );
        }

        if (trimmed.startsWith("#### ")) {
          const headerText = trimmed.replace(/^####\s*/, "");
          return (
            <h5
              key={idx}
              style={{
                margin: "10px 0 4px",
                color: "#4ade80",
                fontSize: "12px",
                fontWeight: 700,
                letterSpacing: "0.4px",
                textTransform: "uppercase",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              <span>⚙️</span>
              <span>{headerText}</span>
            </h5>
          );
        }

        if (trimmed.startsWith("### ") || trimmed.startsWith("## ")) {
          const headerText = trimmed.replace(/^#+\s*/, "");
          return (
            <h4 key={idx} style={{ margin: "8px 0 3px", color: "#6aa8ff", fontSize: "13px", fontWeight: 600 }}>
              {headerText}
            </h4>
          );
        }

        if (trimmed.startsWith("> ")) {
          const quoteText = trimmed.replace(/^>\s*/, "");
          const isWarning = quoteText.includes("Warning") || quoteText.includes("DANGER") || quoteText.includes("Danger");
          return (
            <div
              key={idx}
              style={{
                borderLeft: isWarning ? "3px solid #ff6b6b" : "3px solid #c9a24a",
                background: isWarning ? "rgba(255,107,107,0.1)" : "rgba(201,162,74,0.1)",
                padding: "6px 10px",
                borderRadius: "3px",
                fontSize: "12px",
                color: isWarning ? "#ff9e9e" : "#e6d5a7",
                margin: "4px 0",
              }}
            >
              {renderInlineFormatted(quoteText, onPreviewImage, primaryImageId, annotatedImageId)}
            </div>
          );
        }

        if (trimmed.startsWith("- ") || trimmed.startsWith("* ") || trimmed.startsWith("• ")) {
          const bulletText = line.replace(/^[\s-*•]+/, "");
          return (
            <div key={idx} className="va-chat-bullet-row">
              <span className="va-chat-bullet-dot">•</span>
              <span className="va-chat-bullet-text">
                {renderInlineFormatted(bulletText, onPreviewImage, primaryImageId, annotatedImageId)}
              </span>
            </div>
          );
        }

        return (
          <p key={idx} className="va-chat-p">
            {renderInlineFormatted(line, onPreviewImage, primaryImageId, annotatedImageId)}
          </p>
        );
      })}
    </div>
  );
}

/** Fine-grained voice pipeline display states. */
export type VoiceDisplayState =
  | "WAKE_WORD_LISTENING"
  | "WAITING_FOR_SPEECH"
  | "SPEECH_DETECTED"
  | "SILENCE_WHILE_RECORDING"
  | "PROCESSING"
  | "SPEAKING"
  | "READY";

interface VoiceDisplayInfo {
  title: string;
  subtitle: string;
}

const VOICE_DISPLAY_MAP: Record<VoiceDisplayState, VoiceDisplayInfo> = {
  WAKE_WORD_LISTENING: {
    title: 'Listening for "Okay Kelvin"',
    subtitle: 'Say "Okay Kelvin" or click microphone',
  },
  WAITING_FOR_SPEECH: {
    title: "Listening… Speak now",
    subtitle: "Microphone active — click stop when finished",
  },
  SPEECH_DETECTED: {
    title: "Hearing you…",
    subtitle: "Speaking detected",
  },
  SILENCE_WHILE_RECORDING: {
    title: "Listening…",
    subtitle: "Finishing question if silence continues…",
  },
  PROCESSING: {
    title: "Processing…",
    subtitle: "Transcribing with faster-whisper & querying AI…",
  },
  SPEAKING: {
    title: "Speaking…",
    subtitle: "Playing response…",
  },
  READY: {
    title: "Ready",
    subtitle: "Click the microphone button to speak",
  },
};

/** Spoken when a turn cannot complete. Each one returns the loop to IDLE. */
const NO_SPEECH = "I didn't hear a question. Please try again.";
const STT_FAILED = "Sorry, I couldn't understand that. Please try again.";
const API_FAILED = "Sorry, I'm having trouble processing that right now.";

/** Nothing may hold a stage longer than this; the loop resets if it does. */
const WATCHDOG_MS = 45000;

/**
 * Intents the backend answers with a QUESTION ("…Shall I start your shift on
 * Line 1?") and then holds for a yes/no, with a 30 s expiry.
 *
 * After one of these the operator's reply is a bare "Yes", which the wake-word
 * recogniser discards because it is not "Okay Kelvin" -- so the confirmation
 * could never be given by voice and every voice-started action expired unwritten.
 * Seeing one of these intents opens a single recording turn for the reply.
 *
 * Must match the action intents the backend holds for confirmation
 * (agent/assistant.py, `action_intents`); a new confirmed action added there
 * needs adding here too.
 */
/** Headline shown under the 3D core, one per visual state. */
const CORE_TITLE: Record<CoreState, string> = {
  IDLE: "Ready — Click microphone to speak",
  LISTENING: "Listening… Speak your question",
  HUMAN_SPEAKING: "Hearing you…",
  THINKING: "Thinking…",
  KELVIN_SPEAKING: "Kelvin is speaking…",
  DONE: "Done",
  ERROR: "Something went wrong",
};

/** How long the transient DONE / ERROR looks hold before returning to IDLE. */
const DONE_HOLD_MS = 1400;
const ERROR_HOLD_MS = 1800;

const CONFIRMATION_INTENTS: ReadonlySet<string> = new Set([
  "SHIFT_START",
  "SHIFT_END",
  "LOG_WALL_PRODUCTION",
  "LOG_SHIFT_ISSUE",
  "LOG_SHIFT_HANDOVER",
  "RESOLVE_SHIFT_ISSUE",
]);
/** Upper bound on waiting for an answer to finish speaking. Longer than any
    real answer, so it only ever fires if playback failed to report the end. */
const SPEAK_TIMEOUT_MS = 30000;

/**
 * Developer controls are hidden unless the page is opened with ?dev=1.
 *
 * The operator's screen must have nothing to press. The manual buttons are how
 * the microphone gets debugged when something is wrong, though, so they are
 * hidden rather than deleted -- and they appear on their own if the microphone
 * is actually broken, which is exactly when they are needed.
 */
const DEV_MODE =
  typeof window !== "undefined" && new URLSearchParams(window.location.search).has("dev");

/** Conversation history survives a page reload. */
const HISTORY_KEY = "kelvin6k.conversation";
/** Enough for a shift's questions without growing without bound. */
const HISTORY_LIMIT = 100;
/** How often to check for a new layer-width announcement. The backend does
    the debouncing, so this only has to be often enough to feel immediate. */
const METROLOGY_POLL_MS = 1500;

function loadHistory(): Exchange[] {
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? (JSON.parse(raw) as Exchange[]) : [];
    if (!Array.isArray(parsed)) return [];
    // Stored data from an older build could be missing fields, and a render
    // crash here would take the whole assistant down. Keep only sound rows.
    return parsed.filter(
      (item) =>
        item &&
        typeof item.question === "string" &&
        typeof item.answer === "string" &&
        typeof item.at === "number",
    );
  } catch {
    // Private windows and blocked site data must not break the assistant.
    return [];
  }
}

function saveHistory(history: Exchange[]): void {
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-HISTORY_LIMIT)));
  } catch {
    /* not fatal -- the session simply is not remembered */
  }
}

/** Non-blocking audible chime to acknowledge "Okay Kelvin" immediately. */
function playWakeChime(): void {
  try {
    import("../audio/sharedContext").then(({ getSharedAudioContext, unlockAudioContext }) => {
      const ctx = getSharedAudioContext();
      if (!ctx) return;
      unlockAudioContext();
      if (ctx.state === "suspended") ctx.resume().catch(() => {});
      
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
      osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.12); // A5
      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.15);
    });
  } catch {
    /* AudioContext blocked or unsupported -- non-fatal */
  }
}

export function VoiceAssistant({ autoSpeak: initialAutoSpeak = true }: { autoSpeak?: boolean }) {
  const [history, setHistory] = useState<Exchange[]>(loadHistory);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const submittingRef = useRef(false);
  const [confirming, setConfirming] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [autoSpeak, setAutoSpeak] = useState(initialAutoSpeak);
  const [transcript, setTranscript] = useState("");
  const [serverTts, setServerTts] = useState(true);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [ttsFellBack, setTtsFellBack] = useState(false);
  const [speakingViaServer, setSpeakingViaServer] = useState(false);

  // Lightbox preview modal state
  const [previewModal, setPreviewModal] = useState<PreviewModalData | null>(null);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (previewModal) {
          setPreviewModal(null);
        } else if (isExpanded) {
          setIsExpanded(false);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [previewModal, isExpanded]);

  // File attachments state
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<
    { file: File; previewUrl?: string; kind: "image" | "document" | "text" }[]
  >([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const ALLOWED_EXTENSIONS = new Set([
    "jpg", "jpeg", "png", "webp", "pdf", "docx", "txt", "md", "log", "csv", "json"
  ]);
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const MAX_FILE_COUNT = 4;

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    if (selectedFiles.length + files.length > MAX_FILE_COUNT) {
      setApiError(`Maximum ${MAX_FILE_COUNT} attachments allowed per question.`);
      return;
    }

    const newSelected: File[] = [];
    const newPreviews: { file: File; previewUrl?: string; kind: "image" | "document" | "text" }[] = [];

    for (const f of files) {
      const ext = f.name.split(".").pop()?.toLowerCase() || "";
      if (!ALLOWED_EXTENSIONS.has(ext)) {
        setApiError(`Unsupported file format '.${ext}'. Supported formats: JPG, PNG, WEBP, PDF, DOCX, TXT, MD, LOG, CSV, JSON.`);
        return;
      }
      if (f.size > MAX_FILE_SIZE) {
        setApiError(`File '${f.name}' exceeds the 10MB limit.`);
        return;
      }
      const isImg = ["jpg", "jpeg", "png", "webp"].includes(ext);
      const isDoc = ["pdf", "docx"].includes(ext);
      const kind: "image" | "document" | "text" = isImg ? "image" : isDoc ? "document" : "text";
      const previewUrl = isImg ? URL.createObjectURL(f) : undefined;

      newSelected.push(f);
      newPreviews.push({ file: f, previewUrl, kind });
    }

    setSelectedFiles((prev) => [...prev, ...newSelected]);
    setPreviews((prev) => [...prev, ...newPreviews]);
    setApiError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [selectedFiles]);

  const removeAttachment = useCallback((index: number) => {
    setPreviews((prev) => {
      const item = prev[index];
      if (item?.previewUrl) URL.revokeObjectURL(item.previewUrl);
      return prev.filter((_, i) => i !== index);
    });
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const chatBottomRef = useRef<HTMLDivElement | null>(null);
  const expandedChatBottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const expandedTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const expandedFileInputRef = useRef<HTMLInputElement | null>(null);

  const confirmingRef = useRef(false);
  confirmingRef.current = confirming;
  const autoSpeakRef = useRef(autoSpeak);
  autoSpeakRef.current = autoSpeak;
  const serverTtsRef = useRef(serverTts);
  serverTtsRef.current = serverTts;
  const speakRef = useRef<((text: string) => Promise<void>) | null>(null);
  const speakAnswerTextRef = useRef<((text: string) => Promise<void>) | null>(null);
  const submitRef = useRef<((q: string, via: "text" | "voice") => void) | null>(null);
  const startRecordingRef =
    useRef<((opts?: { autoStopOnSilence?: boolean; isConfirmation?: boolean }) => Promise<boolean>) | null>(null);
  /** True while a wake-word turn is running, so the loop knows to resume. */
  const handsFreeRef = useRef(false);
  /** Mirrors `busy`/recorder activity for callbacks that cannot see state. */
  const busyRef = useRef(false);
  // Highest announcement id already spoken, so a repeat is never re-spoken.
  const lastSpokenAnnouncementRef = useRef(0);

  /** Auto-scroll to bottom of conversation thread as messages arrive or thinking starts */
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    expandedChatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, isThinking]);

  /** Set when the wake word fires, so total turn latency can be reported. */
  const turnStartRef = useRef(0);
  /** Counts completed hands-free turns, to make repeat behaviour visible. */
  const turnCountRef = useRef(0);

  const recorder = useRecorder({
    // Silence detection ends the recording, so the transcript arrives through
    // this callback rather than a promise nobody is awaiting.
    onTranscript: (text) => {
      setTranscript(text);
      submitRef.current?.(text, "voice");
    },
  });
  const player = useAudioPlayer();

  /** Sends a question to the backend -- identical path for text and voice. Supports attachments. */
  const submit = useCallback(async (question: string, via: "text" | "voice", filesToSend?: File[]) => {
    const currentRequestId = ++requestIdRef.current;
    // Abort any previous request (should already be cleared on new chat)
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    const trimmed = question.trim();
    const hasFiles = filesToSend && filesToSend.length > 0;
    if (!trimmed && !hasFiles) {
      if (handsFreeRef.current) {
        handsFreeRef.current = false;
        setConfirming(false);
        confirmingRef.current = false;
        setBusy(false);
        voiceRef.current?.resumeWakeWord();
      }
      return;
    }

    // Set once the answer is known to be a question awaiting yes/no. Read in
    // the finally, AFTER the prompt has finished speaking, so the recording for
    // the reply cannot capture the tail of Kelvin's own voice.
    let awaitingConfirmation = false;

    submittingRef.current = true;
    setIsThinking(true);
    setBusy(true);
    setConfirming(false);
    confirmingRef.current = false;
    setApiError(null);
    try {
      const askedAt = Date.now();
      const response = await ask(trimmed, filesToSend, false, signal);

      // If a newer request started, discard this stale response
      if (requestIdRef.current !== currentRequestId) {
        console.warn('[ASSISTANT] Stale response ignored (newer request started)');
        return;
      }

      const answer = response.answer;

      // Broadcast attendance update so attendance table refreshes without page reload
      if (response.attendance_updated) {
        window.dispatchEvent(new CustomEvent("kelvin-attendance-updated"));
      }
      // Voice only: a request typed into the developer box is answered by
      // typing, and opening the microphone for it would be a surprise.
      awaitingConfirmation =
        via === "voice" && CONFIRMATION_INTENTS.has(response.intent ?? "");
      console.info(`[TIMING] Query + answer : ${((Date.now() - askedAt) / 1000).toFixed(2)}s`);
      console.info("[ASSISTANT] Answer:", answer);

      // Build attachments record for history display
      const exchangeAttachments: ExchangeAttachment[] = (response.attachments || []).map((att) => ({
        filename: att.filename,
        kind: att.kind,
        size_bytes: att.size_bytes,
        previewUrl: previews.find((p) => p.file.name === att.filename)?.previewUrl,
      }));

      // Recording the exchange is isolated in its own try/catch: writing to
      // localStorage can fail (private window, blocked site data, quota), and
      // if that ever threw here it would skip the await below and abandon the
      // turn. History is a convenience; answering is the job.
      try {
        // "Okay Kelvin" alone is the wake phrase, not a question. The router
        // labels it WAKE_WORD, so it is acknowledged but never recorded.
        if (response.intent !== "WAKE_WORD") {
          const displayQuestion = trimmed || (exchangeAttachments.length > 0 ? `[Attached: ${exchangeAttachments.map((a) => a.filename).join(", ")}]` : "Question");
          setHistory((prev) =>
            [
              ...prev,
              {
                question: displayQuestion,
                answer,
                intent: response.intent,
                dataAvailable: response.data_available,
                at: Date.now(),
                attachments: exchangeAttachments.length > 0 ? exchangeAttachments : undefined,
                llm: response.llm,
                fallbackUsed: response.fallback_used,
                primaryImageId: response.primary_image_id,
                annotatedImageId: response.annotated_image_id,
                referenceImageId: response.reference_image_id,
                referenceComponent: response.reference_component,
              },
            ].slice(-HISTORY_LIMIT)
          );
        }
      } catch (historyError) {
        console.warn("[ASSISTANT] Could not record the exchange:", historyError);
      }

      // Immediately clear isThinking and submittingRef now that answer is appended to history!
      setIsThinking(false);
      submittingRef.current = false;

      // SPEAK AFTER RECORDING / RENDERING.
      const speaking = autoSpeakRef.current ? speakRef.current?.(answer) : undefined;

      const spokeAt = Date.now();
      await Promise.race([
        speaking,
        new Promise((resolve) => window.setTimeout(resolve, SPEAK_TIMEOUT_MS)),
      ]);
      console.info(`[TIMING] TTS spoken     : ${((Date.now() - spokeAt) / 1000).toFixed(2)}s`);
      if (turnStartRef.current) {
        console.info(
          `[TIMING] TOTAL TURN     : ${((Date.now() - turnStartRef.current) / 1000).toFixed(2)}s ` +
            `(wake word to end of answer)`,
        );
        turnStartRef.current = 0;
      }
    } catch (error) {
      // If this request was aborted, ignore error
      if (error.name === "AbortError") {
        console.info('[ASSISTANT] Request aborted');
        return;
      }
      setIsThinking(false);
      submittingRef.current = false;
      setApiError(error instanceof Error ? error.message : String(error));
      if (handsFreeRef.current) await speakRef.current?.(API_FAILED);
    } finally {
      // Ensure cleanup only for this request if it's still current
      if (requestIdRef.current === currentRequestId) {
        setIsThinking(false);
        submittingRef.current = false;
        if (awaitingConfirmation) {
          setConfirming(true);
          confirmingRef.current = true;
          handsFreeRef.current = true;
          voiceRef.current?.pauseWakeWord();
          console.info("[ASSISTANT] Confirmation requested — listening for the reply");
          let started = false;
          try {
            playWakeChime();
            await voiceRef.current?.waitForStop();
            started =
              (await startRecordingRef.current?.({
                autoStopOnSilence: true,
                isConfirmation: true,
              })) ?? false;
          } finally {
            setBusy(false);
            if (!started) {
              setConfirming(false);
              confirmingRef.current = false;
              handsFreeRef.current = false;
              voiceRef.current?.resumeWakeWord();
            }
          }
        } else {
          setBusy(false);
          setConfirming(false);
          confirmingRef.current = false;
        }
        // Cleanup abort controller for completed request
        abortControllerRef.current = null;
      }
    }
  }, []);
  submitRef.current = (q, via) => void submit(q, via);

  const voice = useVoice({
    onQuestion: (q) => {
      setTranscript(q);
      void submit(q, "voice");
    },
    onWakeWord: (question: string) => {
      // A late result can arrive just after the wake word was paused. Without
      // this guard it would start a second turn on top of the first, and the
      // two would fight over the microphone.
      if (handsFreeRef.current || busyRef.current) {
        console.info("[WAKE] ignored — a turn is already in progress");
        return;
      }
      setTranscript("");
      setApiError(null);
      // The question arrived with the wake phrase, so useVoice answers it.
      if (question) {
        handsFreeRef.current = true;
        voiceRef.current?.pauseWakeWord();
        return;
      }

      // A bare "Okay Kelvin": run the whole turn hands-free.
      void (async () => {
        turnCountRef.current += 1;
        turnStartRef.current = Date.now();
        console.info(
          `%c[WAKE] "Okay Kelvin" DETECTED — starting turn #${turnCountRef.current}`,
          "color:#4fd18b;font-weight:bold",
        );
        handsFreeRef.current = true;
        let started = false;
        try {
          voiceRef.current?.pauseWakeWord();
          playWakeChime();
          await speakAnswerTextRef.current?.("Hi there, what can I help you with?");
          // Silence detection ON: nobody is going to press STOP.
          started = (await startRecordingRef.current?.({ autoStopOnSilence: true })) ?? false;
        } finally {
          if (!started) {
            handsFreeRef.current = false;
            voiceRef.current?.resumeWakeWord();
          }
        }
      })();
    },
  });
  const voiceRef = useRef<typeof voice | null>(null);
  voiceRef.current = voice;

  /**
   * Speaks text and resolves only when speech has FINISHED.
   *
   * The backend's speak() blocks until playback completes, so a resolved
   * request is a true "stopped talking" signal -- which the hands-free loop
   * needs in order not to record over its own voice.
   */
  const speakAnswerText = useCallback(
    async (text: string): Promise<void> => {
      if (serverTtsRef.current) {
        setSpeakingViaServer(true);
        try {
          await speakOnServer(text);
        } catch (err) {
          setApiError(
            `Backend TTS failed: ${err instanceof Error ? err.message : String(err)} — trying browser audio.`,
          );
          setTtsFellBack(true);
          await player.speak(text);
        } finally {
          setSpeakingViaServer(false);
        }
        return;
      }
      const ok = await player.speak(text);
      if (!ok) {
        setTtsFellBack(true);
        voice.speak(text);
      }
    },
    [player, voice],
  );
  speakRef.current = speakAnswerText;
  speakAnswerTextRef.current = speakAnswerText;

  /**
   * Automatic layer-width announcements from the camera metrology.
   *
   * The backend decides WHETHER something is worth saying (status change,
   * stability window, cooldown) and WHAT to say; this only delivers it, through
   * the SAME speakAnswerText() the assistant already uses. That keeps one voice
   * system, one TTS engine and one SPEAKING state rather than a second
   * assistant talking over the first.
   *
   * Held back while the assistant is busy or mid-conversation: interrupting an
   * operator's own question is worse than a late warning, and in hands-free
   * mode the recorder would capture the announcement as the next question.
   */
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      if (cancelled || busyRef.current || handsFreeRef.current) return;
      if (!autoSpeakRef.current) return;
      try {
        const response = await fetch(`${API_BASE}/api/metrology/announcement`);
        if (!response.ok) return;
        const data = (await response.json()) as MetrologyAnnouncementResponse;
        const announcement = data.announcement;
        if (!announcement || cancelled) return;
        if (announcement.id <= lastSpokenAnnouncementRef.current) return;
        lastSpokenAnnouncementRef.current = announcement.id;
        console.info("[METROLOGY] speaking:", announcement.text);
        // Deaf while speaking, exactly as the wake acknowledgement is. Without
        // this the robot can hear its own alert -- "...Average width is 45.7
        // millimeters" -- and a mis-recognised fragment could trip the wake
        // word, waking the assistant from its own announcement.
        voiceRef.current?.pauseWakeWord();
        try {
          await speakAnswerTextRef.current?.(announcement.text);
        } finally {
          // Only hand the microphone back if the operator has not started a
          // turn in the meantime; resuming mid-question would fight the
          // recorder for the device.
          if (
            !handsFreeRef.current &&
            !confirmingRef.current &&
            !busy &&
            recorder.status !== "recording" &&
            recorder.status !== "transcribing"
          ) {
            voiceRef.current?.resumeWakeWord();
          }
        }
      } catch {
        /* the camera or backend being unreachable must not disturb the
           assistant; the next poll will pick it up */
      }
    };
    const timer = window.setInterval(() => void poll(), METROLOGY_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const startRecording = useCallback(
    async (opts?: { autoStopOnSilence?: boolean; isConfirmation?: boolean }) => {
      setTranscript("");
      setApiError(null);
      return recorder.start(opts);
    },
    [recorder],
  );
  startRecordingRef.current = startRecording;


  useEffect(() => {
    saveHistory(history);
  }, [history]);

  useEffect(() => {
    void getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  // --- failure recovery -------------------------------------------------
  // If a recording captured nothing, no transcript arrives and submit() never
  // runs. Without this the wake word would stay paused and the assistant would
  // go permanently deaf after one failed question.
  useEffect(() => {
    if (!handsFreeRef.current) return;
    if (recorder.status !== "error" && recorder.status !== "denied") return;
    handsFreeRef.current = false;
    setConfirming(false);
    confirmingRef.current = false;
    const text = recorder.error ?? "";
    const nothingSaid = /silen|no sound|too short|didn't hear|no speech|not recognised/i.test(text);
    void (async () => {
      try {
        await speakAnswerTextRef.current?.(nothingSaid ? NO_SPEECH : STT_FAILED);
      } finally {
        voiceRef.current?.resumeWakeWord();
      }
    })();
  }, [recorder.status, recorder.error]);

  // --- derived display state -------------------------------------------
  const listening = recorder.status === "recording";
  const transcribing = recorder.status === "transcribing";
  const speaking = player.speaking || voice.speaking || speakingViaServer;
  const active = listening || transcribing || busy || speaking || confirming;

  // Watchdog: the assistant must never be stuck. If a stage outlives this
  // timer the turn is abandoned and wake-word listening resumes.
  useEffect(() => {
    if (!active) return;
    const timer = window.setTimeout(() => {
      console.warn("[ASSISTANT] Watchdog fired — resetting the conversation loop");
      recorder.cancel();
      setBusy(false);
      setConfirming(false);
      confirmingRef.current = false;
      handsFreeRef.current = false;
      voiceRef.current?.resumeWakeWord();
    }, WATCHDOG_MS);
    return () => window.clearTimeout(timer);
  }, [active, recorder]);


  // Pure derivation of the 7 voice pipeline display states
  const deriveDisplayState = (): VoiceDisplayState => {
    if (speaking) return "SPEAKING";
    if (transcribing || busy) return "PROCESSING";
    if (listening) {
      if (recorder.phase === "speaking") return "SPEECH_DETECTED";
      if (recorder.phase === "trailing") return "SILENCE_WHILE_RECORDING";
      return "WAITING_FOR_SPEECH";
    }
    if (confirming) return "WAITING_FOR_SPEECH";
    if (voice.wakeWordEnabled && voice.recognitionRunning) return "WAKE_WORD_LISTENING";
    return "READY";
  };

  const displayState = deriveDisplayState();
  const displayInfo = VOICE_DISPLAY_MAP[displayState];

  // --- 3D core: transient DONE / ERROR looks -----------------------------
  // Everything else about the core is derived from live state below. These
  // two are moments, not states: DONE when Kelvin finishes speaking an answer
  // that did not fail, ERROR when the recorder or the API reports a failure.
  // Both hold briefly and then fall back to IDLE on their own.
  const [coreFlash, setCoreFlash] = useState<"DONE" | "ERROR" | null>(null);
  const coreFlashTimerRef = useRef<number | null>(null);
  const turnErroredRef = useRef(false);
  const wasSpeakingRef = useRef(false);

  useEffect(() => {
    const flash = (kind: "DONE" | "ERROR", holdMs: number) => {
      if (coreFlashTimerRef.current) window.clearTimeout(coreFlashTimerRef.current);
      setCoreFlash(kind);
      coreFlashTimerRef.current = window.setTimeout(() => {
        setCoreFlash(null);
        coreFlashTimerRef.current = null;
      }, holdMs);
    };

    if (recorder.status === "error" || apiError) {
      turnErroredRef.current = true;
      flash("ERROR", ERROR_HOLD_MS);
    }
    // Falling edge of speech: the answer has been delivered. A turn that
    // already failed (the no-speech message, an API error) ends without a
    // celebration.
    if (wasSpeakingRef.current && !speaking) {
      if (!turnErroredRef.current && !confirmingRef.current) flash("DONE", DONE_HOLD_MS);
      turnErroredRef.current = false;
    }
    wasSpeakingRef.current = speaking;
  }, [speaking, recorder.status, apiError]);

  useEffect(
    () => () => {
      if (coreFlashTimerRef.current) window.clearTimeout(coreFlashTimerRef.current);
    },
    [],
  );

  // Request isolation: unique ID per request and abort controller to cancel
  // in-flight queries when New Chat is clicked.
  const requestIdRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Real microphone level, normalised so the recorder's own speech threshold
  // lands at about a third of full scale. Zero whenever nothing is recording,
  // so the animation can never react to anything but the operator's voice.
  const coreLevel =
    listening || confirming
      ? Math.min(1, recorder.level / Math.max(0.02, recorder.threshold * 3))
      : 0;

  // Live state wins over the transient DONE look; an ERROR shows immediately.
  const coreState: CoreState =
    coreFlash === "ERROR"
      ? "ERROR"
      : speaking
        ? "KELVIN_SPEAKING"
        : transcribing || busy
          ? "THINKING"
          : (listening || confirming) && recorder.phase === "speaking"
            ? "HUMAN_SPEAKING"
            : listening || confirming
              ? "LISTENING"
              : coreFlash === "DONE"
                ? "DONE"
                : "IDLE";

  // "Ready — Say 'Okay Kelvin'" only when the wake word is genuinely running;
  // otherwise keep the existing, more specific wording rather than promise a
  // wake word that is not listening.
  const wakeLive = voice.wakeWordEnabled && voice.recognitionRunning;
  const coreTitle = confirming
    ? recorder.phase === "speaking"
      ? "Hearing confirmation…"
      : "Listening for confirmation…"
    : coreState === "IDLE" && !wakeLive
      ? displayInfo.title
      : CORE_TITLE[coreState];

  // A blocked or missing microphone is unrecoverable without the controls, so
  // reveal them by themselves in that case.
  const micBroken =
    !recorder.supported || recorder.status === "denied" || !voice.supported;
  const showDeveloper = DEV_MODE || micBroken;

  busyRef.current = listening || transcribing || busy || speaking || confirming;

  const latest = history[history.length - 1];

  const clock = (ms: number) => {
    const total = Math.floor(ms / 1000);
    return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
  };
  /**
   * Submits the current composer input (text + files) as a new chat turn.
   * Called by the Send button, the Enter key handler, and the form onSubmit.
   */
  const handleChatSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      const trimmed = input.trim();
      const files = selectedFiles.length > 0 ? [...selectedFiles] : undefined;
      if (!trimmed && !files) return;
      setInput("");
      setSelectedFiles([]);
      setPreviews((prev) => {
        prev.forEach((p) => { if (p.previewUrl) URL.revokeObjectURL(p.previewUrl); });
        return [];
      });
      void submit(trimmed, "text", files);
    },
    [input, selectedFiles, submit],
  );

  /**
   * Starts a genuinely new conversation:
   * 1. Aborts any in-flight backend request so stale responses cannot land.
   * 2. Clears history, composer, attachments and error state.
   * 3. Focuses the composer so the operator can type immediately.
   */
  const handleNewChat = useCallback(() => {
    // Abort and invalidate any pending request
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    requestIdRef.current += 1; // invalidates any in-flight async callbacks

    // Reset conversation state
    setHistory([]);
    setInput("");
    setApiError(null);
    setIsThinking(false);
    setBusy(false);
    setConfirming(false);
    confirmingRef.current = false;
    handsFreeRef.current = false;
    submittingRef.current = false;

    // Clear file attachments
    setPreviews((prev) => {
      prev.forEach((p) => { if (p.previewUrl) URL.revokeObjectURL(p.previewUrl); });
      return [];
    });
    setSelectedFiles([]);

    // Focus the composer
    setTimeout(() => {
      if (isExpanded) {
        expandedTextareaRef.current?.focus();
      } else {
        textareaRef.current?.focus();
      }
    }, 50);
  }, [isExpanded]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleChatSubmit();
      }
    },
    [handleChatSubmit],
  );

  const toggleListening = useCallback(async () => {
    if (recorder.status === "recording") {
      recorder.stop();
      return;
    }
    voiceRef.current?.pauseWakeWord();
    playWakeChime();
    handsFreeRef.current = false;
    await startRecording({ autoStopOnSilence: true });
  }, [recorder, startRecording]);

  const micOk =
    recorder.supported && recorder.status !== "denied" && voice.permissionState !== "denied";

  const components: Array<[string, string, boolean]> = [
    [
      "Microphone",
      micOk
        ? recorder.deviceLabel || `permission ${voice.permissionState}`
        : "permission denied",
      micOk,
    ],
    [
      "Wake Word",
      !voice.supported
        ? "needs Chrome or Edge"
        : !voice.wakeWordEnabled
          ? "off"
          : voice.recognitionRunning
            ? `listening for "Okay Kelvin"`
            : busyRef.current
              ? "paused during this turn"
              : "starting…",
      // ACTIVE only when recognition is genuinely running. Reporting ACTIVE
      // while it was stopped is what hid this fault in the first place.
      voice.supported && voice.wakeWordEnabled && voice.recognitionRunning,
    ],
    [
      "Recognition",
      voice.recognitionRunning ? "running" : "stopped — retrying every 3s",
      voice.recognitionRunning,
    ],
    ["Speech-to-Text", health?.stt ?? "unknown", !!health],
    [
      "Answer Engine",
      health?.llm === "gemini" ? (health.llm_model ?? "gemini") : "offline router",
      !!health,
    ],
    ["Text-to-Speech", health?.tts ?? "unknown", !!health],
  ];

  const renderMessageThread = () => (
    <>
      {history.length === 0 ? (
        <div className="va-chat-welcome">
          <div className="va-chat-welcome-icon">🤖</div>
          <h4>Kelvin Operational Assistant</h4>
          <p>
            {transcript ? `Heard: “${transcript}”` : "Ask operational questions, query employee shifts, or manage machine tasks:"}
          </p>
          <div className="va-chat-suggestions">
            {[
              "Kelvin, start Hamadan's shift",
              "Who is currently on shift?",
              "What is the current print progress?",
              "How long has Hamadan worked?",
              "Who has the most overtime?",
              "What causes a gap between concrete layers?",
              "Can I clear a clogged nozzle while the pump is pressurised?",
            ].map((prompt, i) => (
              <button
                key={i}
                type="button"
                className="va-chat-chip"
                onClick={() => {
                  setInput(prompt);
                  if (isExpanded) {
                    expandedTextareaRef.current?.focus();
                  } else {
                    textareaRef.current?.focus();
                  }
                }}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      ) : (
        history.map((item, index) => (
          <div key={item.at + "-" + index} className="va-chat-turn">
            {/* User Message */}
            <div className="va-chat-msg va-chat-msg-user">
              <div className="va-chat-msg-header">
                <span className="va-chat-role-badge va-role-user">👤 YOU</span>
                <span className="va-chat-msg-time">
                  {new Date(item.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </div>
              <div className="va-chat-bubble va-bubble-user">
                {item.question}
                {item.attachments && item.attachments.length > 0 && (
                  <div className="va-chat-msg-attachments">
                    {item.attachments.map((att, aIdx) => (
                      <div key={aIdx} className="va-msg-attachment-item">
                        {att.kind === "image" && att.previewUrl ? (
                          <img
                            src={att.previewUrl}
                            alt={att.filename}
                            className="va-msg-attachment-thumb"
                            style={{ cursor: "pointer" }}
                            onClick={() =>
                              setPreviewModal({
                                imageUrl: att.previewUrl!,
                                title: att.filename,
                                subtitle: "Uploaded file attachment",
                              })
                            }
                            title="Click to enlarge in preview modal"
                          />
                        ) : (
                          <span>{att.kind === "document" ? "📄" : "📝"}</span>
                        )}
                        <span>{att.filename}</span>
                        <span style={{ color: "#7d8794", fontSize: "10px" }}>
                          ({Math.round(att.size_bytes / 1024)} KB)
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Kelvin Message */}
            <div className="va-chat-msg va-chat-msg-assistant">
              <div className="va-chat-msg-header">
                <span className="va-chat-role-badge va-role-kelvin">🤖 KELVIN</span>
                <span className="va-chat-msg-time">
                  {new Date(item.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
                {item.llm && (
                  <span
                    className={`va-chat-source-tag ${
                      item.llm === "local-attendance"
                        ? "va-source-local"
                        : item.llm === "offline-router"
                        ? "va-source-offline"
                        : item.llm === "rag-knowledge"
                        ? "va-source-rag"
                        : "va-source-gemini"
                    }`}
                  >
                    {item.llm === "local-attendance"
                      ? "📋 LOCAL ATTENDANCE"
                      : item.llm === "offline-router"
                      ? "⚡ OFFLINE ROUTER"
                      : item.llm === "rag-knowledge"
                      ? "📚 RAG HISTORICAL"
                      : item.llm === "gemini-vision"
                      ? "👁️ GEMINI VISION"
                      : item.llm === "gemini"
                      ? "✨ GEMINI AI"
                      : item.llm.toUpperCase()}
                  </span>
                )}
                {item.intent && DEV_MODE && (
                  <span className="va-chat-intent-tag">{item.intent}</span>
                )}
              </div>
              <div className="va-chat-bubble va-bubble-assistant">
                <FormattedMessage
                  text={item.answer}
                  primaryImageId={item.primaryImageId}
                  annotatedImageId={item.annotatedImageId}
                  referenceImageId={item.referenceImageId}
                  onPreviewImage={(modalData) => setPreviewModal(modalData)}
                />
              </div>
            </div>
          </div>
        ))
      )}

      {/* Thinking / Loading Indicator */}
      {(isThinking || transcribing) && (
        <div className="va-chat-turn va-chat-turn-thinking">
          <div className="va-chat-msg va-chat-msg-assistant">
            <div className="va-chat-msg-header">
              <span className="va-chat-role-badge va-role-kelvin">🤖 KELVIN</span>
              <span className="va-chat-status-text">Thinking…</span>
            </div>
            <div className="va-chat-bubble va-bubble-assistant va-bubble-thinking">
              <span className="va-typing-dot" />
              <span className="va-typing-dot" />
              <span className="va-typing-dot" />
            </div>
          </div>
        </div>
      )}
    </>
  );

  const renderComposer = (isExpandedView: boolean) => (
    <form className="va-chat-input-bar" onSubmit={handleChatSubmit}>
      {previews.length > 0 && (
        <div className="va-chat-attachments-tray">
          {previews.map((prev, idx) => (
            <div key={idx} className="va-attachment-preview-chip">
              {prev.kind === "image" && prev.previewUrl ? (
                <img src={prev.previewUrl} alt={prev.file.name} className="va-attachment-thumb" />
              ) : (
                <span className="va-attachment-icon">{prev.kind === "document" ? "📄" : "📝"}</span>
              )}
              <div className="va-attachment-info">
                <span className="va-attachment-name" title={prev.file.name}>{prev.file.name}</span>
                <span className="va-attachment-size">{(prev.file.size / 1024).toFixed(1)} KB</span>
              </div>
              <button
                type="button"
                className="va-attachment-remove-btn"
                onClick={() => removeAttachment(idx)}
                title="Remove attachment"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="va-chat-input-controls">
        <input
          ref={isExpandedView ? expandedFileInputRef : fileInputRef}
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,.webp,.pdf,.docx,.txt,.md,.log,.csv,.json"
          style={{ display: "none" }}
          onChange={handleFileSelect}
        />
        <button
          type="button"
          className="va-chat-attach-btn"
          onClick={() => {
            if (isExpandedView) {
              expandedFileInputRef.current?.click();
            } else {
              fileInputRef.current?.click();
            }
          }}
          disabled={busy || isThinking}
          title="Attach image (JPG, PNG, WEBP), document (PDF, DOCX), or text/log file"
        >
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
          </svg>
        </button>
        <textarea
          ref={isExpandedView ? expandedTextareaRef : textareaRef}
          className="va-chat-textarea"
          value={input}
          placeholder={
            previews.length > 0
              ? "Ask about the attached file(s) or leave blank to analyze... (Enter to send)"
              : "Ask Kelvin anything or issue a shift command... (Enter to send, Shift+Enter for newline)"
          }
          rows={isExpandedView ? 3 : 2}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={busy || isThinking}
        />
        <button
          type="submit"
          className="va-chat-send-btn"
          disabled={busy || isThinking || (!input.trim() && previews.length === 0)}
          title="Send message"
        >
          {isThinking ? (
            <span className="va-spinner" />
          ) : (
            <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          )}
          <span>Send</span>
        </button>
      </div>
    </form>
  );

  return (
    <section className="panel va">
      <header className="va-header">
        <h2>KELVIN6K VOICE ASSISTANT</h2>
        <span className="pill pill-readonly" title="This assistant can only read machine data.">
          READ-ONLY
        </span>
      </header>

      {/* --- the one thing the operator looks at --- */}
      <div className={`va-stage va-stage-${displayState.toLowerCase()}`}>
        <KelvinCore3D state={coreState} level={coreLevel} />

        <p className="va-stage-text" aria-live="polite">
          {coreTitle}
        </p>

        <p className={`va-stage-sub va-stage-sub-${displayState.toLowerCase()}`}>
          <span className="va-dot" />
          {confirming ? "Say 'Yes' or 'No' to confirm" : displayInfo.subtitle}
        </p>

        <div className="va-controls">
          <button 
            type="button"
            className={`va-btn ${listening || confirming ? "va-btn-recording" : "va-btn-idle"}`}
            onClick={toggleListening}
            disabled={speaking || transcribing || busy}
          >
            {listening || confirming ? (
              <>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#ff4d4f", display: "inline-block", marginRight: 4 }} />
                <span>Stop Listening ({clock(recorder.durationMs)})</span>
              </>
            ) : speaking ? (
              <>
                <span>🔊 Kelvin is speaking…</span>
              </>
            ) : transcribing || busy ? (
              <>
                <span>⏳ Thinking…</span>
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                  <line x1="12" y1="19" x2="12" y2="23"></line>
                  <line x1="8" y1="23" x2="16" y2="23"></line>
                </svg>
                <span>Click to Speak</span>
              </>
            )}
          </button>

          <label className="va-ww-toggle">
            <input 
              type="checkbox" 
              checked={voice.wakeWordEnabled} 
              onChange={voice.toggleWakeWord} 
            />
            <span>Wake Word On</span>
          </label>
        </div>

        {/* Real microphone waveform bars, driven purely by existing recorder.level */}
        {listening ? (
          <div className="va-level-wrapper" aria-hidden="true">
            <VoiceLevel
              level={recorder.level}
              threshold={recorder.threshold}
              active={listening}
            />
          </div>
        ) : null}
      </div>

      <div className="va-diagnostic" style={{ margin: "10px", padding: "10px", background: "#f0f0f0", fontSize: "11px", fontFamily: "monospace" }}>
        <strong>DIAGNOSTICS</strong>
        <p style={{ margin: "2px 0" }}>Wake Word: {voice.supported ? (voice.wakeWordEnabled ? "ON" : "OFF") : "UNSUPPORTED"}</p>
        <p style={{ margin: "2px 0" }}>Wake Word State: {voice.recognitionRunning ? "RUNNING" : "STOPPED"} | micStatus: {voice.micStatus} | handsFree: {handsFreeRef.current ? "YES" : "NO"} | busy: {busyRef.current ? "YES" : "NO"}</p>
        <p style={{ margin: "2px 0" }}>Recorder Phase: {recorder.phase}</p>
        <p style={{ margin: "2px 0" }}>Recorder Status: {recorder.status}</p>
        <p style={{ margin: "2px 0" }}>Level: {recorder.level.toFixed(4)} | Peak: {recorder.peakLevel.toFixed(4)} | Threshold: {recorder.threshold.toFixed(4)}</p>
        <p style={{ margin: "2px 0" }}>Debug: {JSON.stringify(recorder.debug)}</p>
        <p style={{ margin: "2px 0", color: "red" }}>{apiError ? `API ERROR: ${apiError}` : ""}</p>
        <p style={{ margin: "2px 0", color: "red" }}>{recorder.error ? `RECORDER ERROR: ${recorder.error}` : ""}</p>
        <p style={{ margin: "2px 0", color: "red" }}>{voice.error ? `VOICE ERROR: ${voice.error}` : ""}</p>
      </div>

      {/* --- ChatGPT-style Operational Chat Interface --- */}
      <div className="va-chat-container">
        <div className="va-chat-header">
          <div className="va-chat-header-title">
            <span className="va-chat-icon">💬</span>
            <h3>OPERATIONAL CHAT & INTELLIGENCE</h3>
          </div>
          <div className="va-chat-header-actions">
            {history.length > 0 && (
              <span className="va-chat-badge">{history.length} {history.length === 1 ? "turn" : "turns"}</span>
            )}
            <button
              type="button"
              className="va-expand-toggle-btn"
              onClick={() => setIsExpanded(true)}
              title="Expand to spacious full workspace overlay (Esc to collapse)"
            >
              ⛶ Expand Workspace
            </button>
            <button
              type="button"
              className="va-new-chat-btn"
              onClick={handleNewChat}
              title="Clear conversation and start a new session"
            >
              + New Chat
            </button>
          </div>
        </div>

        {/* Message Thread (Chronological: Oldest to Newest) */}
        <div className="va-chat-messages" role="log" aria-live="polite">
          {renderMessageThread()}
          <div ref={chatBottomRef} />
        </div>

        {/* Input Bar */}
        {renderComposer(false)}
        <div className="va-chat-input-hint">
          <span>Press <strong>Enter</strong> to send &bull; <strong>Shift+Enter</strong> for newline</span>
        </div>
      </div>

      {apiError ? <p className="va-warn">{apiError}</p> : null}
      {recorder.error ? <p className="va-warn">{recorder.error}</p> : null}
      {!voice.supported ? (
        <p className="va-warn">
          Wake-word detection uses the Web Speech API, which only Chrome and Edge provide. In this
          browser, use START LISTENING under Developer / Diagnostics.
        </p>
      ) : null}

      {/* --- system status --- */}
      <dl className="va-system">
        {components.map(([name, value, ok]) => (
          <div key={name} className="va-system-row">
            <dt>{name}</dt>
            <dd>
              <span className={`va-chip ${ok ? "va-chip-ok" : "va-chip-bad"}`}>
                {ok
                  ? name === "Wake Word"
                    ? "ACTIVE"
                    : name === "Recognition"
                      ? "RUNNING"
                      : "READY"
                  : name === "Recognition"
                    ? "STOPPED"
                    : "ERROR"}
              </span>
              <span className="va-system-detail">{value}</span>
            </dd>
          </div>
        ))}
      </dl>

      {/* --- everything a normal operator never needs --- */}
      {showDeveloper ? (
      <details className="va-dev" open={micBroken}>
        <summary>Developer / Diagnostics</summary>

        <div className="va-dev-body">
          <div className="va-dev-controls">
            <button
              className={`btn ${listening ? "btn-danger" : "btn-primary"}`}
              onClick={() => void toggleListening()}
              disabled={!recorder.supported || transcribing || busy}
            >
              {listening ? `STOP LISTENING (${clock(recorder.durationMs)})` : "START LISTENING"}
            </button>
            <button
              className={`btn ${voice.wakeWordEnabled ? "btn-active" : ""}`}
              onClick={voice.toggleWakeWord}
              disabled={!voice.supported}
              title={`Wake phrases: ${WAKE_WORDS.join(", ")}`}
            >
              {voice.wakeWordEnabled ? "WAKE WORD ON" : "WAKE WORD OFF"}
            </button>
            <button
              className="btn"
              onClick={() => latest && void speakAnswerText(latest.answer)}
              disabled={!latest}
            >
              SPEAK ANSWER
            </button>
          </div>

          <div className="va-dev-toggles">
            <label>
              <input
                type="checkbox"
                checked={autoSpeak}
                onChange={(event) => setAutoSpeak(event.target.checked)}
              />
              <span>Speak answers</span>
            </label>
            <label>
              <input
                type="checkbox"
                checked={serverTts}
                onChange={(event) => setServerTts(event.target.checked)}
              />
              <span>Use backend speaker</span>
            </label>
            {ttsFellBack ? (
              <span className="va-note">Backend audio failed; using browser speech.</span>
            ) : null}
          </div>

          <form
            className="va-dev-ask"
            onSubmit={(event) => {
              event.preventDefault();
              void submit(input, "text");
              setInput("");
            }}
          >
            <input
              type="text"
              value={input}
              placeholder="Type a question (no microphone needed)"
              onChange={(event) => setInput(event.target.value)}
            />
            <button className="btn" type="submit" disabled={busy || !input.trim()}>
              ASK
            </button>
          </form>

          {listening || recorder.audioBytes > 0 ? (
            <pre className="va-debug">
{`MIC DEVICE    : ${recorder.debug.device}
DEVICE ID     : ${recorder.debug.deviceId}
PERMISSION    : ${recorder.debug.permission}
TRACK STATE   : ${recorder.debug.trackState}
TRACK MUTED   : ${recorder.debug.trackMuted}
AUDIO CONTEXT : ${recorder.debug.audioContext}
LIVE RMS      : ${recorder.level.toFixed(4)}
LIVE PEAK     : ${recorder.peakLevel.toFixed(4)}
THRESHOLD     : ${recorder.threshold.toFixed(4)}
MEDIA RECORDER: ${recorder.debug.recorderState}
CHUNKS        : ${recorder.chunkCount}
BLOB SIZE     : ${recorder.audioBytes}
DURATION      : ${(recorder.durationMs / 1000).toFixed(2)}s
CONSTRAINTS   : ${recorder.debug.constraints}`}
            </pre>
          ) : null}

          {recorder.playbackUrl ? (
            <div className="va-playback">
              <span>Play back the last recording</span>
              <audio controls src={recorder.playbackUrl} />
            </div>
          ) : null}

          <IndependentMicTest />
          <MicDiagnostic />

          <button className="btn" onClick={clearHistory} disabled={history.length === 0}>
            CLEAR CONVERSATION
          </button>
        </div>
      </details>
      ) : (
        <p className="va-devhint">
          Hands-free. Open this page with <code>?dev=1</code> for manual controls and diagnostics.
        </p>
      )}

      {/* --- Expanded Full-Screen Overlay Workspace --- */}
      {isExpanded && (
        <div
          className="va-expanded-overlay"
          onClick={(e) => {
            if (e.target === e.currentTarget) setIsExpanded(false);
          }}
          role="dialog"
          aria-modal="true"
          aria-label="Operational Chat Expanded Workspace"
        >
          <div className="va-expanded-dialog">
            <div className="va-expanded-header">
              <div className="va-expanded-header-left">
                <span className="va-chat-icon" style={{ fontSize: "18px" }}>💬</span>
                <div>
                  <h3 style={{ margin: 0, fontSize: "14px", fontWeight: 700, letterSpacing: "0.5px", color: "#e6ecf3" }}>
                    OPERATIONAL CHAT &amp; INTELLIGENCE
                  </h3>
                  <span style={{ fontSize: "11px", color: "#8a99ad" }}>
                    Spacious Industrial Workspace &bull; AI Troubleshooting &amp; Attendance
                  </span>
                </div>
                {history.length > 0 && (
                  <span className="va-chat-badge" style={{ marginLeft: "8px" }}>
                    {history.length} {history.length === 1 ? "turn" : "turns"}
                  </span>
                )}
              </div>
              <div className="va-expanded-header-right">
                <button
                  type="button"
                  className="va-new-chat-btn"
                  onClick={handleNewChat}
                  title="Clear conversation and start a new session"
                >
                  + New Chat
                </button>
                <button
                  type="button"
                  className="va-close-expanded-btn"
                  onClick={() => setIsExpanded(false)}
                  title="Close expanded workspace (Esc)"
                >
                  ✕ Close [Esc]
                </button>
              </div>
            </div>

            <div className="va-expanded-messages" role="log" aria-live="polite">
              <div className="va-expanded-inner-thread">
                {renderMessageThread()}
                <div ref={expandedChatBottomRef} />
              </div>
            </div>

            <div className="va-expanded-composer-bar">
              <div className="va-expanded-composer-inner">
                {renderComposer(true)}
                <div className="va-chat-input-hint">
                  <span>Press <strong>Enter</strong> to send &bull; <strong>Shift+Enter</strong> for newline &bull; <strong>Esc</strong> to collapse</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Lightbox Image Preview Modal */}
      {previewModal && (
        <div
          className="va-lightbox-backdrop"
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(6, 10, 16, 0.88)",
            backdropFilter: "blur(6px)",
            zIndex: 10000,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "20px",
          }}
          onClick={() => setPreviewModal(null)}
        >
          <div
            className="va-lightbox-dialog"
            style={{
              maxWidth: "92vw",
              maxHeight: "92vh",
              background: "#161b22",
              borderRadius: "12px",
              border: "1px solid #30363d",
              boxShadow: "0 12px 36px rgba(0,0,0,0.7)",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "12px 18px",
                borderBottom: "1px solid #21262d",
                background: "#0d1117",
                gap: "12px",
                flexWrap: "wrap",
              }}
            >
              <div>
                <h3 style={{ margin: 0, fontSize: "14px", color: "#e6ecf3", fontWeight: 600 }}>
                  {previewModal.title}
                </h3>
                {previewModal.subtitle && (
                  <p style={{ margin: "2px 0 0", fontSize: "11px", color: "#8b949e" }}>
                    {previewModal.subtitle}
                  </p>
                )}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                {previewModal.alternativeUrl && (
                  <button
                    type="button"
                    onClick={() => {
                      const currentUrl = previewModal.imageUrl;
                      const altUrl = previewModal.alternativeUrl!;
                      const isNowAnnotated = altUrl.includes(previewModal.annotatedImageId || "NONE");
                      setPreviewModal({
                        ...previewModal,
                        imageUrl: altUrl,
                        title: isNowAnnotated ? "AI Defect Annotation Overlay" : "Original Uploaded Defect Image",
                        subtitle: isNowAnnotated
                          ? "AI-assisted visual observation • Not a confirmed root-cause diagnosis"
                          : "Original operator uploaded photo without overlay",
                        alternativeUrl: currentUrl,
                        alternativeLabel: isNowAnnotated ? "Switch to Original Photo" : "Switch to AI Annotation",
                      });
                    }}
                    style={{
                      fontSize: "12px",
                      padding: "4px 10px",
                      borderRadius: "6px",
                      background: "#212836",
                      border: "1px solid #3b4758",
                      color: "#e6ecf3",
                      cursor: "pointer",
                      fontWeight: 500,
                    }}
                  >
                    🔄 {previewModal.alternativeLabel || "Switch Image"}
                  </button>
                )}

                <button
                  type="button"
                  onClick={() => {
                    const fallback = previewModal.isAnnotation
                      ? `kelvin6k_annotation_${previewModal.annotatedImageId || "defect"}.jpg`
                      : previewModal.isReference
                      ? "kelvin6k_reference_component.jpg"
                      : `kelvin6k_defect_${previewModal.primaryImageId || "photo"}.jpg`;
                    downloadIncidentImage(previewModal.imageUrl, fallback);
                  }}
                  style={{
                    fontSize: "12px",
                    padding: "4px 12px",
                    borderRadius: "6px",
                    background: "rgba(74, 222, 128, 0.15)",
                    border: "1px solid rgba(74, 222, 128, 0.4)",
                    color: "#4ade80",
                    cursor: "pointer",
                    fontWeight: 600,
                  }}
                >
                  📥 Download Image
                </button>

                <a
                  href={previewModal.imageUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    fontSize: "12px",
                    padding: "4px 10px",
                    borderRadius: "6px",
                    background: "#21262d",
                    border: "1px solid #30363d",
                    color: "#6aa8ff",
                    textDecoration: "none",
                    cursor: "pointer",
                  }}
                  title="Open raw image directly in a new browser tab"
                >
                  ↗ Open Raw
                </a>

                <button
                  type="button"
                  onClick={() => setPreviewModal(null)}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: "#8b949e",
                    fontSize: "20px",
                    cursor: "pointer",
                    padding: "2px 6px",
                    lineHeight: 1,
                  }}
                  title="Close Preview (Esc)"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Modal Body */}
            <div
              style={{
                padding: "16px",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                background: "#080c12",
                overflow: "auto",
              }}
            >
              <img
                src={previewModal.imageUrl}
                alt={previewModal.title}
                style={{
                  maxWidth: "85vw",
                  maxHeight: "75vh",
                  objectFit: "contain",
                  borderRadius: "6px",
                  boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
                }}
              />
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
