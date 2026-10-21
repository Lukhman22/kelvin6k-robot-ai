/** Alexa-style animated voice orb. Click to toggle listening. */

import "./VoiceOrb.css";

export type AssistantState =
  | "IDLE"
  | "WAKE"
  | "WAKE_WORD_DETECTED"
  | "LISTENING"
  | "PROCESSING"
  | "SPEAKING"
  | "ERROR";

const CLASS_FOR: Record<AssistantState, string> = {
  IDLE: "orb-idle",
  WAKE: "orb-wake",
  WAKE_WORD_DETECTED: "orb-listening",
  LISTENING: "orb-listening",
  PROCESSING: "orb-processing",
  SPEAKING: "orb-speaking",
  ERROR: "orb-error",
};

const LABEL_FOR: Record<AssistantState, string> = {
  IDLE: "IDLE",
  WAKE: "WAKE WORD ON",
  WAKE_WORD_DETECTED: "WAKE WORD DETECTED",
  LISTENING: "LISTENING",
  PROCESSING: "PROCESSING",
  SPEAKING: "SPEAKING",
  ERROR: "ERROR",
};

const HINT_FOR: Record<AssistantState, string> = {
  IDLE: 'Say "Okay Kelvin", or press Start Listening.',
  WAKE: 'Say "Okay Kelvin" followed by your question.',
  WAKE_WORD_DETECTED: "Yes, I'm listening — ask your question.",
  LISTENING: "Listening — ask your question now.",
  PROCESSING: "Querying the machine database and AI…",
  SPEAKING: "Speaking the answer.",
  ERROR: "Something went wrong — see the message below.",
};

const STATE_KEY: Record<AssistantState, string> = {
  IDLE: "idle",
  WAKE: "wake",
  WAKE_WORD_DETECTED: "listening",
  LISTENING: "listening",
  PROCESSING: "processing",
  SPEAKING: "speaking",
  ERROR: "error",
};

function MicIcon() {
  return (
    <svg className="orb-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
         strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="22" />
    </svg>
  );
}

interface Props {
  state: AssistantState;
  disabled?: boolean;
  onToggle: () => void;
}

export function VoiceOrb({ state, disabled, onToggle }: Props) {
  const key = STATE_KEY[state];
  return (
    <div className="orb-area">
      <button
        type="button"
        className={`orb ${CLASS_FOR[state]}`}
        onClick={onToggle}
        disabled={disabled}
        aria-label={`Assistant is ${LABEL_FOR[state]}. Click to toggle listening.`}
      >
        <span className="orb-ring" />
        <span className="orb-ring" />
        <span className="orb-ring" />
        <span className="orb-core" style={{ color: "var(--accent)" }}>
          {state === "SPEAKING" ? (
            <span className="orb-bars" aria-hidden="true">
              <span /><span /><span /><span /><span />
            </span>
          ) : (
            <MicIcon />
          )}
        </span>
      </button>

      <p className={`orb-state orb-state-${key}`}>{LABEL_FOR[state]}</p>
      <p className="orb-hint">
        {state === "WAKE" ? (
          <>Say <span className="orb-wake-word">“Okay Kelvin”</span> followed by your question.</>
        ) : (
          HINT_FOR[state]
        )}
      </p>
    </div>
  );
}
