/**
 * Kelvin 3D AI-Agent Character
 *
 * A professional 3D animated industrial robot character built using
 * CSS 3D transforms (perspective + preserve-3d) and layered vector SVG HUD optics.
 * Zero WebGL/three.js overhead; GPU-composited at 60fps.
 *
 * States & Audio Reactivity:
 *   - IDLE: Subtle 3D floating & mechanical breathing, standby optic sensors.
 *   - LISTENING: Awake emerald/cyan state, ocular sensors wide, leaning forward.
 *   - HUMAN_SPEAKING: Ocular apertures & chest arc core dynamically scale to `level` (mic RMS).
 *   - THINKING: Inquisitive 3D head tilt, amber processing sweep & synaptic scanline.
 *   - KELVIN_SPEAKING: Multi-band vocalizer matrix animated in speech cadence & subtle head nods.
 *   - DONE: Affirmative nod, emerald check-crest flash, smooth ease back to IDLE.
 *   - ERROR: Alert red diagnostics, stabilized warning crossbars.
 */

import type { CSSProperties } from "react";
import "./KelvinCore3D.css";

export type CoreState =
  | "IDLE"
  | "LISTENING"
  | "HUMAN_SPEAKING"
  | "THINKING"
  | "KELVIN_SPEAKING"
  | "DONE"
  | "ERROR";

interface Props {
  state: CoreState;
  /** 0..1, normalised microphone level. Only meaningful while listening. */
  level: number;
}

/** Cybernetic Visor Display: Ocular HUD sensors & voice synthesizer matrix */
function VisorFace({ state }: { state: CoreState }) {
  switch (state) {
    case "IDLE":
      // Standby optic bars: calm, relaxed horizontal energy slits
      return (
        <g className="kc-visor-content kc-face-idle">
          <path className="kc-hud-bracket" d="M26 34 h-6 v20" />
          <path className="kc-hud-bracket" d="M114 34 h6 v20" />
          <rect className="kc-optic-bar kc-bar-left" x="34" y="42" width="24" height="5" rx="2.5" />
          <circle className="kc-optic-dot" cx="46" cy="44.5" r="1.5" />
          <rect className="kc-optic-bar kc-bar-right" x="82" y="42" width="24" height="5" rx="2.5" />
          <circle className="kc-optic-dot" cx="94" cy="44.5" r="1.5" />
          <line className="kc-mouth-standby" x1="58" y1="72" x2="82" y2="72" />
        </g>
      );
    case "LISTENING":
    case "HUMAN_SPEAKING":
      // Attentive circular ocular reticles with dilating pupil apertures and mic sound capture baseline
      return (
        <g className="kc-visor-content kc-face-listen">
          <path className="kc-hud-bracket" d="M22 28 h-6 v12 M22 58 h-6 v-12" />
          <path className="kc-hud-bracket" d="M118 28 h6 v12 M118 58 h6 v-12" />

          {/* Left Eye Reticle & Iris */}
          <g className="kc-eye-group kc-eye-left">
            <circle className="kc-eye-outer-ring" cx="46" cy="43" r="15" />
            <circle className="kc-eye-iris" cx="46" cy="43" r="10" />
            <circle className="kc-eye-pupil" cx="46" cy="43" r="5.5" />
            <circle className="kc-eye-glint" cx="49" cy="40" r="1.8" />
          </g>

          {/* Right Eye Reticle & Iris */}
          <g className="kc-eye-group kc-eye-right">
            <circle className="kc-eye-outer-ring" cx="94" cy="43" r="15" />
            <circle className="kc-eye-iris" cx="94" cy="43" r="10" />
            <circle className="kc-eye-pupil" cx="94" cy="43" r="5.5" />
            <circle className="kc-eye-glint" cx="97" cy="40" r="1.8" />
          </g>

          {/* Audio capture equalizer bars */}
          <g className="kc-mic-spectrum">
            <rect className="kc-spec-bar kc-sbar-1" x="48" y="70" width="3" height="5" rx="1.5" />
            <rect className="kc-spec-bar kc-sbar-2" x="55" y="68" width="3" height="9" rx="1.5" />
            <rect className="kc-spec-bar kc-sbar-3" x="62" y="65" width="3" height="15" rx="1.5" />
            <rect className="kc-spec-bar kc-sbar-4" x="69" y="63" width="3" height="19" rx="1.5" />
            <rect className="kc-spec-bar kc-sbar-5" x="76" y="65" width="3" height="15" rx="1.5" />
            <rect className="kc-spec-bar kc-sbar-6" x="83" y="68" width="3" height="9" rx="1.5" />
            <rect className="kc-spec-bar kc-sbar-7" x="90" y="70" width="3" height="5" rx="1.5" />
          </g>
        </g>
      );
    case "THINKING":
      // Inquisitive angled optics with active synchrotron data scanning line & processing dots
      return (
        <g className="kc-visor-content kc-face-think">
          <line className="kc-scanline-active" x1="20" y1="43" x2="120" y2="43" />
          <g className="kc-eye-group kc-eye-left">
            <rect className="kc-eye-box" x="34" y="32" width="22" height="22" rx="4" />
            <circle className="kc-eye-pupil kc-pupil-shift" cx="47" cy="41" r="5" />
            <circle className="kc-eye-glint" cx="49" cy="39" r="1.5" />
          </g>
          <g className="kc-eye-group kc-eye-right">
            <rect className="kc-eye-box" x="84" y="32" width="22" height="22" rx="4" />
            <circle className="kc-eye-pupil kc-pupil-shift" cx="97" cy="41" r="5" />
            <circle className="kc-eye-glint" cx="99" cy="39" r="1.5" />
          </g>
          <g className="kc-think-dots">
            <circle className="kc-tdot kc-tdot-1" cx="62" cy="72" r="2.5" />
            <circle className="kc-tdot kc-tdot-2" cx="70" cy="72" r="2.5" />
            <circle className="kc-tdot kc-tdot-3" cx="78" cy="72" r="2.5" />
          </g>
        </g>
      );
    case "KELVIN_SPEAKING":
      // Dynamic speech vocalizer matrix with communicative cybernetic optics
      return (
        <g className="kc-visor-content kc-face-speak">
          <g className="kc-eye-group kc-eye-left">
            <circle className="kc-eye-outer-ring" cx="46" cy="41" r="14" />
            <circle className="kc-eye-pupil" cx="46" cy="41" r="5" />
            <circle className="kc-eye-glint" cx="48.5" cy="38.5" r="1.6" />
          </g>
          <g className="kc-eye-group kc-eye-right">
            <circle className="kc-eye-outer-ring" cx="94" cy="41" r="14" />
            <circle className="kc-eye-pupil" cx="94" cy="41" r="5" />
            <circle className="kc-eye-glint" cx="96.5" cy="38.5" r="1.6" />
          </g>
          <g className="kc-vocalizer">
            <rect className="kc-vbar kc-vbar-1" x="42" y="66" width="3.5" height="10" rx="1.75" />
            <rect className="kc-vbar kc-vbar-2" x="49" y="62" width="3.5" height="18" rx="1.75" />
            <rect className="kc-vbar kc-vbar-3" x="56" y="58" width="3.5" height="26" rx="1.75" />
            <rect className="kc-vbar kc-vbar-4" x="63" y="55" width="3.5" height="32" rx="1.75" />
            <rect className="kc-vbar kc-vbar-5" x="70" y="54" width="3.5" height="34" rx="1.75" />
            <rect className="kc-vbar kc-vbar-6" x="77" y="55" width="3.5" height="32" rx="1.75" />
            <rect className="kc-vbar kc-vbar-7" x="84" y="58" width="3.5" height="26" rx="1.75" />
            <rect className="kc-vbar kc-vbar-8" x="91" y="62" width="3.5" height="18" rx="1.75" />
            <rect className="kc-vbar kc-vbar-9" x="98" y="66" width="3.5" height="10" rx="1.75" />
          </g>
        </g>
      );
    case "DONE":
      // Affirmative cybernetic acknowledgment arcs and confident smile crest
      return (
        <g className="kc-visor-content kc-face-done">
          <path className="kc-optic-arc" d="M36 44 q10 -10 20 0" />
          <path className="kc-optic-arc" d="M84 44 q10 -10 20 0" />
          <path className="kc-smile-arc" d="M52 68 q18 12 36 0" />
          <circle className="kc-optic-dot" cx="70" cy="52" r="2" />
        </g>
      );
    case "ERROR":
      return (
        <g className="kc-visor-content kc-face-error">
          <path className="kc-warn-x" d="M37 34 l18 18 M55 34 l-18 18" />
          <path className="kc-warn-x" d="M85 34 l18 18 M103 34 l-18 18" />
          <line className="kc-mouth-standby kc-mouth-error" x1="50" y1="72" x2="90" y2="72" />
        </g>
      );
  }
}

/** Floating 3D State HUD Badge */
function Badge({ state }: { state: CoreState }) {
  switch (state) {
    case "IDLE":
      return (
        <span className="kc-badge kc-badge-zz" aria-hidden="true">
          <b>z</b>
          <b>z</b>
          <b>Z</b>
        </span>
      );
    case "LISTENING":
    case "HUMAN_SPEAKING":
      return (
        <span className="kc-badge kc-chip" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z" />
            <path d="M18.5 11a6.5 6.5 0 0 1-13 0M12 17.5V21" fill="none" />
          </svg>
        </span>
      );
    case "THINKING":
      return (
        <span className="kc-badge kc-badge-dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
      );
    case "KELVIN_SPEAKING":
      return (
        <span className="kc-badge kc-chip" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" />
            <path d="M15.5 9a4 4 0 0 1 0 6M18.2 6.5a7.5 7.5 0 0 1 0 11" fill="none" />
          </svg>
        </span>
      );
    case "DONE":
      return (
        <span className="kc-badge kc-chip kc-chip-pop" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path d="M5 12.5l4.5 4.5L19 7.5" fill="none" />
          </svg>
        </span>
      );
    case "ERROR":
      return (
        <span className="kc-badge kc-chip kc-chip-pop" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path d="M12 3.5l9.5 16.5h-19z" />
            <path d="M12 10v4.5M12 17.2v.3" fill="none" />
          </svg>
        </span>
      );
  }
}

export function KelvinCore3D({ state, level }: Props) {
  // Clamp microphone level 0..1; CSS reads it as a unitless multiplier.
  const lvl = Math.max(0, Math.min(1, level));
  const style = { "--lvl": lvl.toFixed(3) } as CSSProperties;

  return (
    <div className={`kc kc-${state.toLowerCase()}`} style={style} aria-hidden="true">
      <div className="kc-scene">
        {/* Ambient volumetric glow halo */}
        <div className="kc-halo" />

        {/* Orbit resonance rings */}
        <div className="kc-orbit kc-orbit-a">
          <span className="kc-ring" />
        </div>
        <div className="kc-orbit kc-orbit-b">
          <span className="kc-ring" />
        </div>

        {/* Emitted acoustic / energy waves */}
        <div className="kc-waves">
          <span className="kc-wave kc-wave-1" />
          <span className="kc-wave kc-wave-2" />
          <span className="kc-wave kc-wave-3" />
        </div>

        {/* 3D Robot Character Model */}
        <div className="kc-robot">
          {/* Head & Helmet Assembly */}
          <div className="kc-head">
            {/* Ear Audio Sensors */}
            <div className="kc-ear kc-ear-left">
              <span className="kc-ear-light" />
            </div>
            <div className="kc-ear kc-ear-right">
              <span className="kc-ear-light" />
            </div>

            {/* Robot Helmet Chassis */}
            <div className="kc-helmet">
              <div className="kc-helmet-crest" />
              <div className="kc-helmet-rim" />

              {/* Curved 3D Glass Visor with HUD */}
              <div className="kc-visor">
                <div className="kc-visor-sheen" />
                <div className="kc-visor-scanline" />
                <svg className="kc-hud-svg" viewBox="0 0 140 100">
                  <VisorFace state={state} />
                </svg>
              </div>

              <div className="kc-helmet-chin" />
            </div>
          </div>

          {/* Articulated Cybernetic Neck Joint */}
          <div className="kc-neck">
            <span className="kc-neck-band" />
            <span className="kc-neck-band" />
          </div>

          {/* Torso & Shoulder Chassis (Bust) */}
          <div className="kc-torso">
            <div className="kc-shoulder kc-shoulder-left" />
            <div className="kc-shoulder kc-shoulder-right" />

            <div className="kc-chest-plate">
              <span className="kc-torso-badge">K-6000 AI</span>

              {/* Quantum Arc Core / Audio Visualizer Reactor */}
              <div className="kc-arc-core">
                <div className="kc-arc-ring" />
                <div className="kc-arc-lens" />
                <div className="kc-arc-flare" />
              </div>
            </div>
          </div>

          {/* 3D Floating State Badge */}
          <Badge state={state} />
        </div>

        {/* Dynamic Contact Floor Shadow */}
        <div className="kc-shadow" />
      </div>
    </div>
  );
}
