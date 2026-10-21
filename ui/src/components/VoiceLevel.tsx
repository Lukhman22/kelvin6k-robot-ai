/**
 * Live microphone level, drawn as a row of bars.
 *
 * The bars are driven by the real RMS level coming from the AnalyserNode --
 * there is no animation running underneath. When the microphone is silent the
 * bars are flat, which is the honest thing for them to do.
 *
 * A short history is kept so the display scrolls like a waveform rather than
 * every bar moving together.
 */

import { useEffect, useRef } from "react";
import "./VoiceLevel.css";

const BAR_COUNT = 28;

export function VoiceLevel({
  level,
  threshold,
  active,
}: {
  level: number;
  /** Bars at or above this are shown as speech rather than background noise. */
  threshold: number;
  active: boolean;
}) {
  const historyRef = useRef<number[]>(new Array(BAR_COUNT).fill(0));
  const barsRef = useRef<(HTMLSpanElement | null)[]>([]);

  useEffect(() => {
    if (!active) {
      historyRef.current = new Array(BAR_COUNT).fill(0);
      barsRef.current.forEach((bar) => {
        if (bar) {
          bar.style.height = "6%";
          bar.classList.remove("voice-bar-hot");
        }
      });
      return;
    }

    historyRef.current = [...historyRef.current.slice(1), level];

    // Written straight to the DOM: this updates many times a second, and going
    // through React state for it would re-render the whole panel each frame.
    historyRef.current.forEach((value, index) => {
      const bar = barsRef.current[index];
      if (!bar) return;
      const height = Math.max(6, Math.min(100, value * 600));
      bar.style.height = `${height}%`;
      bar.classList.toggle("voice-bar-hot", value >= threshold);
    });
  }, [level, threshold, active]);

  return (
    <div className={`voice-level${active ? " voice-level-active" : ""}`} aria-hidden="true">
      {Array.from({ length: BAR_COUNT }, (_, index) => (
        <span
          key={index}
          className="voice-bar"
          ref={(element) => {
            barsRef.current[index] = element;
          }}
        />
      ))}
    </div>
  );
}
