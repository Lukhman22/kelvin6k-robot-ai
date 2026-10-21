/**
 * Microphone diagnostic panel.
 *
 * Answers one question with evidence rather than guesswork: does any
 * configuration of this microphone actually deliver audio samples to the
 * browser? The winning configuration is saved and used for real recording.
 */

import { useMicDiagnostic } from "../hooks/useMicDiagnostic";
import "./MicDiagnostic.css";

export function MicDiagnostic() {
  const diag = useMicDiagnostic();
  const info = diag.liveInfo;

  return (
    <section className="mic-diag">
      <header className="mic-diag-head">
        <h3>MICROPHONE DIAGNOSTIC</h3>
        <span className="mic-diag-sub">No recording is sent anywhere. This only measures signal.</span>
      </header>

      <label className="mic-diag-row">
        <span className="mic-diag-label">DEVICE</span>
        <select
          className="mic-diag-select"
          value={diag.selectedDeviceId}
          onChange={(event) => diag.setSelectedDeviceId(event.target.value)}
          disabled={diag.probing || diag.monitoring}
        >
          <option value="">(browser default)</option>
          {diag.devices.map((device) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="mic-diag-btn"
          onClick={() => void diag.refreshDevices()}
          disabled={diag.probing}
        >
          REFRESH
        </button>
      </label>

      <div className="mic-diag-actions">
        <button
          type="button"
          className="mic-diag-btn mic-diag-btn-primary"
          onClick={() => void diag.runFullDiagnostic()}
          disabled={diag.probing || diag.monitoring}
        >
          {diag.probing ? "TESTING…" : "RUN FULL DIAGNOSTIC (15 sec — keep talking)"}
        </button>
        <button
          type="button"
          className="mic-diag-btn"
          onClick={() => (diag.monitoring ? diag.stopMonitor() : void diag.startMonitor())}
          disabled={diag.probing}
        >
          {diag.monitoring ? "STOP LIVE METER" : "LIVE METER"}
        </button>
      </div>

      {diag.probing ? (
        <p className="mic-diag-note">
          Testing: <strong>{diag.probingLabel}</strong> — speak continuously while each test runs.
        </p>
      ) : null}

      {diag.error ? <p className="mic-diag-note mic-diag-warn">{diag.error}</p> : null}

      {/* --- live meter --- */}
      {diag.monitoring ? (
        <div className="mic-diag-live">
          <div className="mic-diag-live-head">
            <span className="mic-diag-label">LIVE LEVEL</span>
            <span className="mic-diag-live-value">
              {(diag.level * 100).toFixed(1)}% · peak {(diag.peak * 100).toFixed(1)}%
            </span>
          </div>
          <div className="mic-diag-bar">
            <div
              className={`mic-diag-fill${diag.level < 0.015 ? " mic-diag-fill-low" : ""}`}
              style={{ width: `${Math.min(100, diag.level * 600)}%` }}
            />
          </div>
          {info ? (
            <pre className="mic-diag-info">
{`configuration : ${info.configLabel}
track label   : ${info.trackLabel}
deviceId      : ${info.actualDeviceId}
channels      : ${info.channelCount}
sample rate   : ${info.sampleRate}
readyState    : ${info.readyState}
enabled       : ${info.enabled}
muted         : ${info.muted}
audio context : ${info.contextState}
non-zero PCM  : ${info.nonZeroSamples.toLocaleString()} samples`}
            </pre>
          ) : null}
          {info?.muted ? (
            <p className="mic-diag-note mic-diag-warn">
              The track is MUTED by Windows. The browser holds the device but the operating
              system is sending no audio. This is not something the web page can override.
            </p>
          ) : null}
        </div>
      ) : null}

      {/* --- results table --- */}
      {diag.results.length > 0 ? (
        <div className="mic-diag-results">
          <table>
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Result</th>
                <th>Non-zero samples</th>
                <th>Peak</th>
                <th>Muted</th>
                <th>Ch / Rate</th>
              </tr>
            </thead>
            <tbody>
              {diag.results.map((result) => (
                <tr key={result.configId} className={`verdict-${result.verdict.split(" ")[0].toLowerCase()}`}>
                  <td>{result.label}</td>
                  <td className="mic-diag-verdict">{result.verdict}</td>
                  <td>
                    {result.opened
                      ? `${result.nonZeroSamples.toLocaleString()} / ${result.totalSamples.toLocaleString()}`
                      : "—"}
                  </td>
                  <td>{result.opened ? result.peak.toFixed(4) : "—"}</td>
                  <td>{result.opened ? String(result.mutedAtEnd) : "—"}</td>
                  <td>{result.opened ? `${result.channelCount} / ${result.sampleRate}` : result.error}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {diag.probing ? null : diag.winner ? (
            <p className="mic-diag-note mic-diag-ok">
              Working configuration found: <strong>{diag.winner.label}</strong>. It has been saved
              and will be used by START LISTENING.
            </p>
          ) : (
            <div className="mic-diag-note mic-diag-warn">
              <p>
                <strong>No configuration produced any audio.</strong> Every attempt returned samples
                of exactly zero, so the browser is not the problem — Windows is not delivering sound
                from this microphone. Check, in this order:
              </p>
              <ol>
                <li>
                  Close every Python script that records audio, plus Zoom, Teams and Discord, then
                  run this test again. A USB microphone can be held exclusively by one program.
                </li>
                <li>
                  Windows Settings → System → Sound → Input → click the AB13X microphone → speak and
                  watch the test bar there. If that bar does not move either, no browser code can help.
                </li>
                <li>
                  In the same screen, check the input volume is not 0 and the device is not muted.
                </li>
                <li>Settings → Privacy &amp; security → Microphone → "Let desktop apps access" is ON.</li>
                <li>Try the microphone in a different USB port.</li>
              </ol>
            </div>
          )}
        </div>
      ) : null}
    </section>
  );
}
