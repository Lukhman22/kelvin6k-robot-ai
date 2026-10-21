import { AttendanceCard } from "./components/AttendanceCard";
import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { camera2SnapshotUrl, camera2StreamUrl, getHealth, getHistory, getStatus } from "./api";
import {
  HistoryPanel,
  MaterialCard,
  PrintingDetails,
  ProblemsPanel,
  ProductionCards,
  RobotDetails,
} from "./components/DashboardPanels";
import { Header } from "./components/Header";
import { LiveCamera } from "./components/LiveCamera";
import { MachineStatusPanel } from "./components/Panels";
import { VoiceAssistant } from "./components/VoiceAssistant";
import type { HealthResponse, HistoryResponse, StatusResponse } from "./types";

const STATUS_POLL_MS = 1000;
const HISTORY_POLL_MS = 15000;

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const timers = useRef<number[]>([]);

  const pollStatus = useCallback(async () => {
    try {
      setStatus(await getStatus());
      setConnectionError(null);
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const pollHistory = useCallback(async () => {
    try {
      setHistory(await getHistory());
    } catch {
      /* history is non-critical; the panel shows its own empty state */
    }
  }, []);

  useEffect(() => {
    void pollStatus();
    void pollHistory();
    getHealth().then(setHealth).catch(() => setHealth(null));

    timers.current.push(window.setInterval(() => void pollStatus(), STATUS_POLL_MS));
    timers.current.push(window.setInterval(() => void pollHistory(), HISTORY_POLL_MS));

    const running = timers.current;
    return () => running.forEach((id) => window.clearInterval(id));
  }, [pollStatus, pollHistory]);

  return (
    <div className="app">
      <Header status={status} health={health} connected={!connectionError} />

      {connectionError ? (
        <div className="banner banner-error">
          Cannot reach the Kelvin6K API. Start it with <code>python run_api.py</code>.
          <span className="banner-detail">({connectionError})</span>
        </div>
      ) : null}

      {status && !status.calibrated ? (
        <div className="banner banner-warn">
          <strong>Layer width is not calibrated.</strong> Widths are reported in pixels only —
          set <code>PIXELS_PER_MM</code> in <code>.env</code> for millimetres. No pixel value is
          shown as a millimetre value.
        </div>
      ) : null}

      <div className="legend">
        <span className="badge badge-real">REAL</span> measured
        <span className="badge badge-mock">MOCK</span> simulated — not the real machine
        <span className="badge badge-unavailable">UNAVAILABLE</span> not connected — never guessed
      </div>

      {status ? (
        <main className="grid">
          <div className="col col-left">
            <LiveCamera camera={status.camera} />
            {status.camera2 ? (
              <LiveCamera
                camera={status.camera2}
                peopleCount={status.people_count}
                detections={status.camera2.detections}
                title="Site Camera 2"
                subtitle="Instance Segmentation & People Monitoring"
                streamUrl={camera2StreamUrl}
                snapshotUrl={camera2SnapshotUrl}
                altText="Site Camera 2 live view with YOLO instance segmentation"
              />
            ) : null}
            <ProductionCards production={status.production} />
            <HistoryPanel history={history} />
          </div>

          <div className="col col-mid">
            <MachineStatusPanel vision={status.vision} telemetry={status.machine_telemetry} />
            <PrintingDetails geometry={status.geometry} />
            <MaterialCard material={status.material} />
            <RobotDetails robot={status.robot} />
            <ProblemsPanel problems={status.problems} />
          </div>

          <div className="col col-right">
            <AttendanceCard />
            <VoiceAssistant />
          </div>
        </main>
      ) : (
        <main className="grid">
          <p className="empty-note">Loading machine state…</p>
        </main>
      )}
    </div>
  );
}
