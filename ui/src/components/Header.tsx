/** Dashboard header: robot / machine / camera / communication status. */

import "./Header.css";
import type { HealthResponse, StatusResponse } from "../types";

interface Props {
  status: StatusResponse | null;
  health: HealthResponse | null;
  connected: boolean;
}

function StatusTile({
  label,
  value,
  tone,
  detail,
}: {
  label: string;
  value: string;
  tone: "ok" | "warn" | "bad" | "muted";
  detail?: string;
}) {
  return (
    <div className={`status-tile status-tile-${tone}`} title={detail}>
      <span className="status-tile-label">{label}</span>
      <span className="status-tile-value">
        <span className={`status-dot status-${tone === "ok" ? "live" : tone === "bad" ? "bad" : "ok"}`} />
        {value}
      </span>
    </div>
  );
}

export function Header({ status, health, connected }: Props) {
  const robotOnline = status?.robot.robot_online.value === true;
  const cameraLive = status?.camera.status === "live";
  const machineState = status?.material.machine_state.value ?? null;
  const machineLink = status?.robot.communication.machine_link;

  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          <span className="brand-mark-ring" />
          K6
        </div>
        <div>
          <h1>KELVIN6K ROBOT AI</h1>
          <span className="brand-sub">Industrial Print Monitoring &amp; Voice Assistant</span>
        </div>
      </div>

      <div className="status-tiles">
        <StatusTile
          label="Robot"
          value={robotOnline ? "ONLINE" : "OFFLINE"}
          tone={robotOnline ? "ok" : "bad"}
          detail={status?.robot.robot_status.reason}
        />
        <StatusTile
          label="Machine"
          value={machineState ? String(machineState).toUpperCase() : "UNAVAILABLE"}
          tone={machineState ? "warn" : "muted"}
          detail={
            status?.material.machine_state.provenance === "MOCK"
              ? "Simulated value — not the real machine."
              : status?.material.machine_state.reason
          }
        />
        <StatusTile
          label="Camera"
          value={(status?.camera.status ?? "unknown").toUpperCase()}
          tone={status?.camera.status === "disabled" ? "muted" : (cameraLive ? "ok" : "bad")}
          detail={status?.camera.error ?? undefined}
        />
        <StatusTile
          label="Comms"
          value={connected ? "CONNECTED" : "OFFLINE"}
          tone={connected ? "ok" : "bad"}
          detail={machineLink?.reason}
        />
      </div>

      <div className="header-chips">
        {health ? (
          <>
            <span className="header-chip">LLM: {health.llm}</span>
            <span className="header-chip">STT: {health.stt}</span>
            <span className="header-chip">RAG: {health.rag_chunks} · {health.rag_mode}</span>
          </>
        ) : null}
        <span className="header-chip header-chip-readonly">READ-ONLY</span>
      </div>
    </header>
  );
}
