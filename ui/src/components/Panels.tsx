/** Production, machine (vision), material and alerts panels. */

import type {
  AlertItem,
  MachineTelemetry,
  MaterialStatus,
  Production,
  VisionStatus,
} from "../types";
import { FieldRow } from "./Field";

const oneDp = (n: number) => n.toFixed(1);

export function ProductionPanel({ production }: { production: Production }) {
  const percent = production.percentage.value;
  return (
    <section className="panel">
      <header className="panel-header"><h2>Production Data</h2></header>

      {percent !== null ? (
        <div className="progress-block">
          <div className="progress-bar" role="progressbar" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}>
            <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
          </div>
          <span className="progress-label">{percent.toFixed(1)}% complete</span>
        </div>
      ) : null}

      <FieldRow label="Job" field={production.job_name} />
      <FieldRow label="Printed" field={production.completed_mm} format={oneDp} emphasis />
      <FieldRow label="Remaining" field={production.remaining_mm} format={oneDp} emphasis />
      <FieldRow label="Total length" field={production.total_mm} format={oneDp} />
      <FieldRow label="Current layer" field={production.current_layer} />
      <FieldRow label="Total layers" field={production.total_layers} />
    </section>
  );
}

export function MachineStatusPanel({
  vision,
  telemetry,
}: {
  vision: VisionStatus;
  telemetry: MachineTelemetry;
}) {
  const telemetryItems = [
    telemetry.pump_running,
    telemetry.pump_speed_rpm,
    telemetry.accelerator_running,
    telemetry.accelerator_speed_rpm,
    telemetry.print_speed_mm_s,
  ];
  const allUnavailable = telemetryItems.every(
    (item) => item.provenance === "UNAVAILABLE" || (item.value === null && item.reason)
  );
  const commonReason = telemetryItems[0]?.reason;

  return (
    <section className="panel">
      <header className="panel-header"><h2>Machine Status</h2></header>

      <FieldRow label="Printing" field={vision.printing} format={(v) => (v ? "YES" : "NO")} emphasis />
      <FieldRow label="Print status" field={vision.print_status} />
      <FieldRow label="Layer width" field={vision.layer_width_mm} format={oneDp} emphasis />
      <FieldRow label="Layer width (raw)" field={vision.layer_width_px} format={oneDp} />
      <FieldRow label="Confidence" field={vision.confidence} format={(v) => `${(v * 100).toFixed(0)}%`} />
      <FieldRow label="Motion (ROI)" field={vision.motion_score} format={(v) => v.toFixed(2)} />

      {vision.note ? <p className="panel-note">{vision.note}</p> : null}

      <h3 className="panel-subhead">Drives &amp; pumps</h3>
      {allUnavailable && commonReason ? (
        <>
          <FieldRow label="Pump" field={telemetry.pump_running} format={(v) => (v ? "RUNNING" : "STOPPED")} suppressReason />
          <FieldRow label="Pump speed" field={telemetry.pump_speed_rpm} format={oneDp} suppressReason />
          <FieldRow label="Accelerator" field={telemetry.accelerator_running} format={(v) => (v ? "RUNNING" : "STOPPED")} suppressReason />
          <FieldRow label="Accel. speed" field={telemetry.accelerator_speed_rpm} format={oneDp} suppressReason />
          <FieldRow label="Print speed" field={telemetry.print_speed_mm_s} format={oneDp} suppressReason />
          <p className="panel-note" style={{ marginTop: "6px" }}>{commonReason}</p>
        </>
      ) : (
        <>
          <FieldRow label="Pump" field={telemetry.pump_running} format={(v) => (v ? "RUNNING" : "STOPPED")} />
          <FieldRow label="Pump speed" field={telemetry.pump_speed_rpm} format={oneDp} />
          <FieldRow label="Accelerator" field={telemetry.accelerator_running} format={(v) => (v ? "RUNNING" : "STOPPED")} />
          <FieldRow label="Accel. speed" field={telemetry.accelerator_speed_rpm} format={oneDp} />
          <FieldRow label="Print speed" field={telemetry.print_speed_mm_s} format={oneDp} />
        </>
      )}
    </section>
  );
}

export function MaterialPanel({ material }: { material: MaterialStatus }) {
  return (
    <section className="panel">
      <header className="panel-header"><h2>Material Status</h2></header>

      <FieldRow label="Material flow" field={material.material_flow_percent} format={oneDp} emphasis />
      <FieldRow label="Nozzle temp" field={material.nozzle_temp_c} format={oneDp} />
      <FieldRow label="Bed temp" field={material.bed_temp_c} format={oneDp} />
      <FieldRow label="Machine state" field={material.machine_state} />

      {material.source_note ? <p className="panel-note panel-note-warn">{material.source_note}</p> : null}
    </section>
  );
}

export function AlertsPanel({ alerts }: { alerts: AlertItem[] }) {
  return (
    <section className="panel">
      <header className="panel-header"><h2>Alerts</h2></header>
      {alerts.length === 0 ? (
        <p className="empty-note">No active alerts.</p>
      ) : (
        <ul className="alert-list">
          {alerts.map((alert, index) => (
            <li key={`${alert.message}-${index}`} className={`alert alert-${alert.level}`}>
              <span className="alert-level">{alert.level.toUpperCase()}</span>
              <span className="alert-message">{alert.message}</span>
              <span className="alert-age">{alert.seconds_ago.toFixed(0)}s ago</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
