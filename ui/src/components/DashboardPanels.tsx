/**
 * Printing details, robot details, problems and production history.
 * Values that the backend reports as UNAVAILABLE are rendered as such with
 * their reason -- never replaced with a placeholder number.
 */

import "./DashboardPanels.css";
import type {
  Geometry,
  HistoryResponse,
  MaterialFull,
  Problems,
  ProductionFull,
  RobotInfo,
} from "../types";
import { FieldRow, ProvenanceBadge } from "./Field";

const oneDp = (n: number) => n.toFixed(1);
const twoDp = (n: number) => n.toFixed(2);

/* --- Production cards --- */

export function ProductionCards({ production }: { production: ProductionFull }) {
  const percent = production.percentage.value;
  const status = production.deadline_status.value;

  const cards = [
    { label: "Total Target", field: production.total_mm, format: oneDp },
    { label: "Completed", field: production.completed_mm, format: oneDp },
    { label: "Remaining", field: production.remaining_mm, format: oneDp },
    { label: "Print Rate", field: production.print_rate_mm_per_min, format: twoDp },
    { label: "ETA", field: production.eta },
    { label: "Elapsed", field: production.elapsed },
  ];

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>Production</h2>
        {status ? (
          <span className={`deadline-pill deadline-${status.toLowerCase().replace(/\s+/g, "-")}`}>
            {status}
          </span>
        ) : (
          <ProvenanceBadge provenance={production.deadline_status.provenance} />
        )}
      </header>

      {percent !== null ? (
        <div className="progress-block">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
          </div>
          <span className="progress-label">{percent.toFixed(1)}% complete</span>
        </div>
      ) : null}

      <div className="stat-grid">
        {cards.map((card) => {
          const has = card.field.value !== null && card.field.value !== undefined;
          return (
            <div className="stat-card" key={card.label} title={!has ? card.field.reason : undefined}>
              <span className="stat-label">{card.label}</span>
              <span className={`stat-value${has ? "" : " stat-value-missing"}`}>
                {has
                  ? `${card.format ? card.format(card.field.value as never) : card.field.value}${
                      card.field.unit ? ` ${card.field.unit}` : ""
                    }`
                  : "—"}
              </span>
              <ProvenanceBadge provenance={card.field.provenance} />
            </div>
          );
        })}
      </div>

      <FieldRow label="Job" field={production.job_name} />
      {production.deadline_status.value === null && production.deadline_status.reason ? (
        <p className="panel-note">{production.deadline_status.reason}</p>
      ) : null}
    </section>
  );
}

/* --- Printing details --- */

export function PrintingDetails({ geometry }: { geometry: Geometry }) {
  return (
    <section className="panel">
      <header className="panel-header"><h2>Printing Details</h2></header>
      <FieldRow label="Wall height" field={geometry.wall_height_mm} format={oneDp} />
      <FieldRow label="Wall width" field={geometry.wall_width_mm} format={oneDp} />
      <FieldRow label="Layer width" field={geometry.layer_width_mm} format={twoDp} emphasis />
      <FieldRow label="Layer height" field={geometry.layer_height_mm} format={twoDp} emphasis />
      <FieldRow label="Layer width (raw)" field={geometry.layer_width_px} format={oneDp} />
      <FieldRow label="Layer height (raw)" field={geometry.layer_height_px} format={oneDp} />
      <FieldRow label="Current layer" field={geometry.current_layer} />
      <FieldRow label="Total layers" field={geometry.total_layers} />
    </section>
  );
}

/* --- Material --- */

export function MaterialCard({ material }: { material: MaterialFull }) {
  const ratio = material.material_ratio_percent.value;
  return (
    <section className="panel">
      <header className="panel-header"><h2>Material</h2></header>

      {ratio !== null ? (
        <div className="progress-block">
          <div className="progress-bar">
            <div className="progress-fill progress-fill-warn" style={{ width: `${Math.min(100, ratio)}%` }} />
          </div>
          <span className="progress-label">{ratio.toFixed(1)}% of material used</span>
        </div>
      ) : null}

      <FieldRow label="Material type" field={material.material_type} />
      <FieldRow label="Used" field={material.material_used_kg} format={twoDp} />
      <FieldRow label="Remaining" field={material.material_remaining_kg} format={twoDp} />
      <FieldRow label="Consumption" field={material.consumption_rate_kg_per_hour} format={(v) => v.toFixed(3)} />
      <FieldRow label="Flow" field={material.material_flow_percent} format={oneDp} emphasis />
      <FieldRow label="Nozzle temp" field={material.nozzle_temp_c} format={oneDp} />

      {material.source_note ? <p className="panel-note panel-note-warn">{material.source_note}</p> : null}
    </section>
  );
}

/* --- Robot details --- */

export function RobotDetails({ robot }: { robot: RobotInfo }) {
  const online = robot.robot_online.value === true;
  return (
    <section className="panel">
      <header className="panel-header">
        <h2>Robot Details</h2>
        <span className={`chip-status ${online ? "chip-online" : "chip-offline"}`}>
          <span className={`status-dot status-${online ? "live" : "bad"}`} />
          {online ? "ONLINE" : "OFFLINE"}
        </span>
      </header>

      <FieldRow label="Robot status" field={robot.robot_status} emphasis />
      <FieldRow label="Machine state" field={robot.machine_state} />
      <FieldRow label="Current operation" field={robot.current_operation} emphasis />
      <FieldRow label="Current shift" field={robot.current_shift} />
      <FieldRow label="Current operator" field={robot.current_operator} />

      <h3 className="panel-subhead">Communication</h3>
      <FieldRow label="API" field={robot.communication.api} />
      <FieldRow label="Camera" field={robot.communication.camera} />
      <FieldRow label="Machine link" field={robot.communication.machine_link} />
    </section>
  );
}

/* --- Problems --- */

function ProblemGroup({ title, items }: { title: string; items: Problems["warnings"] }) {
  return (
    <div className="problem-group">
      <h3 className="problem-title">
        {title}
        <span className={`problem-count${items.length ? " problem-count-active" : ""}`}>{items.length}</span>
      </h3>
      {items.length === 0 ? (
        <p className="empty-note">None</p>
      ) : (
        <ul className="alert-list">
          {items.slice(0, 4).map((item, index) => (
            <li key={`${item.message}-${index}`} className={`alert alert-${item.level}`}>
              <span className="alert-message">{item.message}</span>
              <span className="alert-age">{item.seconds_ago.toFixed(0)}s</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ProblemsPanel({ problems }: { problems: Problems }) {
  return (
    <section className="panel">
      <header className="panel-header"><h2>Problems</h2></header>
      <div className="problem-grid">
        <ProblemGroup title="Recent warnings" items={problems.warnings} />
        <ProblemGroup title="Critical alerts" items={problems.critical} />
        <ProblemGroup title="Machine problems" items={problems.machine} />
        <ProblemGroup title="Camera problems" items={problems.camera} />
      </div>
    </section>
  );
}

/* --- Production history + graph --- */

const WINDOW_LABELS: Record<string, string> = {
  "10m": "Last 10 min",
  "1h": "Last 1 hour",
  "8h": "Last 8 hours",
  "24h": "Last 24 hours",
};

function Sparkline({ history }: { history: HistoryResponse }) {
  const points = history.series.filter((p) => p.completed_mm !== null);
  if (points.length < 2) {
    return (
      <div className="graph-empty">
        {history.series_note || "Not enough history yet to draw a graph."}
      </div>
    );
  }

  const values = points.map((p) => p.completed_mm as number);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const width = 100;
  const height = 34;

  const path = points
    .map((p, index) => {
      const x = (index / (points.length - 1)) * width;
      const y = height - (((p.completed_mm as number) - min) / span) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <svg className="graph" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img"
         aria-label="Completed length over the last hour">
      <path className="graph-area" d={`${path} L${width},${height} L0,${height} Z`} />
      <path className="graph-line" d={path} />
    </svg>
  );
}

export function HistoryPanel({ history }: { history: HistoryResponse | null }) {
  return (
    <section className="panel">
      <header className="panel-header">
        <h2>Production History</h2>
        {history ? <ProvenanceBadge provenance={history.series_provenance} /> : null}
      </header>

      {!history ? (
        <p className="empty-note">Loading history…</p>
      ) : (
        <>
          <div className="history-grid">
            {Object.entries(WINDOW_LABELS).map(([key, label]) => {
              const entry = history.windows[key];
              const has = entry && entry.value !== null;
              return (
                <div className="stat-card" key={key} title={!has ? entry?.reason : undefined}>
                  <span className="stat-label">{label}</span>
                  <span className={`stat-value${has ? "" : " stat-value-missing"}`}>
                    {has ? `${(entry.value as number).toFixed(1)} mm` : "—"}
                  </span>
                  <ProvenanceBadge provenance={entry?.provenance ?? "UNAVAILABLE"} />
                </div>
              );
            })}
          </div>

          <h3 className="panel-subhead">Completed length — last hour</h3>
          <Sparkline history={history} />
        </>
      )}
    </section>
  );
}
