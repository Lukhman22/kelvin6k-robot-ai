/**
 * Renders one machine value together with its provenance badge.
 *
 * REAL        measured by the camera/AI pipeline, or entered for the job
 * MOCK        produced by the built-in simulator -- NOT the real machine
 * UNAVAILABLE not connected / not measured -- never replaced with a guess
 */

import type { Field as FieldType, Provenance } from "../types";

export function ProvenanceBadge({ provenance }: { provenance: Provenance }) {
  return (
    <span className={`badge badge-${provenance.toLowerCase()}`} title={provenanceHint(provenance)}>
      {provenance}
    </span>
  );
}

function provenanceHint(provenance: Provenance): string {
  switch (provenance) {
    case "REAL":
      return "Measured from the camera/AI pipeline or the tracked print job.";
    case "MOCK":
      return "Simulated value from the built-in simulator -- not the real machine.";
    default:
      return "Not connected or not measured. No value is being guessed.";
  }
}

interface Props<T> {
  label: string;
  field: FieldType<T>;
  format?: (value: T) => string;
  emphasis?: boolean;
  suppressReason?: boolean;
}

export function FieldRow<T>({ label, field, format, emphasis, suppressReason }: Props<T>) {
  const hasValue = field.value !== null && field.value !== undefined;
  const display = hasValue
    ? `${format ? format(field.value as T) : String(field.value)}${field.unit ? ` ${field.unit}` : ""}`
    : "—";

  return (
    <div className={`field-row${emphasis ? " field-row-emphasis" : ""}`}>
      <span className="field-label">{label}</span>
      <span className="field-value-group">
        <span className={`field-value${hasValue ? "" : " field-value-missing"}`}>{display}</span>
        <ProvenanceBadge provenance={field.provenance} />
      </span>
      {!hasValue && field.reason && !suppressReason ? <p className="field-reason">{field.reason}</p> : null}
    </div>
  );
}
