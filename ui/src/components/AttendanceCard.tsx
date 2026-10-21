import { useEffect, useState } from "react";
import {
  checkInAttendance,
  checkOutAttendance,
  getAttendanceList,
  getOperators,
  createOperator,
  getWallProduction,
  recordWallProduction,
  correctWallProduction,
} from "../api";
import type { AttendanceRecord, WallProductionRecord, WallProductionTotal } from "../types";

export function AttendanceCard() {
  const [activeTab, setActiveTab] = useState<"attendance" | "production">("attendance");
  const [operators, setOperators] = useState<any[]>([]);
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [selectedOp, setSelectedOp] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // New Employee State
  const [isAdding, setIsAdding] = useState(false);
  const [newName, setNewName] = useState("");
  const [newCode, setNewCode] = useState("");
  const [newRole, setNewRole] = useState("operator");

  // Wall Production State
  const [prodScope, setProdScope] = useState<"today" | "this_week" | "all_time">("today");
  const [wallTotals, setWallTotals] = useState<WallProductionTotal | null>(null);
  const [wallRecords, setWallRecords] = useState<WallProductionRecord[]>([]);
  const [prodOp, setProdOp] = useState<string>("");
  const [prodQuantity, setProdQuantity] = useState<number>(1);
  const [prodJobId, setProdJobId] = useState<string>("");
  const [prodNotes, setProdNotes] = useState<string>("");
  const [isSubmittingProd, setIsSubmittingProd] = useState(false);

  // Correction Modal State
  const [correctingRecord, setCorrectingRecord] = useState<WallProductionRecord | null>(null);
  const [correctedQty, setCorrectedQty] = useState<number>(1);
  const [correctionReason, setCorrectionReason] = useState<string>("");
  const [isSubmittingCorrection, setIsSubmittingCorrection] = useState(false);

  useEffect(() => {
    fetchOperators();
    fetchRecords();
    fetchWallData();

    // Refresh records periodically
    const timer = setInterval(() => {
      fetchRecords();
      fetchWallData();
    }, 60000);

    const handleAttendanceUpdated = () => {
      fetchRecords();
      fetchOperators();
      fetchWallData();
    };
    window.addEventListener("kelvin-attendance-updated", handleAttendanceUpdated);

    return () => {
      clearInterval(timer);
      window.removeEventListener("kelvin-attendance-updated", handleAttendanceUpdated);
    };
  }, [prodScope]);

  const fetchOperators = async () => {
    try {
      const res = await getOperators();
      setOperators(res.operators || []);
    } catch (err) {
      console.error("Failed to fetch operators:", err);
    }
  };

  const fetchRecords = async () => {
    try {
      const data = await getAttendanceList(); // Backend defaults to local today
      setRecords(data);
    } catch (err) {
      console.error("Failed to fetch attendance:", err);
    }
  };

  const fetchWallData = async () => {
    try {
      const data = await getWallProduction({ scope: prodScope, include_records: true });
      setWallTotals(data);
      setWallRecords(data.records || []);
    } catch (err) {
      console.error("Failed to fetch wall production:", err);
    }
  };

  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (val === "ADD_NEW") {
      setIsAdding(true);
      setSelectedOp("");
      setError(null);
    } else {
      setSelectedOp(val);
      setIsAdding(false);
    }
  };

  const handleAddEmployee = async () => {
    setError(null);
    setSuccessMsg(null);
    if (!newName.trim()) {
      setError("Employee name is required.");
      return;
    }
    try {
      const res = await createOperator({
        name: newName,
        employee_code: newCode.trim() || undefined,
        role: newRole,
        active: 1,
      });
      await fetchOperators();
      setIsAdding(false);
      setSelectedOp(res.operator_id);
      setNewName("");
      setNewCode("");
      setSuccessMsg(`Employee "${newName}" registered successfully.`);
    } catch (err: any) {
      setError(err.message || "Failed to create employee.");
    }
  };

  const handleCancelAdd = () => {
    setIsAdding(false);
    setSelectedOp("");
    setNewName("");
    setNewCode("");
    setError(null);
  };

  const handleCheckIn = async () => {
    setError(null);
    setSuccessMsg(null);
    try {
      if (!selectedOp) {
        setError("Please select an employee.");
        return;
      }
      await checkInAttendance(selectedOp, undefined);
      setSelectedOp("");
      fetchRecords();
      setSuccessMsg("Shift started successfully.");
    } catch (err: any) {
      setError(err.message || "Failed to check in.");
    }
  };

  const handleCheckOut = async (record: AttendanceRecord) => {
    setError(null);
    setSuccessMsg(null);
    try {
      await checkOutAttendance(record.operator_id, undefined);
      fetchRecords();
      setSuccessMsg(`Shift ended for ${record.employee_name}.`);
    } catch (err: any) {
      setError(err.message || "Failed to check out.");
    }
  };

  const handleRecordWallProduction = async () => {
    setError(null);
    setSuccessMsg(null);
    if (!prodOp) {
      setError("Please select an operator to log completed walls.");
      return;
    }
    const qty = Number(prodQuantity);
    if (!Number.isInteger(qty) || qty <= 0) {
      setError("Wall quantity must be a positive whole integer greater than zero.");
      return;
    }
    setIsSubmittingProd(true);
    try {
      const res = await recordWallProduction({
        operator_id: prodOp,
        quantity: qty,
        job_id: prodJobId.trim() || undefined,
        notes: prodNotes.trim() || undefined,
      });
      setSuccessMsg(`Recorded ${qty} completed wall(s) for ${res.record.operator_name}.`);
      setProdQuantity(1);
      setProdNotes("");
      await fetchWallData();
    } catch (err: any) {
      setError(err.message || "Failed to record wall production.");
    } finally {
      setIsSubmittingProd(false);
    }
  };

  const handleOpenCorrection = (rec: WallProductionRecord) => {
    setCorrectingRecord(rec);
    setCorrectedQty(rec.quantity);
    setCorrectionReason("");
    setError(null);
    setSuccessMsg(null);
  };

  const handleSubmitCorrection = async () => {
    if (!correctingRecord) return;
    setError(null);
    setSuccessMsg(null);
    const qty = Number(correctedQty);
    if (!Number.isInteger(qty) || qty <= 0) {
      setError("Corrected quantity must be a positive integer greater than zero.");
      return;
    }
    setIsSubmittingCorrection(true);
    try {
      await correctWallProduction({
        record_id: correctingRecord.record_id,
        corrected_quantity: qty,
        reason: correctionReason.trim() || "Manual adjustment via dashboard",
      });
      setSuccessMsg(`Record ${correctingRecord.record_id} corrected to ${qty} walls. Total updated.`);
      setCorrectingRecord(null);
      await fetchWallData();
    } catch (err: any) {
      setError(err.message || "Failed to correct wall record.");
    } finally {
      setIsSubmittingCorrection(false);
    }
  };

  const formatTime = (ts: number | null) => {
    if (!ts) return "--";
    return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const formatDateTime = (ts: number | null) => {
    if (!ts) return "--";
    const d = new Date(ts * 1000);
    return `${d.toLocaleDateString("en-GB", { day: "2-digit", month: "short" })} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  };

  return (
    <div className="panel card-attendance">
      <div
        className="panel-header"
        style={{
          marginBottom: "12px",
          borderBottom: "1px solid var(--border)",
          paddingBottom: "10px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "8px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <h2 style={{ display: "flex", alignItems: "center", gap: "8px", margin: 0 }}>
            <span>{activeTab === "attendance" ? "📋" : "🧱"}</span>
            <span>{activeTab === "attendance" ? "Employee Attendance" : "Wall Production Tracking"}</span>
          </h2>
          <span className="badge badge-real">SQLITE RECORD</span>
        </div>

        {/* Tab Toggle */}
        <div style={{ display: "flex", background: "#0d1117", padding: "3px", borderRadius: "6px", border: "1px solid var(--border)", gap: "4px" }}>
          <button
            type="button"
            onClick={() => setActiveTab("attendance")}
            style={{
              padding: "4px 10px",
              fontSize: "11px",
              fontWeight: 600,
              borderRadius: "4px",
              border: "none",
              cursor: "pointer",
              background: activeTab === "attendance" ? "#212836" : "transparent",
              color: activeTab === "attendance" ? "#6aa8ff" : "#8b949e",
            }}
          >
            📋 Shifts
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("production")}
            style={{
              padding: "4px 10px",
              fontSize: "11px",
              fontWeight: 600,
              borderRadius: "4px",
              border: "none",
              cursor: "pointer",
              background: activeTab === "production" ? "#212836" : "transparent",
              color: activeTab === "production" ? "#4ade80" : "#8b949e",
            }}
          >
            🧱 Walls ({wallTotals?.total_quantity ?? 0})
          </button>
        </div>
      </div>

      {error && (
        <div
          style={{
            background: "rgba(239, 68, 68, 0.12)",
            border: "1px solid rgba(239, 68, 68, 0.35)",
            color: "#fca5a5",
            padding: "8px 12px",
            borderRadius: "6px",
            marginBottom: "12px",
            fontSize: "12px",
          }}
        >
          {error}
        </div>
      )}

      {successMsg && (
        <div
          style={{
            background: "rgba(74, 222, 128, 0.12)",
            border: "1px solid rgba(74, 222, 128, 0.35)",
            color: "#86efac",
            padding: "8px 12px",
            borderRadius: "6px",
            marginBottom: "12px",
            fontSize: "12px",
          }}
        >
          {successMsg}
        </div>
      )}

      {/* ============================================================== */}
      {/* TAB 1: ATTENDANCE & SHIFTS                                     */}
      {/* ============================================================== */}
      {activeTab === "attendance" && (
        <div>
          {!isAdding ? (
            <div style={{ marginBottom: "16px", display: "flex", flexDirection: "column", gap: "8px" }}>
              <label style={{ fontSize: "12px", fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                Select Operator
              </label>
              <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                <select
                  value={selectedOp}
                  onChange={handleSelectChange}
                  style={{
                    flex: "1 1 200px",
                    padding: "8px 12px",
                    background: "#0d1117",
                    color: "#e6ecf3",
                    border: "1px solid var(--border)",
                    borderRadius: "6px",
                    fontSize: "13px",
                    outline: "none",
                  }}
                >
                  <option value="">-- Choose Operator --</option>
                  {operators.map((op) => (
                    <option key={op.operator_id} value={op.operator_id}>
                      {op.name} {op.employee_code ? `(${op.employee_code})` : ""}
                    </option>
                  ))}
                  <option disabled>-----------------</option>
                  <option value="ADD_NEW">+ Add New Employee</option>
                </select>

                <button
                  type="button"
                  onClick={handleCheckIn}
                  disabled={!selectedOp}
                  style={{
                    padding: "8px 16px",
                    background: selectedOp ? "#2563eb" : "#1e293b",
                    color: selectedOp ? "#ffffff" : "#64748b",
                    border: "none",
                    cursor: selectedOp ? "pointer" : "not-allowed",
                    borderRadius: "6px",
                    fontSize: "12px",
                    fontWeight: 700,
                    letterSpacing: "0.5px",
                    transition: "background 0.2s",
                  }}
                >
                  START SHIFT
                </button>
              </div>
            </div>
          ) : (
            <div
              style={{
                marginBottom: "16px",
                display: "flex",
                flexDirection: "column",
                gap: "10px",
                padding: "14px",
                background: "#0d1117",
                borderRadius: "8px",
                border: "1px solid var(--border)",
              }}
            >
              <h4 style={{ margin: "0 0 4px 0", color: "#6aa8ff", fontSize: "13px", fontWeight: 700 }}>
                REGISTER NEW EMPLOYEE
              </h4>

              <div>
                <label style={{ fontSize: "11px", color: "var(--muted)", display: "block", marginBottom: "4px" }}>
                  Full Name
                </label>
                <input
                  type="text"
                  value={newName}
                  placeholder="e.g. Anand Sharma"
                  onChange={(e) => setNewName(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "7px 10px",
                    background: "#161b22",
                    color: "white",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                    fontSize: "13px",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: "11px", color: "var(--muted)", display: "block", marginBottom: "4px" }}>
                  Employee Code (Optional)
                </label>
                <input
                  type="text"
                  value={newCode}
                  placeholder="e.g. OP-104"
                  onChange={(e) => setNewCode(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "7px 10px",
                    background: "#161b22",
                    color: "white",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                    fontSize: "13px",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: "11px", color: "var(--muted)", display: "block", marginBottom: "4px" }}>Role</label>
                <select
                  value={newRole}
                  onChange={(e) => setNewRole(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "7px 10px",
                    background: "#161b22",
                    color: "white",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                    fontSize: "13px",
                  }}
                >
                  <option value="operator">Operator</option>
                  <option value="technician">Technician</option>
                  <option value="supervisor">Supervisor</option>
                </select>
              </div>

              <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
                <button
                  type="button"
                  onClick={handleCancelAdd}
                  style={{
                    flex: 1,
                    padding: "7px 12px",
                    background: "#21262d",
                    color: "#c9d1d9",
                    border: "1px solid var(--border)",
                    cursor: "pointer",
                    borderRadius: "4px",
                    fontSize: "12px",
                    fontWeight: 600,
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleAddEmployee}
                  style={{
                    flex: 1,
                    padding: "7px 12px",
                    background: "#16a34a",
                    color: "white",
                    border: "none",
                    cursor: "pointer",
                    borderRadius: "4px",
                    fontSize: "12px",
                    fontWeight: 700,
                  }}
                >
                  Add Employee
                </button>
              </div>
            </div>
          )}

          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <h4 style={{ margin: "0", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--muted)" }}>
                Today's Attendance Roster
              </h4>
              <span style={{ fontSize: "11px", color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
                {new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })}
              </span>
            </div>

            <div style={{ overflowX: "auto", borderRadius: "6px", border: "1px solid var(--border)" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", textAlign: "left" }}>
                <thead>
                  <tr style={{ background: "#0d1117", borderBottom: "1px solid var(--border)", color: "var(--muted)" }}>
                    <th style={{ padding: "8px 10px" }}>Employee</th>
                    <th style={{ padding: "8px 10px" }}>Start</th>
                    <th style={{ padding: "8px 10px" }}>End</th>
                    <th style={{ padding: "8px 10px" }}>Worked</th>
                    <th style={{ padding: "8px 10px" }}>Overtime</th>
                    <th style={{ padding: "8px 10px" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {records.length === 0 ? (
                    <tr>
                      <td colSpan={6} style={{ textAlign: "center", padding: "14px", color: "var(--muted)" }}>
                        No attendance records for today.
                      </td>
                    </tr>
                  ) : (
                    records.map((r) => (
                      <tr key={r.id} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                        <td style={{ padding: "8px 10px", fontWeight: 600, color: "#e6ecf3" }}>{r.employee_name}</td>
                        <td style={{ padding: "8px 10px", color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
                          {formatTime(r.shift_start)}
                        </td>
                        <td style={{ padding: "8px 10px", color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
                          {formatTime(r.shift_end)}
                        </td>
                        <td style={{ padding: "8px 10px", fontVariantNumeric: "tabular-nums" }}>
                          <div style={{ fontWeight: 600, color: "#e6ecf3" }}>{r.worked_display || "0m"}</div>
                          {(r.overtime_seconds || 0) > 0 && (
                            <div style={{ fontSize: "10px", color: "#fbbf24", fontWeight: 600 }}>
                              +{r.overtime_display} OT
                            </div>
                          )}
                        </td>
                        <td
                          style={{
                            padding: "8px 10px",
                            color: (r.overtime_seconds || 0) > 0 ? "#fbbf24" : "var(--muted)",
                            fontVariantNumeric: "tabular-nums",
                          }}
                        >
                          {r.overtime_display || "0m"}
                        </td>
                        <td style={{ padding: "8px 10px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            <span
                              style={{
                                padding: "2px 7px",
                                borderRadius: "4px",
                                background: r.status === "ACTIVE" ? "rgba(74, 222, 128, 0.15)" : "rgba(148, 163, 184, 0.15)",
                                color: r.status === "ACTIVE" ? "#4ade80" : "#94a3b8",
                                border: `1px solid ${
                                  r.status === "ACTIVE" ? "rgba(74, 222, 128, 0.35)" : "rgba(148, 163, 184, 0.25)"
                                }`,
                                fontSize: "11px",
                                fontWeight: 700,
                                display: "inline-flex",
                                alignItems: "center",
                                gap: "4px",
                              }}
                            >
                              {r.status === "ACTIVE" ? "🟢 ACTIVE" : "COMPLETED"}
                            </span>
                            {r.status === "ACTIVE" && (
                              <button
                                type="button"
                                onClick={() => handleCheckOut(r)}
                                style={{
                                  padding: "2px 7px",
                                  background: "rgba(248, 113, 113, 0.15)",
                                  color: "#f87171",
                                  border: "1px solid rgba(248, 113, 113, 0.35)",
                                  cursor: "pointer",
                                  borderRadius: "4px",
                                  fontSize: "10px",
                                  fontWeight: 700,
                                }}
                                title="Clock out this operator"
                              >
                                END SHIFT
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ============================================================== */}
      {/* TAB 2: WALL PRODUCTION TRACKING                                */}
      {/* ============================================================== */}
      {activeTab === "production" && (
        <div>
          {/* Production Summary Metric Banner */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: "10px",
              marginBottom: "14px",
            }}
          >
            <div
              style={{
                background: "#0d1117",
                border: "1px solid rgba(74, 222, 128, 0.3)",
                borderRadius: "8px",
                padding: "10px 14px",
                display: "flex",
                flexDirection: "column",
                gap: "4px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "11px", color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                  Completed Walls
                </span>
                <div style={{ display: "flex", gap: "4px" }}>
                  {(["today", "this_week", "all_time"] as const).map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setProdScope(s)}
                      style={{
                        fontSize: "9px",
                        padding: "2px 6px",
                        borderRadius: "3px",
                        border: "none",
                        cursor: "pointer",
                        background: prodScope === s ? "#16a34a" : "#21262d",
                        color: prodScope === s ? "white" : "#8b949e",
                        fontWeight: prodScope === s ? 700 : 500,
                        textTransform: "uppercase",
                      }}
                    >
                      {s === "this_week" ? "Week" : s === "all_time" ? "All" : "Today"}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: "8px" }}>
                <span style={{ fontSize: "24px", fontWeight: 800, color: "#4ade80", fontVariantNumeric: "tabular-nums" }}>
                  {wallTotals?.total_quantity ?? 0}
                </span>
                <span style={{ fontSize: "12px", color: "#8a99ad" }}>
                  walls printed across {wallTotals?.record_count ?? 0} logs
                </span>
              </div>
            </div>

            {/* Operator distribution preview */}
            <div
              style={{
                background: "#0d1117",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                padding: "10px 14px",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
              }}
            >
              <span style={{ fontSize: "11px", color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>
                Operator Output ({prodScope.replace("_", " ")})
              </span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {wallTotals?.by_operator && wallTotals.by_operator.length > 0 ? (
                  wallTotals.by_operator.map((op) => (
                    <span
                      key={op.operator_id}
                      style={{
                        fontSize: "11px",
                        background: "#161b22",
                        border: "1px solid #30363d",
                        padding: "2px 8px",
                        borderRadius: "4px",
                        color: "#e6ecf3",
                      }}
                    >
                      <strong>{op.operator_name}:</strong> {op.total_quantity}
                    </span>
                  ))
                ) : (
                  <span style={{ fontSize: "11px", color: "var(--muted)", fontStyle: "italic" }}>
                    No walls recorded for this period yet.
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Quick Record Form */}
          <div
            style={{
              background: "#0d1117",
              border: "1px solid var(--border)",
              borderRadius: "8px",
              padding: "12px",
              marginBottom: "16px",
            }}
          >
            <h4 style={{ margin: "0 0 10px 0", fontSize: "12px", color: "#6aa8ff", textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Log Completed Wall Units
            </h4>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "8px", marginBottom: "8px" }}>
              <div>
                <label style={{ fontSize: "10px", color: "var(--muted)", display: "block", marginBottom: "3px" }}>Operator</label>
                <select
                  value={prodOp}
                  onChange={(e) => setProdOp(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    background: "#161b22",
                    color: "#e6ecf3",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                    fontSize: "12px",
                  }}
                >
                  <option value="">-- Choose Operator --</option>
                  {operators.map((op) => (
                    <option key={op.operator_id} value={op.name}>
                      {op.name} {op.employee_code ? `(${op.employee_code})` : ""}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ fontSize: "10px", color: "var(--muted)", display: "block", marginBottom: "3px" }}>Walls Completed</label>
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={prodQuantity}
                  onChange={(e) => setProdQuantity(Math.max(1, parseInt(e.target.value, 10) || 1))}
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    background: "#161b22",
                    color: "#e6ecf3",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                    fontSize: "12px",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: "10px", color: "var(--muted)", display: "block", marginBottom: "3px" }}>Job ID (Optional)</label>
                <input
                  type="text"
                  placeholder="e.g. WALL-JOB-01"
                  value={prodJobId}
                  onChange={(e) => setProdJobId(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    background: "#161b22",
                    color: "#e6ecf3",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                    fontSize: "12px",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: "10px", color: "var(--muted)", display: "block", marginBottom: "3px" }}>Notes (Optional)</label>
                <input
                  type="text"
                  placeholder="e.g. Standard 3m wall section"
                  value={prodNotes}
                  onChange={(e) => setProdNotes(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    background: "#161b22",
                    color: "#e6ecf3",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                    fontSize: "12px",
                  }}
                />
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button
                type="button"
                onClick={handleRecordWallProduction}
                disabled={!prodOp || isSubmittingProd}
                style={{
                  padding: "6px 14px",
                  background: prodOp ? "#16a34a" : "#1e293b",
                  color: prodOp ? "white" : "#64748b",
                  border: "none",
                  cursor: prodOp ? "pointer" : "not-allowed",
                  borderRadius: "4px",
                  fontSize: "12px",
                  fontWeight: 700,
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <span>🧱</span>
                <span>{isSubmittingProd ? "Logging..." : "Log Completed Walls"}</span>
              </button>
            </div>
          </div>

          {/* Wall Production Records Table */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <h4 style={{ margin: 0, fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--muted)" }}>
                Wall Production Log ({prodScope.replace("_", " ")})
              </h4>
              <button
                type="button"
                onClick={fetchWallData}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "#6aa8ff",
                  cursor: "pointer",
                  fontSize: "11px",
                }}
              >
                🔄 Refresh
              </button>
            </div>

            <div style={{ overflowX: "auto", borderRadius: "6px", border: "1px solid var(--border)" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", textAlign: "left" }}>
                <thead>
                  <tr style={{ background: "#0d1117", borderBottom: "1px solid var(--border)", color: "var(--muted)" }}>
                    <th style={{ padding: "8px 10px" }}>Record ID</th>
                    <th style={{ padding: "8px 10px" }}>Operator</th>
                    <th style={{ padding: "8px 10px" }}>Quantity</th>
                    <th style={{ padding: "8px 10px" }}>Logged At</th>
                    <th style={{ padding: "8px 10px" }}>Job / Shift</th>
                    <th style={{ padding: "8px 10px" }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {wallRecords.length === 0 ? (
                    <tr>
                      <td colSpan={6} style={{ textAlign: "center", padding: "14px", color: "var(--muted)" }}>
                        No wall production records found for this period.
                      </td>
                    </tr>
                  ) : (
                    wallRecords.map((r) => (
                      <tr key={r.record_id} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                        <td style={{ padding: "8px 10px", color: "#8a99ad", fontFamily: "monospace", fontSize: "11px" }}>
                          {r.record_id}
                        </td>
                        <td style={{ padding: "8px 10px", fontWeight: 600, color: "#e6ecf3" }}>
                          {r.operator_name} {r.employee_code ? `(${r.employee_code})` : ""}
                        </td>
                        <td style={{ padding: "8px 10px", fontVariantNumeric: "tabular-nums" }}>
                          <span
                            style={{
                              background: "rgba(74, 222, 128, 0.15)",
                              color: "#4ade80",
                              padding: "2px 7px",
                              borderRadius: "4px",
                              fontWeight: 700,
                              border: "1px solid rgba(74, 222, 128, 0.3)",
                            }}
                          >
                            {r.quantity} {r.quantity === 1 ? "wall" : "walls"}
                          </span>
                        </td>
                        <td style={{ padding: "8px 10px", color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
                          {formatDateTime(r.created_at)}
                        </td>
                        <td style={{ padding: "8px 10px", color: "var(--muted)", fontSize: "11px" }}>
                          {r.job_id || "Standard"} {r.shift_id ? `(Shift #${r.shift_id})` : ""}
                          {r.notes ? ` • ${r.notes}` : ""}
                        </td>
                        <td style={{ padding: "8px 10px" }}>
                          <button
                            type="button"
                            onClick={() => handleOpenCorrection(r)}
                            style={{
                              padding: "2px 7px",
                              background: "rgba(251, 191, 36, 0.12)",
                              color: "#fbbf24",
                              border: "1px solid rgba(251, 191, 36, 0.35)",
                              borderRadius: "4px",
                              cursor: "pointer",
                              fontSize: "11px",
                              fontWeight: 600,
                            }}
                            title="Correct this wall entry"
                          >
                            ✏️ Correct
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Audit Correction Modal */}
          {correctingRecord && (
            <div
              style={{
                position: "fixed",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: "rgba(0, 0, 0, 0.8)",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                zIndex: 9999,
                padding: "16px",
              }}
              onClick={() => setCorrectingRecord(null)}
            >
              <div
                style={{
                  background: "#161b22",
                  border: "1px solid var(--border)",
                  borderRadius: "8px",
                  padding: "18px",
                  width: "100%",
                  maxWidth: "420px",
                  boxShadow: "0 8px 24px rgba(0,0,0,0.6)",
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                  <h3 style={{ margin: 0, fontSize: "14px", color: "#fbbf24", display: "flex", alignItems: "center", gap: "6px" }}>
                    <span>✏️</span>
                    <span>Audit Correction for {correctingRecord.record_id}</span>
                  </h3>
                  <button
                    type="button"
                    onClick={() => setCorrectingRecord(null)}
                    style={{ background: "transparent", border: "none", color: "#8b949e", cursor: "pointer", fontSize: "16px" }}
                  >
                    ✕
                  </button>
                </div>

                <div
                  style={{
                    background: "#0d1117",
                    padding: "10px",
                    borderRadius: "6px",
                    marginBottom: "12px",
                    fontSize: "12px",
                    color: "#8a99ad",
                  }}
                >
                  <div>
                    <strong>Operator:</strong> {correctingRecord.operator_name}
                  </div>
                  <div>
                    <strong>Original Logged Count:</strong> {correctingRecord.quantity} walls
                  </div>
                  <div>
                    <strong>Original Date:</strong> {formatDateTime(correctingRecord.created_at)}
                  </div>
                </div>

                <div style={{ marginBottom: "12px" }}>
                  <label style={{ fontSize: "11px", color: "var(--muted)", display: "block", marginBottom: "4px" }}>
                    New Corrected Quantity (Positive Integer)
                  </label>
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={correctedQty}
                    onChange={(e) => setCorrectedQty(Math.max(1, parseInt(e.target.value, 10) || 1))}
                    style={{
                      width: "100%",
                      padding: "8px 10px",
                      background: "#0d1117",
                      color: "white",
                      border: "1px solid var(--border)",
                      borderRadius: "4px",
                      fontSize: "13px",
                    }}
                  />
                </div>

                <div style={{ marginBottom: "16px" }}>
                  <label style={{ fontSize: "11px", color: "var(--muted)", display: "block", marginBottom: "4px" }}>
                    Reason for Correction (Audit Log)
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Recounted by supervisor; 1 wall was incomplete"
                    value={correctionReason}
                    onChange={(e) => setCorrectionReason(e.target.value)}
                    style={{
                      width: "100%",
                      padding: "8px 10px",
                      background: "#0d1117",
                      color: "white",
                      border: "1px solid var(--border)",
                      borderRadius: "4px",
                      fontSize: "13px",
                    }}
                  />
                </div>

                <p style={{ fontSize: "10px", color: "#8b949e", margin: "0 0 14px", fontStyle: "italic" }}>
                  🔒 Audit Compliance: The original record will be marked corrected and preserved in the audit trail. Totals
                  will accurately reflect the new count without duplicate inflation.
                </p>

                <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
                  <button
                    type="button"
                    onClick={() => setCorrectingRecord(null)}
                    style={{
                      padding: "6px 12px",
                      background: "#21262d",
                      color: "#c9d1d9",
                      border: "1px solid var(--border)",
                      borderRadius: "4px",
                      cursor: "pointer",
                      fontSize: "12px",
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleSubmitCorrection}
                    disabled={isSubmittingCorrection}
                    style={{
                      padding: "6px 14px",
                      background: "#d97706",
                      color: "white",
                      border: "none",
                      borderRadius: "4px",
                      cursor: "pointer",
                      fontSize: "12px",
                      fontWeight: 700,
                    }}
                  >
                    {isSubmittingCorrection ? "Saving..." : "Save Correction"}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
