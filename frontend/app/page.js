"use client";

import { useState, useEffect, useCallback } from "react";
import PatientSearch from "../components/PatientSearch";
import RiskGauge from "../components/RiskGauge";
import ActivityChart from "../components/ActivityChart";

const API_BASE = "/api";

export default function Dashboard() {
  const [patients, setPatients] = useState([]);
  const [atRiskData, setAtRiskData] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [activity, setActivity] = useState(null);
  const [loading, setLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState(null);
  const [riskFilter, setRiskFilter] = useState(null);

  // ── Filtered patients for table ────────────────────────────
  const filteredPatients = atRiskData?.patients
    ? riskFilter
      ? atRiskData.patients.filter((p) => p.risk_label === riskFilter)
      : atRiskData.patients
    : [];

  // ── Health check ──────────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then(setHealthStatus)
      .catch(() => setHealthStatus({ status: "offline" }));
  }, []);

  // ── Load at-risk patients on mount ────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/patients/at-risk?limit=500`)
      .then((r) => {
        if (!r.ok) throw new Error("API error");
        return r.json();
      })
      .then((data) => {
        // Ensure patients array exists
        if (data && !data.patients) data.patients = [];
        setAtRiskData(data);
      })
      .catch((err) => {
        console.error("Failed to load at-risk data:", err);
        setAtRiskData(null);
      });
  }, []);

  // ── Select a patient ──────────────────────────────────────
  const handleSelectPatient = useCallback(async (patientId) => {
    setSelectedPatient(patientId);
    setLoading(true);
    setPrediction(null);
    setActivity(null);

    try {
      const [predRes, actRes] = await Promise.all([
        fetch(`${API_BASE}/predict-churn`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ patient_id: patientId }),
        }),
        fetch(`${API_BASE}/patient/${patientId}/activity`),
      ]);

      if (predRes.ok) {
        const predData = await predRes.json();
        setPrediction(predData);
      }
      if (actRes.ok) {
        const actData = await actRes.json();
        setActivity(actData);
      }
    } catch (err) {
      console.error("Failed to fetch patient data:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Risk color helper ─────────────────────────────────────
  const riskColor = (label) => {
    if (label === "High") return "var(--risk-high)";
    if (label === "Medium") return "var(--risk-medium)";
    return "var(--risk-low)";
  };

  const riskClass = (label) => (label || "low").toLowerCase();

  return (
    <div className="app-container">
      {/* ── Header ─────────────────────────────────────────── */}
      <header className="header fade-in">
        <div className="header-left">
          <div className="header-icon">🧠</div>
          <div>
            <h1>Patient Churn Prediction</h1>
            <p>ML-powered risk analysis dashboard</p>
          </div>
        </div>
        <div className="status-badge">
          <span
            className={`status-dot ${
              healthStatus?.status === "healthy" ? "" : "error"
            }`}
          />
          {healthStatus?.status === "healthy"
            ? "System Online"
            : healthStatus?.status === "degraded"
            ? "Degraded"
            : "Connecting..."}
        </div>
      </header>

      {/* ── Stats Overview ─────────────────────────────────── */}
      {atRiskData && (
        <div className="stats-grid fade-in fade-in-delay-1">
          <div
            className={`stat-card teal ${riskFilter === null ? "active-filter" : ""}`}
            onClick={() => setRiskFilter(null)}
            style={{ cursor: "pointer" }}
          >
            <div className="stat-label">Total Patients</div>
            <div className="stat-value teal">{atRiskData.total_patients}</div>
          </div>
          <div
            className={`stat-card rose ${riskFilter === "High" ? "active-filter" : ""}`}
            onClick={() => setRiskFilter(riskFilter === "High" ? null : "High")}
            style={{ cursor: "pointer" }}
          >
            <div className="stat-label">High Risk</div>
            <div className="stat-value rose">{atRiskData.high_risk}</div>
          </div>
          <div
            className={`stat-card amber ${riskFilter === "Medium" ? "active-filter" : ""}`}
            onClick={() => setRiskFilter(riskFilter === "Medium" ? null : "Medium")}
            style={{ cursor: "pointer" }}
          >
            <div className="stat-label">Medium Risk</div>
            <div className="stat-value amber">{atRiskData.medium_risk}</div>
          </div>
          <div
            className={`stat-card purple ${riskFilter === "Low" ? "active-filter" : ""}`}
            onClick={() => setRiskFilter(riskFilter === "Low" ? null : "Low")}
            style={{ cursor: "pointer" }}
          >
            <div className="stat-label">Low Risk</div>
            <div className="stat-value purple">{atRiskData.low_risk}</div>
          </div>
        </div>
      )}

      {/* ── Main Content Grid ──────────────────────────────── */}
      <div className="main-grid fade-in fade-in-delay-2">
        {/* Left: Search + Patient List */}
        <div className="search-section">
          <div className="card">
            <div className="card-header">
              <span className="card-title">
                <span className="icon">👤</span> Select Patient
              </span>
            </div>
            <PatientSearch
              onSelect={handleSelectPatient}
              selectedId={selectedPatient}
            />
          </div>
        </div>

        {/* Right: Prediction Result */}
        <div className="card">
          {!selectedPatient && !loading && (
            <div className="empty-state">
              <div className="icon">🔍</div>
              <div className="title">Select a Patient</div>
              <div className="subtitle">
                Choose a patient from the list to view their churn risk
                prediction and activity trends.
              </div>
            </div>
          )}

          {loading && (
            <div className="loading-container">
              <div className="spinner" />
              <div className="loading-text">Analyzing patient data...</div>
            </div>
          )}

          {prediction && !loading && (
            <div className="gauge-section fade-in">
              <RiskGauge
                probability={prediction.churn_probability}
                riskLabel={prediction.risk_label}
              />

              <div
                className={`risk-display ${riskClass(prediction.risk_label)}`}
              >
                {prediction.risk_label} Risk
              </div>

              {/* Key Features */}
              {prediction.key_features && (
                <div style={{ width: "100%" }}>
                  <div
                    className="card-title"
                    style={{ marginBottom: 14, marginTop: 8 }}
                  >
                    <span className="icon">📊</span> Key Indicators
                  </div>
                  <div className="features-grid">
                    <div className="feature-item">
                      <div className="feature-label">Meals Logged</div>
                      <div className="feature-value">
                        {Math.round(prediction.key_features.total_meal_logs)}
                      </div>
                    </div>
                    <div className="feature-item">
                      <div className="feature-label">Weight Logs</div>
                      <div className="feature-value">
                        {Math.round(prediction.key_features.total_weight_logs)}
                      </div>
                    </div>
                    <div className="feature-item">
                      <div className="feature-label">Exercise Logs</div>
                      <div className="feature-value">
                        {Math.round(
                          prediction.key_features.total_exercise_logs
                        )}
                      </div>
                    </div>
                    <div className="feature-item">
                      <div className="feature-label">Meal Compliance</div>
                      <div className="feature-value">
                        {(
                          prediction.key_features.meal_compliance_rate * 100
                        ).toFixed(0)}
                        %
                      </div>
                    </div>
                    <div className="feature-item">
                      <div className="feature-label">Completion Rate</div>
                      <div className="feature-value">
                        {(
                          prediction.key_features.collection_completion_rate *
                          100
                        ).toFixed(0)}
                        %
                      </div>
                    </div>
                    <div className="feature-item">
                      <div className="feature-label">Days Enrolled</div>
                      <div className="feature-value">
                        {Math.round(
                          prediction.key_features.enrollment_age_days
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Activity Chart ─────────────────────────────────── */}
      {activity && !loading && (
        <div className="card chart-section fade-in fade-in-delay-3">
          <div className="card-header">
            <span className="card-title">
              <span className="icon">📈</span> Activity Timeline (Last 30 Days)
            </span>
            {activity.patient_name && (
              <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
                {activity.patient_name}
              </span>
            )}
          </div>
          <div className="chart-container">
            <ActivityChart data={activity.activity_timeline} />
          </div>
        </div>
      )}

      {/* ── At-Risk Patients Table ─────────────────────────── */}
      {atRiskData && atRiskData.patients && atRiskData.patients.length > 0 && (
        <div className="card fade-in fade-in-delay-4">
          <div className="card-header">
            <span className="card-title">
              <span className="icon">⚠️</span>
              {riskFilter
                ? `${riskFilter} Risk Patients`
                : "All Patients — Risk Ranking"}
            </span>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
              {filteredPatients.length} patients
              {riskFilter && (
                <button
                  onClick={() => setRiskFilter(null)}
                  style={{
                    marginLeft: 8, background: "var(--glass-border)",
                    border: "none", color: "var(--text-secondary)",
                    padding: "2px 8px", borderRadius: 4, cursor: "pointer",
                    fontSize: 11,
                  }}
                >
                  Clear Filter ✕
                </button>
              )}
            </span>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="patients-table">
              <thead>
                <tr>
                  <th>Patient Name</th>
                  <th>Patient ID</th>
                  <th>Churn Probability</th>
                  <th>Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {filteredPatients.slice(0, 50).map((p) => (
                  <tr
                    key={p.patient_id}
                    onClick={() => handleSelectPatient(p.patient_id)}
                    style={{
                      background:
                        selectedPatient === p.patient_id
                          ? "var(--accent-teal-glow)"
                          : undefined,
                    }}
                  >
                    <td style={{ fontWeight: 500 }}>
                      {p.patient_name || "—"}
                    </td>
                    <td style={{ fontFamily: "'Courier New', monospace", fontSize: 12 }}>
                      {p.patient_id.substring(0, 8)}...
                    </td>
                    <td>
                      <div className="prob-bar">
                        <div className="prob-bar-track">
                          <div
                            className="prob-bar-fill"
                            style={{
                              width: `${p.churn_probability * 100}%`,
                              background: riskColor(p.risk_label),
                            }}
                          />
                        </div>
                        <span
                          className="prob-bar-value"
                          style={{ color: riskColor(p.risk_label) }}
                        >
                          {(p.churn_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td>
                      <span className={`risk-badge ${riskClass(p.risk_label)}`}>
                        {p.risk_label}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
