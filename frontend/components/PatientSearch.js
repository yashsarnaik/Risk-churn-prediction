"use client";

import { useState, useEffect, useRef } from "react";

const API_BASE = "/api";

export default function PatientSearch({ onSelect, selectedId }) {
  const [query, setQuery] = useState("");
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef(null);

  // ── Fetch patients on query change (debounced) ────────────
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(() => {
      setLoading(true);
      fetch(`${API_BASE}/patients/search?q=${encodeURIComponent(query)}&limit=20`)
        .then((r) => r.json())
        .then((data) => {
          setPatients(data);
          setLoading(false);
        })
        .catch(() => {
          setPatients([]);
          setLoading(false);
        });
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query]);

  return (
    <div>
      <div className="search-input-wrapper">
        <span className="search-icon">🔍</span>
        <input
          type="text"
          className="search-input"
          placeholder="Search by name or patient ID..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          id="patient-search-input"
        />
      </div>

      <div className="patient-list" style={{ marginTop: 14 }}>
        {loading && (
          <div style={{ textAlign: "center", padding: 20, color: "var(--text-muted)", fontSize: 13 }}>
            Searching...
          </div>
        )}

        {!loading && patients.length === 0 && (
          <div style={{ textAlign: "center", padding: 20, color: "var(--text-muted)", fontSize: 13 }}>
            No patients found
          </div>
        )}

        {!loading &&
          patients.map((p) => (
            <div
              key={p.patient_id}
              className={`patient-item ${selectedId === p.patient_id ? "active" : ""}`}
              onClick={() => onSelect(p.patient_id)}
              id={`patient-item-${p.patient_id.substring(0, 8)}`}
            >
              <div className="patient-item-info">
                <span className="patient-item-name">
                  {p.first_name || p.last_name
                    ? `${p.first_name || ""} ${p.last_name || ""}`.trim()
                    : "Unnamed Patient"}
                </span>
                <span className="patient-item-id">
                  {p.patient_id.substring(0, 12)}...
                </span>
              </div>
              {p.created_at && (
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  {p.created_at}
                </span>
              )}
            </div>
          ))}
      </div>
    </div>
  );
}
