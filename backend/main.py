"""
FastAPI Application — Patient Churn Prediction API
===================================================
Serves churn predictions from pre-computed synthetic demo data.
Optimized for showcasing to stakeholders.
"""

import json
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .config import settings
from .schemas import (
    PredictChurnRequest,
    ChurnPredictionResponse,
    PatientsAtRiskResponse,
    PatientRiskItem,
    PatientActivityResponse,
    ActivityLogItem,
    HealthResponse,
    PatientSearchItem,
)

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "models"

# ── App Setup ────────────────────────────────────────────────────
app = FastAPI(
    title="Patient Churn Prediction API",
    description="ML-powered API to predict patient churn risk",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cached Data ──────────────────────────────────────────────────
_predictions_cache = None
_timelines_cache = None
_patient_names = None
_model_loaded = False


def load_caches():
    """Load pre-computed predictions and timelines from JSON."""
    global _predictions_cache, _timelines_cache, _patient_names, _model_loaded

    try:
        pred_file = DATA_DIR / "predictions_cache.json"
        timeline_file = DATA_DIR / "activity_timelines.json"
        names_file = DATA_DIR / "patient_names.json"

        if pred_file.exists():
            with open(pred_file, "r", encoding="utf-8") as f:
                _predictions_cache = json.load(f)

        if timeline_file.exists():
            with open(timeline_file, "r", encoding="utf-8") as f:
                _timelines_cache = json.load(f)

        if names_file.exists():
            with open(names_file, "r", encoding="utf-8") as f:
                _patient_names = json.load(f)

        model_file = MODEL_DIR / "churn_model.joblib"
        _model_loaded = model_file.exists()

        print(f"[OK] Loaded {len(_predictions_cache or [])} predictions")
        print(f"[OK] Loaded {len(_timelines_cache or {})} timelines")

    except Exception as e:
        print(f"[WARN] Error loading caches: {e}")


# Load on startup
load_caches()


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _predictions_cache and _model_loaded else "degraded",
        model_loaded=_model_loaded,
        database_connected=True,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict-churn", response_model=ChurnPredictionResponse)
def predict_churn(request: PredictChurnRequest):
    """Predict churn risk for a single patient."""
    if not _predictions_cache:
        raise HTTPException(status_code=503, detail="Model not loaded. Run generate_and_train.py first.")

    # Find patient in cache
    patient = None
    for p in _predictions_cache:
        if p["patient_id"] == request.patient_id:
            patient = p
            break

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return ChurnPredictionResponse(
        patient_id=patient["patient_id"],
        churn_probability=patient["churn_probability"],
        churn_prediction=patient["churn_prediction"],
        risk_label=patient["risk_label"],
        key_features={
            "total_meal_logs": patient.get("total_meal_logs", 0),
            "total_weight_logs": patient.get("total_weight_logs", 0),
            "total_exercise_logs": patient.get("total_exercise_logs", 0),
            "meal_compliance_rate": patient.get("meal_compliance_rate", 0),
            "collection_completion_rate": patient.get("collection_completion_rate", 0),
            "engagement_score": patient.get("engagement_score", 0),
            "enrollment_age_days": patient.get("enrollment_age_days", 0),
            "days_since_last_meal": patient.get("days_since_last_meal", 0),
            "days_since_last_weight": patient.get("days_since_last_weight", 0),
        },
    )


@app.get("/patients/at-risk", response_model=PatientsAtRiskResponse)
def get_patients_at_risk(
    limit: int = Query(100, ge=1, le=500),
    risk_filter: Optional[str] = Query(None, pattern="^(Low|Medium|High)$"),
):
    """Get all patients ranked by churn risk."""
    if not _predictions_cache:
        raise HTTPException(status_code=503, detail="No predictions available")

    results = _predictions_cache.copy()
    if risk_filter:
        results = [r for r in results if r["risk_label"] == risk_filter]

    results = results[:limit]

    high = sum(1 for r in results if r["risk_label"] == "High")
    medium = sum(1 for r in results if r["risk_label"] == "Medium")
    low = sum(1 for r in results if r["risk_label"] == "Low")

    return PatientsAtRiskResponse(
        total_patients=len(results),
        high_risk=high,
        medium_risk=medium,
        low_risk=low,
        patients=[
            PatientRiskItem(
                patient_id=r["patient_id"],
                patient_name=r.get("patient_name"),
                churn_probability=r["churn_probability"],
                churn_prediction=r["churn_prediction"],
                risk_label=r["risk_label"],
            )
            for r in results
        ],
    )


@app.get("/patient/{patient_id}/activity", response_model=PatientActivityResponse)
def get_patient_activity(patient_id: str):
    """Get activity timeline for a patient."""
    if not _timelines_cache:
        raise HTTPException(status_code=503, detail="No timeline data available")

    timeline_data = _timelines_cache.get(patient_id)
    if not timeline_data:
        raise HTTPException(status_code=404, detail="Patient not found")

    return PatientActivityResponse(
        patient_id=timeline_data["patient_id"],
        patient_name=timeline_data.get("patient_name"),
        enrollment_date=timeline_data.get("enrollment_date"),
        activity_timeline=[
            ActivityLogItem(**t) for t in timeline_data["activity_timeline"]
        ],
        summary=timeline_data["summary"],
    )


@app.get("/patients/search")
def search_patients(
    q: str = Query("", min_length=0),
    limit: int = Query(20, ge=1, le=100),
):
    """Search patients by name or ID prefix."""
    if not _predictions_cache:
        return []

    results = []
    query_lower = q.lower()

    for p in _predictions_cache:
        pid = p["patient_id"]
        name = (_patient_names or {}).get(pid, "") if _patient_names else ""

        if not q or query_lower in pid.lower() or query_lower in name.lower():
            name_parts = name.split(" ", 1) if name else ["", ""]
            results.append(
                PatientSearchItem(
                    patient_id=pid,
                    first_name=name_parts[0] if name_parts else None,
                    last_name=name_parts[1] if len(name_parts) > 1 else None,
                    created_at="2026-03-15",
                )
            )

        if len(results) >= limit:
            break

    return results


@app.get("/model/metrics")
def get_model_metrics():
    """Return saved model evaluation metrics."""
    metrics_file = MODEL_DIR / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
