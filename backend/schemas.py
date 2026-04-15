"""
Pydantic Schemas
================
Request and response models for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# ── Request Models ───────────────────────────────────────────────

class PredictChurnRequest(BaseModel):
    patient_id: str = Field(..., description="Patient UUID")


# ── Response Models ──────────────────────────────────────────────

class ChurnPredictionResponse(BaseModel):
    patient_id: str
    churn_probability: Optional[float] = None
    churn_prediction: Optional[int] = None
    risk_label: Optional[str] = None
    key_features: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PatientRiskItem(BaseModel):
    patient_id: str
    patient_name: Optional[str] = None
    churn_probability: float
    churn_prediction: int
    risk_label: str


class PatientsAtRiskResponse(BaseModel):
    total_patients: int
    high_risk: int
    medium_risk: int
    low_risk: int
    patients: List[PatientRiskItem]


class ActivityLogItem(BaseModel):
    date: str
    meal_count: int = 0
    weight_logged: bool = False
    exercise_count: int = 0
    collection_progress: int = 0


class PatientActivityResponse(BaseModel):
    patient_id: str
    patient_name: Optional[str] = None
    enrollment_date: Optional[str] = None
    activity_timeline: List[ActivityLogItem]
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str


class PatientSearchItem(BaseModel):
    patient_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    created_at: Optional[str] = None
