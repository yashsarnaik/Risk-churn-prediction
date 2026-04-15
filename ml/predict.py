"""
Prediction Pipeline
====================
Shared prediction logic used by both batch prediction and the FastAPI endpoint.
Loads the trained model and applies the same feature engineering pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
import psycopg2
import joblib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "drapp"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123"),
}

MODEL_DIR = Path(__file__).resolve().parent / "models"
SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def load_model():
    """Load the trained model, scaler, and feature columns."""
    model = joblib.load(MODEL_DIR / "churn_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    with open(MODEL_DIR / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


def get_single_patient_sql() -> str:
    """SQL query for a single patient's features (same as training query but filtered)."""
    base_sql = open(SQL_DIR / "feature_extraction.sql", "r", encoding="utf-8").read()
    # Add a WHERE clause filter for a specific patient
    # We inject a placeholder that will be replaced
    return base_sql


def fetch_patient_features(patient_id: str) -> pd.DataFrame:
    """Fetch feature vector for a single patient from the database."""
    sql = get_single_patient_sql()

    # Add WHERE clause before ORDER BY
    sql = sql.replace(
        "ORDER BY p.created_at DESC;",
        f"WHERE p.id = '{patient_id}'\nORDER BY p.created_at DESC;"
    )

    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql_query(sql, conn)
    conn.close()

    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as training (without churn label)."""
    df = df.copy()

    # Activity counts
    activity_cols = ["total_meal_logs", "total_weight_logs", "total_exercise_logs", "total_collection_items"]
    existing = [c for c in activity_cols if c in df.columns]
    df["total_activity_count"] = sum(df[c] for c in existing)

    recency_cols = [
        "days_since_last_meal",
        "days_since_last_weight",
        "days_since_last_exercise",
        "days_since_last_collection",
    ]

    # Engagement score
    for col in ["total_meal_logs", "total_weight_logs", "total_exercise_logs", "total_collection_items"]:
        if col not in df.columns:
            df[col] = 0

    max_vals = {
        "total_meal_logs": max(df["total_meal_logs"].max(), 1),
        "total_weight_logs": max(df["total_weight_logs"].max(), 1),
        "total_exercise_logs": max(df["total_exercise_logs"].max(), 1),
        "total_collection_items": max(df["total_collection_items"].max(), 1),
    }

    df["engagement_score"] = (
        0.3 * (df["total_meal_logs"] / max_vals["total_meal_logs"])
        + 0.2 * (df["total_weight_logs"] / max_vals["total_weight_logs"])
        + 0.2 * (df["total_exercise_logs"] / max_vals["total_exercise_logs"])
        + 0.3 * (df["total_collection_items"] / max_vals["total_collection_items"])
    )

    # Recency decay scores
    def decay_score(days, max_days=365):
        capped = np.minimum(days, max_days)
        return np.exp(-0.05 * capped)

    for col in recency_cols:
        if col in df.columns:
            decay_col = col.replace("days_since_last_", "") + "_recency_score"
            df[decay_col] = decay_score(df[col])

    # Logging rates
    if "distinct_meal_days" in df.columns and "enrollment_age_days" in df.columns:
        df["meal_logging_rate"] = np.where(
            df["enrollment_age_days"] > 0,
            df["distinct_meal_days"] / df["enrollment_age_days"],
            0,
        )

    if "distinct_weight_days" in df.columns and "enrollment_age_days" in df.columns:
        df["weight_logging_rate"] = np.where(
            df["enrollment_age_days"] > 0,
            df["distinct_weight_days"] / df["enrollment_age_days"],
            0,
        )

    # Recent activity flag
    df["has_recent_meals"] = (df.get("meals_last_7_days", pd.Series([0])) > 0).astype(int)

    # Provider count
    provider_cols = ["has_doctor", "has_nutritionist", "has_fitness_coach"]
    existing_providers = [c for c in provider_cols if c in df.columns]
    if existing_providers:
        df["provider_count"] = sum(df[c] for c in existing_providers)

    # Fill NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    return df


def predict_churn(patient_id: str) -> dict:
    """
    Main prediction function.
    Fetches patient data, applies feature engineering, runs model.
    Returns prediction result dictionary.
    """
    model, scaler, feature_cols = load_model()

    # Fetch data
    df = fetch_patient_features(patient_id)
    if df.empty:
        return {
            "patient_id": patient_id,
            "error": "Patient not found",
            "churn_probability": None,
            "risk_label": None,
        }

    # Apply feature engineering
    df = apply_feature_engineering(df)

    # Ensure all required features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Extract features in correct order
    X = df[feature_cols].values

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    probability = float(model.predict_proba(X_scaled)[0, 1])
    prediction = int(model.predict(X_scaled)[0])

    # Risk label
    if probability < 0.3:
        risk_label = "Low"
    elif probability < 0.6:
        risk_label = "Medium"
    else:
        risk_label = "High"

    # Extract key features for explanation
    feature_values = dict(zip(feature_cols, X[0]))

    return {
        "patient_id": patient_id,
        "churn_probability": round(probability, 4),
        "churn_prediction": prediction,
        "risk_label": risk_label,
        "key_features": {
            "total_meal_logs": feature_values.get("total_meal_logs", 0),
            "total_weight_logs": feature_values.get("total_weight_logs", 0),
            "total_exercise_logs": feature_values.get("total_exercise_logs", 0),
            "days_since_last_meal": feature_values.get("days_since_last_meal", 9999),
            "days_since_last_weight": feature_values.get("days_since_last_weight", 9999),
            "meal_compliance_rate": feature_values.get("meal_compliance_rate", 0),
            "collection_completion_rate": feature_values.get("collection_completion_rate", 0),
            "engagement_score": feature_values.get("engagement_score", 0),
            "enrollment_age_days": feature_values.get("enrollment_age_days", 0),
        },
    }


def predict_all_patients() -> list:
    """Run prediction for all patients."""
    model, scaler, feature_cols = load_model()

    # Fetch all patients
    sql = open(SQL_DIR / "feature_extraction.sql", "r", encoding="utf-8").read()
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql_query(sql, conn)
    conn.close()

    if df.empty:
        return []

    patient_ids = df["patient_id"].tolist()
    df = apply_feature_engineering(df)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = model.predict(X_scaled)

    results = []
    for i, pid in enumerate(patient_ids):
        prob = float(probabilities[i])
        if prob < 0.3:
            risk = "Low"
        elif prob < 0.6:
            risk = "Medium"
        else:
            risk = "High"

        results.append({
            "patient_id": str(pid),
            "churn_probability": round(prob, 4),
            "churn_prediction": int(predictions[i]),
            "risk_label": risk,
        })

    # Sort by risk (highest first)
    results.sort(key=lambda x: x["churn_probability"], reverse=True)
    return results


if __name__ == "__main__":
    # Test with a sample patient
    import sys
    if len(sys.argv) > 1:
        result = predict_churn(sys.argv[1])
    else:
        print("Running batch prediction for all patients...")
        results = predict_all_patients()
        print(f"\nPredictions for {len(results)} patients:")
        for r in results[:10]:
            print(f"  {r['patient_id'][:8]}... → {r['risk_label']:6s} ({r['churn_probability']:.1%})")
