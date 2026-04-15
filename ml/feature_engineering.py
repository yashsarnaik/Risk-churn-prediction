"""
Feature Engineering Module
==========================
Loads the raw extracted CSV, applies transformations,
engineers the churn label, and prepares data for model training.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CHURN_THRESHOLD_DAYS = int(os.getenv("CHURN_THRESHOLD_DAYS", 14))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_FILE = DATA_DIR / "churn_dataset.csv"
PROCESSED_FILE = DATA_DIR / "churn_processed.csv"

# Features to exclude from model input
EXCLUDE_COLS = ["patient_id"]

# Columns representing "days since last activity" — high value = inactive
RECENCY_COLS = [
    "days_since_last_meal",
    "days_since_last_weight",
    "days_since_last_exercise",
    "days_since_last_collection",
]

# Activity count columns
ACTIVITY_COLS = [
    "total_meal_logs",
    "total_weight_logs",
    "total_exercise_logs",
    "total_collection_items",
]


def load_raw_data() -> pd.DataFrame:
    """Load the raw extracted dataset."""
    df = pd.read_csv(RAW_FILE)
    print(f"📂 Loaded {len(df)} records from {RAW_FILE}")
    return df


def create_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define churn label based on patient inactivity.

    A patient is considered 'churned' if:
    1. Their minimum "days since last activity" across all channels
       exceeds the threshold (default: 14 days), OR
    2. They have zero activity across all log types.
    """
    df = df.copy()

    # Replace sentinel 9999 values with NaN for recency calculation
    for col in RECENCY_COLS:
        if col in df.columns:
            df[col] = df[col].replace(9999, np.nan)

    # Calculate the minimum recency across all activity types
    # (how recently they did ANYTHING)
    recency_df = df[RECENCY_COLS].copy()
    df["min_days_since_activity"] = recency_df.min(axis=1)

    # Total activity across all channels
    df["total_activity_count"] = sum(
        df[col] for col in ACTIVITY_COLS if col in df.columns
    )

    # Churn definition:
    # - No activity at all (total_activity_count == 0), OR
    # - All activity channels are older than threshold
    df["is_churned"] = (
        (df["total_activity_count"] == 0)
        | (df["min_days_since_activity"].isna())  # No activity at all
        | (df["min_days_since_activity"] > CHURN_THRESHOLD_DAYS)
    ).astype(int)

    # Restore sentinel values for model features (0 activity = max inactivity)
    for col in RECENCY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(9999)

    print(f"\n🏷️  Churn Label Distribution (threshold={CHURN_THRESHOLD_DAYS} days):")
    print(f"   Churned (1):     {df['is_churned'].sum()}")
    print(f"   Not Churned (0): {(df['is_churned'] == 0).sum()}")
    print(f"   Churn Rate:      {df['is_churned'].mean():.1%}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional derived features."""
    df = df.copy()

    # ── Activity Engagement Score ────────────────────────────────
    # Weighted combination of activity counts (normalized)
    max_meals = df["total_meal_logs"].max() if df["total_meal_logs"].max() > 0 else 1
    max_weights = df["total_weight_logs"].max() if df["total_weight_logs"].max() > 0 else 1
    max_exercises = df["total_exercise_logs"].max() if df["total_exercise_logs"].max() > 0 else 1
    max_collections = df["total_collection_items"].max() if df["total_collection_items"].max() > 0 else 1

    df["engagement_score"] = (
        0.3 * (df["total_meal_logs"] / max_meals)
        + 0.2 * (df["total_weight_logs"] / max_weights)
        + 0.2 * (df["total_exercise_logs"] / max_exercises)
        + 0.3 * (df["total_collection_items"] / max_collections)
    )

    # ── Recency Decay Score ──────────────────────────────────────
    # Lower = more recent activity = better
    def decay_score(days, max_days=365):
        """Exponential decay: recent activity → high score."""
        capped = np.minimum(days, max_days)
        return np.exp(-0.05 * capped)

    for col in RECENCY_COLS:
        if col in df.columns:
            decay_col = col.replace("days_since_last_", "") + "_recency_score"
            df[decay_col] = decay_score(df[col])

    # ── Meal Logging Consistency ─────────────────────────────────
    if "distinct_meal_days" in df.columns and "enrollment_age_days" in df.columns:
        df["meal_logging_rate"] = np.where(
            df["enrollment_age_days"] > 0,
            df["distinct_meal_days"] / df["enrollment_age_days"],
            0,
        )

    # ── Weight Monitoring Frequency ──────────────────────────────
    if "distinct_weight_days" in df.columns and "enrollment_age_days" in df.columns:
        df["weight_logging_rate"] = np.where(
            df["enrollment_age_days"] > 0,
            df["distinct_weight_days"] / df["enrollment_age_days"],
            0,
        )

    # ── Has Any Recent Activity (last 7 days) ────────────────────
    df["has_recent_meals"] = (df.get("meals_last_7_days", 0) > 0).astype(int)

    # ── Provider Assignment Count ────────────────────────────────
    provider_cols = ["has_doctor", "has_nutritionist", "has_fitness_coach"]
    existing_provider_cols = [c for c in provider_cols if c in df.columns]
    if existing_provider_cols:
        df["provider_count"] = sum(df[c] for c in existing_provider_cols)

    return df


def handle_missing_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and clip outliers."""
    df = df.copy()

    # Fill remaining NaN with 0 for count features, median for continuous
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if "count" in col or "total" in col or "days" in col:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())

    # Clip outliers at 1st and 99th percentile for continuous features
    clip_cols = [
        c for c in numeric_cols
        if c not in EXCLUDE_COLS + ["is_churned", "gender_encoded"]
        and "encoded" not in c and "flag" not in c
    ]
    for col in clip_cols:
        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(q01, q99)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature columns for model training."""
    exclude = EXCLUDE_COLS + [
        "is_churned",
        "min_days_since_activity",
        "total_activity_count",
    ]
    return [c for c in df.columns if c not in exclude]


def process_data() -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = load_raw_data()
    df = create_churn_label(df)
    df = engineer_features(df)
    df = handle_missing_and_outliers(df)

    # Save processed data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"\n💾 Saved processed dataset to: {PROCESSED_FILE}")
    print(f"   Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    process_data()
