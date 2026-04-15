"""
Synthetic Data Generator + Model Trainer
=========================================
Generates realistic patient churn data for demo/showcase purposes.
Creates 500 patients with realistic behavioral patterns, trains an
XGBoost model, and prepares everything for the dashboard.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import random
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"

np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════

NUM_PATIENTS = 500
TODAY = datetime(2026, 4, 15)

# Patient archetypes — logging rates are expressed as fractions of enrollment days
# E.g., meal_day_rate=0.9 means they log meals on 90% of enrolled days
# meals_per_day=3 means ~3 meal entries per active day (breakfast, lunch, dinner)
ARCHETYPES = {
    "highly_engaged": {
        "weight": 0.25,   # 25% of patients
        "churn_prob": 0.05,
        "enrollment_days_range": (60, 210),
        # Meals: log 3 meals/day on 85-98% of days
        "meal_day_rate_range": (0.85, 0.98),
        "meals_per_day_range": (2.5, 3.0),
        "meal_compliance_range": (0.75, 0.95),
        "calories_range": (1400, 2100),
        "protein_range": (50, 90),
        # Weight: log daily or near-daily
        "weight_day_rate_range": (0.7, 0.95),
        "weight_change_range": (-6, -1),
        # Exercise: 5-7 days/week
        "exercise_day_rate_range": (0.65, 0.90),
        "exercise_minutes_range": (25, 60),
        # Collections: high completion
        "collection_day_rate_range": (0.5, 0.8),
        "completion_rate_range": (0.65, 0.95),
        # Recency: very recent
        "days_since_meal_range": (0, 1),
        "days_since_weight_range": (0, 2),
        "days_since_exercise_range": (0, 2),
        "days_since_collection_range": (0, 3),
    },
    "moderate_engaged": {
        "weight": 0.25,
        "churn_prob": 0.18,
        "enrollment_days_range": (30, 150),
        "meal_day_rate_range": (0.50, 0.80),
        "meals_per_day_range": (1.5, 2.5),
        "meal_compliance_range": (0.45, 0.75),
        "calories_range": (1100, 2400),
        "protein_range": (25, 65),
        "weight_day_rate_range": (0.3, 0.6),
        "weight_change_range": (-4, 1),
        "exercise_day_rate_range": (0.25, 0.55),
        "exercise_minutes_range": (15, 45),
        "collection_day_rate_range": (0.2, 0.5),
        "completion_rate_range": (0.30, 0.65),
        "days_since_meal_range": (1, 5),
        "days_since_weight_range": (2, 7),
        "days_since_exercise_range": (2, 8),
        "days_since_collection_range": (2, 7),
    },
    "at_risk": {
        "weight": 0.25,
        "churn_prob": 0.55,
        "enrollment_days_range": (14, 90),
        "meal_day_rate_range": (0.10, 0.35),
        "meals_per_day_range": (1.0, 2.0),
        "meal_compliance_range": (0.10, 0.40),
        "calories_range": (800, 2600),
        "protein_range": (10, 45),
        "weight_day_rate_range": (0.05, 0.20),
        "weight_change_range": (-1, 3),
        "exercise_day_rate_range": (0.02, 0.15),
        "exercise_minutes_range": (10, 30),
        "collection_day_rate_range": (0.05, 0.20),
        "completion_rate_range": (0.05, 0.30),
        "days_since_meal_range": (8, 20),
        "days_since_weight_range": (10, 25),
        "days_since_exercise_range": (12, 35),
        "days_since_collection_range": (8, 25),
    },
    "disengaged": {
        "weight": 0.25,
        "churn_prob": 0.92,
        "enrollment_days_range": (7, 60),
        "meal_day_rate_range": (0.0, 0.10),
        "meals_per_day_range": (0.5, 1.5),
        "meal_compliance_range": (0.0, 0.15),
        "calories_range": (0, 2200),
        "protein_range": (0, 25),
        "weight_day_rate_range": (0.0, 0.08),
        "weight_change_range": (0, 5),
        "exercise_day_rate_range": (0.0, 0.05),
        "exercise_minutes_range": (0, 20),
        "collection_day_rate_range": (0.0, 0.05),
        "completion_rate_range": (0.0, 0.10),
        "days_since_meal_range": (18, 60),
        "days_since_weight_range": (20, 60),
        "days_since_exercise_range": (25, 60),
        "days_since_collection_range": (18, 60),
    },
}

# Indian names for realistic patient display
FIRST_NAMES_MALE = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
    "Ayaan", "Krishna", "Ishaan", "Shaurya", "Atharv", "Advik", "Pranav",
    "Advait", "Dhruv", "Kabir", "Ritvik", "Aarush", "Kayaan", "Darsh",
    "Virat", "Rudra", "Arnav", "Samar", "Yash", "Rohan", "Kartik",
    "Rahul", "Amit", "Suresh", "Rajesh", "Vikram", "Nikhil", "Ankit",
    "Gaurav", "Manoj", "Deepak", "Ravi", "Sanjay", "Pradeep", "Ajay",
]

FIRST_NAMES_FEMALE = [
    "Ananya", "Aanya", "Aadhya", "Saanvi", "Myra", "Diya", "Prisha",
    "Riya", "Aarohi", "Anvi", "Anika", "Kavya", "Sara", "Kiara",
    "Navya", "Aisha", "Tara", "Pihu", "Zara", "Pari", "Ira",
    "Sneha", "Pooja", "Priya", "Neha", "Swati", "Anjali", "Megha",
    "Nisha", "Divya", "Sunita", "Rekha", "Meera", "Lakshmi", "Sita",
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Joshi",
    "Mehta", "Shah", "Reddy", "Nair", "Iyer", "Rao", "Desai",
    "Kulkarni", "Patil", "Pillai", "Menon", "Agarwal", "Malhotra",
    "Chopra", "Kapoor", "Saxena", "Tiwari", "Pandey", "Mishra",
    "Dubey", "Srivastava", "Chauhan", "Yadav", "Thakur", "Bhatia",
]


def rand_range(lo, hi):
    return lo + (hi - lo) * np.random.random()


def rand_int_range(lo, hi):
    return int(lo + (hi - lo) * np.random.random())


def generate_uuid():
    import uuid
    return str(uuid.uuid4())


def generate_patients():
    """Generate synthetic patient data. Churn is DERIVED from behavior, never random."""
    print("[GEN] Generating synthetic patient data...")

    records = []
    patient_names = {}

    for i in range(NUM_PATIENTS):
        # Pick archetype
        r = np.random.random()
        cumulative = 0
        archetype = None
        for name, config in ARCHETYPES.items():
            cumulative += config["weight"]
            if r <= cumulative:
                archetype = config
                archetype_name = name
                break

        # Demographics
        gender = np.random.choice([0, 1], p=[0.45, 0.55])
        age = int(np.clip(np.random.normal(42, 14), 18, 75))
        height = np.clip(np.random.normal(165, 10), 140, 195)
        weight = np.clip(np.random.normal(82, 18), 45, 150)
        has_medical = np.random.choice([0, 1], p=[0.35, 0.65])
        has_allergies = np.random.choice([0, 1], p=[0.7, 0.3])

        first = random.choice(FIRST_NAMES_MALE if gender == 1 else FIRST_NAMES_FEMALE)
        last = random.choice(LAST_NAMES)
        pid = generate_uuid()
        patient_names[pid] = f"{first} {last}"

        # Enrollment & providers
        enrollment_days = rand_int_range(*archetype["enrollment_days_range"])
        is_active = 1 if np.random.random() > 0.05 else 0
        has_doctor = np.random.choice([0, 1], p=[0.15, 0.85])
        has_nutritionist = np.random.choice([0, 1], p=[0.20, 0.80])
        has_fitness_coach = np.random.choice([0, 1], p=[0.40, 0.60])

        # ── Generate ALL features from archetype rates ──────────
        meal_day_rate = rand_range(*archetype["meal_day_rate_range"])
        meals_per_day = rand_range(*archetype["meals_per_day_range"])
        distinct_meal_days = max(0, int(enrollment_days * meal_day_rate))
        total_meals = max(0, int(distinct_meal_days * meals_per_day))
        avg_calories = rand_range(*archetype["calories_range"])
        avg_protein = rand_range(*archetype["protein_range"])
        avg_carbs = rand_range(30, 120)
        avg_fat = rand_range(10, 60)
        distinct_meal_types = min(7, rand_int_range(1, min(total_meals + 1, 8)))
        meal_compliance = rand_range(*archetype["meal_compliance_range"])
        days_since_meal = rand_int_range(*archetype["days_since_meal_range"])

        if days_since_meal <= 7:
            meals_7d = max(0, int(7 * meal_day_rate * meals_per_day * rand_range(0.6, 1.0)))
        else:
            meals_7d = 0
        if days_since_meal <= 14:
            meals_14d = max(meals_7d, int(14 * meal_day_rate * meals_per_day * rand_range(0.5, 0.9)))
        else:
            meals_14d = 0

        weight_day_rate = rand_range(*archetype["weight_day_rate_range"])
        total_weights = max(0, int(enrollment_days * weight_day_rate))
        distinct_weight_days = min(total_weights, max(0, int(total_weights * rand_range(0.85, 1.0))))
        weight_change = rand_range(*archetype["weight_change_range"])
        weight_range_kg = abs(weight_change) + rand_range(0, 2)
        avg_bmi = weight / ((height / 100) ** 2) + rand_range(-1, 1)
        days_since_weight = rand_int_range(*archetype["days_since_weight_range"])

        exercise_day_rate = rand_range(*archetype["exercise_day_rate_range"])
        total_exercises = max(0, int(enrollment_days * exercise_day_rate))
        distinct_exercise_days = min(total_exercises, max(0, int(total_exercises * rand_range(0.8, 1.0))))
        avg_exercise_min = rand_range(*archetype["exercise_minutes_range"])
        total_exercise_min = int(total_exercises * avg_exercise_min)
        total_cals_burned = round(total_exercise_min * rand_range(4, 7), 1)
        total_steps = total_exercises * rand_int_range(2000, 8000)
        days_since_exercise = rand_int_range(*archetype["days_since_exercise_range"])
        distinct_exercise_types = min(5, rand_int_range(1, min(total_exercises + 1, 6)))

        collection_day_rate = rand_range(*archetype["collection_day_rate_range"])
        total_collection = max(0, int(enrollment_days * collection_day_rate))
        completion_rate = rand_range(*archetype["completion_rate_range"])
        completed_items = int(total_collection * completion_rate)
        avg_progress = completion_rate * 100 * rand_range(0.7, 1.0)
        unlocked_items = min(total_collection, completed_items + rand_int_range(0, max(1, total_collection - completed_items)))
        days_since_collection = rand_int_range(*archetype["days_since_collection_range"])

        # ═══════════════════════════════════════════════════════════
        # DERIVE CHURN FROM ACTUAL BEHAVIOR — NOT RANDOM
        # Rule: low logs + high recency = churned. Period.
        # ═══════════════════════════════════════════════════════════
        churn_score = 0.0

        # LOW total logs → likely to churn
        if total_meals < 15:
            churn_score += 0.25
        elif total_meals < 40:
            churn_score += 0.10

        if total_weights < 5:
            churn_score += 0.10
        if total_exercises < 5:
            churn_score += 0.10

        # HIGH recency (haven't logged recently) → likely to churn
        if days_since_meal > 14:
            churn_score += 0.20
        elif days_since_meal > 7:
            churn_score += 0.10

        if days_since_weight > 14:
            churn_score += 0.10
        if days_since_exercise > 14:
            churn_score += 0.10

        # LOW compliance → likely to churn
        if meal_compliance < 0.3:
            churn_score += 0.10
        if completion_rate < 0.2:
            churn_score += 0.05

        # Add small noise so it's not perfectly deterministic
        churn_score += np.random.normal(0, 0.05)
        churn_score = np.clip(churn_score, 0, 1)

        # Threshold: score >= 0.40 → churned
        is_churned = 1 if churn_score >= 0.40 else 0

        records.append({
            "patient_id": pid,
            "patient_name": patient_names[pid],
            "gender_encoded": gender,
            "age_years": age,
            "height_cm": round(height, 1),
            "initial_weight_kg": round(weight, 1),
            "has_medical_conditions": has_medical,
            "has_allergies": has_allergies,
            "enrollment_age_days": enrollment_days,
            "is_enrolled_active": is_active,
            "has_doctor": has_doctor,
            "has_nutritionist": has_nutritionist,
            "has_fitness_coach": has_fitness_coach,
            "total_meal_logs": total_meals,
            "distinct_meal_days": distinct_meal_days,
            "avg_calories_per_meal": round(avg_calories, 1),
            "avg_protein_g": round(avg_protein, 1),
            "avg_carbs_g": round(avg_carbs, 1),
            "avg_fat_g": round(avg_fat, 1),
            "distinct_meal_types": distinct_meal_types,
            "meal_compliance_rate": round(meal_compliance, 4),
            "days_since_last_meal": days_since_meal,
            "meals_last_7_days": meals_7d,
            "meals_last_14_days": meals_14d,
            "total_weight_logs": total_weights,
            "distinct_weight_days": distinct_weight_days,
            "weight_range_kg": round(weight_range_kg, 2),
            "avg_bmi": round(avg_bmi, 1),
            "days_since_last_weight": days_since_weight,
            "weight_change_kg": round(weight_change, 2),
            "total_exercise_logs": total_exercises,
            "distinct_exercise_days": distinct_exercise_days,
            "total_exercise_minutes": total_exercise_min,
            "avg_exercise_minutes": round(avg_exercise_min, 1),
            "total_calories_burned": total_cals_burned,
            "total_steps": total_steps,
            "days_since_last_exercise": days_since_exercise,
            "distinct_exercise_types": distinct_exercise_types,
            "total_collection_items": total_collection,
            "completed_items": completed_items,
            "collection_completion_rate": round(completion_rate, 4),
            "avg_progress_percent": round(avg_progress, 1),
            "unlocked_items": unlocked_items,
            "days_since_last_collection": days_since_collection,
            "is_churned": is_churned,
        })

    df = pd.DataFrame(records)

    active = df[df["is_churned"] == 0]
    churned = df[df["is_churned"] == 1]
    print(f"   Generated {len(df)} patients")
    print(f"   Churn rate: {df['is_churned'].mean():.1%}")
    print(f"   Churned: {churned.shape[0]}, Active: {active.shape[0]}")
    print(f"\n   Active patient averages:")
    print(f"      Enrollment days: {active['enrollment_age_days'].mean():.0f}")
    print(f"      Meal logs:       {active['total_meal_logs'].mean():.0f}")
    print(f"      Weight logs:     {active['total_weight_logs'].mean():.0f}")
    print(f"      Exercise logs:   {active['total_exercise_logs'].mean():.0f}")
    print(f"      Compliance:      {active['meal_compliance_rate'].mean():.1%}")
    print(f"      Days since meal: {active['days_since_last_meal'].mean():.1f}")
    print(f"\n   Churned patient averages:")
    print(f"      Enrollment days: {churned['enrollment_age_days'].mean():.0f}")
    print(f"      Meal logs:       {churned['total_meal_logs'].mean():.0f}")
    print(f"      Weight logs:     {churned['total_weight_logs'].mean():.0f}")
    print(f"      Exercise logs:   {churned['total_exercise_logs'].mean():.0f}")
    print(f"      Compliance:      {churned['meal_compliance_rate'].mean():.1%}")
    print(f"      Days since meal: {churned['days_since_last_meal'].mean():.1f}")

    return df, patient_names


# ══════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def engineer_features(df):
    """Add derived features to the dataset."""
    df = df.copy()

    # Engagement score
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
    for col in ["days_since_last_meal", "days_since_last_weight",
                "days_since_last_exercise", "days_since_last_collection"]:
        decay_col = col.replace("days_since_last_", "") + "_recency_score"
        df[decay_col] = np.exp(-0.05 * np.minimum(df[col], 365))

    # Logging rates
    df["meal_logging_rate"] = np.where(
        df["enrollment_age_days"] > 0,
        df["distinct_meal_days"] / df["enrollment_age_days"], 0)
    df["weight_logging_rate"] = np.where(
        df["enrollment_age_days"] > 0,
        df["distinct_weight_days"] / df["enrollment_age_days"], 0)

    # Recent activity flag
    df["has_recent_meals"] = (df["meals_last_7_days"] > 0).astype(int)

    # Provider count
    df["provider_count"] = df["has_doctor"] + df["has_nutritionist"] + df["has_fitness_coach"]

    # Overall recency (minimum across all channels)
    df["min_days_since_activity"] = df[
        ["days_since_last_meal", "days_since_last_weight",
         "days_since_last_exercise", "days_since_last_collection"]
    ].min(axis=1)

    # Total activity count
    df["total_activity_count"] = (
        df["total_meal_logs"] + df["total_weight_logs"] +
        df["total_exercise_logs"] + df["total_collection_items"]
    )

    return df


# ══════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════

FEATURE_COLS = None   # Will be set during training
EXCLUDE_COLS = ["patient_id", "patient_name", "is_churned",
                "min_days_since_activity", "total_activity_count"]


def train_model(df):
    """Train and evaluate models."""
    global FEATURE_COLS

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    FEATURE_COLS = feature_cols

    X = df[feature_cols].values
    y = df["is_churned"].values

    print(f"\n📐 Feature matrix: {X.shape}")
    print(f"   Target: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Train/Test: {len(X_train)} / {len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train models
    models = {}

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_s, y_train)
    models["Logistic Regression"] = lr

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5,
        min_samples_leaf=3, random_state=42, class_weight="balanced")
    rf.fit(X_train_s, y_train)
    models["Random Forest"] = rf

    if HAS_XGBOOST:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=200, max_depth=5, min_child_weight=3,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42, eval_metric="logloss", use_label_encoder=False)
        xgb.fit(X_train_s, y_train)
        models["XGBoost"] = xgb

    # Evaluate
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)

    best_name, best_f1 = None, -1
    all_metrics = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0

        # Manual CV for XGBoost compatibility
        n_splits = min(5, min(np.bincount(y_train)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="f1")
        except (AttributeError, TypeError):
            cv_list = []
            for tr_idx, val_idx in cv.split(X_train_s, y_train):
                clone = model.__class__(**model.get_params())
                clone.fit(X_train_s[tr_idx], y_train[tr_idx])
                vp = clone.predict(X_train_s[val_idx])
                cv_list.append(f1_score(y_train[val_idx], vp, zero_division=0))
            cv_scores = np.array(cv_list)

        metrics = {
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
        }
        all_metrics[name] = metrics

        print(f"\n📊 {name}:")
        for k, v in metrics.items():
            print(f"   {k:15s}: {v}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    best_model = models[best_name]
    print(f"\n🏆 Best Model: {best_name} (F1={best_f1:.4f})")

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(feature_cols))

    feat_imp = pd.DataFrame({
        "feature": feature_cols, "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n📈 Top 15 Features:")
    for _, row in feat_imp.head(15).iterrows():
        bar = "█" * int(row["importance"] * 40 / max(feat_imp["importance"].max(), 1e-9))
        print(f"   {row['feature']:35s} {bar} {row['importance']:.4f}")

    # ── Save plots ───────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feat_imp.head(15), x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title("Top 15 Feature Importances - Churn Prediction", fontsize=14)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(DATA_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    y_pred_best = best_model.predict(X_test_s)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Active", "Churned"], yticklabels=["Active", "Churned"])
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(DATA_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=["Active", "Churned"]))

    # ── Save artifacts ───────────────────────────────────────────
    joblib.dump(best_model, MODEL_DIR / "churn_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"best_model": best_name, "metrics": all_metrics}, f, indent=2)

    print(f"\n💾 Model saved to: {MODEL_DIR / 'churn_model.joblib'}")
    print(f"💾 Scaler saved to: {MODEL_DIR / 'scaler.joblib'}")

    return best_model, scaler, feature_cols, all_metrics


# ══════════════════════════════════════════════════════════════════
# 4. GENERATE DEMO PREDICTIONS
# ══════════════════════════════════════════════════════════════════

def generate_predictions(df, model, scaler, feature_cols, patient_names):
    """Generate predictions for all patients and save for the API."""
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    probas = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    results = []
    for i in range(len(df)):
        prob = float(probas[i])
        if prob < 0.3:
            risk = "Low"
        elif prob < 0.6:
            risk = "Medium"
        else:
            risk = "High"

        results.append({
            "patient_id": df.iloc[i]["patient_id"],
            "patient_name": df.iloc[i]["patient_name"],
            "churn_probability": round(prob, 4),
            "churn_prediction": int(preds[i]),
            "risk_label": risk,
            "total_meal_logs": int(df.iloc[i]["total_meal_logs"]),
            "total_weight_logs": int(df.iloc[i]["total_weight_logs"]),
            "total_exercise_logs": int(df.iloc[i]["total_exercise_logs"]),
            "meal_compliance_rate": float(df.iloc[i]["meal_compliance_rate"]),
            "collection_completion_rate": float(df.iloc[i]["collection_completion_rate"]),
            "engagement_score": float(df.iloc[i]["engagement_score"]),
            "enrollment_age_days": int(df.iloc[i]["enrollment_age_days"]),
            "days_since_last_meal": int(df.iloc[i]["days_since_last_meal"]),
            "days_since_last_weight": int(df.iloc[i]["days_since_last_weight"]),
        })

    results.sort(key=lambda x: x["churn_probability"], reverse=True)

    # Save predictions cache
    with open(DATA_DIR / "predictions_cache.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save patient names mapping
    with open(DATA_DIR / "patient_names.json", "w") as f:
        json.dump(patient_names, f, indent=2)

    high = sum(1 for r in results if r["risk_label"] == "High")
    med = sum(1 for r in results if r["risk_label"] == "Medium")
    low = sum(1 for r in results if r["risk_label"] == "Low")

    print(f"\n📊 Prediction Summary:")
    print(f"   High Risk:   {high} patients")
    print(f"   Medium Risk: {med} patients")
    print(f"   Low Risk:    {low} patients")

    return results


# ══════════════════════════════════════════════════════════════════
# 5. GENERATE ACTIVITY TIMELINE DATA
# ══════════════════════════════════════════════════════════════════

def generate_activity_timelines(df, patient_names):
    """Generate 30-day activity timelines for each patient."""
    timelines = {}

    for _, row in df.iterrows():
        pid = row["patient_id"]
        is_churned = row["is_churned"]

        timeline = []
        for day_offset in range(30, -1, -1):
            date = (TODAY - timedelta(days=day_offset)).strftime("%Y-%m-%d")

            # Activity probability decreases for churned patients
            if is_churned:
                # Activity mostly in earlier days, dropping off
                activity_prob = max(0, 0.4 - (30 - day_offset) * 0.02)
            else:
                activity_prob = 0.3 + row["meal_compliance_rate"] * 0.4

            meal_count = np.random.binomial(3, min(activity_prob, 0.9))
            weight_logged = np.random.random() < activity_prob * 0.3
            exercise_count = np.random.binomial(1, min(activity_prob * 0.5, 0.8))
            collection = np.random.binomial(2, min(activity_prob * 0.6, 0.9))

            timeline.append({
                "date": date,
                "meal_count": int(meal_count),
                "weight_logged": bool(weight_logged),
                "exercise_count": int(exercise_count),
                "collection_progress": int(collection),
            })

        timelines[pid] = {
            "patient_id": pid,
            "patient_name": patient_names.get(pid, "Unknown"),
            "enrollment_date": (TODAY - timedelta(days=int(row["enrollment_age_days"]))).strftime("%Y-%m-%d"),
            "activity_timeline": timeline,
            "summary": {
                "period_days": 31,
                "active_days": sum(1 for t in timeline if t["meal_count"] > 0 or t["weight_logged"] or t["exercise_count"] > 0),
                "total_meals_logged": sum(t["meal_count"] for t in timeline),
                "weight_log_days": sum(1 for t in timeline if t["weight_logged"]),
                "total_exercises": sum(t["exercise_count"] for t in timeline),
                "activity_rate": round(
                    sum(1 for t in timeline if t["meal_count"] > 0 or t["weight_logged"] or t["exercise_count"] > 0) / 31, 4),
            },
        }

    # Save timelines
    with open(DATA_DIR / "activity_timelines.json", "w") as f:
        json.dump(timelines, f)

    print(f"📅 Generated activity timelines for {len(timelines)} patients")
    return timelines


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PATIENT CHURN PREDICTION — DEMO DATA PIPELINE")
    print("=" * 60)

    # 1. Generate data
    df, patient_names = generate_patients()

    # 2. Engineer features
    print("\n⚙️  Engineering features...")
    df = engineer_features(df)

    # 3. Save raw + processed data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "churn_dataset.csv", index=False)
    df.to_csv(DATA_DIR / "churn_processed.csv", index=False)
    print(f"💾 Saved dataset: {DATA_DIR / 'churn_dataset.csv'} ({df.shape})")

    # 4. Train model
    print("\n🤖 Training models...")
    model, scaler, feature_cols, metrics = train_model(df)

    # 5. Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = generate_predictions(df, model, scaler, feature_cols, patient_names)

    # 6. Generate timelines
    print("\n📅 Generating activity timelines...")
    timelines = generate_activity_timelines(df, patient_names)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE ✅")
    print("=" * 60)
    print(f"\n  Files generated:")
    print(f"    📁 {DATA_DIR / 'churn_dataset.csv'}")
    print(f"    📁 {DATA_DIR / 'churn_processed.csv'}")
    print(f"    📁 {DATA_DIR / 'predictions_cache.json'}")
    print(f"    📁 {DATA_DIR / 'activity_timelines.json'}")
    print(f"    📁 {DATA_DIR / 'patient_names.json'}")
    print(f"    📁 {MODEL_DIR / 'churn_model.joblib'}")
    print(f"    📁 {MODEL_DIR / 'scaler.joblib'}")
    print(f"    📁 {DATA_DIR / 'feature_importance.png'}")
    print(f"    📁 {DATA_DIR / 'confusion_matrix.png'}")
    print(f"\n  🚀 Restart the FastAPI server to use the new model!")


if __name__ == "__main__":
    main()
