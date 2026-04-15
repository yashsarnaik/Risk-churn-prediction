"""
Model Training Script
=====================
Trains an XGBoost classifier on the processed churn dataset.
Includes evaluation, cross-validation, and feature importance analysis.
Saves the trained model and preprocessing artifacts.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv

import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed; falling back to RandomForest")

from feature_engineering import process_data, get_feature_columns, EXCLUDE_COLS

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILE = MODEL_DIR / "churn_model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"
FEATURES_FILE = MODEL_DIR / "feature_columns.json"
METRICS_FILE = MODEL_DIR / "metrics.json"
REPORT_DIR = Path(__file__).resolve().parent.parent / "data"


def train_and_evaluate():
    """Full training pipeline."""

    # ── 1. Load processed data ───────────────────────────────────
    print("=" * 60)
    print("  PATIENT CHURN PREDICTION — MODEL TRAINING")
    print("=" * 60)

    df = process_data()
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df["is_churned"].values

    print(f"\n📐 Feature matrix shape: {X.shape}")
    print(f"   Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 2. Train-Test Split ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n🔀 Train/Test split: {len(X_train)} / {len(X_test)}")

    # ── 3. Scale Features ────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 4. Train Models ──────────────────────────────────────────
    models = {}

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = lr

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train_scaled, y_train)
    models["Random Forest"] = rf

    # XGBoost
    if HAS_XGBOOST:
        # Calculate scale_pos_weight for imbalanced data
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            min_child_weight=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        xgb.fit(X_train_scaled, y_train)
        models["XGBoost"] = xgb

    # ── 5. Evaluate All Models ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)

    best_model_name = None
    best_f1 = -1
    all_metrics = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = 0.0

        # Cross-validation (with fallback for XGBoost/sklearn compatibility)
        n_splits = min(5, min(np.bincount(y_train)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
        except (AttributeError, TypeError):
            # Manual CV fallback for XGBoost compatibility with sklearn 1.6+
            cv_scores_list = []
            for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                clone_model = model.__class__(**model.get_params())
                clone_model.fit(X_train_scaled[train_idx], y_train[train_idx])
                val_pred = clone_model.predict(X_train_scaled[val_idx])
                cv_scores_list.append(f1_score(y_train[val_idx], val_pred, zero_division=0))
            cv_scores = np.array(cv_scores_list)

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
        }
        all_metrics[name] = metrics

        print(f"\n📊 {name}:")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {auc:.4f}")
        print(f"   CV F1:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # ── 6. Select Best Model ─────────────────────────────────────
    best_model = models[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name} (F1={best_f1:.4f})")

    # ── 7. Feature Importance ────────────────────────────────────
    print("\n📈 Feature Importance (Top 15):")
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(feature_cols))

    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    for _, row in feat_imp.head(15).iterrows():
        bar = "█" * int(row["importance"] * 50 / max(feat_imp["importance"].max(), 1e-9))
        print(f"   {row['feature']:35s} {bar} {row['importance']:.4f}")

    # Save feature importance plot
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_feats = feat_imp.head(15)
    sns.barplot(
        data=top_feats, x="importance", y="feature",
        palette="viridis", ax=ax
    )
    ax.set_title("Top 15 Feature Importances — Churn Prediction", fontsize=14)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"\n📊 Feature importance plot saved to: {REPORT_DIR / 'feature_importance.png'}")

    # Confusion Matrix plot
    y_pred_best = best_model.predict(X_test_scaled)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Active", "Churned"],
                yticklabels=["Active", "Churned"])
    ax.set_title(f"Confusion Matrix — {best_model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"📊 Confusion matrix saved to: {REPORT_DIR / 'confusion_matrix.png'}")

    # ── 8. Save Artifacts ────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_FILE)
    print(f"\n💾 Model saved to: {MODEL_FILE}")

    joblib.dump(scaler, SCALER_FILE)
    print(f"💾 Scaler saved to: {SCALER_FILE}")

    with open(FEATURES_FILE, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"💾 Feature columns saved to: {FEATURES_FILE}")

    with open(METRICS_FILE, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "metrics": all_metrics,
        }, f, indent=2)
    print(f"💾 Metrics saved to: {METRICS_FILE}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE ✅")
    print("=" * 60)

    return best_model, scaler, feature_cols


if __name__ == "__main__":
    train_and_evaluate()
