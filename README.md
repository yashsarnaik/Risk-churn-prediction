# Patient Churn Prediction — ML Pipeline

An end-to-end machine learning system that predicts which patients are likely to drop out of a health program, built with PostgreSQL, Python (XGBoost), FastAPI, and Next.js.

---

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌────────────────┐     ┌──────────────┐
│  PostgreSQL  │────▶│  ML Pipeline   │────▶│  FastAPI API   │────▶│  Next.js UI  │
│  (6 tables)  │     │  (XGBoost)     │     │  (4 endpoints) │     │  (Dashboard) │
└──────────────┘     └────────────────┘     └────────────────┘     └──────────────┘
```

## Folder Structure

```
├── .env                          # Database & config
├── requirements.txt              # Python dependencies
├── sql/
│   └── feature_extraction.sql    # CTE-based feature query (40+ features)
├── ml/
│   ├── extract_data.py           # DB → CSV extraction
│   ├── feature_engineering.py    # Feature transforms + churn labeling
│   ├── train_model.py            # XGBoost / RF / LR training
│   ├── predict.py                # Prediction pipeline
│   └── models/                   # Saved model artifacts
│       ├── churn_model.joblib
│       ├── scaler.joblib
│       ├── feature_columns.json
│       └── metrics.json
├── data/
│   ├── churn_dataset.csv         # Raw extracted features
│   ├── churn_processed.csv       # Engineered features
│   ├── feature_importance.png    # Feature importance chart
│   └── confusion_matrix.png     # Confusion matrix chart
├── backend/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   ├── config.py                 # Settings from .env
│   ├── database.py               # DB connection utilities
│   └── schemas.py                # Pydantic models
└── frontend/
    ├── package.json
    ├── next.config.js            # API proxy config
    ├── app/
    │   ├── layout.js
    │   ├── page.js               # Dashboard page
    │   └── globals.css           # Design system
    └── components/
        ├── PatientSearch.js
        ├── RiskGauge.js
        └── ActivityChart.js
```

---

## Quick Start

### 1. Environment Setup

```bash
# .env is already configured with your DB credentials
# Verify .env has correct values:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drapp
DB_USER=postgres
DB_PASSWORD=123
```

### 2. Install Python Dependencies (with uv)

```bash
uv venv
uv pip install -r requirements.txt
```

### 3. Run the ML Pipeline

```bash
# Step 1: Extract features from PostgreSQL
cd ml
python extract_data.py

# Step 2: Train the model (auto-runs feature engineering)
python train_model.py
```

This will:
- Extract 40+ features for 327 patients
- Engineer churn labels (14-day inactivity threshold)
- Train XGBoost, Random Forest, and Logistic Regression
- Select the best model based on F1 score
- Save model artifacts to `ml/models/`
- Generate evaluation charts in `data/`

### 4. Start the FastAPI Backend

```bash
# From the project root directory
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 5. Start the Frontend

```bash
cd frontend
npm install   # or: yarn install
npm run dev
```

The dashboard will be available at `http://localhost:3000`

---

## API Reference

### `GET /health`
Health check endpoint.

```bash
curl http://localhost:8000/health
```

### `POST /predict-churn`
Predict churn risk for a specific patient.

```bash
curl -X POST http://localhost:8000/predict-churn \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "ab5ac142-1c54-4f0a-8600-f261ee04862c"}'
```

**Response:**
```json
{
  "patient_id": "ab5ac142-...",
  "churn_probability": 0.7234,
  "churn_prediction": 1,
  "risk_label": "High",
  "key_features": {
    "total_meal_logs": 0,
    "total_weight_logs": 1,
    "meal_compliance_rate": 0,
    "engagement_score": 0.12,
    "enrollment_age_days": 45
  }
}
```

### `GET /patients/at-risk`
Get all patients ranked by churn risk.

```bash
curl "http://localhost:8000/patients/at-risk?limit=10"
curl "http://localhost:8000/patients/at-risk?risk_filter=High"
```

### `GET /patient/{patient_id}/activity`
Get daily activity timeline for a patient (last 30 days).

```bash
curl http://localhost:8000/patient/ab5ac142-1c54-4f0a-8600-f261ee04862c/activity
```

### `GET /patients/search`
Search patients by name or ID.

```bash
curl "http://localhost:8000/patients/search?q=Ajay"
```

### `GET /model/metrics`
Get saved model evaluation metrics.

```bash
curl http://localhost:8000/model/metrics
```

---

## Features Extracted (40+)

| Category | Features |
|---|---|
| **Demographics** | gender, age, height, weight, medical conditions, allergies |
| **Enrollment** | enrollment age, active status, assigned providers |
| **Meal Logs** | total logs, distinct days, compliance rate, calories, macros, recency, 7/14-day counts |
| **Weight Logs** | total logs, weight change, BMI, recency |
| **Exercise Logs** | total logs, duration, calories burned, steps, recency |
| **Collection Progress** | items completed, completion rate, progress %, recency |
| **Engineered** | engagement score, recency decay scores, logging rates, provider count |

## Churn Definition

A patient is classified as **churned** if:
- They have **zero activity** across all channels (meals, weight, exercise, collections), **OR**
- Their most recent activity across **all channels** is older than **14 days**

This threshold is configurable via `CHURN_THRESHOLD_DAYS` in `.env`.

---

## Model Details

- **Primary**: XGBoost with regularization (max_depth=4, min_child_weight=3)
- **Comparison**: Random Forest, Logistic Regression
- **Selection**: Best model chosen by F1 score
- **Validation**: Stratified 80/20 split + K-fold cross-validation
- **Class balancing**: `scale_pos_weight` (XGBoost) / `class_weight='balanced'` (RF/LR)
