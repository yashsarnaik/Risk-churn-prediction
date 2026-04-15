"""
Data Extraction Script
======================
Connects to PostgreSQL, runs the feature extraction SQL query,
and exports the result as a CSV file for model training.
"""

import os
import sys
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "drapp"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123"),
}

SQL_FILE = Path(__file__).resolve().parent.parent / "sql" / "feature_extraction.sql"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "churn_dataset.csv"


def load_sql_query() -> str:
    """Read the feature extraction SQL from file."""
    with open(SQL_FILE, "r", encoding="utf-8") as f:
        return f.read()


def extract_data() -> pd.DataFrame:
    """Connect to PostgreSQL, execute feature query, return DataFrame."""
    print("📡 Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)

    print("⚙️  Running feature extraction query...")
    sql = load_sql_query()
    df = pd.read_sql_query(sql, conn)

    conn.close()
    print(f"✅ Extracted {len(df)} patient records with {len(df.columns)} features")
    return df


def save_csv(df: pd.DataFrame) -> None:
    """Save DataFrame to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved dataset to: {OUTPUT_FILE}")


def main():
    df = extract_data()

    # Quick data summary
    print("\n📊 Dataset Summary:")
    print(f"   Patients:  {len(df)}")
    print(f"   Features:  {len(df.columns)}")
    print(f"   Columns:   {list(df.columns)}")
    print(f"\n   Null counts per column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts[null_counts > 0].items():
        print(f"     {col}: {count}")

    save_csv(df)
    return df


if __name__ == "__main__":
    main()
