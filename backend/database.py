"""
Database Module
===============
Provides sync and async database connection utilities.
"""

import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from .config import settings


@contextmanager
def get_db_connection():
    """Synchronous database connection context manager."""
    conn = psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_query(sql: str, params: dict = None) -> list:
    """Execute a query and return results as list of dicts."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
