from __future__ import annotations

import json
from typing import Any

import pandas as pd
import psycopg2
import streamlit as st

from ui.config import DB_URL


def get_conn():
    """
    Create and return a new Postgres connection.

    Why a new connection each time:
    - Keeps the code simple (no global connection/pooling in the dashboard).
    - Using `with get_conn() as conn:` ensures the connection is closed automatically.

    Note:
    - If your dashboard scales up, you may want connection pooling.
    - For a student project / small dashboard, this approach is fine.
    """
    return psycopg2.connect(DB_URL)


@st.cache_data(ttl=5)
def fetch_recent_predictions(limit: int = 200) -> pd.DataFrame:
    """
    Fetch the most recent prediction logs from the database.

    Why we cache:
    - Streamlit reruns the script very often (every click, slider move, etc.).
    - Caching for a few seconds reduces repeated identical DB queries.

    Args:
        limit:
            Max number of rows to return (sorted newest -> oldest).

    Returns:
        Pandas DataFrame of recent prediction logs for display in the UI.
    """
    sql = """
      SELECT
        created_at,
        request_json,
        response_json,
        predicted_label,
        failure_probability,
        risk_level
      FROM predictions
      ORDER BY created_at DESC
      LIMIT %s
    """

    # Use a context manager so the connection closes even if an error happens
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=(limit,))


def safe_json_dumps(value: Any) -> str:
    """
    Convert values from the DB into a display-friendly JSON string.

    Why this is needed:
    - request_json/response_json columns may come back as dicts, JSON strings, or plain text.
    - Streamlit tables look cleaner if we convert them to a consistent string format.

    Handles:
    - None -> ""
    - dict/list -> JSON string
    - string -> tries to parse JSON, otherwise returns the string
    - other types -> str(value)

    Returns:
        A compact JSON string if possible, otherwise a safe string version.
    """
    # Handle NULLs from DB
    if value is None:
        return ""

    # Already parsed JSON (common when using JSONB)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    # If it's a string, it might be JSON — try parsing it.
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            # Not JSON, return the original string
            return value

    # Fallback for numbers, timestamps, etc.
    return str(value)
