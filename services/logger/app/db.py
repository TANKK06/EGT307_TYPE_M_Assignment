from __future__ import annotations

import os
import psycopg2
from psycopg2.extras import Json

# Connection string injected via Kubernetes Secret / env var
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Cached global connection (simple approach; autocommit enabled below)
_conn = None


def get_conn():
    """
    Get (or create) a PostgreSQL connection for the logger service.

    - Reuses a global connection for efficiency.
    - Reconnects automatically if connection is closed.
    - Uses autocommit so INSERT happens immediately (no explicit commit needed).
    """
    global _conn
    if _conn is None or _conn.closed != 0:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set for logger service")
        _conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)  # fast fail if DB is unreachable
        _conn.autocommit = True
    return _conn


def _safe_get(d: dict, key: str):
    """Small helper: safely read a key from a dict (returns None if missing)."""
    return d.get(key)


def insert_prediction_log(request_json: dict, response_json: dict, model_path: str | None):
    """
    Insert a prediction record into the `predictions` table.

    What gets stored:
    - Structured columns (type, temperatures, rpm, torque, wear, outputs) for fast filtering/dashboarding
    - Raw JSON payloads (request_json, response_json) for debugging/audit/replay
    - model_path so you know which model produced the output

    Notes:
    - request_json keys come from inference API:
      Type, Air_temperature_C, Process_temperature_C, Rotational_speed_rpm, Torque_Nm, Tool_wear_min
    """
    conn = get_conn()

    # Normalize request payload (avoid NoneType errors)
    req = request_json or {}

    # Extract input feature values from request JSON (may be None if missing)
    type_ = _safe_get(req, "Type")
    air = _safe_get(req, "Air_temperature_C")
    proc = _safe_get(req, "Process_temperature_C")
    rpm = _safe_get(req, "Rotational_speed_rpm")
    torque = _safe_get(req, "Torque_Nm")
    wear = _safe_get(req, "Tool_wear_min")

    # Insert into DB (Json() adapts python dict -> JSONB)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions (
                type,
                air_temperature_c,
                process_temperature_c,
                rotational_speed_rpm,
                torque_nm,
                tool_wear_min,
                request_json,
                response_json,
                failure_probability,
                predicted_label,
                risk_level,
                model_path
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                type_,
                air,
                proc,
                rpm,
                torque,
                wear,
                Json(request_json),                         # raw request payload (JSONB)
                Json(response_json),                        # raw response payload (JSONB)
                response_json.get("failure_probability"),   # output: probability
                response_json.get("predicted_label"),       # output: label
                response_json.get("risk_level"),            # output: risk band
                model_path,                                 # which model artifact was used
            ),
        )
