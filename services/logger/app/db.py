from __future__ import annotations

import os
import psycopg2
from psycopg2.extras import Json

# Database connection string is provided via env var (Docker Compose / K8s Secret)
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Keep a single global connection to reduce overhead (simple connection pooling)
_conn = None


def get_conn():
    """
    Get (or create) a Postgres connection for the logger service.

    Behavior:
    - Reuses a global connection if it's still open.
    - Reconnects automatically if the connection was closed.
    - Sets autocommit=True so each INSERT is committed immediately.

    Raises:
        RuntimeError: if DATABASE_URL is not set.
    """
    global _conn

    # psycopg2 connection has `.closed` attribute:
    # 0 = open, non-zero = closed
    if _conn is None or _conn.closed != 0:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set for logger service")

        # connect_timeout prevents the service from hanging too long if DB is down
        _conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)

        # Autocommit so we don't need explicit conn.commit() after each insert
        _conn.autocommit = True

    return _conn


def _safe_get(d: dict, key: str):
    """
    Small helper to fetch a key from a dict safely.
    (Keeps code tidy and avoids KeyError.)
    """
    return d.get(key)


def insert_prediction_log(request_json: dict, response_json: dict, model_path: str | None):
    """
    Insert a prediction log row into the 'predictions' table.

    What we store:
    1) Raw JSON request + response (for traceability and debugging)
    2) Extracted input features (so dashboard can filter/query easily)
    3) Extracted outputs (probability, label, risk)
    4) model_path (so we know which model version made the prediction)

    Notes:
    - request_json keys are expected to come from inference:
      Type, Air_temperature_C, Process_temperature_C, Rotational_speed_rpm, Torque_Nm, Tool_wear_min
    """
    conn = get_conn()

    # Ensure we always work with a dict (avoid None issues)
    req = request_json or {}

    # -----------------------------
    # Extract input fields from request JSON
    # -----------------------------
    type_ = _safe_get(req, "Type")
    air = _safe_get(req, "Air_temperature_C")
    proc = _safe_get(req, "Process_temperature_C")
    rpm = _safe_get(req, "Rotational_speed_rpm")
    torque = _safe_get(req, "Torque_Nm")
    wear = _safe_get(req, "Tool_wear_min")

    # -----------------------------
    # Insert into DB
    # -----------------------------
    # Json(...) ensures psycopg2 stores dict as JSONB properly.
    # We insert both raw JSON and extracted columns for easy dashboards.
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
                Json(request_json),     
                Json(response_json),      
                response_json.get("failure_probability"),
                response_json.get("predicted_label"),
                response_json.get("risk_level"),
                model_path,
            ),
        )
