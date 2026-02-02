from __future__ import annotations

import os
import psycopg2
from psycopg2.extras import Json

DATABASE_URL = os.getenv("DATABASE_URL", "")

_conn = None


def get_conn():
    global _conn
    if _conn is None or _conn.closed != 0:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set for logger service")
        _conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)
        _conn.autocommit = True
    return _conn


def _safe_get(d: dict, key: str):
    return d.get(key)


def insert_prediction_log(request_json: dict, response_json: dict, model_path: str | None):
    """
    Writes a row into predictions table INCLUDING the input feature columns.
    Your request_json keys come from inference: Type, Air_temperature_C, ...
    """
    conn = get_conn()
    req = request_json or {}

    type_ = _safe_get(req, "Type")
    air = _safe_get(req, "Air_temperature_C")
    proc = _safe_get(req, "Process_temperature_C")
    rpm = _safe_get(req, "Rotational_speed_rpm")
    torque = _safe_get(req, "Torque_Nm")
    wear = _safe_get(req, "Tool_wear_min")

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
