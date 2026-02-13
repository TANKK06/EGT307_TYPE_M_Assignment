# services/dashboard/app.py
from __future__ import annotations

import io
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2
import requests
import streamlit as st

# ---------------- Service URLs (Kubernetes/Docker service hostnames) ----------------
# These are typically injected via ConfigMap/Secret in Kubernetes.
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8000/predict")
INFERENCE_HEALTH_URL = os.getenv("INFERENCE_HEALTH_URL", "http://inference:8000/health")

LOGGER_HEALTH_URL = os.getenv("LOGGER_HEALTH_URL", "http://logger:8001/health")

# ✅ Correct hostname uses dash: batch-predict
BATCH_PREDICT_URL = os.getenv("BATCH_PREDICT_URL", "http://batch-predict:8002/predict-file")
BATCH_PREDICT_HEALTH_URL = os.getenv("BATCH_PREDICT_HEALTH_URL", "http://batch-predict:8002/health")

TRAINER_URL = os.getenv("TRAINER_URL", "http://trainer:8003/train")
TRAINER_HEALTH_URL = os.getenv("TRAINER_HEALTH_URL", "http://trainer:8003/health")

# Database URL comes from Secret in Kubernetes (fallback is local/dev)
DB_URL = os.getenv("DATABASE_URL", "postgresql://pm_user:pm_pass@db:5432/pm_db")

# ---------------- Timezones ----------------
# ✅ Singapore timezone for display; many DB timestamps are stored in UTC
SG_TZ = ZoneInfo("Asia/Singapore")
UTC_TZ = ZoneInfo("UTC")

# ---------------- Page config + CSS ----------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Custom CSS: cards + badges + spacing
st.markdown(
    """
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
h1 { margin-bottom: 0.25rem; }
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }

/* Cards */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.03);
}
.card h3 { margin: 0 0 6px 0; font-size: 1.05rem; }
.card p { margin: 0; color: rgba(255,255,255,0.75); }

/* Badges */
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.85rem;
  margin-left: 8px;
}
.badge-up { background: rgba(34,197,94,0.18); color: rgb(34,197,94); border: 1px solid rgba(34,197,94,0.35); }
.badge-down { background: rgba(239,68,68,0.18); color: rgb(239,68,68); border: 1px solid rgba(239,68,68,0.35); }
.badge-warn { background: rgba(245,158,11,0.18); color: rgb(245,158,11); border: 1px solid rgba(245,158,11,0.35); }

/* Buttons spacing */
div.stButton>button { border-radius: 10px; padding: 0.55rem 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
st.title("Predictive Maintenance Dashboard")
st.caption("Predict • Logs • Batch Predict • Train Model • System Status")

# ---------------- Helpers (UI + Time) ----------------
def to_sg_time(dt):
    """
    Convert a datetime (naive or tz-aware) to Asia/Singapore time.

    Rules:
    - If dt is naive (no tzinfo), assume it is UTC (common for DB timestamps).
    - If dt is a string, parse it first.
    """
    if dt is None or pd.isna(dt):
        return dt

    # string -> datetime
    if isinstance(dt, str):
        dt = pd.to_datetime(dt, errors="coerce")

    # pandas Timestamp -> python datetime
    if hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()

    if dt is None:
        return dt

    # If timezone is missing, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)

    return dt.astimezone(SG_TZ)


def badge(ok: bool, label_up: str = "UP", label_down: str = "DOWN") -> str:
    """Return a small HTML badge showing UP/DOWN status."""
    if ok:
        return f'<span class="badge badge-up">✅ {label_up}</span>'
    return f'<span class="badge badge-down">❌ {label_down}</span>'


def card_html(title: str, badge_html: str, subtitle: str, lines: list[str]) -> str:
    """Generate a consistent HTML card for the UI."""
    lines_html = "".join([f"<p>{ln}</p>" for ln in lines if ln is not None])
    return f"""
<div class="card">
  <h3>{title} {badge_html}</h3>
  <p class="small-muted">{subtitle}</p>
  <div style="margin-top: 8px;">{lines_html}</div>
</div>
"""


# ---------------- DB Helpers ----------------
def get_conn():
    """Create a new PostgreSQL connection using psycopg2."""
    return psycopg2.connect(DB_URL)


def db_health() -> tuple[bool, str | None]:
    """
    Quick DB health check: run SELECT 1.
    Returns (ok, error_message).
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
        return True, None
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=10, show_spinner=False)
def fetch_recent_predictions_cached(limit: int) -> pd.DataFrame:
    """
    Fetch recent prediction logs from the database.

    Cached for 10 seconds to avoid spamming the DB when user refreshes UI frequently.
    """
    q = """
    SELECT
        id,
        created_at,
        type,
        air_temperature_c,
        process_temperature_c,
        rotational_speed_rpm,
        torque_nm,
        tool_wear_min,
        predicted_label,
        failure_probability,
        risk_level,
        model_path,
        request_json,
        response_json
    FROM predictions
    ORDER BY id DESC
    LIMIT %s;
    """
    with get_conn() as conn:
        df = pd.read_sql(q, conn, params=(limit,))
    return df


# ---------------- API Helpers ----------------
def call_inference(payload: dict) -> dict:
    """Send a single prediction request to the inference API."""
    r = requests.post(INFERENCE_URL, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def check_url(url: str) -> tuple[bool, int | None, float | None, str | None, dict | None]:
    """
    Check a health endpoint and measure latency.

    Returns:
        (ok, status_code, latency_ms, error_message, json_body)
    """
    try:
        t0 = datetime.now(SG_TZ)
        r = requests.get(url, timeout=5)
        latency_ms = (datetime.now(SG_TZ) - t0).total_seconds() * 1000.0
        r.raise_for_status()

        body = None
        try:
            body = r.json()
        except Exception:
            body = None

        return True, r.status_code, latency_ms, None, body

    except Exception as e:
        code = None
        if hasattr(e, "response") and e.response is not None:
            try:
                code = e.response.status_code
            except Exception:
                code = None
        return False, code, None, str(e), None


# ---------------- Sidebar Navigation ----------------
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "",
        [
            "Predict (Single)",
            "Logs (Database)",
            "Batch Predict (Upload CSV)",
            "Train New Model (Upload CSV)",
            "System Status",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

# ===================== PAGE 1: Predict (Single) =====================
if page == "Predict (Single)":
    st.subheader("Single Prediction")

    left, right = st.columns([1.1, 1.2], gap="large")

    with left:
        st.markdown("### Inputs")
        with st.form("predict_form", clear_on_submit=False):
            Type = st.selectbox("Type", ["L", "M", "H"], index=1)

            Air_temperature_C = st.number_input("Air temperature (°C)", value=27.0, step=0.1)
            Process_temperature_C = st.number_input("Process temperature (°C)", value=38.0, step=0.1)

            Rotational_speed_rpm = st.number_input("Rotational speed (rpm)", value=1500.0, step=10.0)
            Torque_Nm = st.number_input("Torque (Nm)", value=40.0, step=0.5)
            Tool_wear_min = st.number_input("Tool wear (min)", value=120, step=1)

            submitted = st.form_submit_button("Predict")

    with right:
        st.markdown("### Result")
        st.markdown(
            card_html(
                "How it works",
                '<span class="badge badge-warn">ℹ️ Flow</span>',
                "What happens after you click Predict",
                [
                    "1) Dashboard → Inference `/predict`",
                    "2) Inference → Logger (background)",
                    "3) Logger → Postgres (stores row)",
                    "4) View it in Logs page",
                ],
            ),
            unsafe_allow_html=True,
        )

        if submitted:
            # Build payload exactly as inference expects
            payload = {
                "Type": Type,
                "Air_temperature_C": float(Air_temperature_C),
                "Process_temperature_C": float(Process_temperature_C),
                "Rotational_speed_rpm": float(Rotational_speed_rpm),
                "Torque_Nm": float(Torque_Nm),
                "Tool_wear_min": int(Tool_wear_min),
            }

            try:
                with st.spinner("Calling inference..."):
                    result = call_inference(payload)

                # Extract outputs with safe defaults
                prob = float(result.get("failure_probability", 0.0))
                label = int(result.get("predicted_label", 0))
                risk = str(result.get("risk_level", "unknown"))

                # Show key results as metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Failure probability", f"{prob:.4f}")
                m2.metric("Predicted label", str(label))
                m3.metric("Risk level", risk)

                # Optional raw request/response JSON view
                with st.expander("Request / Response JSON", expanded=False):
                    st.json({"request": payload, "response": result})

                st.success("Done. Check **Logs** to confirm it was saved.")
            except Exception as e:
                st.error(f"Inference call failed: {e}")

# ===================== PAGE 2: Logs =====================
elif page == "Logs (Database)":
    st.subheader("Prediction Logs")

    top = st.columns([1, 1, 2], gap="large")
    with top[0]:
        limit = st.slider("Rows to show", min_value=10, max_value=500, value=50, step=10)
    with top[1]:
        refresh = st.button("Refresh", use_container_width=True)
    with top[2]:
        st.caption(f"DB: {DB_URL}")

    # Clear cache when user clicks refresh
    if refresh:
        fetch_recent_predictions_cached.clear()

    try:
        with st.spinner("Loading logs..."):
            df = fetch_recent_predictions_cached(limit)

        # Convert timestamps to SG time for display
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").apply(to_sg_time)

        if df.empty:
            st.info("No logs yet. Make a prediction first.")
        else:
            df_display = df.copy()

            # JSON columns -> pretty string for dataframe display
            df_display["request_json"] = df_display["request_json"].apply(lambda x: json.dumps(x, ensure_ascii=False))
            df_display["response_json"] = df_display["response_json"].apply(lambda x: json.dumps(x, ensure_ascii=False))

            tab1, tab2, tab3 = st.tabs(["Table", "Stats", "Trend"])

            with tab1:
                st.dataframe(df_display, use_container_width=True, height=420)

            with tab2:
                total = len(df)
                fail_rate = (df["predicted_label"] == 1).mean() if total else 0.0
                avg_prob = float(df["failure_probability"].mean()) if total else 0.0
                high_risk = int((df["risk_level"] == "high").sum()) if total else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows shown", total)
                c2.metric("Failure rate", f"{fail_rate:.1%}")
                c3.metric("Avg probability", f"{avg_prob:.4f}")
                c4.metric("High risk count", high_risk)

            with tab3:
                trend = df[["created_at", "failure_probability"]].copy()
                trend = trend.sort_values("created_at")
                trend.set_index("created_at", inplace=True)
                st.line_chart(trend)

    except Exception as e:
        st.error(f"Failed to load logs from DB: {e}")

# ===================== PAGE 3: Batch Predict =====================
elif page == "Batch Predict (Upload CSV)":
    st.subheader("Batch Prediction")
    st.caption("Upload a CSV → calls inference per row → downloads predictions.csv")

    st.markdown("**Required columns:**")
    st.code(
        "Type, Air_temperature_C, Process_temperature_C, Rotational_speed_rpm, Torque_Nm, Tool_wear_min",
        language="text",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_preview = pd.read_csv(uploaded)
            st.markdown("### Preview")
            st.dataframe(df_preview.head(12), use_container_width=True)

            colA, colB = st.columns([1, 2])
            with colA:
                run = st.button("Run batch prediction", use_container_width=True)

            if run:
                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                with st.spinner("Running batch prediction..."):
                    r = requests.post(BATCH_PREDICT_URL, files=files, timeout=300)
                    r.raise_for_status()

                st.success("Batch prediction completed!")
                st.download_button(
                    label="Download predictions.csv",
                    data=r.content,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# ===================== PAGE 4: Train New Model =====================
elif page == "Train New Model (Upload CSV)":
    st.subheader("Train a New Model")
    st.caption("Upload training CSV → trainer saves model → inference reloads")

    st.markdown("**Required columns:**")
    st.code(
        "Target, Type, Air_temperature_C, Process_temperature_C, Rotational_speed_rpm, Torque_Nm, Tool_wear_min",
        language="text",
    )

    uploaded = st.file_uploader("Upload Training CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_preview = pd.read_csv(uploaded)
            st.markdown("### Preview")
            st.dataframe(df_preview.head(12), use_container_width=True)

            colA, colB = st.columns([1, 2])
            with colA:
                train = st.button("Train model", use_container_width=True)

            if train:
                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                with st.spinner("Training... this may take a while"):
                    r = requests.post(TRAINER_URL, files=files, timeout=300)
                    r.raise_for_status()
                    out = r.json()

                st.success("Training completed!")
                st.json(out)
                st.info("Go to **System Status** to verify all services are UP, then run a new prediction.")

        except Exception as e:
            st.error(f"Training failed: {e}")

# ===================== PAGE 5: System Status =====================
elif page == "System Status":
    st.subheader("System Status")

    # Health endpoints to check
    checks = [
        ("Inference API", INFERENCE_HEALTH_URL),
        ("Logger Service", LOGGER_HEALTH_URL),
        ("Batch Predict", BATCH_PREDICT_HEALTH_URL),
        ("Trainer", TRAINER_HEALTH_URL),
    ]

    results = {}
    for name, url in checks:
        ok, code, latency_ms, err, body = check_url(url)
        results[name] = {
            "ok": ok,
            "code": code,
            "latency_ms": latency_ms,
            "err": err,
            "body": body,
            "url": url,
        }

    # DB check (separate because it's not HTTP)
    ok_db, db_err = db_health()

    # Dashboard is always "up" if you can see this page
    up_count = sum(1 for v in results.values() if v["ok"]) + (1 if ok_db else 0) + 1
    total_count = len(results) + 2  # db + dashboard

    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    c1.metric("Services UP", f"{up_count}/{total_count}")
    c2.metric("DB", "UP" if ok_db else "DOWN")
    c3.metric("Last check (SG)", datetime.now(SG_TZ).strftime("%Y-%m-%d %H:%M:%S"))

    st.markdown("### Services")

    grid = st.columns(3, gap="large")

    # --- Inference card ---
    inf = results["Inference API"]
    grid[0].markdown(
        card_html(
            "Inference API",
            badge(inf["ok"]),
            inf["url"],
            [
                f"HTTP: {inf['code']}" if inf["code"] is not None else "HTTP: -",
                f"Latency: {inf['latency_ms']:.0f} ms" if inf["latency_ms"] is not None else "Latency: -",
                (f"Error: {inf['err']}" if not inf["ok"] else "Healthy ✅"),
            ],
        ),
        unsafe_allow_html=True,
    )
    with grid[0].expander("Details", expanded=False):
        st.json(inf["body"]) if inf["body"] is not None else st.code(inf["err"] or "No JSON returned.")

    # --- Logger card ---
    log = results["Logger Service"]
    grid[1].markdown(
        card_html(
            "Logger Service",
            badge(log["ok"]),
            log["url"],
            [
                f"HTTP: {log['code']}" if log["code"] is not None else "HTTP: -",
                f"Latency: {log['latency_ms']:.0f} ms" if log["latency_ms"] is not None else "Latency: -",
                (f"Error: {log['err']}" if not log["ok"] else "Healthy ✅"),
            ],
        ),
        unsafe_allow_html=True,
    )
    with grid[1].expander("Details", expanded=False):
        st.json(log["body"]) if log["body"] is not None else st.code(log["err"] or "No JSON returned.")

    # --- Batch card ---
    bat = results["Batch Predict"]
    grid[2].markdown(
        card_html(
            "Batch Predict",
            badge(bat["ok"]),
            bat["url"],
            [
                f"HTTP: {bat['code']}" if bat["code"] is not None else "HTTP: -",
                f"Latency: {bat['latency_ms']:.0f} ms" if bat["latency_ms"] is not None else "Latency: -",
                (f"Error: {bat['err']}" if not bat["ok"] else "Healthy ✅"),
            ],
        ),
        unsafe_allow_html=True,
    )
    with grid[2].expander("Details", expanded=False):
        st.json(bat["body"]) if bat["body"] is not None else st.code(bat["err"] or "No JSON returned.")

    # Second row of cards
    grid2 = st.columns(3, gap="large")

    # --- Trainer card ---
    tr = results["Trainer"]
    grid2[0].markdown(
        card_html(
            "Trainer",
            badge(tr["ok"]),
            tr["url"],
            [
                f"HTTP: {tr['code']}" if tr["code"] is not None else "HTTP: -",
                f"Latency: {tr['latency_ms']:.0f} ms" if tr["latency_ms"] is not None else "Latency: -",
                (f"Error: {tr['err']}" if not tr["ok"] else "Healthy ✅"),
            ],
        ),
        unsafe_allow_html=True,
    )
    with grid2[0].expander("Details", expanded=False):
        st.json(tr["body"]) if tr["body"] is not None else st.code(tr["err"] or "No JSON returned.")

    # --- DB card ---
    grid2[1].markdown(
        card_html(
            "Postgres DB",
            badge(ok_db),
            "postgres (psycopg2 SELECT 1)",
            [
                "Connected ✅" if ok_db else "Connection failed ❌",
                (f"Error: {db_err}" if not ok_db else "Healthy ✅"),
            ],
        ),
        unsafe_allow_html=True,
    )
    with grid2[1].expander("Details", expanded=False):
        st.json({"status": "ok"}) if ok_db else st.code(db_err or "unknown error")

    # --- Dashboard card ---
    grid2[2].markdown(
        card_html(
            "Dashboard",
            '<span class="badge badge-up">✅ UP</span>',
            "self",
            ["Running ✅"],
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Troubleshooting")
    st.code(
        "docker compose ps\n"
        "docker compose logs inference --tail 80\n"
        "docker compose logs logger --tail 80\n"
        "docker compose logs batch-predict --tail 80\n"
        "docker compose logs trainer --tail 80\n"
        "docker compose logs db --tail 80\n"
        "docker compose logs dashboard --tail 80\n",
        language="text",
    )

    # Button to refresh statuses instantly
    if st.button("Refresh statuses"):
        st.rerun()
