from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
from ui.components import card, section_title, status_badge
from ui.config import (
    BATCH_PREDICT_HEALTH_URL,
    BATCH_PREDICT_URL,
    INFERENCE_HEALTH_URL,
    INFERENCE_URL,
    LOGGER_HEALTH_URL,
    TRAINER_HEALTH_URL,
    TRAINER_URL,
    SG_TZ,
)
from ui.db import fetch_recent_predictions, safe_json_dumps

# Create one API client instance to reuse HTTP connections (faster + cleaner).
client = ApiClient(timeout=30)


def _show_api_result(result, success_title: str = "Success", error_title: str = "Error") -> None:
    """
    Standard way to display API call results in Streamlit.

    - If result.ok:
        show a green success message + display JSON response (if any)
    - If result.ok is False:
        show a red error message + print debug details (error string)

    This keeps all pages consistent and avoids repeating the same UI code everywhere.
    """
    if result.ok:
        st.success(success_title)
        if result.data is not None:
            st.json(result.data)
    else:
        st.error(error_title)
        st.code(result.error or "Unknown error")


def render_predict() -> None:
    """
    Page: Single Prediction

    Collects sensor inputs from the user and sends them to the inference service.
    """
    section_title(
        "Single Prediction",
        "Enter sensor readings and get a prediction from the inference service.",
    )

    # Using a form ensures the API call only happens when user clicks "Predict"
    # (Streamlit reruns on every input change otherwise).
    with st.form("predict_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        # --- Column 1: Temperatures ---
        with col1:
            # NOTE: Your inference API schema expects Air_temperature_C / Process_temperature_C
            # but this UI currently collects Kelvin. Make sure inference supports Kelvin
            # or change these inputs to °C to match PredictRequest.
            air_temp = st.number_input("Air temperature [K]", value=300.0, step=0.1)
            proc_temp = st.number_input("Process temperature [K]", value=310.0, step=0.1)

        # --- Column 2: Speed + torque ---
        with col2:
            rot_speed = st.number_input("Rotational speed [rpm]", value=1500.0, step=1.0)
            torque = st.number_input("Torque [Nm]", value=40.0, step=0.1)

        # --- Column 3: Wear + type ---
        with col3:
            tool_wear = st.number_input("Tool wear [min]", value=0.0, step=1.0)
            machine_type = st.selectbox("Machine type", ["L", "M", "H"], index=1)

        submitted = st.form_submit_button("Predict")

    # Stop if user hasn't clicked the submit button yet
    if not submitted:
        return

    # Payload keys MUST match what inference expects.
    # WARNING: This payload uses Kelvin column names; your PredictRequest expects *_C fields.
    # Fix by changing to:
    #   "Air_temperature_C": air_temp,
    #   "Process_temperature_C": proc_temp,
    # (and ensure the values are in °C).
    payload = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type": machine_type,
    }

    # Call inference API
    with st.spinner("Calling inference service..."):
        res = client.post_json(INFERENCE_URL, payload)

    _show_api_result(res, success_title="Prediction received", error_title="Prediction failed")


def render_logs() -> None:
    """
    Page: Prediction Logs

    Fetch recent logs from Postgres and display them in a table.
    """
    section_title("Prediction Logs", "View recent predictions recorded in Postgres.")

    # User controls how many rows to fetch from the DB
    limit = st.slider("Number of rows", min_value=50, max_value=1000, value=200, step=50)

    # Cached DB query (ttl=5s) to reduce load during Streamlit reruns
    df = fetch_recent_predictions(limit=limit)
    if df.empty:
        st.info("No logs yet.")
        return

    # Copy to avoid mutating the cached dataframe
    df_display = df.copy()

    # JSONB columns are not always pretty in dataframe view,
    # so we convert to compact JSON strings.
    df_display["request_json"] = df_display["request_json"].apply(safe_json_dumps)
    df_display["response_json"] = df_display["response_json"].apply(safe_json_dumps)

    st.dataframe(df_display, use_container_width=True)

    # Quick stats panel under expander
    with st.expander("Quick stats"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows shown", len(df_display))

        # Only show risk counts if column exists
        if "risk_level" in df_display.columns:
            counts = df_display["risk_level"].value_counts(dropna=False)
            col2.metric("Risk levels", len(counts))
            col3.write(counts)


def render_batch_predict() -> None:
    """
    Page: Batch Predict

    Upload a CSV and send it to the batch-predict service.
    The service returns predictions for each row.
    """
    section_title("Batch Predict", "Upload a CSV and get predictions for all rows.")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if not uploaded:
        # Friendly info card when no file is selected
        card(
            "Tip",
            "Your CSV should contain the columns required by batch-predict "
            "(e.g., Type, Air_temperature_C, Process_temperature_C, Rotational_speed_rpm, Torque_Nm, Tool_wear_min).",
        )
        return

    # Button prevents auto-triggering API call when file changes
    if st.button("Run batch prediction"):
        with st.spinner("Uploading and processing..."):
            res = client.post_file(BATCH_PREDICT_URL, uploaded.getvalue(), uploaded.name)

        _show_api_result(res, success_title="Batch prediction complete", error_title="Batch prediction failed")


def render_train() -> None:
    """
    Page: Train New Model

    Upload a training CSV and send it to the trainer service.
    Trainer retrains and saves a new model to shared storage, then triggers inference reload.
    """
    section_title("Train New Model", "Upload a training CSV to retrain the model via the trainer service.")

    # Use a unique key so this uploader doesn't conflict with the batch uploader
    uploaded = st.file_uploader("Upload training CSV", type=["csv"], key="train_csv")
    if not uploaded:
        card(
            "Tip",
            "Training may take time. Ensure the trainer service is running and sharing model storage with inference.",
        )
        return

    if st.button("Start training"):
        with st.spinner("Starting training..."):
            res = client.post_file(TRAINER_URL, uploaded.getvalue(), uploaded.name)

        _show_api_result(res, success_title="Training started", error_title="Training failed")


def render_status() -> None:
    """
    Page: System Status

    Calls /health endpoints for each service and displays their status.
    """
    section_title("System Status", "Quick health checks for services.")

    # Services to check (name, health_url)
    services = [
        ("Inference", INFERENCE_HEALTH_URL),
        ("Batch Predict", BATCH_PREDICT_HEALTH_URL),
        ("Trainer", TRAINER_HEALTH_URL),
        ("Logger", LOGGER_HEALTH_URL),
    ]

    rows = []
    for name, url in services:
        # Use ApiClient wrapper so errors are handled consistently
        res = client.get_health(url)

        # Details: show a short preview of JSON/text for debugging
        details = ""
        if res.data is not None:
            if isinstance(res.data, (dict, list)):
                details = json.dumps(res.data, ensure_ascii=False)[:220]
            else:
                details = str(res.data)[:220]
        if res.error:
            details = str(res.error)[:220]

        rows.append(
            {
                "Service": name,
                "Status": "OK" if res.ok else "Down",
                "Details": details,
            }
        )

    df = pd.DataFrame(rows)

    # Summary card: overall OK only if every service is OK
    all_ok = all(r["Status"] == "OK" for r in rows)
    st.markdown(
        f"""
        <div class="pm-card">
          <div style="font-weight:700;">Health Summary</div>
          <div class="pm-subtle" style="margin-top:6px;">
            {status_badge(all_ok)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed table of each service
    st.dataframe(df, use_container_width=True)

    # Show timestamp for when health checks were run (Singapore time)
    st.caption(f"Checked at {datetime.now(SG_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
