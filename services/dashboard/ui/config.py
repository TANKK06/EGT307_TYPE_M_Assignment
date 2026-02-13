"""
Central configuration for the Streamlit dashboard.

Why this file exists:
- Keeps all service URLs, DB connection strings, and timezones in ONE place
- Makes it easy to run in different environments (local Docker, Kubernetes, etc.)
  by overriding values via environment variables.

Best practice:
- In Docker/K8s, set the env vars instead of editing the code.
"""

import os
from zoneinfo import ZoneInfo

# -----------------------------
# Service URLs (override via env vars)
# -----------------------------
# These default URLs use Docker Compose service names (inference, batch-predict, trainer, logger).
# In Kubernetes, you typically keep similar names (Service names), or override via ConfigMap.

# Inference API:
# - INFERENCE_URL: endpoint used for single prediction (dashboard -> inference)
# - INFERENCE_HEALTH_URL: endpoint used to show "UP/DOWN" status in the UI
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8000/predict")
INFERENCE_HEALTH_URL = os.getenv("INFERENCE_HEALTH_URL", "http://inference:8000/health")

# Batch prediction service:
# - BATCH_PREDICT_URL: upload CSV endpoint (dashboard -> batch-predict)
# - BATCH_PREDICT_HEALTH_URL: status endpoint for System Status page
BATCH_PREDICT_URL = os.getenv("BATCH_PREDICT_URL", "http://batch-predict:8002/predict-file")
BATCH_PREDICT_HEALTH_URL = os.getenv(
    "BATCH_PREDICT_HEALTH_URL",
    "http://batch-predict:8002/health",
)

# Trainer service:
# - TRAINER_URL: upload CSV -> retrain endpoint (dashboard -> trainer)
# - TRAINER_HEALTH_URL: status endpoint for System Status page
TRAINER_URL = os.getenv("TRAINER_URL", "http://trainer:8003/train")
TRAINER_HEALTH_URL = os.getenv("TRAINER_HEALTH_URL", "http://trainer:8003/health")

# Logger service:
# - LOGGER_HEALTH_URL: status endpoint (logger itself may be "up" even if DB is down)
LOGGER_HEALTH_URL = os.getenv("LOGGER_HEALTH_URL", "http://logger:8001/health")

# -----------------------------
# Database
# -----------------------------
# Postgres connection string used by the dashboard to READ prediction logs.
# Notes:
# - In Compose, hostname "db" resolves automatically inside the compose network.
# - In Kubernetes, set DATABASE_URL via a Secret for safety (don’t hardcode passwords).
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://pm_user:pm_pass@db:5432/pm_db",
)

# -----------------------------
# Timezones
# -----------------------------
# Used to display timestamps consistently in the dashboard:
# - DB often stores timestamps in UTC (or tz-aware TIMESTAMPTZ).
# - We convert to Singapore time for display.
SG_TZ = ZoneInfo("Asia/Singapore")
UTC_TZ = ZoneInfo("UTC")
