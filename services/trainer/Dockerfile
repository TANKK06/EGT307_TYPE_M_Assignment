# Dockerfile for the Trainer service (FastAPI)
# Purpose:
# - Accept a training CSV upload
# - Run the existing ML training pipeline inside /app/model/src
# - Save best_model.joblib (and metrics.json) into a shared model store volume
# - Ask the inference service to reload the updated model

FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# -----------------------------
# Install Python dependencies
# -----------------------------
# Copy requirements first so Docker can cache pip install.
COPY services/trainer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# -----------------------------
# Copy ML training pipeline code
# -----------------------------
# This includes model/src/* (config.py, main.py, training.py, etc.)
# and any default artifacts/data that are shipped with the repo.
COPY model /app/model

# -----------------------------
# Copy trainer service API code
# -----------------------------
# This is the FastAPI service that exposes /train endpoint.
COPY services/trainer/app /app/app

# -----------------------------
# Python import path setup
# -----------------------------
# The training service imports files like:
#   from config import Config
#   from main import main as pipeline_main
# Those modules live in /app/model/src, so we add it to PYTHONPATH.
ENV PYTHONPATH=/app/model/src

# Document which port the trainer service listens on
EXPOSE 8003

# Start FastAPI using uvicorn
# app.main:app means:
#   - module: app/main.py
#   - FastAPI instance: app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
