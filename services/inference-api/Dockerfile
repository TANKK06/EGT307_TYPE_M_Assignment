# Dockerfile for the Inference API service (FastAPI + sklearn model)
# This container serves predictions on port 8000 and loads a trained model from MODEL_PATH.

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# -----------------------------
# Install dependencies
# -----------------------------
# Copy requirements first so Docker can cache dependency installation.
# (If only code changes, Docker won't reinstall packages.)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# -----------------------------
# Copy application code + model files
# -----------------------------
# Copy FastAPI code (e.g., app/main.py, schemas.py, etc.)
COPY app /app/app

# Copy the bundled model folder into the image.
# Note: In Kubernetes, you may override this by mounting a PVC at /app/model_store
# and pointing MODEL_PATH there.
COPY model /app/model

# The container will listen on 8000 (matches Kubernetes containerPort and Service targetPort)
EXPOSE 8000

# Default model path inside the container.
# Can be overridden at runtime (e.g., Kubernetes env var in Deployment).
ENV MODEL_PATH=/app/model/best_model.joblib

# Start the FastAPI server
# app.main:app means:
#   - module: app/main.py
#   - FastAPI object: app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
