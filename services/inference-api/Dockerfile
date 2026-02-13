# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# ---- Install dependencies ----
# Copy requirements first so Docker can cache the pip install layer
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy application code + default model ----
# Copy FastAPI inference service code
COPY app /app/app

# Copy a default model into the image (useful for local/dev)
# In Kubernetes you may override MODEL_PATH to point at a PVC-mounted model_store.
COPY model /app/model

# ---- Runtime settings ----
# Inference API port
EXPOSE 8000                               

# Default model location (can be overridden by env var MODEL_PATH)
ENV MODEL_PATH=/app/model/best_model.joblib

# Start the FastAPI inference app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
