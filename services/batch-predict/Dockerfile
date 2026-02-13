# services/batch-predict/Dockerfile
# Dockerfile for the Batch Prediction service (FastAPI).
# This builds a lightweight container that runs uvicorn on port 8002.

# Use a small official Python base image (Debian slim) to keep image size smaller
FROM python:3.11-slim

# Set the working directory inside the container
# All subsequent paths (COPY/RUN) are relative to /app
WORKDIR /app

# -----------------------------
# Install Python dependencies
# -----------------------------
# Copy requirements file first so Docker can cache the pip install layer.
# (If your code changes but requirements.txt doesn't, Docker won't re-install packages.)
COPY services/batch-predict/requirements.txt /app/requirements.txt

# Install dependencies without caching to reduce image size
RUN pip install --no-cache-dir -r /app/requirements.txt

# -----------------------------
# Copy application source code
# -----------------------------
# Copy the FastAPI app code into the container.
# NOTE: folder name is "batch-predict" (dash), not "batch_predict" (underscore).
COPY services/batch-predict/app /app/app

# Expose the port the service listens on (for documentation; K8s uses containerPort/Service)
EXPOSE 8002

# Start the FastAPI server using uvicorn
# app.main:app means:
#   - Python module: app/main.py
#   - FastAPI instance variable: app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
