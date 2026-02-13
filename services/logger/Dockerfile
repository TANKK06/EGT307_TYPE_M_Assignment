# Dockerfile for the Logger service (FastAPI)
# This service receives logs from inference and writes them to Postgres.

FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# -----------------------------
# Install dependencies
# -----------------------------
# Copy requirements first so Docker can cache dependency installation.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# -----------------------------
# Copy application code
# -----------------------------
# Copy the FastAPI logger app package (contains main.py, db.py, schemas.py, etc.)
COPY app /app/app

# Document the port the service listens on
EXPOSE 8001

# Start FastAPI with uvicorn
# app.main:app means:
#   - module: app/main.py
#   - FastAPI instance variable: app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
