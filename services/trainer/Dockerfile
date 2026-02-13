# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# ---- Install dependencies ----
# Copy requirements first so Docker can cache dependency installation
COPY services/trainer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy training pipeline code ----
# Copy the ML pipeline source (config.py, main.py, preprocessing, etc.) + any default artifacts
COPY model /app/model

# ---- Copy trainer service code ----
# Copy the FastAPI trainer microservice (handles CSV upload + triggers training)
COPY services/trainer/app /app/app

# ---- Python import path ----
# Ensure the trainer container can import the training pipeline modules from /app/model/src
# Example imports: from config import Config, from main import main
ENV PYTHONPATH=/app/model/src

# ---- Runtime settings ----
# Trainer API port
EXPOSE 8003                              

# Start the FastAPI trainer app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
