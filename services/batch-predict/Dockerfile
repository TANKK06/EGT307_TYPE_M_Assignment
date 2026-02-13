# services/batch-predict/Dockerfile
# Small Python base image (faster pull than full)
FROM python:3.11-slim                   

# All commands run from /app
WORKDIR /app                            

# ---- Install dependencies ----
# Copy requirements first so Docker can cache pip install when code changes
COPY services/batch-predict/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt  # Install Python packages

# ---- Copy application code ----
# Copy the batch-predict service source code into the container
COPY services/batch-predict/app /app/app

# ---- Runtime settings ----
# Container listens on port 8002

EXPOSE 8002                             
# Start FastAPI using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
