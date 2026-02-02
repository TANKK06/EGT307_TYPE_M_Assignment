# services/batch-predict/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY services/batch-predict/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy service code  ✅ (dash, not underscore)
COPY services/batch-predict/app /app/app

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
