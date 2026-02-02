FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY services/trainer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy training pipeline + artifacts
COPY model /app/model

# Copy trainer service code
COPY services/trainer/app /app/app

# Make sure Python can import config.py, data.py, etc from /app/model/src
ENV PYTHONPATH=/app/model/src

EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
