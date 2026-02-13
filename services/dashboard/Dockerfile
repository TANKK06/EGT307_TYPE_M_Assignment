# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# ---- Install dependencies ----
# Copy requirements first so Docker can cache the dependency layer
COPY requirements.txt /app/requirements.txt

# Install Python packages listed in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy application code ----
# Copy the Streamlit dashboard script into the container
COPY app.py /app/app.py

# ---- Runtime settings ----
# Streamlit runs on port 8501 by default
EXPOSE 8501

# Start the Streamlit app and listen on all interfaces so Kubernetes can reach it
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
