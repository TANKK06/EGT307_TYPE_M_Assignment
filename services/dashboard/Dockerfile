# Dockerfile for the Streamlit Dashboard service
# Purpose:
# - Install Python dependencies
# - Copy the Streamlit app (app.py) and UI assets (ui/ folder)
# - Run Streamlit on port 8501 so it can be accessed from the host / Kubernetes service

FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# -----------------------------
# Install dependencies
# -----------------------------
# Copy requirements first so Docker can cache the pip install layer.
# If your code changes but requirements.txt stays the same, Docker won't reinstall packages.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# -----------------------------
# Copy application code + static assets
# -----------------------------
# Copy the main Streamlit script
COPY app.py /app/app.py

# Copy additional UI files (images, css, components, etc.)
COPY ui /app/ui

# Document the Streamlit port (Compose/K8s still maps it explicitly)
EXPOSE 8501

# -----------------------------
# Start Streamlit server
# -----------------------------
# --server.address=0.0.0.0 makes it reachable outside the container
# --server.port=8501 matches the containerPort/Service port used in your deployment
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
