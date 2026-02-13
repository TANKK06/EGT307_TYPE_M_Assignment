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
# Copy the logger service source code into the container
COPY app /app/app

# ---- Runtime settings ----
# Logger API port
EXPOSE 8001                          

# Start the FastAPI app using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
