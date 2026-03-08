# Base Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app/src
ENV SIM_ENGINE=pybullet

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV, Pybullet, and Open3D
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the correct port for Google Cloud Run (defaults to 8080)
EXPOSE 8080

# Run the FastAPI server via Uvicorn with the port bound dynamically for Cloud Run
CMD ["uvicorn", "welding_simulator.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
