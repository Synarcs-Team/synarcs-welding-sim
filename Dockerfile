FROM python:3.11-slim

# Install system dependencies required for OpenCV, PyBullet, and ffmpeg
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement and install Python dependencies
# open3d is specifically installed via pip here, ensuring a compatible wheel exists for python 3.11
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir open3d

# Copy the entire project
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Set environment variable to use PyBullet engine instead of Isaac Sim
ENV SIM_ENGINE=pybullet
# Set PYTHONPATH so modules resolve correctly
ENV PYTHONPATH=/app/src

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "welding_simulator.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
