# Base image with CUDA support for GPU acceleration
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /app/logs /app/data/temp /app/data/models

# Upgrade pip and build tools
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA support (compatible with PyTorch 2.6+)
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure correct permissions
RUN chmod -R 755 /app

# Make scripts executable
RUN chmod +x run_patched.py

# Expose the port the app runs on
EXPOSE 8000

# Define volumes for data persistence
VOLUME ["/app/data/models", "/app/data/temp", "/app/logs"]

# Set the entrypoint using run_patched.py to handle PyTorch 2.6+ compatibility
CMD ["python3", "run_patched.py"]