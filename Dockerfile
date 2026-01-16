# Base image with CUDA support - compatible with typical cluster environments
# Using PyTorch 2.1 which is stable and supports recent CUDA versions
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
# ffmpeg is required for whisper/librosa audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Default command
CMD ["python", "src/data_processing.py"]
