# ─────────────────────────────────────────────────────────────────────
# VoxCPM Local API Server — Dockerfile
# ─────────────────────────────────────────────────────────────────────
# Default base image is lightweight python:3.10-slim.
# For CUDA GPU acceleration, build with:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime -t voxcpm-api .
# ─────────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

LABEL maintainer="OpenBMB <openbmb@gmail.com>"
LABEL description="VoxCPM Local REST & Streaming API Server"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/root/.cache/huggingface

# System audio libraries, git, curl, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libsndfile1 \
        ffmpeg \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Layer 1: Copy metadata & install dependencies (cached unless pyproject.toml changes)
COPY pyproject.toml /app/
RUN mkdir -p /app/src/voxcpm && echo '__version__ = "0.0.0"' > /app/src/voxcpm/__init__.py
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Install dependencies plus FastAPI, Uvicorn, and Python Multipart
RUN pip install --no-cache-dir -e . "fastapi" "uvicorn[standard]" "python-multipart"

# Layer 2: Copy full project codebase
COPY . /app/

# Reinstall voxcpm in editable mode with actual source code
RUN pip install --no-cache-dir -e .

# Create cache and output directories
RUN mkdir -p /root/.cache/huggingface /app/outputs

EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default startup command
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]
