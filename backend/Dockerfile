# Python FastAPI Backend for Language Annotation Tool
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install system dependencies required for building wheels and PostgreSQL support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip, pre-install lightweight CPU-only PyTorch (avoids downloading 4GB+ unused CUDA libraries),
# then install remaining backend dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Expose backend API port
EXPOSE 8000

# Run FastAPI via Uvicorn listening on all network interfaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
