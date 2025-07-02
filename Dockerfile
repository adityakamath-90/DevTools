# DevTools - AI-Powered Kotlin Documentation & Test Generator
# Multi-stage build for optimized production image

FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY main.py .

# Create directories for input/output/data
RUN mkdir -p /app/input /app/output /app/data /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Default environment variables (can be overridden)
ENV OLLAMA_API_URL=http://ollama:11434/api/generate
ENV MODEL_NAME=codellama:instruct

# Make main.py executable
RUN chmod +x main.py

# Health check to ensure the application is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.path.insert(0, '/app/src'); from LLMClient import LLMClient" || exit 1

# Default command - can be overridden in docker-compose or docker run
CMD ["python3", "main.py", "--help"]
