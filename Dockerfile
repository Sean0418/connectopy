# =============================================================================
# Connectopy Analysis Pipeline - Docker Image
# =============================================================================
# Multi-stage build for optimized image size
#
# Usage:
#   docker build -t connectopy .
#   docker run -v /path/to/data:/app/data -v /path/to/output:/app/output connectopy
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and build package
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy and install the package
COPY pyproject.toml .
COPY connectopy/ connectopy/
COPY Runners/ Runners/
RUN pip install --no-cache-dir .

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal image for running the pipeline
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

LABEL org.opencontainers.image.title="Connectopy Analysis"
LABEL org.opencontainers.image.description="Reproducible connectomics analysis pipeline for brain connectivity data"
LABEL org.opencontainers.image.source="https://github.com/Sean0418/Brain-Connectome"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY connectopy/ connectopy/
COPY Runners/ Runners/
COPY Documentation/ Documentation/
COPY README.md .
COPY pyproject.toml .

# Create directories for data and output (to be mounted)
RUN mkdir -p /app/data /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: run the pipeline
ENTRYPOINT ["python", "Runners/run_pipeline.py"]
CMD ["--help"]
