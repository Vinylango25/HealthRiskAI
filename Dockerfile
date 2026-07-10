# =============================================================================
# HealthRisk AI — Multi-stage Dockerfile
# Stage 1: builder  — installs all Python dependencies
# Stage 2: final    — lean runtime image
# =============================================================================

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install all dependencies into /install
# Using requirements.txt directly ensures all transitive deps are resolved
COPY requirements.txt ./

RUN pip install \
    --no-cache-dir \
    --prefix=/install \
    --no-build-isolation \
    -r requirements.txt

# ── Stage 2: final ────────────────────────────────────────────────────────────
FROM python:3.10-slim AS final

LABEL maintainer="HealthRisk AI Team <contact@healthrisk.ai>"
LABEL version="0.1.0"
LABEL description="HealthRisk AI — clinical risk modelling & financial analytics platform"

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1001 healthrisk \
    && useradd --uid 1001 --gid healthrisk --shell /bin/bash --create-home healthrisk

# Copy application source
COPY --chown=healthrisk:healthrisk . .

# Create required runtime directories
RUN mkdir -p /app/mlruns /app/mlartifacts /app/logs /app/data/raw /app/data/processed \
    && chown -R healthrisk:healthrisk /app

USER healthrisk

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — start the FastAPI application via uvicorn
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
