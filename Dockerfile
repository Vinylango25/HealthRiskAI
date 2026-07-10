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

# Upgrade pip and install wheel / setuptools first for faster builds
RUN pip install --upgrade pip setuptools wheel

# Copy only the dependency manifest to maximise Docker layer caching.
# The full source is copied in the final stage.
COPY pyproject.toml ./

# Export a plain requirements.txt from pyproject.toml so we can install
# without requiring the full Poetry toolchain at runtime.
RUN pip install poetry==1.8.3 \
    && poetry config virtualenvs.create false \
    && poetry export --without-hashes --format=requirements.txt \
       --output requirements.txt \
    || echo "Poetry export fallback — installing from pyproject.toml directly"

# Install dependencies into a dedicated prefix so they are easy to COPY
RUN pip install --prefix=/install \
    --no-cache-dir \
    typing_extensions==4.12.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scipy==1.13.1 \
    scikit-learn==1.5.1 \
    statsmodels==0.14.2 \
    xgboost==2.1.0 \
    lightgbm==4.4.0 \
    catboost==1.2.5 \
    lifelines==0.29.0 \
    shap==0.45.1 \
    lime==0.2.0.1 \
    mlflow==2.14.1 \
    requests==2.32.3 \
    httpx==0.27.0 \
    aiohttp==3.9.5 \
    beautifulsoup4==4.12.3 \
    pdfplumber==0.11.1 \
    python-dotenv==1.0.1 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    loguru==0.7.2 \
    spacy==3.7.5 \
    plotly==5.22.0 \
    matplotlib==3.9.1 \
    seaborn==0.13.2 \
    fastapi \
    uvicorn[standard] \
    psycopg2-binary \
    sqlalchemy \
    alembic

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
