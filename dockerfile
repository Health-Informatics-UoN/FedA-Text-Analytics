FROM python:3.10-slim

# Basic Python hygiene
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system deps
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single static binary)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Make sure uv is on PATH
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy only dependency metadata first (better caching)
COPY pyproject.toml uv.lock* ./

# Install deps into system Python
RUN uv pip install --system --no-cache-dir .

# Copy application code
COPY . .

# Funnel/TES output location
RUN mkdir -p /outputs

ENTRYPOINT ["python", "tre_modelserve_run_once.py"]
