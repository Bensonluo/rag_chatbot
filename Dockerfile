# Multi-stage Dockerfile for RAG Chatbot
# Stage 1: Builder
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Use Chinese PyPI mirror for faster downloads
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
# Increase pip timeout for slow networks
ENV PIP_DEFAULT_TIMEOUT=300

# Use Chinese apt mirror for faster downloads
RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true

# Install system dependencies with build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Build the project in its own directory — never at /. With the default
# workdir (/), `pip install .` makes setuptools' package discovery
# (where=["."]) walk the whole container filesystem (/proc, /sys, the
# venv itself), pinning a core at 100% indefinitely.
WORKDIR /build

# Upgrade pip and install build tools FIRST (separate layer for caching)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --upgrade pip setuptools wheel

# pyproject.toml is the dependency source of truth. Dependencies and
# project code live in separate layers so code-only changes never
# re-download the dependency stack:
#   1. metadata + lock → dependency install (cached across app/ edits)
#   2. app source → --no-deps project install (seconds, not minutes)
COPY pyproject.toml README.md LICENSE ./
COPY requirements-lock.txt ./
# The lock (fresh python-3.11 resolve, see its header) pins the same
# versions the old `-c requirements-lock.txt .` resolution produced,
# so builds stay reproducible and pip-audit keeps a stable input.
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    -r requirements-lock.txt
# The lock is authored as a constraint set, not a guaranteed platform
# closure (see its header) — pip check turns that assumption into a
# build-time invariant: a missing dependency fails the build loudly
# instead of shipping an image that imports crash at runtime.
RUN pip check

COPY app ./app
RUN pip install --no-cache-dir --no-deps .

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Use Chinese apt mirror for faster downloads
RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true

# Install runtime dependencies (including netcat for health checks)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    libpq5 \
    netcat-openbsd \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/app /data && \
    chown -R appuser:appuser /home/appuser /data

# Set working directory
WORKDIR /home/appuser/app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Copy entrypoint script
COPY --chown=appuser:appuser scripts/entrypoint.sh /home/appuser/app/scripts/entrypoint.sh

# Make entrypoint executable
USER root
RUN chmod +x /home/appuser/app/scripts/entrypoint.sh
USER appuser

# Run application via entrypoint
CMD ["/home/appuser/app/scripts/entrypoint.sh"]
