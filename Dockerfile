# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — dependency builder
# Install Python packages into a clean prefix so Stage 2 stays small.
# Separating build from runtime means the final image never contains
# pip, wheel, or compiler artefacts.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Install OS-level build tools needed by some Python packages (e.g. albumentations, Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer until requirements change,
# so a code-only edit skips the pip install step entirely.
COPY requirements.txt .

# Install into /install/site-packages so we can copy them cleanly
RUN pip install --no-cache-dir --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.0+cpu \
        torchvision==0.18.0+cpu \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt \
    # FastAPI stack (not in existing requirements.txt)
    && pip install --no-cache-dir --prefix=/install \
        fastapi==0.111.0 \
        uvicorn[standard]==0.29.0 \
        python-multipart==0.0.9


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — lean runtime image
# Only the app code + installed packages, nothing else.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime OS libs (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY model.py dataset.py losses.py main.py ./

# model.pt is expected to be present at container run-time.
# Mount it via:  docker run -v /path/to/model.pt:/app/model.pt ...
# Or COPY it here if you want it baked into the image:
# COPY model.pt .

# Non-root user — principle of least privilege
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser /app
USER appuser

# Expose the API port
EXPOSE 8000

# Docker HEALTHCHECK — calls our /health endpoint every 30 s.
# unhealthy after 3 consecutive failures (90 s total).
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:8000/health', timeout=8); \
         sys.exit(0 if r.status == 200 else 1)"

# Uvicorn with 2 workers.
# Increase --workers on multi-core machines; keep 1 if GPU memory is shared.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
