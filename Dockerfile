# ═══════════════════════════════════════════════════════════════
#  Dengue CDSS — Dockerfile
#  Works on: Render, Railway, Fly.io, any Docker-based host
#  Streamlit Cloud: still uses packages.txt (ignores this file)
# ═══════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── System-level dependencies ────────────────────────────────
# Install Tesseract OCR binary + English language pack +
# Poppler (pdf2image) + OpenCV system libs in a single layer.
# This is the permanent, platform-agnostic fix — no PATH issues,
# no missing binary at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Confirm binary location (visible in Render build logs)
RUN which tesseract && tesseract --version

# ── App setup ────────────────────────────────────────────────
WORKDIR /app

# Copy requirements first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# ── Streamlit config ─────────────────────────────────────────
EXPOSE 8501

# Render injects $PORT at runtime; fall back to 8501 for local Docker
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false