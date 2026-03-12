# ═══════════════════════════════════════════════════════════════
#  Dengue CDSS — Dockerfile
#  Uses full python image (not slim) — ensures apt-get works
#  correctly on Render and all Docker-based hosts
# ═══════════════════════════════════════════════════════════════

FROM python:3.11

# ── System-level dependencies ────────────────────────────────
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Confirm Tesseract installed (visible in build logs)
RUN tesseract --version

# ── App setup ────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Run ──────────────────────────────────────────────────────
EXPOSE 8501

CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false