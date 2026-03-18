# ═══════════════════════════════════════════════════════════════════════════
#  Dengue CDSS — Dockerfile 
#  Strategy: bake pre-trained models INTO the image at build time.
#
#  Cold-start time:
#    Before: 45-90s  (train_model.main() runs on every Render spin-up)
#    After:  < 0.5s  (joblib.load() reads pkl files already in the image)
#
#  Build stages:
#    [1] Install OS packages  (tesseract, poppler, libgl)
#    [2] Install Python deps  (pip install requirements.txt)
#    [3] Copy CSV + train_model.py → RUN training → pkl files generated
#    [4] Copy remaining app files
#    [5] Start Streamlit
# ═══════════════════════════════════════════════════════════════════════════

FROM python:3.11

# ── [1] OS-level dependencies ────────────────────────────────────────────────
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
        libgl1 \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Confirm Tesseract binary — visible in Render build logs
RUN tesseract --version

# ── [2] Python dependencies ──────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── [3] Train models at BUILD TIME — bake pkl files into image ───────────────
#  Only the files needed for training are copied first.
#  This layer is cached by Docker — if CSV and train_model.py haven't changed,
#  Docker reuses the cached layer and skips retraining on subsequent deploys.
COPY dengue_data_cleaned_debug.csv .
COPY train_model.py .
RUN python train_model.py && \
    echo "✅ Models baked into image:" && \
    ls -lh models/

# ── [4] Copy remaining application files ────────────────────────────────────
COPY . .

# ── [5] Start Streamlit ──────────────────────────────────────────────────────
EXPOSE 8501
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false