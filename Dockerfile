# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Kokoro-82M (Chloé) StyleTTS2 fine-tune — self-contained GPU training image
# Target: Akash Console GPU template (console.akash.network/templates)
#
# Fully ENV-DRIVEN: every run parameter comes from container env vars, so the
# same public image slots into a Console template with NO SDL / Dockerfile edits.
# The Console template form only needs:  image ref + GPU(A100) + the env vars.
#
# Build (where a Docker daemon exists, e.g. CI):
#   docker build -t ghcr.io/ivangegovdve-sudo/kokoro-train:latest .
# Push (after `docker login ghcr.io`):
#   docker push ghcr.io/ivangegovdve-sudo/kokoro-train:latest
#
# Recommended path on THIS machine (no local Docker): GitHub Actions —
#   see .github/workflows/build-kokoro-image.yml  (builds + pushes to GHCR free)
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.title="kokoro-train" \
      org.opencontainers.image.description="Kokoro-82M / Chloé StyleTTS2 fine-tune runner (env-driven, Akash-ready)" \
      org.opencontainers.image.source="https://github.com/ivangegovdve-sudo/kokoro-train"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/StyleTTS2 \
    HF_HOME=/tmp/hf_cache \
    NLTK_DATA=/usr/share/nltk_data

# ── System deps (espeak-ng = phonemizer backend; libsndfile/ffmpeg = audio) ──
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        git wget curl espeak-ng libsndfile1 ffmpeg build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps. boto3/pyyaml/huggingface_hub/nltk are REQUIRED by train.py ──
#    (no GCS dep — this image talks to Cloudflare R2 via boto3 / S3 API)
RUN pip install --no-cache-dir \
        openai-whisper huggingface_hub phonemizer librosa soundfile \
        einops munch pyyaml matplotlib tqdm transformers accelerate \
        numba scipy nltk inflect unidecode boto3

# ── StyleTTS2 baked into the image so the container starts without a clone ──
#    Includes Utils/ (ASR, JDC F0, PLBERT) that train.py references by path.
RUN git clone --depth 1 https://github.com/yl4579/StyleTTS2 /app/StyleTTS2 \
    && pip install --no-cache-dir -r /app/StyleTTS2/requirements.txt --ignore-requires-python || true

# ── Pre-fetch NLTK data so runtime needs no extra network round-trips ──
RUN python -m nltk.downloader -d ${NLTK_DATA} averaged_perceptron_tagger cmudict

WORKDIR /app
COPY train.py /app/train.py

# All config arrives via env vars (EPOCHS, LR, BATCH, SPEAKER, R2_*, HF_*,
# R2_DATASET_PREFIX, R2_OUTPUT_PREFIX, DATA_DIR, OUTPUT_DIR). No CMD args needed.
ENTRYPOINT ["python", "/app/train.py"]
