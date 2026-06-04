# kokoro-train — Akash-ready StyleTTS2 fine-tune image

Self-contained, **env-driven** GPU training container for the Kokoro-82M (Chloé)
StyleTTS2 fine-tune. Designed to drop into an **Akash Console GPU template**
(`console.akash.network/templates`) with no SDL or Dockerfile editing — every
run parameter is a container env var.

## Image ref
```
ghcr.io/ivangegovdve-sudo/kokoro-train:latest
```

## What's baked in
- PyTorch 2.2.0 + CUDA 12.1 runtime base (verified pullable on Docker Hub)
- StyleTTS2 (`yl4579/StyleTTS2`, incl. Utils/ ASR + JDC + PLBERT assets)
- `train.py` entrypoint (downloads dataset + Kokoro weights, writes config, trains, uploads checkpoints)
- All Python deps incl. `boto3` (R2/S3) and pre-fetched NLTK data

## Env vars (set these in the Console template form)
| Var | Required | Default | Notes |
|-----|----------|---------|-------|
| `R2_ACCOUNT_ID` | yes (R2 path) | – | `ef99690c7870af1f058778934d233b80` |
| `R2_ACCESS_KEY` | yes | – | R2 **write-capable** token id |
| `R2_SECRET_KEY` | yes | – | R2 **write-capable** token secret |
| `R2_BUCKET` | yes | `kokoro-training` | use existing `chloe-models` |
| `R2_DATASET_PREFIX` | no | `dataset` | input prefix inside the bucket |
| `R2_OUTPUT_PREFIX` | no | `model_output` | checkpoint output prefix |
| `EPOCHS` | no | `100` | 200 / 300 / 150 per run |
| `LR` | no | `1e-5` | |
| `BATCH` | no | `2` | |
| `SPEAKER` | no | `chloe` | |
| `DATA_DIR` | no | – | set to skip R2 download (local/mounted data) |
| `OUTPUT_DIR` | no | `/tmp/output` | |
| `HF_TOKEN` | no | – | only if HF rate-limited |

## Build + push
This machine has **no Docker daemon**, so build via CI:
1. Push this folder to a GitHub repo `ivangegovdve-sudo/kokoro-train`.
2. The included workflow `.github/workflows/build-kokoro-image.yml` builds and
   pushes `ghcr.io/ivangegovdve-sudo/kokoro-train:latest` using `GITHUB_TOKEN`.
3. Set the GHCR package visibility to **public** so Akash providers can pull it.

Or, on any machine with Docker:
```bash
TOKEN=$(gcloud secrets versions access latest --secret=ghcr-pat --project=forest-family-cloud)
echo "$TOKEN" | docker login ghcr.io -u ivangegovdve-sudo --password-stdin
docker build -t ghcr.io/ivangegovdve-sudo/kokoro-train:latest .
docker push  ghcr.io/ivangegovdve-sudo/kokoro-train:latest
```
The vaulted `ghcr-pat` token is verified valid with `write:packages` scope.
