#!/usr/bin/env python3
"""
Kokoro-82M fine-tune runner — standalone script for containerised training.

Entrypoint for Dockerfile.train.  Handles:
  1. Download dataset from Cloudflare R2  (or use /data if already mounted)
  2. Download Kokoro-82M weights from HuggingFace
  3. Clone StyleTTS2 (already in image — skip if present)
  4. Build config_ft.yml
  5. Run StyleTTS2 train_finetune.py for N epochs
  6. Upload checkpoints to R2  (or HF Hub if HF_TOKEN given)

Required env vars:
  R2_ACCOUNT_ID   Cloudflare R2 account ID
  R2_ACCESS_KEY   R2 S3-compatible access key
  R2_SECRET_KEY   R2 S3-compatible secret key
  R2_BUCKET       R2 bucket name (default: kokoro-training)

Optional env vars:
  HF_TOKEN        HuggingFace token for private models / HF Hub upload
  HF_HOME         Cache dir for HF weights  (default: /tmp/hf_cache)
  DATA_DIR        If set, skip R2 download and use this local path
  EPOCHS          Number of training epochs  (default: 100)
  LR              Learning rate  (default: 1e-5)
  BATCH           Batch size  (default: 2)
  SPEAKER         Speaker label in metadata.csv  (default: chloe)
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile
import time

import yaml

# ── CLI args (allow override from both env and flags) ─────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",  type=int,   default=int(os.environ.get("EPOCHS",  100)))
parser.add_argument("--lr",      type=float, default=float(os.environ.get("LR",    1e-5)))
parser.add_argument("--batch",   type=int,   default=int(os.environ.get("BATCH",   2)))
parser.add_argument("--speaker", type=str,   default=os.environ.get("SPEAKER",    "chloe"))
parser.add_argument("--data",    type=str,   default=os.environ.get("DATA_DIR",   ""))
parser.add_argument("--output",  type=str,   default=os.environ.get("OUTPUT_DIR", "/tmp/output"))
args = parser.parse_args()

WORK_DIR      = "/tmp/kokoro_ft"
STYLETTS2_DIR = "/app/StyleTTS2"   # pre-cloned in Dockerfile
DATA_DIR      = args.data or os.path.join(WORK_DIR, "dataset")
OUTPUT_DIR    = args.output
KOKORO_DIR    = os.path.join(WORK_DIR, "kokoro_weights")
SAMPLE_RATE   = 24000
EPOCHS        = args.epochs
LR            = args.lr
BATCH         = args.batch
SPEAKER       = args.speaker

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")
R2_BUCKET     = os.environ.get("R2_BUCKET", "kokoro-training")
R2_DATASET_PREFIX = os.environ.get("R2_DATASET_PREFIX", "dataset").strip("/")
R2_OUTPUT_PREFIX  = os.environ.get("R2_OUTPUT_PREFIX",  "model_output").strip("/")
HF_TOKEN      = os.environ.get("HF_TOKEN", "")

for d in [WORK_DIR, DATA_DIR, OUTPUT_DIR, KOKORO_DIR]:
    os.makedirs(d, exist_ok=True)


def sh(cmd, check=True, **kwargs):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, **kwargs)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")
    return r


# ── 1. Dataset: download from R2 if not already present ──────────────────────
metadata_path = os.path.join(DATA_DIR, "metadata.csv")
if not os.path.exists(metadata_path):
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY]):
        raise EnvironmentError(
            "Dataset not found at DATA_DIR and R2 credentials not set.\n"
            "Set R2_ACCOUNT_ID / R2_ACCESS_KEY / R2_SECRET_KEY or mount dataset at DATA_DIR."
        )
    print(f"==> Downloading dataset from R2 bucket {R2_BUCKET}...", flush=True)
    # Install boto3 if missing (not in base image)
    sh("pip install -q boto3")
    import boto3
    endpoint = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )
    paginator = s3.get_paginator("list_objects_v2")
    n_dl = 0
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=f"{R2_DATASET_PREFIX}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # strip the R2 prefix so files land under WORK_DIR/dataset/...
            rel = key[len(R2_DATASET_PREFIX):].lstrip("/")
            local = os.path.join(WORK_DIR, "dataset", rel)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            if not os.path.exists(local):
                s3.download_file(R2_BUCKET, key, local)
                n_dl += 1
    print(f"==> Downloaded {n_dl} objects from r2://{R2_BUCKET}/{R2_DATASET_PREFIX}/", flush=True)
    DATA_DIR = os.path.join(WORK_DIR, "dataset")
    print(f"==> Dataset downloaded to {DATA_DIR}", flush=True)
else:
    print(f"==> Dataset already at {DATA_DIR}", flush=True)

# ── 2. Download Kokoro-82M pretrained weights ─────────────────────────────────
weight_file = os.path.join(KOKORO_DIR, "kokoro-v1_0.pth")
if not os.path.exists(weight_file):
    print("==> Downloading Kokoro-82M weights from HuggingFace...", flush=True)
    from huggingface_hub import hf_hub_download
    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    hf_hub_download(
        repo_id="hexgrad/Kokoro-82M",
        filename="kokoro-v1_0.pth",
        local_dir=KOKORO_DIR,
        **kwargs,
    )
    size_mb = os.path.getsize(weight_file) / 1024**2
    print(f"==> Downloaded kokoro-v1_0.pth ({size_mb:.1f} MB)", flush=True)
else:
    print(f"==> Weights already present: {weight_file}", flush=True)

# ── 3. StyleTTS2 — clone only if not pre-baked by Dockerfile ─────────────────
train_script = os.path.join(STYLETTS2_DIR, "train_finetune.py")
if not os.path.exists(train_script):
    print("==> Cloning StyleTTS2 (not pre-baked in image)...", flush=True)
    sh(f"git clone --depth 1 https://github.com/yl4579/StyleTTS2 {STYLETTS2_DIR}")
    req = os.path.join(STYLETTS2_DIR, "requirements.txt")
    if os.path.exists(req):
        sh(f"pip install -q -r {req} --ignore-requires-python", check=False)

if STYLETTS2_DIR not in sys.path:
    sys.path.insert(0, STYLETTS2_DIR)

# ── 4. Build train/val lists ──────────────────────────────────────────────────
wavs_dir = os.path.join(DATA_DIR, "wavs")
rows = []
skipped = 0
with open(metadata_path, encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="|"):
        if len(row) < 2:
            skipped += 1
            continue
        stem, text = row[0].strip(), row[1].strip()
        wav = os.path.join(wavs_dir, stem + ".wav")
        if not os.path.exists(wav):
            wav = os.path.join(wavs_dir, stem)
        if not os.path.exists(wav) or len(text.split()) < 3:
            skipped += 1
            continue
        rows.append((wav, text))

print(f"==> Valid samples: {len(rows)}  (skipped: {skipped})", flush=True)
if len(rows) < 5:
    raise ValueError(f"Only {len(rows)} valid samples — check metadata.csv.")

random.seed(42)
random.shuffle(rows)
n_val = max(1, len(rows) // 10)
val_rows, train_rows = rows[:n_val], rows[n_val:]

train_list = os.path.join(WORK_DIR, "train_list.txt")
val_list   = os.path.join(WORK_DIR, "val_list.txt")
with open(train_list, "w") as f:
    f.writelines(f"{w}|{t}\n" for w, t in train_rows)
with open(val_list, "w") as f:
    f.writelines(f"{w}|{t}\n" for w, t in val_rows)

print(f"==> Train: {len(train_rows)} | Val: {len(val_rows)}", flush=True)

# ── 5. Write config_ft.yml ────────────────────────────────────────────────────
epochs_1st = max(1, int(EPOCHS * 0.8))
epochs_2nd = max(1, EPOCHS - epochs_1st)
total_steps_1st = int(epochs_1st * len(train_rows) / max(1, BATCH))

ft_config = {
    "log_dir":      OUTPUT_DIR,
    "save_freq":    max(1, EPOCHS // 10),
    "log_interval": 10,
    "device":       "cuda",
    "epochs_1st":   epochs_1st,
    "epochs_2nd":   epochs_2nd,
    "batch_size":   BATCH,
    "max_len":      200,
    "pretrained_model":             weight_file,
    "second_stage_load_pretrained": True,
    "train_data": train_list,
    "val_data":   val_list,
    "OOD_data":   val_list,
    "preprocess_params": {
        "sr": SAMPLE_RATE,
        "spect_params": {"n_fft": 2048, "win_length": 1200, "hop_length": 300},
    },
    "optimizer_params": {
        "lr": LR, "beta1": 0.0, "beta2": 0.99, "weight_decay": 1e-4,
    },
    "lr_scheduler_params": {
        "start_factor": 1.0, "end_factor": 0.1, "total_iters": total_steps_1st,
    },
    "model_params": {
        "args": {"dim_in": 64, "hidden_dim": 512, "max_conv_dim": 512,
                 "n_layer": 3, "n_mels": 80},
        "n_mels": 80, "sampling_rate": SAMPLE_RATE, "segment_size": 8192,
        "style_dim": 128, "max_conv_dim": 512, "hidden_dim": 512,
        "n_layer": 3, "dim_in": 64, "multispeaker": False,
    },
    "loss_params": {
        "lambda_mel": 5.0, "lambda_gen": 1.0, "lambda_slm": 1.0,
        "lambda_mono": 1.0, "lambda_s": 1.0, "lambda_F0": 1.0,
        "lambda_norm": 1.0, "lambda_dur": 1.0, "lambda_ce": 20.0,
        "lambda_sty": 1.0, "lambda_diff": 1.0, "diff_epoch": 0, "joint_epoch": 0,
    },
    "slmadv_params": {
        "min_len": 400, "max_len": 500, "batch_percentage": 0.5,
        "iter": 10, "thresh": 5, "scale": 0.01, "sig": 1.5,
    },
    # Required paths discovered during prior Colab run — must be present
    "ASR_config": f"{STYLETTS2_DIR}/Utils/ASR/config.yml",
    "ASR_path":   f"{STYLETTS2_DIR}/Utils/ASR/epoch_00080.pth",
    "F0_path":    f"{STYLETTS2_DIR}/Utils/JDC/bst.t7",
    "PLBERT_dir": f"{STYLETTS2_DIR}/Utils/PLBERT",
}

config_path = os.path.join(WORK_DIR, "config_ft.yml")
with open(config_path, "w") as f:
    yaml.dump(ft_config, f, default_flow_style=False, allow_unicode=True)
print(f"==> Config written: {config_path}", flush=True)

# ── 6. NLTK data ──────────────────────────────────────────────────────────────
import nltk
for pkg in ("averaged_perceptron_tagger", "cmudict"):
    nltk.download(pkg, quiet=True)

# ── 7. Run training ───────────────────────────────────────────────────────────
print("=" * 65, flush=True)
print(f"  Kokoro-82M Fine-Tuning  |  Speaker: {SPEAKER}", flush=True)
print(f"  Epochs: {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH}", flush=True)
print("=" * 65, flush=True)

t0 = time.time()
r = subprocess.run(
    [sys.executable, train_script, "--config_path", config_path],
    cwd=STYLETTS2_DIR,
)
elapsed = time.time() - t0
print(f"\nTraining runtime: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)", flush=True)

# ── 8. Inventory checkpoints ──────────────────────────────────────────────────
checkpoints = sorted([
    f for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".pth") or f.endswith(".pt")
])
print(f"\nCheckpoints found: {len(checkpoints)}", flush=True)
for c in checkpoints[-5:]:
    size = os.path.getsize(os.path.join(OUTPUT_DIR, c)) / 1024**2
    print(f"  {c}  ({size:.1f} MB)", flush=True)

if r.returncode != 0:
    print(f"\nWARNING: training exited with code {r.returncode}", flush=True)
    sys.exit(r.returncode)

# ── 9. Upload to R2 ───────────────────────────────────────────────────────────
if all([R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY]) and checkpoints:
    print("\n==> Uploading checkpoints to R2...", flush=True)
    import boto3
    endpoint = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )
    for fname in checkpoints:
        local_path = os.path.join(OUTPUT_DIR, fname)
        r2_key = f"{R2_OUTPUT_PREFIX}/{fname}"
        s3.upload_file(local_path, R2_BUCKET, r2_key)
        size_mb = os.path.getsize(local_path) / 1024**2
        print(f"  Uploaded: r2://{R2_BUCKET}/{r2_key}  ({size_mb:.1f} MB)", flush=True)
    print("==> Upload complete.", flush=True)
elif HF_TOKEN and checkpoints:
    # Fallback: push latest checkpoint to HF Hub
    print("\n==> Uploading latest checkpoint to HuggingFace Hub...", flush=True)
    sh("pip install -q huggingface_hub")
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    latest = checkpoints[-1]
    local_path = os.path.join(OUTPUT_DIR, latest)
    # Requires HF_REPO env var like "youruser/kokoro-chloe-finetune"
    hf_repo = os.environ.get("HF_REPO", "")
    if hf_repo:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=latest,
            repo_id=hf_repo,
            repo_type="model",
        )
        print(f"==> Uploaded to hf.co/{hf_repo}/{latest}", flush=True)
    else:
        print("==> HF_TOKEN set but HF_REPO not set — checkpoint stays local.", flush=True)
else:
    print("\nWARNING: No R2 credentials or HF_TOKEN — checkpoint NOT uploaded.", flush=True)
    print(f"Checkpoint is at: {OUTPUT_DIR}/{checkpoints[-1] if checkpoints else '(none)'}", flush=True)

print("\nDone.", flush=True)
