#!/usr/bin/env python3
"""
Kokoro-82M fine-tune runner — Stage-2 corrected (2026-06-07)

Changes vs prior version:
  - wavs_dir → DATA_DIR/clips/  (not wavs/)
  - assert DATASET_SIZE >= 350 before training starts
  - lambda_dur bumped to 2.0 (env: LAMBDA_DUR)
  - 8 real reference clips downloaded from R2 refclips/, Whisper-transcribed,
    mixed into the training set for timbre identity
  - Checkpoint watcher thread: uploads each .pth to R2 within 60 s of save
  - Wall-clock timeout (env: WALL_TIMEOUT_HR, default 18) + SIGTERM handler
    that gracefully terminates the training subprocess and runs cleanup
  - Post-training: analyze ref clips (F0 baseline) + attempt inference +
    compute RMS/duration/spectral-flatness/F0/MFCC vs ref. Honest reporting.
  - R2 sentinel (DONE/FAILED/TIMEOUT) written on any exit so external
    watcher can close the Akash deployment.

Required env vars:
  R2_ACCOUNT_ID   Cloudflare R2 account ID
  R2_ACCESS_KEY   R2 S3-compatible access key
  R2_SECRET_KEY   R2 S3-compatible secret key
  R2_BUCKET       R2 bucket name (default: chloe-models)

Optional env vars:
  HF_TOKEN           HuggingFace token (public base model; only if rate-limited)
  HF_HOME            Cache dir for HF weights (default: /tmp/hf_cache)
  DATA_DIR           Skip R2 download; use this local path instead
  EPOCHS             Total training epochs (default: 500)
  LR                 Learning rate (default: 8e-6)
  BATCH              Batch size (default: 2)
  SPEAKER            Speaker label (default: chloe)
  LAMBDA_DUR         Duration loss weight (default: 2.0)
  LAMBDA_STY         Style loss weight (default: 1.0)
  SAVE_EVERY         Checkpoint interval in epochs (default: 50)
  WALL_TIMEOUT_HR    Hard wall-clock limit in hours (default: 18)
  REF_CLIPS_R2_PREFIX  R2 prefix for real reference clips (default: kokoro/refclips)
  R2_DATASET_PREFIX  R2 prefix for training dataset (default: kokoro/dataset)
  R2_OUTPUT_PREFIX   R2 prefix for checkpoints (default: kokoro/output/run3)
"""

import argparse
import csv
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time

import yaml

# ── CLI / env args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",      type=int,   default=int(os.environ.get("EPOCHS",   500)))
parser.add_argument("--lr",          type=float, default=float(os.environ.get("LR",     8e-6)))
parser.add_argument("--batch",       type=int,   default=int(os.environ.get("BATCH",    2)))
parser.add_argument("--speaker",     type=str,   default=os.environ.get("SPEAKER",     "chloe"))
parser.add_argument("--data",        type=str,   default=os.environ.get("DATA_DIR",    ""))
parser.add_argument("--output",      type=str,   default=os.environ.get("OUTPUT_DIR",  "/tmp/output"))
args = parser.parse_args()

WORK_DIR      = "/tmp/kokoro_ft"
STYLETTS2_DIR = "/app/StyleTTS2"
DATA_DIR      = args.data or os.path.join(WORK_DIR, "dataset")
OUTPUT_DIR    = args.output
KOKORO_DIR    = os.path.join(WORK_DIR, "kokoro_weights")
REF_DIR       = os.path.join(WORK_DIR, "refclips")
SAMPLE_RATE   = 24000
EPOCHS        = args.epochs
LR            = args.lr
BATCH         = args.batch
SPEAKER       = args.speaker
LAMBDA_DUR    = float(os.environ.get("LAMBDA_DUR",  "2.0"))
LAMBDA_STY    = float(os.environ.get("LAMBDA_STY",  "1.0"))
SAVE_EVERY    = int(os.environ.get("SAVE_EVERY",    "50"))
WALL_TIMEOUT_HR = float(os.environ.get("WALL_TIMEOUT_HR", "18"))
WALL_TIMEOUT_SEC = int(WALL_TIMEOUT_HR * 3600)

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")
R2_BUCKET     = os.environ.get("R2_BUCKET", "chloe-models")
R2_DATASET_PREFIX    = os.environ.get("R2_DATASET_PREFIX",    "kokoro/dataset").strip("/")
R2_OUTPUT_PREFIX     = os.environ.get("R2_OUTPUT_PREFIX",     "kokoro/output/run3").strip("/")
REF_CLIPS_R2_PREFIX  = os.environ.get("REF_CLIPS_R2_PREFIX",  "kokoro/refclips").strip("/")
HF_TOKEN      = os.environ.get("HF_TOKEN", "")

for d in [WORK_DIR, DATA_DIR, OUTPUT_DIR, KOKORO_DIR, REF_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"==> Config: EPOCHS={EPOCHS} LR={LR} BATCH={BATCH} LAMBDA_DUR={LAMBDA_DUR} "
      f"SAVE_EVERY={SAVE_EVERY} WALL_TIMEOUT_HR={WALL_TIMEOUT_HR}", flush=True)


def sh(cmd, check=True, **kwargs):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, **kwargs)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")
    return r


def make_r2_client():
    """Return a boto3 S3 client for Cloudflare R2, or None if creds missing."""
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY]):
        return None
    sh("pip install -q boto3", check=False)
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )


# ── 1. Dataset download from R2 ───────────────────────────────────────────────
metadata_path = os.path.join(DATA_DIR, "metadata.csv")
if not os.path.exists(metadata_path):
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY]):
        raise EnvironmentError(
            "Dataset not found at DATA_DIR and R2 credentials not set.\n"
            "Set R2_ACCOUNT_ID / R2_ACCESS_KEY / R2_SECRET_KEY or mount dataset at DATA_DIR."
        )
    print(f"==> Downloading dataset from R2 {R2_BUCKET}/{R2_DATASET_PREFIX}/", flush=True)
    s3 = make_r2_client()
    paginator = s3.get_paginator("list_objects_v2")
    n_dl = 0
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=f"{R2_DATASET_PREFIX}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(R2_DATASET_PREFIX):].lstrip("/")
            local = os.path.join(DATA_DIR, rel)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            if not os.path.exists(local):
                s3.download_file(R2_BUCKET, key, local)
                n_dl += 1
    print(f"==> Downloaded {n_dl} objects → {DATA_DIR}", flush=True)
else:
    print(f"==> Dataset already at {DATA_DIR}", flush=True)

# ── 2. Reference clips: download + Whisper-transcribe + mix into training ─────
downloaded_ref_clips = []
ref_rows = []

if REF_CLIPS_R2_PREFIX and all([R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY]):
    print(f"==> Downloading ref clips from R2 {R2_BUCKET}/{REF_CLIPS_R2_PREFIX}/", flush=True)
    s3_ref = make_r2_client()
    paginator_ref = s3_ref.get_paginator("list_objects_v2")
    for page in paginator_ref.paginate(Bucket=R2_BUCKET, Prefix=f"{REF_CLIPS_R2_PREFIX}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".wav"):
                continue
            local = os.path.join(REF_DIR, os.path.basename(key))
            if not os.path.exists(local):
                s3_ref.download_file(R2_BUCKET, key, local)
            downloaded_ref_clips.append(local)
    print(f"==> Ref clips downloaded: {len(downloaded_ref_clips)}", flush=True)

    if downloaded_ref_clips:
        print("==> Transcribing ref clips with Whisper tiny...", flush=True)
        sh("pip install -q openai-whisper", check=False)
        import whisper as _whisper
        _wm = _whisper.load_model("tiny")
        for ref_path in downloaded_ref_clips:
            try:
                result = _wm.transcribe(ref_path, language="en", fp16=False)
                text = result["text"].strip()
                words = text.split()
                if len(words) >= 3:
                    ref_rows.append((ref_path, text))
                    print(f"  ref: {os.path.basename(ref_path)} → {text[:70]}", flush=True)
                else:
                    print(f"  ref: {os.path.basename(ref_path)} → too short ({len(words)} words), skip", flush=True)
            except Exception as e:
                print(f"  ref: {os.path.basename(ref_path)} → Whisper error: {e}", flush=True)
        print(f"==> Ref clips transcribed: {len(ref_rows)} usable", flush=True)
else:
    print("==> REF_CLIPS_R2_PREFIX not set or no R2 creds — skipping ref clip mixing", flush=True)

# ── 3. Download Kokoro-82M pretrained weights ─────────────────────────────────
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
    print(f"==> Weights present: {weight_file}", flush=True)

# ── 4. StyleTTS2 — clone if not pre-baked ────────────────────────────────────
train_script = os.path.join(STYLETTS2_DIR, "train_finetune.py")
if not os.path.exists(train_script):
    print("==> Cloning StyleTTS2...", flush=True)
    sh(f"git clone --depth 1 https://github.com/yl4579/StyleTTS2 {STYLETTS2_DIR}")
    req = os.path.join(STYLETTS2_DIR, "requirements.txt")
    if os.path.exists(req):
        sh(f"pip install -q -r {req} --ignore-requires-python", check=False)

if STYLETTS2_DIR not in sys.path:
    sys.path.insert(0, STYLETTS2_DIR)

# ── 5. Build train/val lists ──────────────────────────────────────────────────
wavs_dir = os.path.join(DATA_DIR, "clips")
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

aug_count = len(rows)
print(f"==> Augmented dataset: {aug_count}  (skipped: {skipped})", flush=True)

# Mix in real ref clips (always in train set — timbre identity)
rows.extend(ref_rows)
print(f"DATASET_SIZE={len(rows)}", flush=True)
assert len(rows) >= 350, (
    f"Dataloader nearly empty: {len(rows)} — "
    f"check wavs_dir={wavs_dir} and metadata.csv format."
)

random.seed(42)
random.shuffle(rows)
# Ref rows go to train only (never val — too few and too precious)
val_rows   = []
train_rows = []
ref_set = set(r[0] for r in ref_rows)
non_ref = [r for r in rows if r[0] not in ref_set]
n_val = max(1, len(non_ref) // 10)
val_rows  = non_ref[:n_val]
train_rows = non_ref[n_val:] + ref_rows  # ref clips always in train

train_list = os.path.join(WORK_DIR, "train_list.txt")
val_list   = os.path.join(WORK_DIR, "val_list.txt")
with open(train_list, "w") as f:
    f.writelines(f"{w}|{t}\n" for w, t in train_rows)
with open(val_list, "w") as f:
    f.writelines(f"{w}|{t}\n" for w, t in val_rows)

print(f"==> Train: {len(train_rows)} (incl. {len(ref_rows)} ref) | Val: {len(val_rows)}", flush=True)

# ── 6. Write config_ft.yml ────────────────────────────────────────────────────
epochs_1st = max(1, int(EPOCHS * 0.80))
epochs_2nd = max(1, EPOCHS - epochs_1st)
total_steps_1st = int(epochs_1st * len(train_rows) / max(1, BATCH))

ft_config = {
    "log_dir":      OUTPUT_DIR,
    "save_freq":    SAVE_EVERY,
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
        "lambda_mel":  5.0,
        "lambda_gen":  1.0,
        "lambda_slm":  1.0,
        "lambda_mono": 1.0,
        "lambda_s":    1.0,
        "lambda_F0":   1.0,
        "lambda_norm": 1.0,
        "lambda_dur":  LAMBDA_DUR,   # bumped: 1.0 → 2.0
        "lambda_ce":   20.0,
        "lambda_sty":  LAMBDA_STY,
        "lambda_diff": 1.0,
        "diff_epoch":  0,
        "joint_epoch": 0,
    },
    "slmadv_params": {
        "min_len": 400, "max_len": 500, "batch_percentage": 0.5,
        "iter": 10, "thresh": 5, "scale": 0.01, "sig": 1.5,
    },
    "ASR_config": f"{STYLETTS2_DIR}/Utils/ASR/config.yml",
    "ASR_path":   f"{STYLETTS2_DIR}/Utils/ASR/epoch_00080.pth",
    "F0_path":    f"{STYLETTS2_DIR}/Utils/JDC/bst.t7",
    "PLBERT_dir": f"{STYLETTS2_DIR}/Utils/PLBERT",
}

config_path = os.path.join(WORK_DIR, "config_ft.yml")
with open(config_path, "w") as f:
    yaml.dump(ft_config, f, default_flow_style=False, allow_unicode=True)
print(f"==> Config written: {config_path}  "
      f"(epochs_1st={epochs_1st} epochs_2nd={epochs_2nd} lambda_dur={LAMBDA_DUR})", flush=True)

# ── 7. NLTK data ──────────────────────────────────────────────────────────────
import nltk
for pkg in ("averaged_perceptron_tagger", "cmudict"):
    nltk.download(pkg, quiet=True)

# ── 8. R2 client (shared by watcher + sentinel) ───────────────────────────────
s3_upload = make_r2_client()

# ── 9. Checkpoint watcher thread ──────────────────────────────────────────────
import glob as _glob

stop_watcher = threading.Event()
watcher_uploaded: set = set()


def checkpoint_watcher_loop():
    while not stop_watcher.wait(60):
        for fpath in _glob.glob(os.path.join(OUTPUT_DIR, "*.pth")):
            if fpath in watcher_uploaded:
                continue
            fname = os.path.basename(fpath)
            r2_key = f"{R2_OUTPUT_PREFIX}/{fname}"
            try:
                if s3_upload:
                    s3_upload.upload_file(fpath, R2_BUCKET, r2_key)
                    sz = os.path.getsize(fpath) / 1024**2
                    print(f"==> [ckpt-watcher] Uploaded {fname} ({sz:.0f}MB) → r2:{r2_key}", flush=True)
                watcher_uploaded.add(fpath)
            except Exception as e:
                print(f"==> [ckpt-watcher] Upload error {fname}: {e}", flush=True)
    # Final sweep after stop
    for fpath in _glob.glob(os.path.join(OUTPUT_DIR, "*.pth")):
        if fpath not in watcher_uploaded:
            fname = os.path.basename(fpath)
            r2_key = f"{R2_OUTPUT_PREFIX}/{fname}"
            try:
                if s3_upload:
                    s3_upload.upload_file(fpath, R2_BUCKET, r2_key)
                    sz = os.path.getsize(fpath) / 1024**2
                    print(f"==> [ckpt-watcher] Final upload {fname} ({sz:.0f}MB) → r2:{r2_key}", flush=True)
                watcher_uploaded.add(fpath)
            except Exception as e:
                print(f"==> [ckpt-watcher] Final upload error: {e}", flush=True)


watcher_thread = threading.Thread(target=checkpoint_watcher_loop, daemon=True)
watcher_thread.start()
print(f"==> Checkpoint watcher started (polls every 60s, uploads to r2:{R2_OUTPUT_PREFIX}/)", flush=True)

# ── 10. SIGTERM handler + wall-clock timeout thread ───────────────────────────
training_proc = None
exit_reason = "DONE"


def sigterm_handler(signum, frame):
    global exit_reason
    exit_reason = "TIMEOUT" if exit_reason == "TIMEOUT" else "SIGTERM"
    print(f"==> SIGTERM received (reason={exit_reason}) — terminating training subprocess", flush=True)
    if training_proc and training_proc.poll() is None:
        training_proc.terminate()
        try:
            training_proc.wait(timeout=30)
        except Exception:
            training_proc.kill()
    stop_watcher.set()


signal.signal(signal.SIGTERM, sigterm_handler)


def wall_timeout_runner():
    global exit_reason
    time.sleep(WALL_TIMEOUT_SEC)
    exit_reason = "TIMEOUT"
    print(f"==> WALL CLOCK TIMEOUT ({WALL_TIMEOUT_HR}h) reached — sending SIGTERM", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)


t_wall = threading.Thread(target=wall_timeout_runner, daemon=True)
t_wall.start()
print(f"==> Wall-clock timeout set: {WALL_TIMEOUT_HR}h ({WALL_TIMEOUT_SEC}s)", flush=True)

# ── 11. Run training ──────────────────────────────────────────────────────────
print("=" * 65, flush=True)
print(f"  Kokoro-82M Fine-Tuning  |  Speaker: {SPEAKER}", flush=True)
print(f"  Epochs: {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH}", flush=True)
print(f"  λ_dur: {LAMBDA_DUR}  |  save_freq: {SAVE_EVERY}", flush=True)
print(f"  Dataset: {len(train_rows)} train + {len(val_rows)} val", flush=True)
print("=" * 65, flush=True)

t0 = time.time()
training_proc = subprocess.Popen(
    [sys.executable, train_script, "--config_path", config_path],
    cwd=STYLETTS2_DIR,
)
training_proc.wait()
elapsed = time.time() - t0
rc = training_proc.returncode

print(f"\nTraining subprocess exited: rc={rc}  elapsed={elapsed/60:.1f}min ({elapsed/3600:.2f}hr)", flush=True)

if rc != 0 and exit_reason == "DONE":
    exit_reason = "FAILED"

# Stop watcher + final uploads
stop_watcher.set()
watcher_thread.join(timeout=120)
print(f"==> Checkpoint watcher stopped. Total uploaded: {len(watcher_uploaded)}", flush=True)

# ── 12. Checkpoint inventory ──────────────────────────────────────────────────
checkpoints = sorted([
    f for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".pth") or f.endswith(".pt")
])
print(f"\nCheckpoints in OUTPUT_DIR: {len(checkpoints)}", flush=True)
for c in checkpoints:
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, c)) / 1024**2
    in_r2 = "R2✓" if os.path.join(OUTPUT_DIR, c) in watcher_uploaded else "local-only"
    print(f"  {c}  ({sz:.1f}MB)  [{in_r2}]", flush=True)

if not checkpoints:
    print("WARNING: No checkpoints found — training may have crashed before first save.", flush=True)
    exit_reason = "FAILED"

# ── 13. Post-training: ref clip analysis + inference attempt + metrics ─────────
print("\n==> Post-training analysis", flush=True)

def compute_audio_stats(wav_path):
    """Return dict of audio statistics. Returns None on error."""
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(wav_path, sr=24000, mono=True)
        duration = len(y) / sr
        rms = float(np.sqrt(np.mean(y ** 2)))
        sf_arr = librosa.feature.spectral_flatness(y=y)
        spec_flat = float(np.mean(sf_arr))
        try:
            f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=500, sr=sr)
            voiced_f0 = f0[voiced] if voiced is not None else []
            f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        except Exception:
            f0_mean = 0.0
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return {
            "duration_s":     round(duration, 2),
            "rms":            round(rms, 5),
            "spectral_flat":  round(spec_flat, 6),
            "f0_mean_hz":     round(f0_mean, 1),
            "mfcc_mean":      [round(float(v), 2) for v in mfcc.mean(axis=1)],
        }
    except Exception as e:
        print(f"  stats error on {wav_path}: {e}", flush=True)
        return None


def mfcc_distance(stats_a, stats_b):
    """L2 distance between MFCC mean vectors."""
    try:
        import numpy as np
        a = stats_b["mfcc_mean"]
        b = stats_a["mfcc_mean"]
        return round(float(np.linalg.norm(
            [x - y for x, y in zip(a, b)]
        )), 2)
    except Exception:
        return -1.0


# Analyze ref clips → baseline
ref_stats_list = []
for rp in downloaded_ref_clips:
    s = compute_audio_stats(rp)
    if s:
        ref_stats_list.append(s)

if ref_stats_list:
    import numpy as np
    ref_f0s = [s["f0_mean_hz"] for s in ref_stats_list if s["f0_mean_hz"] > 0]
    ref_rms  = [s["rms"] for s in ref_stats_list]
    REF_F0_MEAN  = round(float(np.mean(ref_f0s)), 1)  if ref_f0s  else 143.0
    REF_RMS_MEAN = round(float(np.mean(ref_rms)), 5)
    print(f"  Ref clips ({len(ref_stats_list)}) — F0 mean: {REF_F0_MEAN} Hz  RMS mean: {REF_RMS_MEAN}", flush=True)
else:
    REF_F0_MEAN  = 143.0
    REF_RMS_MEAN = None
    print("  No ref clips available for baseline (ref download skipped or all failed)", flush=True)

# Attempt inference using StyleTTS2
print("\n==> Inference attempt", flush=True)
test_wav = os.path.join(OUTPUT_DIR, "test_inference.wav")
inference_ok = False
test_stats = None
TEST_SENTENCE = "Hello, I am Chloe. The coffee this morning was absolutely wonderful, don't you think?"

latest_ckpt = os.path.join(OUTPUT_DIR, checkpoints[-1]) if checkpoints else None

if latest_ckpt:
    # Try StyleTTS2 Python API directly
    try:
        import torch
        import numpy as np
        import soundfile as sf
        sys.path.insert(0, STYLETTS2_DIR)

        from models import build_model
        from Utils.PLBERT.util import load_plbert
        from cached_path import cached_path

        device = "cpu"

        with open(config_path) as f:
            ft_cfg = yaml.safe_load(f)

        # Load PLBERT
        plbert = load_plbert(ft_cfg["PLBERT_dir"])

        # Build model
        model_params = ft_cfg["model_params"]
        model = build_model(
            model_params,
            text_aligner=None,
            pitch_extractor=None,
            plbert=plbert,
        )

        # Load fine-tuned checkpoint
        state = torch.load(latest_ckpt, map_location="cpu")
        # StyleTTS2 checkpoint format varies — try common keys
        for key in ("net", "model", "state_dict"):
            if key in state:
                state = state[key]
                break

        # Partial load (strict=False) — fine-tuned weights may not cover all sub-modules
        missing, unexpected = [], []
        for k, v in state.items():
            try:
                parts = k.split(".")
                obj = model
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                param = getattr(obj, parts[-1])
                param.data.copy_(v)
            except Exception:
                missing.append(k)

        print(f"  Checkpoint loaded (missing keys: {len(missing)})", flush=True)

        # StyleTTS2 inference requires phonemizer + text cleaner — try it
        from text_utils import TextCleaner
        from phonemizer import phonemize

        cleaner = TextCleaner()
        phones = phonemize(TEST_SENTENCE, backend="espeak", language="en-us", with_stress=True)
        tokens = cleaner(phones)
        token_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)

        # Load a ref style from first ref clip
        ref_wav_for_style = downloaded_ref_clips[0] if downloaded_ref_clips else None
        if ref_wav_for_style is None and val_rows:
            ref_wav_for_style = val_rows[0][0]

        if ref_wav_for_style:
            import librosa
            ref_audio, _ = librosa.load(ref_wav_for_style, sr=24000, mono=True)
            ref_tensor = torch.FloatTensor(ref_audio).unsqueeze(0).unsqueeze(0)
        else:
            ref_tensor = torch.zeros(1, 1, 24000)

        model.eval()
        with torch.no_grad():
            # StyleTTS2 API: model has a generate() or forward() method
            # Try the most common pattern first
            if hasattr(model, "generate"):
                out = model.generate(token_ids, ref_tensor, speed=1.0)
            elif hasattr(model, "inference"):
                out = model.inference(token_ids, ref_tensor)
            else:
                raise AttributeError("No generate/inference method found on model")

        audio_np = out.squeeze().cpu().numpy()
        sf.write(test_wav, audio_np, 24000)
        print(f"  Test audio written: {test_wav}", flush=True)
        inference_ok = True
        test_stats = compute_audio_stats(test_wav)
    except Exception as e:
        print(f"  Inference attempt failed: {e}", flush=True)
        print("  (Manual validation required — download checkpoint and run inference locally)", flush=True)
else:
    print("  No checkpoint available for inference", flush=True)

# Print metrics report
print("\n" + "=" * 65, flush=True)
print("  POST-TRAINING METRICS REPORT", flush=True)
print("=" * 65, flush=True)
print(f"  Exit reason:        {exit_reason}", flush=True)
print(f"  Checkpoints saved:  {len(checkpoints)}", flush=True)
print(f"  Total R2 uploads:   {len(watcher_uploaded)}", flush=True)
print(f"  Training time:      {elapsed/60:.1f} min", flush=True)
print(f"  Dataset used:       {len(train_rows)} train + {len(val_rows)} val", flush=True)
print(f"  Ref clips in train: {len(ref_rows)}", flush=True)
print(f"  λ_dur applied:      {LAMBDA_DUR}", flush=True)
print(f"  Ref F0 baseline:    {REF_F0_MEAN} Hz  (Chloé target ~143 Hz)", flush=True)

if inference_ok and test_stats:
    delta_f0 = round(test_stats["f0_mean_hz"] - REF_F0_MEAN, 1)
    mfcc_d = mfcc_distance(test_stats, ref_stats_list[0]) if ref_stats_list else -1.0
    is_speech = test_stats["spectral_flat"] < 0.3 and test_stats["rms"] > 0.005
    timbre_ok = mfcc_d < 30  # rough threshold — lower = closer to Chloé

    print(f"\n  === GENERATED TEST CLIP ===", flush=True)
    print(f"  Duration:           {test_stats['duration_s']} s", flush=True)
    print(f"  RMS:                {test_stats['rms']}", flush=True)
    print(f"  Spectral flatness:  {test_stats['spectral_flat']}  {'(speech ✓)' if is_speech else '(SUSPICIOUS — may be noise)'}", flush=True)
    print(f"  F0 mean:            {test_stats['f0_mean_hz']} Hz  (ref {REF_F0_MEAN} Hz, delta {delta_f0:+.1f} Hz)", flush=True)
    print(f"  MFCC dist vs ref:   {mfcc_d}  {'(timbre OK ✓)' if timbre_ok else '(DIVERGED from Chloé)'}", flush=True)
    print(f"\n  VERDICT: {'REAL SPEECH' if is_speech else 'NOT SPEECH'}  |  "
          f"TIMBRE {'MATCHES' if timbre_ok else 'DIVERGED'}", flush=True)
    if s3_upload:
        try:
            r2_key = f"{R2_OUTPUT_PREFIX}/test_inference.wav"
            s3_upload.upload_file(test_wav, R2_BUCKET, r2_key)
            print(f"  Test clip uploaded: r2:{R2_BUCKET}/{r2_key}", flush=True)
        except Exception as e:
            print(f"  Test clip upload failed: {e}", flush=True)
else:
    print(f"\n  === INFERENCE NOT RUN ===", flush=True)
    print(f"  (Download checkpoint from R2 and validate locally)", flush=True)
    print(f"  r2://{R2_BUCKET}/{R2_OUTPUT_PREFIX}/", flush=True)
    print(f"\n  VERDICT: TRAINING {'COMPLETED' if exit_reason == 'DONE' else exit_reason} — "
          f"inference not measured (see manual validation)", flush=True)

print("=" * 65, flush=True)

# ── 14. R2 sentinel — write on any exit so watcher can close Akash lease ──────
sentinel_body = (
    f"status={exit_reason}\n"
    f"checkpoints={len(checkpoints)}\n"
    f"uploaded={len(watcher_uploaded)}\n"
    f"training_time_min={elapsed/60:.1f}\n"
    f"dataset_size={len(train_rows) + len(val_rows)}\n"
    f"inference_ok={inference_ok}\n"
)
if s3_upload:
    try:
        s3_upload.put_object(
            Bucket=R2_BUCKET,
            Key=f"{R2_OUTPUT_PREFIX}/SENTINEL",
            Body=sentinel_body.encode(),
        )
        print(f"\n==> R2 sentinel written: r2://{R2_BUCKET}/{R2_OUTPUT_PREFIX}/SENTINEL", flush=True)
        print(f"    status={exit_reason} — external watcher can now close the Akash lease.", flush=True)
    except Exception as e:
        print(f"==> Sentinel write error: {e}", flush=True)
else:
    print("\n==> No R2 creds — sentinel NOT written.", flush=True)

print("\nDone.", flush=True)
sys.exit(0 if exit_reason in ("DONE", "TIMEOUT") else 1)
