#!/usr/bin/env bash
# One-time setup on a rented GPU pod (RunPod or similar), run from /workspace.
# Needs two environment variables, set in the pod's terminal before running:
#   GITHUB_TOKEN  read-only fine-grained token for this repository (never echoed)
#   ZIP_URL       a download link for s512.zip (for example a Google Drive link shared
#                 as "anyone with the link"); gdown handles Drive links.
# Result: /workspace/explainable-dr-grading with the processed images in data/processed/s512.
set -euo pipefail
: "${GITHUB_TOKEN:?set GITHUB_TOKEN first}"
: "${ZIP_URL:?set ZIP_URL first}"
REPO="Kyojur0/explainable-dr-grading"
cd /workspace
if [ ! -d explainable-dr-grading ]; then
  git clone --depth 1 "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO}.git" explainable-dr-grading
  git -C explainable-dr-grading remote set-url origin "https://github.com/${REPO}.git"  # drop the token from .git/config
fi
cd explainable-dr-grading
echo "commit: $(git rev-parse HEAD)"
# Install through the interpreter that runs training (pip can belong to another Python).
# timm 1.0.11 cannot import huggingface_hub 1.x, so it is pinned below 1.0 (seen on a pod, 2026-10-20).
python -m pip install -q "timm==1.0.11" "scikit-learn==1.7.2" "huggingface_hub<1.0" safetensors PyYAML pandas Pillow gdown
python -m pip install -q --no-deps -e .
python -c "import timm; timm.create_model('efficientnet_b0.ra_in1k', pretrained=True); print('pretrained weights OK')"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "nproc: $(nproc) (may be the host; pass CPUS=<allocated vCPU> to run_parallel.sh)"
mkdir -p data/processed
if [ ! -f data/processed/s512/manifest.csv ]; then
  # Newer gdown versions dropped --fuzzy; they accept a Drive file id, taken here from the link.
  FILE_ID=$(python -c "import re,sys; m=re.search(r'/d/([^/?]+)|[?&]id=([^&]+)', sys.argv[1]); print(m.group(1) or m.group(2))" "$ZIP_URL")
  gdown "$FILE_ID" -O /workspace/s512.zip
  python -c "import zipfile; zipfile.ZipFile('/workspace/s512.zip').extractall('data/processed')"
fi
python - <<'PY'
import os, pandas as pd
rows = len(pd.read_csv("data/processed/s512/manifest.csv"))
jpgs = sum(f.endswith(".jpg") for f in os.listdir("data/processed/s512"))
assert rows == jpgs, (rows, jpgs)
print(f"{jpgs} images match the manifest")
PY
