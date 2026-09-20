#!/usr/bin/env bash
# Run once on a fresh RunPod pod. Assumes a PyTorch image (torch preinstalled).
#
#   bash infra/runpod_bootstrap.sh
#
# Recommended pod: 1x A40 (48GB) or 1x A100 40GB. LoRA on Qwen2.5-3B needs
# ~20GB; a 24GB card works. Full fine-tuning needs 80GB.
set -euo pipefail

echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

pip install -q -r infra/requirements.txt

# Pre-download the base model so training does not stall on a cold cache.
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-3B-Instruct")
print("model cached at", p)
PY

# Validate the data against the real chat template BEFORE spending GPU time.
python scripts/check_formatting.py
python data/verifier/gsm8k_verifier.py --selftest
echo "bootstrap OK"
