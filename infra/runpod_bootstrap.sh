#!/usr/bin/env bash
# Run once on a fresh RunPod pod. Assumes a PyTorch image (torch preinstalled).
#
#   bash infra/runpod_bootstrap.sh
#
# Recommended pod: 1x A40 (48GB) or 1x A100 40GB. LoRA on Qwen2.5-3B needs
# ~20GB; a 24GB card works. Full fine-tuning needs 80GB.
set -euo pipefail

# Put the Hugging Face cache on the PERSISTENT volume, not the container disk.
# Container disk is wiped when the pod is stopped; without this you re-download
# 6.2 GB of weights every time you resume.
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
grep -q 'HF_HOME=/workspace/hf_cache' ~/.bashrc 2>/dev/null \
  || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

case "$PWD" in
  /workspace/*) : ;;
  *) echo "WARNING: you are in $PWD, not under /workspace." >&2
     echo "         Work here is lost if the pod is STOPPED. Clone into /workspace." >&2 ;;
esac

echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "== disk =="; df -h /workspace / | sed 's/^/  /'

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
