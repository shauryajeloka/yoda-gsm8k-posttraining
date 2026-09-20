#!/usr/bin/env bash
# Parity row: the SFT model under the SAME answer-format instruction that was
# used to produce base-fmt.
#
# Without this row, base-fmt (81.4%) and sft-epoch3 (61.4%) differ in the
# prompt as well as in the model, so the gap between them confounds "what SFT
# did to the model" with "what the instruction did to the prompt". With it,
# base-fmt vs sft-epoch3-fmt is a clean, format-matched model comparison in
# which the fallback extraction rules never fire for either side.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=outputs/sft-lora/checkpoint-339        # epoch 3
test -f "$CKPT/adapter_model.safetensors" || { echo "missing $CKPT" >&2; exit 1; }

python scripts/generate.py --adapter "$CKPT" \
  --prompts data/math/gsm8k_eval.jsonl \
  --out outputs/sft-epoch3-fmt/gsm8k_eval.jsonl \
  --batch-size 64 \
  --prompt-suffix '

End your reply with "Final answer: <number>".'

echo PARITY_DONE
