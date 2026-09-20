#!/usr/bin/env bash
# Does BREVITY alone explain the -9pp GSM8K drop from SFT?
#
# SFT changed two things at once: the model got terse (175 -> 44 words) and it
# got a persona. This isolates brevity by constraining the BASE model (no
# adapter, no persona) to answer in the SFT length band, on the same 500
# prompts, same greedy decoding.
#
#   arm A  format only        -> controls for the answer-format instruction
#   arm B  format + brevity   -> adds the length constraint on top of A
#
# A ~ 70% and B ~ 61% would mean length, not persona, is the mechanism, and
# lengthening the SFT rewrites is the right fix. B ~ 70% would mean brevity is
# innocent and rewriting 1,500 examples would be wasted effort.
set -euo pipefail
BASE=Qwen/Qwen2.5-3B-Instruct
FMT="End your reply with \"Final answer: <number>\"."

python scripts/generate.py --model "$BASE" \
  --prompts data/math/gsm8k_eval.jsonl \
  --out outputs/base-fmt/gsm8k_eval.jsonl \
  --batch-size 64 --system-prompt qwen_default \
  --prompt-suffix "

$FMT"

python scripts/generate.py --model "$BASE" \
  --prompts data/math/gsm8k_eval.jsonl \
  --out outputs/base-fmt-brief/gsm8k_eval.jsonl \
  --batch-size 64 --system-prompt qwen_default \
  --prompt-suffix "

Answer in at most 40 words. $FMT"

echo; echo "===== BREVITY CONTROL ====="
python scripts/eval_math.py \
    outputs/base-bs64/gsm8k_eval.jsonl \
    outputs/base-fmt/gsm8k_eval.jsonl \
    outputs/base-fmt-brief/gsm8k_eval.jsonl \
    outputs/sft-epoch3/gsm8k_eval.jsonl
