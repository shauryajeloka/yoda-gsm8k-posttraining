#!/usr/bin/env bash
# Is the persona the problem at all?
#
# Every result so far confounds three things: the Yoda voice, the short
# targets, and the fact that GSM8K reference solutions are a WEAKER reasoner
# than Qwen2.5-3B-Instruct (3.5 arithmetic steps against the base model's
# 6.62). Nothing run so far separates them, because there has never been a
# no-persona arm.
#
#   flatmath          1500 GSM8K reference solutions, zero persona data
#   flatmath_plusgen  the same, plus the same 407 general persona examples,
#                     so the mix matches the Yoda SFT run exactly
#
# If these drop to roughly the same ~61% the Yoda SFT did, the persona is
# innocent and the cause is SFT on reference-quality traces. If they hold near
# base, the persona is implicated after all.
#
# Step 3 generates the base model's OWN chain-of-thought over the 1500
# TRAINING problems. That is the raw material for the likely fix: targets that
# keep the model's own reasoning depth instead of GSM8K's shallower chain.
set -euo pipefail
cd "$(dirname "$0")/.."

for ARM in flatmath flatmath_plusgen; do
  echo "===== training $ARM ====="
  python scripts/train_sft.py \
      --data "data/combined/sft_train_${ARM}.jsonl" \
      --output-dir "outputs/sft-${ARM}" --epochs 3
  echo "===== generating $ARM ====="
  python scripts/generate.py --adapter "outputs/sft-${ARM}" \
      --prompts data/math/gsm8k_eval.jsonl \
      --out "outputs/${ARM}/gsm8k_eval.jsonl" --batch-size 64
  python scripts/generate.py --adapter "outputs/sft-${ARM}" \
      --prompts data/persona/persona_eval_prompts.jsonl \
      --out "outputs/${ARM}/persona_eval.jsonl" --batch-size 64
done

echo "===== base CoT over the 1500 TRAINING problems (material for the fix) ====="
python scripts/generate.py --model Qwen/Qwen2.5-3B-Instruct \
    --prompts data/math/gsm8k_train_prompts.jsonl \
    --out outputs/base-train-cot/gsm8k_train.jsonl \
    --batch-size 64 --system-prompt qwen_default --max-new-tokens 1024

echo FLAT_DONE
