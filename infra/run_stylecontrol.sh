#!/usr/bin/env bash
# The missing control: what does the YODA STYLING itself cost?
#
# yodadistill (68.4%) differs from selfdistill (82.4%) in three ways at once --
# the Yoda restyling, 563 examples instead of 1,392, and the 407 general-prosa
# mix -- so the write-up's "~10 pts for the restyling" was arithmetic on
# estimates, not a measurement. This closes that gap with a minimal pair:
#
#   plain563     the SAME 563 problems' plain self-distilled traces + the SAME
#                407 general examples, same recipe
#   yodadistill  identical in every respect except the maths targets are the
#                Yoda restylings of those same traces
#
# The paired difference between these two arms IS the styling cost (bundled
# with the restyler's compression, 165 -> 127 words, which was never separated).
# As a bonus, plain563 vs selfdistill measures the dataset-size effect in the
# right data regime for the first time.
#
# One known asymmetry, chosen deliberately: --max-len 1536 here (matching the
# selfdistill arm) because plain traces are longer than their restylings and
# truncating targets would bias the control downward; yodadistill trained at
# 1024, where its shorter targets already fit untruncated.
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=plain563

echo "===== training $ARM ====="
python scripts/train_sft.py \
    --data data/combined/sft_train_plain563.jsonl \
    --output-dir outputs/sft-${ARM} \
    --epochs 3 --max-len 1536

echo "===== generating on BOTH frozen eval sets ====="
python scripts/generate.py --adapter outputs/sft-${ARM} \
    --prompts data/math/gsm8k_eval.jsonl \
    --out outputs/${ARM}/gsm8k_eval.jsonl --batch-size 48
python scripts/generate.py --adapter outputs/sft-${ARM} \
    --prompts data/persona/persona_eval_prompts.jsonl \
    --out outputs/${ARM}/persona_eval.jsonl --batch-size 64

echo
echo "===== MATH: the styling cost, measured directly ====="
python scripts/eval_math.py \
    outputs/selfdistill/gsm8k_eval.jsonl \
    outputs/${ARM}/gsm8k_eval.jsonl \
    outputs/yodadistill/gsm8k_eval.jsonl

echo
echo "===== PERSONA: confirms the styling is what carries the voice ====="
python scripts/persona_classifier.py --model outputs/persona_clf.json \
    --score plain563=outputs/${ARM}/persona_eval.jsonl \
    --score yodadistill=outputs/yodadistill/persona_eval.jsonl

echo STYLECONTROL_DONE
