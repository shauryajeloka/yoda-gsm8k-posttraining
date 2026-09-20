#!/usr/bin/env bash
# Week 1 end to end: baseline metrics -> SFT -> post-training metrics.
# Detached and resumable, so it survives an SSH disconnect:
#
#   tmux new -s week1 'bash infra/run_week1.sh 2>&1 | tee outputs/week1.log'
#   # detach with ctrl-b d ; reattach with: tmux attach -t week1
#
# Judge scoring is NOT run here -- it needs an API key and is cheap to run
# afterwards from anywhere, including your laptop.
set -euo pipefail
BASE=Qwen/Qwen2.5-3B-Instruct
ADAPTER=outputs/sft-lora

echo "===== 1/4  BASELINE generations (must happen before training) ====="
for SET in \
  "data/persona/persona_eval_prompts.jsonl persona_eval" \
  "data/math/gsm8k_eval.jsonl gsm8k_eval" \
  "data/math/gsm8k_persona_math_eval.jsonl persona_math_eval"; do
  set -- $SET
  python scripts/generate.py --model "$BASE" --prompts "$1" \
      --out "outputs/base/$2.jsonl" --system-prompt qwen_default
done

echo "===== 2/4  SFT ====="
# Resumable: generations skip themselves if present (see generate.py), and
# training is skipped if the adapter is already there. Safe to re-run the whole
# script after any interruption.
if [ -f "$ADAPTER/epoch_checkpoints.json" ]; then
  echo "  $ADAPTER already trained, skipping. Delete it to retrain."
else
  python scripts/train_sft.py --output-dir "$ADAPTER" --epochs 3
fi

echo "===== 3/4  POST-SFT generations ====="
for SET in \
  "data/persona/persona_eval_prompts.jsonl persona_eval" \
  "data/math/gsm8k_eval.jsonl gsm8k_eval" \
  "data/math/gsm8k_persona_math_eval.jsonl persona_math_eval"; do
  set -- $SET
  python scripts/generate.py --adapter "$ADAPTER" --prompts "$1" \
      --out "outputs/sft/$2.jsonl"
done

echo "===== 4/4  Metrics that need no API key ====="
python scripts/eval_math.py outputs/base/gsm8k_eval.jsonl outputs/sft/gsm8k_eval.jsonl
python scripts/persona_similarity.py \
    --candidate base=outputs/base/persona_eval.jsonl \
    --candidate sft=outputs/sft/persona_eval.jsonl
# Fit on the content-controlled math contrast ONLY. Passing
# --extra-negatives outputs/base/persona_eval.jsonl and then scoring that same
# file trains the classifier on rows it is about to grade, which drives the
# base score toward 0 and inflates the base-vs-SFT gap. For domain-matched
# negatives use --extra-negatives WITH --holdout-out, and score the held-out
# half rather than the whole file.
python scripts/persona_classifier.py --fit outputs/persona_clf.json
python scripts/persona_classifier.py \
    --score base=outputs/base/persona_eval.jsonl \
    --score sft=outputs/sft/persona_eval.jsonl

echo
echo "Remaining (needs ANTHROPIC_API_KEY, runnable anywhere):"
echo "  python scripts/eval_persona.py outputs/base/persona_eval.jsonl outputs/sft/persona_eval.jsonl"
echo "  python scripts/eval_persona.py outputs/base/persona_math_eval.jsonl outputs/sft/persona_math_eval.jsonl"
