#!/usr/bin/env bash
# Epoch sweep: evaluate the model after EACH epoch, not just at the end.
#
# Why this is worth the extra GPU time on a 1,907-example dataset: persona SFT
# can lock in the voice within one epoch and then start overfitting, and the
# interesting question -- does math accuracy fall as the voice strengthens? --
# is only visible if you have more than two points on the curve. The output is
# a persona-vs-accuracy trajectory rather than a single before/after pair,
# which is a much better answer to the trade-off question in the writeup.
#
#   tmux new -s sweep 'bash infra/run_epoch_sweep.sh 2>&1 | tee outputs/sweep.log'
#
# Assumes infra/run_week1.sh has already produced outputs/base/* and trained
# outputs/sft-lora with --epochs 3.
set -euo pipefail

BASE=Qwen/Qwen2.5-3B-Instruct
ADAPTER=outputs/sft-lora

if [ ! -f "$ADAPTER/epoch_checkpoints.json" ]; then
  echo "No epoch_checkpoints.json -- run infra/run_week1.sh first." >&2
  exit 1
fi

# One generation pass per epoch, over all three frozen sets.
python - <<'PY' > /tmp/epoch_paths.txt
import json
for m in json.load(open("outputs/sft-lora/epoch_checkpoints.json")):
    if "note" not in m:                     # skip the duplicate final save
        print(f"{int(m['epoch'])}\t{m['path']}")
PY

while IFS=$'\t' read -r EPOCH CKPT; do
  echo "===== epoch $EPOCH : $CKPT ====="
  for SET in \
    "data/persona/persona_eval_prompts.jsonl persona_eval" \
    "data/math/gsm8k_eval.jsonl gsm8k_eval" \
    "data/math/gsm8k_persona_math_eval.jsonl persona_math_eval"; do
    set -- $SET
    OUT="outputs/sft-epoch$EPOCH/$2.jsonl"
    [ -f "$OUT" ] && { echo "  $OUT exists, skipping"; continue; }
    python scripts/generate.py --adapter "$CKPT" --prompts "$1" --out "$OUT"
  done
done < /tmp/epoch_paths.txt

echo
echo "===== MATH ACCURACY BY EPOCH ====="
python scripts/eval_math.py \
    outputs/base/gsm8k_eval.jsonl \
    outputs/sft-epoch1/gsm8k_eval.jsonl \
    outputs/sft-epoch2/gsm8k_eval.jsonl \
    outputs/sft-epoch3/gsm8k_eval.jsonl

echo
echo "===== PERSONA (no API key needed) BY EPOCH ====="
python scripts/persona_similarity.py \
    --candidate base=outputs/base/persona_eval.jsonl \
    --candidate epoch1=outputs/sft-epoch1/persona_eval.jsonl \
    --candidate epoch2=outputs/sft-epoch2/persona_eval.jsonl \
    --candidate epoch3=outputs/sft-epoch3/persona_eval.jsonl

python scripts/persona_classifier.py \
    --score base=outputs/base/persona_eval.jsonl \
    --score epoch1=outputs/sft-epoch1/persona_eval.jsonl \
    --score epoch2=outputs/sft-epoch2/persona_eval.jsonl \
    --score epoch3=outputs/sft-epoch3/persona_eval.jsonl

cat <<'EOF'

===== REMAINING: judge scoring (needs ANTHROPIC_API_KEY) =====
Roughly 600 judge calls for the full sweep. Smoke-test on 20 first:

  python scripts/eval_persona.py --limit 20 outputs/sft-epoch1/persona_eval.jsonl

then the whole thing:

  python scripts/eval_persona.py \
      outputs/base/persona_eval.jsonl \
      outputs/sft-epoch1/persona_eval.jsonl \
      outputs/sft-epoch2/persona_eval.jsonl \
      outputs/sft-epoch3/persona_eval.jsonl

Pick the epoch for the Week 1 row on the PERSONA/ACCURACY trade-off you see,
not on validation loss alone -- and say in the writeup which you picked and why.
EOF
