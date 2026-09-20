#!/usr/bin/env bash
# Matched-pair experiment: does LENGTH of the SFT targets explain the math drop?
#
# Two LoRA runs over the SAME problems and the SAME general-persona examples.
# The only difference is how long the math answers are:
#
#   v1_control   50.3 words per answer   (the original v1 rewrites)
#   v2_long     121.4 words per answer   (re-authored, same arithmetic)
#
# Because the problem set, the mix and every hyperparameter are identical, any
# gap between the two arms is attributable to target length and not to data
# size, topic coverage or persona content. That is what the earlier base-model
# brevity control could not establish: it showed brevity HURTS a model that is
# already good, not that training LONGER helps a model we are fine-tuning.
set -euo pipefail
cd "$(dirname "$0")/.."

for ARM in v1_control v2_long; do
  echo "===== training $ARM ====="
  python scripts/train_sft.py \
      --data "data/combined/sft_train_${ARM}.jsonl" \
      --output-dir "outputs/sft-${ARM}" \
      --epochs 3
done

for ARM in v1_control v2_long; do
  echo "===== generating $ARM ====="
  python scripts/generate.py --adapter "outputs/sft-${ARM}" \
      --prompts data/math/gsm8k_eval.jsonl \
      --out "outputs/${ARM}/gsm8k_eval.jsonl" --batch-size 64
  python scripts/generate.py --adapter "outputs/sft-${ARM}" \
      --prompts data/persona/persona_eval_prompts.jsonl \
      --out "outputs/${ARM}/persona_eval.jsonl" --batch-size 64
done

echo "===== MATH ====="
python scripts/eval_math.py \
    outputs/base-uncapped/gsm8k_eval.jsonl \
    outputs/v1_control/gsm8k_eval.jsonl \
    outputs/v2_long/gsm8k_eval.jsonl

echo "===== PERSONA ====="
python scripts/persona_classifier.py --model outputs/persona_clf.json \
    --score base=outputs/base/persona_eval.jsonl \
    --score v1_control=outputs/v1_control/persona_eval.jsonl \
    --score v2_long=outputs/v2_long/persona_eval.jsonl

python - <<'PY'
import json, statistics
for a in ("base-uncapped","v1_control","v2_long"):
    w=[len(json.loads(l)["response"].split()) for l in open(f"outputs/{a}/gsm8k_eval.jsonl")]
    print(f"{a:14s} mean {statistics.mean(w):6.1f} words")
PY
echo V2_DONE
