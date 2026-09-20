#!/usr/bin/env bash
# Does the 113-example pilot generalise? A scaling curve for the SHORT (v1)
# rewrites at 146 / 300 / 800 / 1500 math examples, nested, with the general
# persona set held fixed at 407 throughout.
#
# The v2 long-form pilot trained on only 113 examples, so a fair objection is
# that it was simply underpowered -- maybe length helps once there is enough
# data. This curve answers that without authoring anything: if accuracy is
# roughly flat in n, then a small-n result is informative and the pilot stands.
# If it climbs steeply, the pilot was underpowered and the v2 arm needs to be
# re-run at scale before any conclusion about length can be drawn.
set -euo pipefail
cd "$(dirname "$0")/.."

for N in 146 300 800 1500; do
  echo "===== training scale$N ====="
  python scripts/train_sft.py \
      --data "data/combined/sft_train_scale${N}.jsonl" \
      --output-dir "outputs/sft-scale${N}" --epochs 3
  echo "===== generating scale$N ====="
  python scripts/generate.py --adapter "outputs/sft-scale${N}" \
      --prompts data/math/gsm8k_eval.jsonl \
      --out "outputs/scale${N}/gsm8k_eval.jsonl" --batch-size 64
done
echo SCALE_DONE
