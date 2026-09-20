#!/usr/bin/env bash
# Step 2 of the fix: is the 20-point SFT loss caused by TARGET QUALITY?
#
# A perfectly matched pair. Same 1392 GSM8K training problems, no persona data
# in either arm, identical recipe and hyperparameters. The only thing that
# differs is whose reasoning the target contains:
#
#   selfdistill   the base model's OWN verified-correct chain of thought
#                 170.4 words, 6.10 arithmetic steps
#   flatref       GSM8K's reference solution for the same problem
#                  51.4 words, 3.46 arithmetic steps
#
# Both are filtered to problems the base model solves correctly, so the two
# arms see an identical problem set -- this is not a difficulty confound.
#
# If selfdistill holds near the base model's 82% while flatref drops to ~62%,
# the cause is confirmed as target quality: SFT on GSM8K references teaches a
# strong model to imitate a weaker reasoner. Only then is it worth Yodifying.
set -euo pipefail
cd "$(dirname "$0")/.."

for ARM in selfdistill flatref; do
  echo "===== training $ARM ====="
  python scripts/train_sft.py \
      --data "data/combined/sft_train_${ARM}.jsonl" \
      --output-dir "outputs/sft-${ARM}" --epochs 3 --max-len 1536
  echo "===== generating $ARM ====="
  python scripts/generate.py --adapter "outputs/sft-${ARM}" \
      --prompts data/math/gsm8k_eval.jsonl \
      --out "outputs/${ARM}/gsm8k_eval.jsonl" --batch-size 48
done
echo SELFDISTILL_DONE
