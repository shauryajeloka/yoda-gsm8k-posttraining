#!/usr/bin/env bash
# The base model was hitting --max-new-tokens 400 on 84/500 GSM8K items (and
# base+fmt on 17). A truncated response cannot state an answer, so the cap was
# scoring the verbose arms as wrong for running long rather than for reasoning
# badly -- a bias that falls entirely on the BASE rows, which are exactly the
# rows the whole comparison is measured against.
#
# Regenerate the affected arms with a cap high enough that nothing is cut
# (max observed was 400/400, so 1024 leaves real headroom). The SFT arms are
# untouched: their longest response is 400 tokens on a single item and their
# mean is ~80, so the cap never bound them.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=Qwen/Qwen2.5-3B-Instruct

python scripts/generate.py --model "$BASE" \
  --prompts data/math/gsm8k_eval.jsonl \
  --out outputs/base-uncapped/gsm8k_eval.jsonl \
  --batch-size 64 --system-prompt qwen_default --max-new-tokens 1024

python scripts/generate.py --model "$BASE" \
  --prompts data/math/gsm8k_eval.jsonl \
  --out outputs/base-fmt-uncapped/gsm8k_eval.jsonl \
  --batch-size 64 --system-prompt qwen_default --max-new-tokens 1024 \
  --prompt-suffix '

End your reply with "Final answer: <number>".'

# The one SFT item that hit the cap, re-run for symmetry.
python scripts/generate.py --adapter outputs/sft-lora/checkpoint-339 \
  --prompts data/math/gsm8k_eval.jsonl \
  --out outputs/sft-epoch3-uncapped/gsm8k_eval.jsonl \
  --batch-size 64 --max-new-tokens 1024

echo UNCAPPED_DONE
