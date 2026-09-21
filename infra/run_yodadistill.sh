#!/usr/bin/env bash
# The Week-1 question, asked properly: does WHOSE reasoning you imitate explain
# the 20-point math drop, once the persona is held constant?
#
# Two Yoda arms, both ending in a Yoda voice, differing only in the source of
# the chain of thought that was Yodified:
#
#   sft-epoch3      GSM8K's own reference answers, restyled into Yoda.
#                   51.4 words, someone else's reasoning.        -> 61.4%
#   sft-yodadistill the BASE MODEL's own verified-correct CoT, restyled into
#                   Yoda with a step-preservation guard.
#                   126.7 words, its own reasoning.              -> ?
#
# Both mix in the identical 407 general-persona examples, so persona supervision
# is constant across the pair. The no-persona controls already showed the base
# model's own CoT costs nothing (selfdistill 82.4% vs base 82.0%, p=0.91) while
# GSM8K references cost 19 points (flatref 63.0%). This arm asks whether that
# protection survives the Yoda restyling -- i.e. whether you can have the
# persona without paying the capability.
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=yodadistill

echo "===== training $ARM ====="
python scripts/train_sft.py \
    --data data/combined/sft_train_yodadistill.jsonl \
    --output-dir outputs/sft-${ARM} \
    --epochs 3 --max-len 1024

echo "===== generating $ARM on BOTH frozen eval sets ====="
python scripts/generate.py --adapter outputs/sft-${ARM} \
    --prompts data/math/gsm8k_eval.jsonl \
    --out outputs/${ARM}/gsm8k_eval.jsonl --batch-size 64
python scripts/generate.py --adapter outputs/sft-${ARM} \
    --prompts data/persona/persona_eval_prompts.jsonl \
    --out outputs/${ARM}/persona_eval.jsonl --batch-size 64

echo
echo "===== MATH: the headline comparison ====="
# base-uncapped and sft-epoch3-uncapped are the 1024-token regenerations, so
# every arm here is compared under an identical generation budget.
python scripts/eval_math.py \
    outputs/base-uncapped/gsm8k_eval.jsonl \
    outputs/sft-epoch3-uncapped/gsm8k_eval.jsonl \
    outputs/flatmath/gsm8k_eval.jsonl \
    outputs/selfdistill/gsm8k_eval.jsonl \
    outputs/${ARM}/gsm8k_eval.jsonl

echo
echo "===== MATH: new arm vs the Week-1 Yoda arm, head to head ====="
python scripts/eval_math.py \
    outputs/sft-epoch3-uncapped/gsm8k_eval.jsonl \
    outputs/${ARM}/gsm8k_eval.jsonl

echo
echo "===== PERSONA ====="
python scripts/persona_classifier.py --model outputs/persona_clf.json \
    --score base=outputs/base/persona_eval.jsonl \
    --score week1_sft=outputs/sft-epoch3/persona_eval.jsonl \
    --score ${ARM}=outputs/${ARM}/persona_eval.jsonl

python scripts/persona_similarity.py \
    --candidate base=outputs/base/persona_eval.jsonl \
    --candidate week1_sft=outputs/sft-epoch3/persona_eval.jsonl \
    --candidate ${ARM}=outputs/${ARM}/persona_eval.jsonl

echo
echo "===== generated-response shape (is it still reasoning at length?) ====="
python - <<'PY'
import json, statistics, re
EQ=re.compile(r"[\d,.()][\d,.\s+\-*/x×÷()]*?\s*=\s*-?[\d,]*\.?\d+")
for a in ("base-uncapped","sft-epoch3-uncapped","flatmath","selfdistill","yodadistill"):
    try:
        rows=[json.loads(l)["response"] for l in open(f"outputs/{a}/gsm8k_eval.jsonl")]
    except FileNotFoundError:
        continue
    w=statistics.mean(len(r.split()) for r in rows)
    e=statistics.mean(len(EQ.findall(r)) for r in rows)
    print(f"{a:22s} {w:6.1f} words   {e:5.2f} equations")
PY
echo YODADISTILL_DONE
