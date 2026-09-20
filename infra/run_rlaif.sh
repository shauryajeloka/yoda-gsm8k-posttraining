#!/usr/bin/env bash
# Week 2: RLAIF on top of SFT, then evaluate against the SFT baseline.
#
# Trains with GRPO against the style classifier, then regenerates BOTH frozen
# eval sets so the Week 2 comparison is like-for-like with SFT:
#
#   persona_eval  -> did the persona improve?
#   gsm8k_eval    -> what did it cost the maths?
#
# The second one is not optional. Persona-only RL has every incentive to drift
# away from the reasoning the SFT model retained, and a Week 2 writeup that
# reports only the persona gain is reporting half the result.
#
# The LLM judge is NOT used as the reward -- it stays held out so that the
# judge score remains an independent check on whether the classifier reward
# was gamed. Run scripts/eval_persona.py afterwards (needs ANTHROPIC_API_KEY)
# to get that number.
set -euo pipefail
cd "$(dirname "$0")/.."

SFT=${SFT:-outputs/sft-lora}
OUT=${OUT:-outputs/rlaif-lora}
STEPS=${STEPS:-200}
BETA=${BETA:-0.05}

echo "===== RLAIF (GRPO, persona classifier reward) ====="
python scripts/train_rlaif.py \
    --adapter "$SFT" --out "$OUT" \
    --steps "$STEPS" --beta "$BETA" \
    --prompts-per-step 4 --group-size 6 --max-new-tokens 128

echo "===== generating on the frozen eval sets ====="
python scripts/generate.py --adapter "$OUT" \
    --prompts data/persona/persona_eval_prompts.jsonl \
    --out outputs/rlaif/persona_eval.jsonl --batch-size 64
python scripts/generate.py --adapter "$OUT" \
    --prompts data/math/gsm8k_eval.jsonl \
    --out outputs/rlaif/gsm8k_eval.jsonl --batch-size 64

echo
echo "===== PERSONA: base -> SFT -> RLAIF ====="
python scripts/persona_classifier.py --model outputs/persona_clf.json \
    --score base=outputs/base/persona_eval.jsonl \
    --score sft=outputs/sft-epoch3/persona_eval.jsonl \
    --score rlaif=outputs/rlaif/persona_eval.jsonl

python scripts/persona_similarity.py \
    --candidate base=outputs/base/persona_eval.jsonl \
    --candidate sft=outputs/sft-epoch3/persona_eval.jsonl \
    --candidate rlaif=outputs/rlaif/persona_eval.jsonl

echo
echo "===== MATH: what RLAIF cost ====="
python scripts/eval_math.py \
    outputs/base-uncapped/gsm8k_eval.jsonl \
    outputs/sft-epoch3/gsm8k_eval.jsonl \
    outputs/rlaif/gsm8k_eval.jsonl

echo
echo "===== REWARD-HACKING DIAGNOSTICS ====="
python - <<'PY'
import json, statistics, sys
sys.path.insert(0, "scripts")
from persona_similarity import features
for name in ("sft-epoch3", "rlaif"):
    p = f"outputs/{'sft-epoch3' if name=='sft-epoch3' else 'rlaif'}/persona_eval.jsonl"
    rows = [json.loads(l)["response"] for l in open(p)]
    w = statistics.mean(len(r.split()) for r in rows)
    c = statistics.mean(features(r)[0] for r in rows)
    uniq = statistics.mean(len(set(r.lower().split())) / max(len(r.split()), 1)
                           for r in rows)
    print(f"{name:11s} words {w:6.1f}   inversion cues {c:5.2f}   "
          f"type/token {uniq:.3f}")
print()
print("Cues climbing far above SFT, or type/token collapsing, means the")
print("policy is gaming the linear reward rather than improving the voice.")
print("The LLM judge is the arbiter: run scripts/eval_persona.py on")
print("outputs/sft-epoch3/persona_eval.jsonl and outputs/rlaif/persona_eval.jsonl.")
PY
echo RLAIF_DONE
