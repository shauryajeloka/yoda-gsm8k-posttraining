#!/usr/bin/env bash
# Week 2: RLAIF on top of SFT, with an LLM supplying the feedback.
#
#   1. sanity-check the reward discriminates on-policy
#   2. GRPO against the Claude judge + guardrails
#   3. regenerate BOTH frozen eval sets and score
#
# REWARD: Claude Haiku scores each completion alone against the persona rubric,
# then flat penalties apply for Star Wars vocabulary, self-naming and filler
# spam. Validated on a degradation harness with known-correct orderings
# (scripts/validate_judge.py): 0.77 voice-stops-halfway, 0.73 word-salad,
# 0.93 Star-Wars-splice, 0.95 "Yoda I am".
#
# Three earlier rewards were built, measured and rejected -- see the docstring
# in scripts/train_rlaif.py. The short version: a local 7B judge answered
# comparative prompts from POSITION rather than style (46.9% slot-A vs 25%
# chance), and even scored pointwise it PAID the policy +0.16 for appending
# "Yoda I am" despite being told not to in the prompt.
#
# INDEPENDENCE: Haiku scores, Sonnet reports (eval_persona.py). Different
# models, same developer -- partial independence, and the writeup says so.
#
# CREDENTIALS: sourced from /workspace/.env, which the user writes directly on
# the pod. The key is never passed on a command line, never logged, and never
# handled by the assistant. Rotate it after the run: this is a rented machine.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f /workspace/.env ]; then
    set -a; . /workspace/.env; set +a
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY not set and /workspace/.env absent or empty."
    echo "Write it on the pod yourself; do not pass it on a command line."
    exit 2
fi

SFT=${SFT:-outputs/sft-yodadistill}
OUT=${OUT:-outputs/rlaif-lora}
JUDGE=${JUDGE:-claude-haiku-4-5-20251001}
STEPS=${STEPS:-150}
BETA=${BETA:-0.05}

echo "===== 1/3  reward sanity check on-policy ====="
# A reward with ~0 variance INSIDE a group yields identically-zero advantages
# and teaches nothing, however good it looks in aggregate.
python scripts/claude_reward.py --model "$JUDGE" \
    --samples work/rm_samples.jsonl --limit 40

echo
echo "===== 2/3  GRPO against the Claude judge ====="
python scripts/train_rlaif.py \
    --adapter "$SFT" --out "$OUT" \
    --reward-kind claude --claude-model "$JUDGE" \
    --steps "$STEPS" --beta "$BETA" \
    --prompts-per-step 4 --group-size 6 --max-new-tokens 192

echo
echo "===== 3/3  regenerating BOTH frozen eval sets ====="
# Persona alone is half a result: persona-only RL has every incentive to drift
# away from the reasoning SFT retained, so the maths is measured too.
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
    --score week1_sft=outputs/sft-epoch3/persona_eval.jsonl \
    --score yodadistill=outputs/yodadistill/persona_eval.jsonl \
    --score rlaif=outputs/rlaif/persona_eval.jsonl

echo
echo "===== MATH: what RLAIF cost ====="
python scripts/eval_math.py \
    outputs/base-uncapped/gsm8k_eval.jsonl \
    outputs/sft-epoch3-uncapped/gsm8k_eval.jsonl \
    outputs/yodadistill/gsm8k_eval.jsonl \
    outputs/rlaif/gsm8k_eval.jsonl

echo
echo "===== REWARD-HACKING DIAGNOSTICS ====="
python - <<'PY'
import json, re, statistics, sys
sys.path.insert(0, "scripts")
from persona_similarity import features
from claude_reward import SW_RE, NAME_RE, FILLER_RE
for name in ("yodadistill", "rlaif"):
    rows = [json.loads(l)["response"] for l in open(f"outputs/{name}/persona_eval.jsonl")]
    n = len(rows)
    w = statistics.mean(len(r.split()) for r in rows)
    c = statistics.mean(features(r)[0] for r in rows)
    uniq = statistics.mean(len(set(r.lower().split())) / max(len(r.split()), 1) for r in rows)
    sw = sum(bool(SW_RE.search(r)) for r in rows) / n
    nm = sum(bool(NAME_RE.search(r)) for r in rows) / n
    fl = statistics.mean(len(FILLER_RE.findall(r)) for r in rows)
    print(f"{name:13s} words {w:6.1f}  cues {c:5.2f}  type/token {uniq:.3f}  "
          f"starwars {sw:5.1%}  says-Yoda {nm:5.1%}  filler {fl:4.2f}")
print()
print("The guardrails make Star Wars and self-naming costly, so those columns")
print("should stay near zero. Cue density climbing far above the SFT row, or")
print("type/token collapsing, is hacking the part of the reward that is a")
print("judgement call. The arbiter is scripts/eval_persona.py (Sonnet, never")
print("trained against).")
PY
echo RLAIF_FULL_DONE
