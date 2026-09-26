#!/usr/bin/env bash
# Week 3: RLVR as a two-arm ablation, both arms starting from the RLAIF policy.
#
#   Arm A  verifier only            R = verify(answer)
#   Arm B  verifier + persona       R = verify(answer) + 0.5 * persona
#
# Identical in every other respect (same filtered prompt pool, seed, steps,
# KL anchor, sampling), so the paired difference between them is the effect of
# the persona term alone.
#
# Pre-registered prediction, from the styling control: speaking Yoda on maths
# answers costs ~15 points of accuracy, so Arm A's reward pays the policy to
# drop the voice exactly there. Arm A should therefore gain maths AND lose
# persona-on-maths; Arm B should gain less maths and hold the voice. If Arm A
# keeps its voice anyway, the KL anchor alone suffices -- also worth knowing.
#
# Sizing decisions (each audited against the RLAIF run's settings):
#   --max-new-tokens 512   RLAIF used 192, which would truncate 40% of the
#                          policy's maths answers; the verifier scores a
#                          truncated answer wrong, so a low cap secretly
#                          rewards shorter reasoning. 400 looked sufficient
#                          on greedy outputs (0.4%) but truncated 8% at the
#                          sampling temperature in the smoke test; 512.
#   --micro-batch 6        gradient accumulation in group-sized chunks, exact
#                          w.r.t. the full-batch loss, sized for a 48GB card.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ST=outputs/rlvr_status.log
mark() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$ST"; }

START=${START:-outputs/rlaif-lora}
STEPS=${STEPS:-200}
COMMON="--adapter $START --rlvr-prompts work/rlvr_prompts.jsonl
        --steps $STEPS --beta 0.05 --prompts-per-step 4 --group-size 6
        --max-new-tokens 512 --micro-batch 6 --save-every 50"

gen_evals() {  # $1 = adapter dir, $2 = output arm name
  python scripts/generate.py --adapter "$1" --prompts data/math/gsm8k_eval.jsonl \
      --out "outputs/$2/gsm8k_eval.jsonl" --batch-size 48
  python scripts/generate.py --adapter "$1" \
      --prompts data/persona/persona_eval_prompts.jsonl \
      --out "outputs/$2/persona_eval.jsonl" --batch-size 64
}

if [ ! -s work/rlvr_prompts.jsonl ]; then
  mark "PREPASS_START"
  python scripts/rlvr_prepass.py --adapter "$START" --n 800 --g 6 --max-new-tokens 512
  mark "PREPASS_DONE"
fi

if [ ! -f outputs/rlvr-verifier/gsm8k_eval.jsonl ]; then
  mark "ARM_A_START"
  python scripts/train_rlaif.py $COMMON --reward-kind verifier \
      --out outputs/rlvr-verifier-lora
  mark "ARM_A_TRAINED"
  gen_evals outputs/rlvr-verifier-lora rlvr-verifier
  mark "ARM_A_DONE"
fi

if [ ! -f outputs/rlvr-combined/gsm8k_eval.jsonl ]; then
  if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi
  [ -n "${ANTHROPIC_API_KEY:-}" ] || { mark "ARM_B_SKIPPED no key"; exit 0; }
  mark "ARM_B_START"
  python scripts/train_rlaif.py $COMMON --reward-kind combined \
      --lambda-persona 0.5 --claude-model claude-haiku-4-5-20251001 \
      --out outputs/rlvr-combined-lora
  mark "ARM_B_TRAINED"
  gen_evals outputs/rlvr-combined-lora rlvr-combined
  mark "ARM_B_DONE"
fi
mark "RLVR_ALL_DONE"
