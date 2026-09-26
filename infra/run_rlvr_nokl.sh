#!/usr/bin/env bash
# Arm C: Arm A (verifier-only RLVR) with the KL penalty removed, beta = 0.
#
# Question: was it the KL anchor that held the Yoda voice on maths answers in
# Arm A? Everything except beta matches Arm A -- same 511-problem pool, seed,
# 200 steps, lr, group size, cap -- so the paired difference is the anchor.
#
# Caution specific to this trainer: we take one gradient step per batch, so the
# PPO ratio is always 1 and ratio clipping never activates. With beta = 0 the
# only limits on step size are lr 1e-5 and grad-norm clipping at 1.0 -- looser
# than published KL-free recipes, which keep ratio clipping. KL to the RLAIF
# policy is still computed and logged every step, so drift is measured even
# though it is not penalised.
#
# Pre-registered prediction: maths >= Arm A (73.0%), voice on maths answers
# below Arm A (2.60), general-chat voice below Arm A (2.83).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ST=outputs/rlvr_status.log
mark() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$ST"; }

mark "ARM_C_START"
python scripts/train_rlaif.py --adapter outputs/rlaif-lora \
    --rlvr-prompts work/rlvr_prompts.jsonl --reward-kind verifier \
    --steps 200 --beta 0.0 --prompts-per-step 4 --group-size 6 \
    --max-new-tokens 512 --micro-batch 6 --save-every 50 \
    --out outputs/rlvr-nokl-lora
mark "ARM_C_TRAINED"
python scripts/generate.py --adapter outputs/rlvr-nokl-lora \
    --prompts data/math/gsm8k_eval.jsonl --out outputs/rlvr-nokl/gsm8k_eval.jsonl --batch-size 48
python scripts/generate.py --adapter outputs/rlvr-nokl-lora \
    --prompts data/persona/persona_eval_prompts.jsonl --out outputs/rlvr-nokl/persona_eval.jsonl --batch-size 64
mark "ARM_C_DONE"
