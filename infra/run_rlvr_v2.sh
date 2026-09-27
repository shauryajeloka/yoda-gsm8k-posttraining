#!/usr/bin/env bash
# RLVR from the RLAIF checkpoint as it was actually trained and evaluated.
#
# Arms A-C merged RLAIF's LoRA into the bf16 weights before attaching their
# own. bf16 cannot hold an update that small: 84% of RLAIF's weight entries
# round away, and the merged model sits 0.0083 nats/token from RLAIF on
# general-chat answers (scripts/merge_precision.py), more than twice as far as
# Arm A's whole run then moved it. Every arm here keeps RLAIF as a frozen,
# unmerged LoRA (--parent-unmerged), so the start and the KL reference are the
# RLAIF model in the Checkpoint-2 tables.
#
#   A2     verifier only, otherwise identical to Arm A
#   blend  RLAIF + a * (A2 - RLAIF), a in {0.25, 0.5, 0.75}
#
# The persona-term and KL ablations on this start are infra/run_rlvr_v2_ablations.sh.
# (An Arm D, general prompts scored by the persona reward, was planned here to
# repair a general-chat voice loss; A2 showed that loss was the merge, so it was
# dropped. The routed reward it would have used is still in train_rlaif.py.)
#
# Pre-registered: A2 gains maths like Arm A. If the merge explained Arm A's
# general-chat loss, A2's general-chat voice stays near RLAIF's 3.07.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ST=outputs/rlvr_status.log
mark() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$ST"; }

pm_subset() {  # the 150 persona x maths rows, verbatim from $1/gsm8k_eval.jsonl
  python - "$1" <<'PY'
import json, sys
d = sys.argv[1]
ids = [json.loads(l)["id"] for l in open("data/math/gsm8k_persona_math_eval.jsonl")]
g = {json.loads(l)["id"]: l for l in open(f"{d}/gsm8k_eval.jsonl")}
open(f"{d}/persona_math_eval.jsonl", "w").writelines(g[i] for i in ids)
print(f"{d}/persona_math_eval.jsonl: {len(ids)} rows")
PY
}

gen_evals() {  # $1 adapter, $2 output dir, $3 adapter scale
  python scripts/generate.py --adapter "$1" --adapter-scale "$3" \
      --prompts data/math/gsm8k_eval.jsonl --out "$2/gsm8k_eval.jsonl" --batch-size 48
  python scripts/generate.py --adapter "$1" --adapter-scale "$3" \
      --prompts data/persona/persona_eval_prompts.jsonl \
      --out "$2/persona_eval.jsonl" --batch-size 64
  pm_subset "$2"
}

COMMON="--adapter outputs/rlaif-lora --parent-unmerged
        --rlvr-prompts work/rlvr_prompts.jsonl
        --beta 0.05 --prompts-per-step 4 --group-size 6
        --max-new-tokens 512 --micro-batch 6"

# RLAIF regenerated on this pod: the noise floor for every comparison below.
mark "RLAIF_REGEN_START"
gen_evals outputs/rlaif-lora outputs/rlaif-regen 1.0
mark "RLAIF_REGEN_DONE"

mark "ARM_A2_START"
python scripts/train_rlaif.py $COMMON --reward-kind verifier --steps 200 \
    --save-every 50 --out outputs/rlvr-verifier-v2-lora
mark "ARM_A2_TRAINED"
gen_evals outputs/rlvr-verifier-v2-lora outputs/rlvr-verifier-v2 1.0
mark "ARM_A2_DONE"

mark "BLEND_START"
for pct in 25 50 75; do
  gen_evals outputs/rlvr-verifier-v2-lora "outputs/blend-v2-$pct" "0.$pct"
done
mark "BLEND_DONE"
