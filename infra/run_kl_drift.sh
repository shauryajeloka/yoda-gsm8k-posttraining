#!/usr/bin/env bash
# After Arm C: build its persona-on-maths subset (the 150 frozen persona x maths
# ids, rows copied verbatim from its GSM8K eval answers, same as Arms A and B),
# then measure where each RLVR arm drifted from RLAIF (scripts/kl_drift.py).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
ST=outputs/rlvr_status.log
mark() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$ST"; }

python - <<'PY'
import json
ids = [json.loads(l)["id"] for l in open("data/math/gsm8k_persona_math_eval.jsonl")]
g = {json.loads(l)["id"]: l for l in open("outputs/rlvr-nokl/gsm8k_eval.jsonl")}
with open("outputs/rlvr-nokl/persona_math_eval.jsonl", "w") as f:
    f.writelines(g[i] for i in ids)
print(f"persona_math_eval: {len(ids)} rows")
PY

mark "KL_DRIFT_START"
python scripts/kl_drift.py \
    outputs/rlvr-verifier-lora:outputs/rlvr-verifier \
    outputs/rlvr-combined-lora:outputs/rlvr-combined \
    outputs/rlvr-nokl-lora:outputs/rlvr-nokl
mark "KL_DRIFT_DONE"
