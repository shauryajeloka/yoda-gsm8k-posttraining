#!/usr/bin/env bash
# How much of each stage survives being merged into bf16 weights, and what
# that did to the RLVR comparisons. These were run by hand on the third pod
# (2026-09-26) while the pipeline waited; this script replays them in order.
#
#   outputs/merge_precision.json   weight-level loss + exact KL, merged vs trained
#   outputs/rlaif-merged/          RLAIF merged into bf16 = the start of Arms A-C
#                                  (scale 0 of an Arm-A-style adapter)
#   outputs/sft-merged/            SFT merged into bf16 = the start of RLAIF
#   outputs/kl_drift_a2.json       where Arm A2 moved, relative to true RLAIF
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache}

python scripts/merge_precision.py

pm_subset() {
  python - "$1" <<'PY'
import json, sys
d = sys.argv[1]
ids = [json.loads(l)["id"] for l in open("data/math/gsm8k_persona_math_eval.jsonl")]
g = {json.loads(l)["id"]: l for l in open(f"{d}/gsm8k_eval.jsonl")}
open(f"{d}/persona_math_eval.jsonl", "w").writelines(g[i] for i in ids)
PY
}

# rlvr-verifier-lora records no parent_unmerged, so generate.py merges SFT and
# RLAIF into bf16 exactly as the Arm A trainer did; scale 0 removes Arm A.
for spec in "outputs/rlvr-verifier-lora:outputs/rlaif-merged" \
            "outputs/rlaif-lora:outputs/sft-merged"; do
  adapter=${spec%%:*}; out=${spec##*:}
  python scripts/generate.py --adapter "$adapter" --adapter-scale 0 \
      --prompts data/math/gsm8k_eval.jsonl --out "$out/gsm8k_eval.jsonl" --batch-size 48
  python scripts/generate.py --adapter "$adapter" --adapter-scale 0 \
      --prompts data/persona/persona_eval_prompts.jsonl --out "$out/persona_eval.jsonl" --batch-size 64
done
pm_subset outputs/rlaif-merged

python scripts/kl_drift.py outputs/rlvr-verifier-v2-lora:outputs/rlvr-verifier-v2 \
    --out outputs/kl_drift_a2.json
