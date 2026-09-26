#!/usr/bin/env bash
# Two follow-ups to the Checkpoint-3 finding that the general-chat voice erodes
# under maths-only RLVR whatever the KL penalty does.
#
# 1. Blend. RLAIF + alpha * (Arm A's RLVR update), alpha in {0.25, 0.5, 0.75}.
#    No training. Does the maths gain come apart from the voice loss along that
#    one direction? Prediction: roughly a straight-line trade-off, because the
#    update is small (KL ~0.004) and the model should respond almost linearly.
#
# 2. Arm D. Arm A plus general prompts, each reward sent where it can separate
#    a group's samples: the verifier on 3 maths prompts per step, the
#    Checkpoint-2 persona reward (Haiku + guardrails) on 1 general prompt.
#    Everything else as Arm A: RLAIF start, 511-problem pool, seed, beta 0.05,
#    200 steps, G=6, lr 1e-5, 512-token cap.
#    Prediction: maths near Arm A (73.0%), general-chat voice near RLAIF (3.07)
#    rather than Arm A (2.83). If the voice still drops, our within-group
#    explanation for Arm B's failure is wrong.
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

# ---- 1. blend -------------------------------------------------------------
mark "BLEND_CHECK"
# Scale 0 must reproduce RLAIF, not Arm A, or the scaling is not being applied.
head -48 data/math/gsm8k_eval.jsonl > work/gsm8k_first48.jsonl
python scripts/generate.py --adapter outputs/rlvr-verifier-lora --adapter-scale 0 \
    --prompts work/gsm8k_first48.jsonl --out work/blend_check_scale0.jsonl \
    --batch-size 48 --overwrite
python - <<'PY'
import json
# First 150 characters, not whole answers: RLAIF is merged in bf16 here but
# was applied unmerged when its eval was generated, so one flipped token late
# in a long answer should not count as a mismatch.
jl = lambda p: [json.loads(l)["response"][:150] for l in open(p)]
s0 = jl("work/blend_check_scale0.jsonl")
ra = jl("outputs/rlaif/gsm8k_eval.jsonl")[:48]
aa = jl("outputs/rlvr-verifier/gsm8k_eval.jsonl")[:48]
m_r = sum(a == b for a, b in zip(s0, ra)); m_a = sum(a == b for a, b in zip(s0, aa))
print(f"scale 0, first 150 chars: same as RLAIF {m_r}/48, same as Arm A {m_a}/48")
assert m_r >= m_a + 8, "scale 0 is not closer to RLAIF than to Arm A"
PY
mark "BLEND_START"
for pct in 25 50 75; do
  gen_evals outputs/rlvr-verifier-lora "outputs/blend-$pct" "0.$pct"
done
mark "BLEND_DONE"

# ---- 2. Arm D -------------------------------------------------------------
load_env() { if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi; }
load_env
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  mark "ARM_D_WAITING_FOR_KEY"
  until [ -n "${ANTHROPIC_API_KEY:-}" ]; do sleep 30; load_env; done
fi

ARGS="--adapter outputs/rlaif-lora --rlvr-prompts work/rlvr_prompts.jsonl
      --prompts data/persona/general_yoda_train.jsonl
      --reward-kind routed --general-per-step 1
      --claude-model claude-haiku-4-5-20251001
      --beta 0.05 --prompts-per-step 4 --group-size 6
      --max-new-tokens 512 --micro-batch 6"

mark "ARM_D_SMOKE"
python scripts/train_rlaif.py $ARGS --steps 2 --save-every 1000 \
    --out /workspace/smoke-routed
python - <<'PY'
import json, math
rows = [json.loads(l) for l in open("/workspace/smoke-routed/rlaif_log.jsonl")]
assert len(rows) == 2, rows
for r in rows:
    for k in ("verifier", "persona", "kl_maths", "kl_general", "probe_general",
              "words_general", "mixed_groups"):
        assert k in r and math.isfinite(r[k]), (k, r)
    assert 0 <= r["persona"] <= 1 and not r["skipped"], r
print("smoke OK:", {k: round(rows[-1][k], 3) for k in
                    ("verifier", "persona", "kl_general", "words_general", "mixed_groups")})
PY

mark "ARM_D_START"
python scripts/train_rlaif.py $ARGS --steps 200 --save-every 50 \
    --out outputs/rlvr-routed-lora
mark "ARM_D_TRAINED"
gen_evals outputs/rlvr-routed-lora outputs/rlvr-routed 1.0
python scripts/kl_drift.py outputs/rlvr-routed-lora:outputs/rlvr-routed \
    --out outputs/kl_drift_routed.json
mark "ARM_D_DONE"
