#!/usr/bin/env bash
# Arms B and C rerun from RLAIF itself (--parent-unmerged), so the persona-term
# and KL-anchor ablations compare with Arm A2 on a correct start. One arm per
# pod:
#
#   ARM=B2 bash infra/run_rlvr_v2_ablations.sh   verifier + 0.5 * persona
#                                                (Haiku + guardrails), beta 0.05
#   ARM=C2 bash infra/run_rlvr_v2_ablations.sh   verifier only, beta 0
#
# Everything else as A2: RLAIF start, 511-problem pool, seed, 200 steps, G=6,
# lr 1e-5, 512-token cap. Predictions carried over from the merged-start arms:
# B2 below A2 on maths with no persona gain; C2 matches A2 on maths and voice.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ST=outputs/rlvr_status.log
mark() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$ST"; }

case "${ARM:-}" in
  B2) EXTRA="--reward-kind combined --lambda-persona 0.5
             --claude-model claude-haiku-4-5-20251001 --beta 0.05"
      OUT=rlvr-combined-v2; NEED_KEY=1 ;;
  C2) EXTRA="--reward-kind verifier --beta 0.0"
      OUT=rlvr-nokl-v2; NEED_KEY=0 ;;
  *) echo "set ARM=B2 or ARM=C2" >&2; exit 2 ;;
esac
COMMON="--adapter outputs/rlaif-lora --parent-unmerged
        --rlvr-prompts work/rlvr_prompts.jsonl
        --prompts-per-step 4 --group-size 6 --max-new-tokens 512 --micro-batch 6"

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

# A2 regenerated on this pod first: between pods, greedy generation alone moves
# GSM8K by ~2 points, so each ablation is compared with an A2 from its own GPU.
python scripts/generate.py --adapter outputs/rlvr-verifier-v2-lora \
    --prompts data/math/gsm8k_eval.jsonl --out "outputs/rlvr-verifier-v2-on-$ARM/gsm8k_eval.jsonl" --batch-size 48
python scripts/generate.py --adapter outputs/rlvr-verifier-v2-lora \
    --prompts data/persona/persona_eval_prompts.jsonl \
    --out "outputs/rlvr-verifier-v2-on-$ARM/persona_eval.jsonl" --batch-size 64
pm_subset "outputs/rlvr-verifier-v2-on-$ARM"
mark "${ARM}_A2_REGEN_DONE"

if [ "$NEED_KEY" = 1 ]; then
  load_env() { if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi; }
  load_env
  if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    mark "${ARM}_WAITING_FOR_KEY"
    until [ -n "${ANTHROPIC_API_KEY:-}" ]; do sleep 30; load_env; done
  fi
fi

mark "${ARM}_SMOKE"
python scripts/train_rlaif.py $COMMON $EXTRA --steps 2 --save-every 1000 \
    --out "/workspace/smoke-$ARM"
python - "$ARM" <<'PY'
import json, math, sys
rows = [json.loads(l) for l in open(f"/workspace/smoke-{sys.argv[1]}/rlaif_log.jsonl")]
cfg = json.load(open(f"/workspace/smoke-{sys.argv[1]}/training_config.json"))
assert len(rows) == 2 and rows[0]["kl"] == 0.0, rows
assert cfg["parent_unmerged"] is True, cfg
for r in rows:
    assert math.isfinite(r["reward"]) and math.isfinite(r["kl"]) and not r["skipped"], r
    if sys.argv[1] == "B2":
        assert 0 <= r["persona"] <= 1, r
print("smoke OK:", {k: round(v, 4) for k, v in rows[-1].items()
                    if k in ("reward", "verifier", "persona", "kl", "mixed_groups")})
PY

mark "${ARM}_START"
python scripts/train_rlaif.py $COMMON $EXTRA --steps 200 --save-every 50 \
    --out "outputs/$OUT-lora"
mark "${ARM}_TRAINED"
python scripts/generate.py --adapter "outputs/$OUT-lora" \
    --prompts data/math/gsm8k_eval.jsonl --out "outputs/$OUT/gsm8k_eval.jsonl" --batch-size 48
python scripts/generate.py --adapter "outputs/$OUT-lora" \
    --prompts data/persona/persona_eval_prompts.jsonl \
    --out "outputs/$OUT/persona_eval.jsonl" --batch-size 64
pm_subset "outputs/$OUT"
python scripts/kl_drift.py "outputs/$OUT-lora:outputs/$OUT" --out "outputs/kl_drift_$ARM.json"
mark "${ARM}_DONE"
