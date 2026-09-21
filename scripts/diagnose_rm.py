#!/usr/bin/env python3
# SUPERSEDED -- kept for the negative result it produced, not for reuse.
#
# Part of the abandoned "distil judge preferences into a Bradley-Terry reward
# model" pipeline. It was built, run, and measured, and the measurement is the
# reason it is not used:
#
#   Qwen2.5-7B ranking 4 on-policy completions agreed with itself only 0.590
#   of the time under order reversal, and put whatever sat in slot A first
#   46.9% of the time (chance 25%). Two-way comparison was WORSE: 0.493 --
#   a coin flip -- with 72.9% of votes going to whichever came first. The
#   reward model distilled from those labels reached 0.631 held-out pairwise
#   accuracy against a 0.552 linear baseline and failed its gate.
#
#   Root cause: any comparative prompt puts several candidates in one context
#   window, and this judge answered from position rather than style.
#
# The live path scores each completion ALONE (scripts/llm_reward.py, or
# scripts/claude_reward.py), which removes position structurally. Distillation
# is also unnecessary at this scale -- GRPO needs ~3,600 judgements total, and
# the whole point of RLAIF is that an AI judge CAN be queried online.
#
# Do not delete: work/rm_pairs_biased.jsonl and outputs/reward-model-biased
# are the evidence behind the Week 2 writeup's position-bias section.

"""Why did the reward model miss its gate?

Held-out pairwise accuracy was 0.631 against a 0.75 floor, with 0.943 on
training pairs. Three candidate causes, and they call for opposite fixes, so
guessing is expensive:

  A. JUDGE NOISE. If the 7B judge disagrees with itself when shown the same
     group twice, its labels have a consistency ceiling and no reward model
     can score above it. Fix: a better judge, or an easier judging task.
  B. IRREDUCIBLE FINE DISTINCTIONS. Ranking 4 completions forces the judge to
     separate adjacent items that may genuinely be equivalent. If accuracy is
     high on distant pairs (best vs worst) and chance-level on adjacent ones,
     the signal is real and only the near-ties are noise. Fix: train on
     confident pairs only.
  C. OVERFITTING. 8.8M LoRA params on 2,076 pairs. Fix: shrink the adapter,
     fewer epochs, more data.

This measures A and B. C is already evidenced by the 0.943/0.631 gap.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def rm_scores(rm_path, rows, batch_size=16, max_len=640):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    cfg = json.loads(Path(rm_path, "adapter_config.json").read_text())
    base = cfg["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(rm_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, torch_dtype=torch.float32)
    m.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(m, rm_path).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    def score(pairs_side):
        out = []
        with torch.no_grad():
            for i in range(0, len(pairs_side), batch_size):
                b = pairs_side[i:i + batch_size]
                enc = tok([tok.apply_chat_template(
                              [{"role": "user", "content": q},
                               {"role": "assistant", "content": t}],
                              tokenize=False) for q, t in b],
                          return_tensors="pt", padding=True, truncation=True,
                          max_length=max_len).to(dev)
                out.extend(model(**enc).logits.squeeze(-1).float().tolist())
        return out

    ch = score([(r["prompt"], r["chosen"]) for r in rows])
    rj = score([(r["prompt"], r["rejected"]) for r in rows])
    return ch, rj


def linear_scores(rows):
    from persona_similarity import features
    from persona_classifier import apply_std
    m = json.loads(Path("outputs/persona_clf.json").read_text())

    def s(t):
        x = apply_std(features(t), m["mu"], m["sd"])
        return m["b"] + sum(w * xi for w, xi in zip(m["w"], x))

    return [s(r["chosen"]) for r in rows], [s(r["rejected"]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="work/rm_pairs.jsonl")
    ap.add_argument("--rm", default="outputs/reward-model")
    ap.add_argument("--metrics", default="outputs/reward-model/rm_metrics.json")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    # Reproduce train_reward_model.py's split exactly.
    groups = sorted({p["group_id"] for p in pairs})
    random.Random(args.seed).shuffle(groups)
    val_ids = set(groups[:max(1, int(len(groups) * args.val_frac))])
    val = [p for p in pairs if p["group_id"] in val_ids]
    print(f"val pairs: {len(val)} over {len(val_ids)} groups\n")

    ch, rj = rm_scores(args.rm, val)
    lch, lrj = linear_scores(val)

    print("=== B: accuracy by rank gap (1 = adjacent in the judge's order) ===")
    print(f"{'gap':>4} {'n':>5} {'reward model':>13} {'linear':>8}")
    by = defaultdict(lambda: [0, 0, 0])
    for r, a, b, la, lb in zip(val, ch, rj, lch, lrj):
        g = r.get("rank_gap", 0)
        by[g][0] += 1
        by[g][1] += int(a > b)
        by[g][2] += int(la > lb)
    for g in sorted(by):
        n, ok, lok = by[g]
        print(f"{g:>4} {n:>5} {ok/n:>13.3f} {lok/n:>8.3f}")
    tot = sum(v[0] for v in by.values())
    print(f"{'all':>4} {tot:>5} {sum(v[1] for v in by.values())/tot:>13.3f} "
          f"{sum(v[2] for v in by.values())/tot:>8.3f}")
    print()
    print("A large gap-3 vs gap-1 spread means the judge's coarse preferences are")
    print("learnable and only its adjacent calls are noise -- in which case the fix")
    print("is to train on confident pairs, not to collect more of the same.")


if __name__ == "__main__":
    main()
