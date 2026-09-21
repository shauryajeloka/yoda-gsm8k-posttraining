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

"""Distil the LLM judge's preferences into a small neural reward model.

Bradley-Terry over pairs: the model emits a scalar r(x) per completion and is
trained so the judge's preferred completion scores higher,

    loss = -log sigmoid( r(chosen) - r(rejected) )

Why a neural model rather than refitting the existing logistic regression on
the judge's labels: that model is 51 hand-designed style features, so it can
only ever express what those features measure -- and those features are
exactly what RL learns to stuff. It already sits at the ceiling on-policy
(median P 0.933, 34% at >=0.99, p90 1.000). Refitting it on better labels
does not give it the capacity to rank completions it cannot tell apart.

THE GATE. Two checks decide whether this model is worth using at all, both on
pairs held out BY PROMPT:

  1. held-out pairwise accuracy >= --gate-acc (default 0.75);
  2. it must BEAT the 51-feature linear classifier on the same held-out pairs.

The second matters more than the first. If the linear model ranks the judge's
preferences just as well, the neural model is adding nothing and the honest
move is to say so rather than ship a more complicated reward.

Splitting BY PROMPT is not a detail: all 6 pairs from one group share two of
four completions, so a pair-level split puts near-identical pairs on both
sides and reports an accuracy that is mostly memorisation.

    python scripts/train_reward_model.py --pairs work/rm_pairs.jsonl \
        --out outputs/reward-model
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_pairs(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def split_by_group(pairs, val_frac, seed):
    groups = sorted({p["group_id"] for p in pairs})
    random.Random(seed).shuffle(groups)
    n_val = max(1, int(len(groups) * val_frac))
    val_ids = set(groups[:n_val])
    tr = [p for p in pairs if p["group_id"] not in val_ids]
    va = [p for p in pairs if p["group_id"] in val_ids]
    return tr, va


def linear_baseline(val):
    """How well the existing 51-feature classifier ranks the judge's pairs."""
    from persona_similarity import features
    from persona_classifier import apply_std
    m = json.loads(Path("outputs/persona_clf.json").read_text())

    def score(t):
        x = apply_std(features(t), m["mu"], m["sd"])
        return m["b"] + sum(w * xi for w, xi in zip(m["w"], x))

    ok = sum(1 for p in val if score(p["chosen"]) > score(p["rejected"]))
    return ok / max(len(val), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--pairs", default="work/rm_pairs.jsonl")
    ap.add_argument("--out", default="outputs/reward-model")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--val-frac", type=float, default=0.15)
    # r=16 over all seven projections gave 8.8M trainable params on ~2k pairs
    # and overfitted hard: 0.943 train against 0.631 held-out. Attention-only
    # at r=8 is ~4x smaller, which is the right size for a dataset this small.
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    # An absolute floor is the wrong gate when the labels are themselves noisy:
    # the reward model cannot rank better than the judge agrees with itself.
    # If work/judge_consistency.json exists, the floor becomes a fraction of
    # that measured ceiling instead.
    ap.add_argument("--gate-acc", type=float, default=0.75)
    ap.add_argument("--gate-frac-of-ceiling", type=float, default=0.90,
                    help="required share of the judge's own self-consistency")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = load_pairs(args.pairs)
    tr, va = split_by_group(pairs, args.val_frac, args.seed)
    n_groups = len({p["group_id"] for p in pairs})
    print(f"{len(pairs)} pairs over {n_groups} groups")
    print(f"  train {len(tr)} pairs / {len({p['group_id'] for p in tr})} groups")
    print(f"  val   {len(va)} pairs / {len({p['group_id'] for p in va})} groups"
          f"   (split by prompt, not by pair)")

    base_acc = linear_baseline(va)
    print(f"\nbaseline: 51-feature linear classifier on the same val pairs: "
          f"{base_acc:.3f}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=1, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
        bias="none", task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    def encode(prompt, completion):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": completion}],
            tokenize=False)

    def score_batch(texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len).to(dev)
        return model(**enc).logits.squeeze(-1)

    @torch.no_grad()
    def evaluate(rows):
        model.eval()
        ok = 0
        for i in range(0, len(rows), args.batch_size):
            b = rows[i:i + args.batch_size]
            rc = score_batch([encode(p["prompt"], p["chosen"]) for p in b])
            rr = score_batch([encode(p["prompt"], p["rejected"]) for p in b])
            ok += (rc > rr).sum().item()
        model.train()
        return ok / max(len(rows), 1)

    steps = max(1, int(len(tr) / args.batch_size * args.epochs))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    print(f"\ntraining {steps} steps")
    order = list(range(len(tr)))
    t0, step, best = time.time(), 0, 0.0
    model.train()
    while step < steps:
        random.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            if step >= steps:
                break
            b = [tr[j] for j in order[i:i + args.batch_size]]
            rc = score_batch([encode(p["prompt"], p["chosen"]) for p in b])
            rr = score_batch([encode(p["prompt"], p["rejected"]) for p in b])
            loss = -F.logsigmoid(rc - rr).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0:
                print(f"  step {step}/{steps}  loss {loss.item():.4f}  "
                      f"({time.time()-t0:.0f}s)", end="\r")

    acc = evaluate(va)
    tr_acc = evaluate(tr[:len(va)])

    # The judge's self-consistency is the ceiling: labels that only agree with
    # themselves X% of the time cannot support a model that ranks better than
    # X%. Gate against that when it has been measured.
    ceiling = None
    cpath = Path("work/judge_consistency.json")
    if cpath.exists():
        ceiling = json.loads(cpath.read_text())["self_consistency"]
        floor = args.gate_frac_of_ceiling * ceiling
    else:
        floor = args.gate_acc

    print(f"\n\n=== GATE ===")
    print(f"  held-out pairwise accuracy   {acc:.3f}")
    print(f"  train-subset accuracy        {tr_acc:.3f}"
          f"   (gap {tr_acc-acc:+.3f} -- large means overfitting)")
    print(f"  51-feature linear baseline   {base_acc:.3f}")
    if ceiling is not None:
        print(f"  judge self-consistency       {ceiling:.3f}  (the ceiling)")
        print(f"  required floor               {floor:.3f}  "
              f"({args.gate_frac_of_ceiling:.0%} of ceiling)")
    else:
        print(f"  required floor               {floor:.3f}  (absolute)")

    passed = acc >= floor and acc > base_acc
    if acc < floor:
        print(f"  FAIL: below the floor.")
    if acc <= base_acc:
        print(f"  FAIL: does not beat the linear baseline -- a more complicated "
              f"reward that ranks no better is not worth shipping.")
    if ceiling is not None and ceiling < 0.60:
        print(f"  NOTE: the ceiling itself is near chance. The bottleneck is "
              f"the judge, not the reward model or the amount of data.")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (Path(args.out) / "rm_metrics.json").write_text(json.dumps({
        "val_pairwise_accuracy": acc,
        "train_subset_accuracy": tr_acc,
        "linear_baseline_accuracy": base_acc,
        "judge_self_consistency": ceiling,
        "required_floor": floor,
        "passed": passed,
        "n_pairs": len(pairs),
        "n_groups": n_groups,
        "n_train_pairs": len(tr),
        "n_val_pairs": len(va),
        "base_model": args.base,
        "split": "by_group_id",
    }, indent=2))
    print(f"\n-> {args.out}")
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
