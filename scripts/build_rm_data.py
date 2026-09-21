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

"""Sample ON-POLICY completions to train the reward model on.

The reward model is queried during GRPO on completions drawn from the SFT
policy at temperature 1.0, so that is exactly the distribution it must be
able to rank. Training it on "Yoda prose vs base-model prose" instead would
be easy and useless: the existing 51-feature linear classifier already scores
0.985 held-out on that task, and it STILL saturates on-policy -- median
P(persona) 0.933, 34% of completions pinned at >=0.99, p90 = 1.000. A third
of every GRPO group is invisible to it.

The discrimination we actually need -- good Yoda against better Yoda -- only
exists in samples from the policy itself. That is what this produces.

    python scripts/build_rm_data.py --adapter outputs/sft-yodadistill \
        --out work/rm_samples.jsonl --k 4
"""

import argparse
import json
import random
import time
from pathlib import Path


def load_prompts(path, frozen_eval):
    """Training prompts, with the frozen eval set excluded by assertion.

    Sampling reward-model data from the prompts we later evaluate on would
    contaminate the Week 2 comparison just as surely as training on them, so
    this refuses rather than warns.
    """
    import re

    def norm(t):
        return re.sub(r"\W+", " ", t.lower()).strip()

    prompts = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        prompts.append(r.get("prompt") or r["messages"][0]["content"])

    held = set()
    for line in open(frozen_eval, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            held.add(norm(r.get("prompt") or r.get("question")))
    clash = [p for p in prompts if norm(p) in held]
    if clash:
        raise SystemExit(
            f"{len(clash)} reward-model prompts also appear in the frozen eval "
            f"set ({frozen_eval}). First: {clash[0]!r}")
    return prompts


def build_stack(model, adapter):
    """Rebuild the adapter stack recorded in training_config.json.

    Same discipline as generate.py: an adapter trained on top of a merged
    parent must be loaded onto that parent, or the model is silently neither
    thing.
    """
    from peft import PeftModel

    chain, cur, seen = [], adapter, set()
    while cur:
        chain.append(cur)
        if cur in seen:
            raise SystemExit(f"cycle in adapter init_from chain at {cur}")
        seen.add(cur)
        cfg = Path(cur, "training_config.json")
        cur = json.loads(cfg.read_text()).get("init_from") if cfg.exists() else None
    for parent in reversed(chain[1:]):
        print(f"  stacking parent adapter: {parent}")
        model = PeftModel.from_pretrained(model, parent).merge_and_unload()
    return PeftModel.from_pretrained(model, chain[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="outputs/sft-yodadistill")
    ap.add_argument("--prompts", default="data/persona/general_yoda_train.jsonl")
    ap.add_argument("--frozen-eval", default="data/persona/persona_eval_prompts.jsonl")
    ap.add_argument("--out", default="work/rm_samples.jsonl")
    ap.add_argument("--k", type=int, default=4,
                    help="completions per prompt. K=4 yields C(4,2)=6 pairs "
                         "from a single judge call, 6x cheaper than pairwise.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="must match the GRPO sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = load_prompts(args.prompts, args.frozen_eval)
    print(f"{len(prompts)} prompts (disjoint from the frozen eval set), "
          f"K={args.k} -> {len(prompts) * args.k} completions")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model = build_stack(model, args.adapter)
    model.eval()

    # Each prompt is repeated K times in one batch so the K samples for a
    # prompt are drawn independently under the same conditions.
    jobs = [(i, p) for i, p in enumerate(prompts) for _ in range(args.k)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    t0 = time.time()
    for i in range(0, len(jobs), args.batch_size):
        batch = jobs[i:i + args.batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for _, p in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, pad_token_id=tok.pad_token_id)
        for (pid, prompt), seq in zip(batch, out):
            text = tok.decode(seq[enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            groups.setdefault(pid, {"prompt": prompt, "completions": []})
            groups[pid]["completions"].append(text)
        done = min(i + args.batch_size, len(jobs))
        print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", end="\r")

    # Degenerate groups teach nothing: if every sample is identical the judge
    # has nothing to rank and the pairs would be pure noise.
    kept = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for pid in sorted(groups):
            g = groups[pid]
            if len({c.strip() for c in g["completions"] if c.strip()}) < 2:
                continue
            fh.write(json.dumps({"group_id": pid, **g}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"\nwrote {args.out}: {kept}/{len(groups)} groups "
          f"({len(groups)-kept} dropped as degenerate) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
