#!/usr/bin/env python3
"""Difficulty pre-pass for RLVR: keep problems the CURRENT policy solves sometimes.

GRPO learns only from groups whose rewards vary. With a binary verifier, a
problem the policy always solves gives rewards [1,1,1,1,1,1]; one it never
solves gives [0,0,0,0,0,0]. Both have zero variance, zero advantage, and zero
gradient -- the same failure as a reward pinned at its ceiling. So "hard" is
defined relative to the policy, not the dataset: sample each candidate G times
at the training temperature and keep those with between 1 and G-1 correct.

The pool is GSM8K *train* problems never used anywhere in this project: not in
the 1,500 sampled for self-distillation (whose ids are raw line indices), not
in any SFT dataset, and -- trivially, since it is the test split -- not in the
frozen eval sets. Leakage checks refuse rather than warn.

    python scripts/rlvr_prepass.py --adapter outputs/rlaif-lora --n 800 --g 6
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("data/verifier")))
from gsm8k_verifier import extract_ground_truth, verify  # noqa: E402


def norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def fresh_pool():
    raw = [json.loads(l) for l in open("data/raw/gsm8k_train.jsonl", encoding="utf-8")]
    used_idx = {int(json.loads(l)["id"].split("_")[-1])
                for l in open("outputs/base-train-cot/gsm8k_train.jsonl", encoding="utf-8")}
    used_q = set()
    for f in Path("data/combined").glob("*.jsonl"):
        for l in open(f, encoding="utf-8"):
            m = json.loads(l).get("messages")
            if m:
                used_q.add(norm(m[0]["content"]))
    held = {norm(json.loads(l)["question"])
            for l in open("data/math/gsm8k_eval.jsonl", encoding="utf-8")}
    pool = []
    for i, r in enumerate(raw):
        q = r["question"].strip()
        if i in used_idx or norm(q) in used_q:
            continue
        if norm(q) in held:
            raise SystemExit(f"train problem {i} appears in the frozen eval set")
        pool.append({"id": f"gsm8k_train_{i:05d}", "prompt": q,
                     "ground_truth": extract_ground_truth(r["answer"])})
    return pool


def load_policy(model_id, adapter):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    # Same chain rebuild as train_rlaif.py / generate.py: base-most first.
    chain, cur = [], adapter
    while cur:
        chain.append(cur)
        cfg = Path(cur, "training_config.json")
        cur = json.loads(cfg.read_text()).get("init_from") if cfg.exists() else None
    for a in reversed(chain):
        print(f"  merging adapter: {a}")
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="outputs/rlaif-lora")
    ap.add_argument("--n", type=int, default=800, help="candidate problems")
    ap.add_argument("--g", type=int, default=6, help="samples per problem = GRPO group size")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--prompts-per-batch", type=int, default=32)
    ap.add_argument("--out", default="work/rlvr_prompts.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    pool = fresh_pool()
    random.Random(args.seed).shuffle(pool)
    cands = pool[:args.n]
    print(f"{len(pool)} fresh train problems; sampling {len(cands)} x {args.g}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_policy(args.model, args.adapter)
    torch.manual_seed(args.seed)

    t0 = time.time()
    hist = [0] * (args.g + 1)
    kept, n_trunc, n_samp = [], 0, 0
    for b in range(0, len(cands), args.prompts_per_batch):
        chunk = cands[b:b + args.prompts_per_batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": c["prompt"]}],
                                         tokenize=False, add_generation_prompt=True)
                 for c in chunk for _ in range(args.g)]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        n_trunc += int(((gen != tok.pad_token_id).sum(1) >= args.max_new_tokens).sum())
        n_samp += gen.shape[0]
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for i, c in enumerate(chunk):
            k = sum(verify(d, c["ground_truth"], question=c["prompt"])["correct"]
                    for d in dec[i * args.g:(i + 1) * args.g])
            hist[k] += 1
            if 0 < k < args.g:
                kept.append({**c, "n_correct": k, "g": args.g})
        print(f"  {min(b + args.prompts_per_batch, len(cands))}/{len(cands)} "
              f"kept={len(kept)} ({time.time() - t0:.0f}s)", end="\r")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"candidates": len(cands), "g": args.g, "kept": len(kept),
             "hist_n_correct": hist, "sampled_accuracy":
                 sum(i * h for i, h in enumerate(hist)) / (len(cands) * args.g),
             "truncated_frac": n_trunc / max(n_samp, 1),
             "adapter": args.adapter, "temperature": args.temperature,
             "max_new_tokens": args.max_new_tokens, "seed": args.seed}
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nkept {len(kept)}/{len(cands)} mixed-outcome problems -> {args.out}")
    print(f"n_correct histogram (0..{args.g}): {hist}")
    print(f"sampled accuracy {stats['sampled_accuracy']:.3f}, "
          f"truncated {stats['truncated_frac']:.3%}")


if __name__ == "__main__":
    main()
