#!/usr/bin/env python3
"""Where did each RLVR arm move away from the RLAIF policy?

The KL penalty in train_rlaif.py is computed on completions sampled for the
training prompts, which in RLVR are all maths. The Checkpoint-3 reading is
that this anchors maths behaviour and leaves general chat free to drift. This
script measures that directly instead of inferring it from persona scores.

For every arm, on teacher-forced text, it computes the exact per-token

    KL( arm || RLAIF ) = sum_v p_arm(v) * (log p_arm(v) - log p_rlaif(v))

over the full vocabulary at each response position, then averages over
positions. "Adapter on" is the arm and "adapter off" is RLAIF, exactly the
policy/reference pair the trainer used. Two text sets per domain:

    own    the arm's own greedy eval answers
    rlaif  RLAIF's greedy eval answers (same text for every arm)

Domains are the 150 persona-on-maths items and the 150 general persona prompts.
This is exact KL on greedy text, not the k3 estimate on temperature-1 samples
that the training log records, so the two are not on the same scale; compare
arms and domains within this script only.

    python scripts/kl_drift.py \
        outputs/rlvr-verifier-lora:outputs/rlvr-verifier \
        outputs/rlvr-combined-lora:outputs/rlvr-combined \
        outputs/rlvr-nokl-lora:outputs/rlvr-nokl
"""

import argparse
import json
import random
import statistics as st
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
DOMAINS = {"maths": "persona_math_eval.jsonl", "general": "persona_eval.jsonl"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def chain_of(adapter):
    """init_from ancestors of `adapter`, base-most first, excluding itself."""
    out, cur = [], json.loads(Path(adapter, "training_config.json").read_text())["init_from"]
    while cur:
        out.append(cur)
        p = Path(cur, "training_config.json")
        cur = json.loads(p.read_text()).get("init_from") if p.exists() else None
    return list(reversed(out))


def encode(tok, rows):
    items = []
    for r in rows:
        g = r["generation"]
        assert g["system_prompt"] == "qwen_default" and not g.get("prompt_suffix"), g
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                    tokenize=False, add_generation_prompt=True)
        pid = tok(p, add_special_tokens=False)["input_ids"]
        rid = tok(r["response"], add_special_tokens=False)["input_ids"]
        items.append((pid, rid))
    return items


@torch.no_grad()
def seq_kls(model, tok, items, bs):
    """Mean per-token exact KL(adapter-on || adapter-off) for each item."""
    res = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        L = max(len(p) + len(r) for p, r in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        resp = torch.zeros((len(chunk), L), dtype=torch.bool)
        for j, (p, r) in enumerate(chunk):          # right-pad
            ids[j, :len(p) + len(r)] = torch.tensor(p + r)
            att[j, :len(p) + len(r)] = 1
            # position t predicts token t+1: response tokens are predicted
            # from positions len(p)-1 .. len(p)+len(r)-2
            resp[j, len(p) - 1:len(p) + len(r) - 1] = True
        ids, att, resp = ids.cuda(), att.cuda(), resp.cuda()
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
        with model.disable_adapter():
            lq = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)          # [B, L]
        del lp, lq
        for j in range(len(chunk)):
            res.append(float(kl[j][resp[j]].mean()))
    return res


def boot(d, n=4000, seed=0):
    rng = random.Random(seed)
    bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return [bs[int(.025 * n)], bs[int(.975 * n)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="adapter_dir:generations_dir")
    ap.add_argument("--ref-gen", default="outputs/rlaif")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="outputs/kl_drift.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.pad_token or tok.eos_token
    summary = {}
    for spec in args.arms:
        adapter, gen = spec.split(":")
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                     device_map="cuda")
        for a in chain_of(adapter):
            model = PeftModel.from_pretrained(model, a).merge_and_unload()
        model = PeftModel.from_pretrained(model, adapter).eval()
        rec = {}
        for dom, fn in DOMAINS.items():
            for src, d in (("own", gen), ("rlaif", args.ref_gen)):
                k = seq_kls(model, tok, encode(tok, jl(Path(d, fn))), args.batch_size)
                rec[f"{dom}/{src}"] = {"mean": st.mean(k), "ci": boot(k), "n": len(k)}
                print(f"{adapter:30s} {dom:8s} {src:6s} KL/token {st.mean(k):.5f} "
                      f"CI [{rec[f'{dom}/{src}']['ci'][0]:.5f}, {rec[f'{dom}/{src}']['ci'][1]:.5f}]",
                      flush=True)
        for src in ("own", "rlaif"):
            rec[f"general_over_maths/{src}"] = rec[f"general/{src}"]["mean"] / rec[f"maths/{src}"]["mean"]
        summary[adapter] = rec
        del model
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
