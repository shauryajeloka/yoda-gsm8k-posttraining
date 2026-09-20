#!/usr/bin/env python3
"""
Validate the SFT data against the real Qwen2.5 chat template, before any GPU
is rented. Needs only transformers + jinja2 -- no torch, no GPU.

Checks
  1. the template is prefix-consistent, so train_sft.py's label masking is exact
  2. no example is truncated at --max-len
  3. reports the loss-bearing token share and the system prompt actually used

Usage
    python scripts/check_formatting.py [--max-len 640] [--system-prompt none]
"""

import argparse
import json
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/combined/sft_train.jsonl")
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--system-prompt", default="qwen_default",
                    choices=["qwen_default", "none"])
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    prefix_ok = 0
    tot, resp, trunc = [], [], 0
    for r in rows:
        msgs = list(r["messages"])
        if args.system_prompt == "none":
            msgs = [{"role": "system", "content": ""}] + msgs
        p = tok.apply_chat_template(msgs[:-1], tokenize=True,
                                    add_generation_prompt=True)
        f = tok.apply_chat_template(msgs, tokenize=True)
        prefix_ok += list(f[:len(p)]) == list(p)
        tot.append(len(f))
        resp.append(len(f) - len(p))
        trunc += len(f) > args.max_len

    print(f"model            {args.model}")
    print(f"system prompt    {args.system_prompt}")
    sys_msg = tok.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False).split("<|im_start|>user")[0]
    print(f"  -> rendered as: {sys_msg.strip()[:110]!r}")
    print(f"\nexamples                    {len(rows)}")
    print(f"prefix-consistent           {prefix_ok}/{len(rows)}  "
          f"{'(label masking is exact)' if prefix_ok == len(rows) else '*** MASKING WOULD BE WRONG ***'}")
    print(f"truncated at max_len={args.max_len}    {trunc}  (must be 0)")
    print(f"total tokens    mean {st.mean(tot):6.1f}  max {max(tot)}")
    print(f"loss tokens     mean {st.mean(resp):6.1f}  max {max(resp)}  "
          f"({sum(resp)/sum(tot):.0%} of tokens carry loss)")
    print(f"\ntokens per epoch            {sum(tot):,}")
    assert prefix_ok == len(rows) and trunc == 0, "formatting check FAILED"
    print("\nformatting check PASSED")


if __name__ == "__main__":
    main()
