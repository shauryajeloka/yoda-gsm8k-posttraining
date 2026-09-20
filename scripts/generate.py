#!/usr/bin/env python3
"""
Generate completions for a frozen prompt set, from the base model or from an
SFT/RL checkpoint. This is the one place completions are produced, so every
stage is generated identically.

Fixed by default and recorded in the output file:
  * greedy decoding (temperature 0) -- sampling noise would otherwise be
    confounded with the difference between stages;
  * the system-prompt policy, read from the checkpoint's training_config.json
    so evaluation cannot silently disagree with training.

Usage
    # base model (the "Base Qwen" row -- run this BEFORE training)
    python scripts/generate.py --model Qwen/Qwen2.5-3B-Instruct \
        --prompts data/persona/persona_eval_prompts.jsonl \
        --out outputs/base/persona_eval.jsonl

    # after SFT
    python scripts/generate.py --adapter outputs/sft-lora \
        --prompts data/math/gsm8k_eval.jsonl \
        --out outputs/sft/gsm8k_eval.jsonl
"""

import argparse
import json
import time
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def read_prompts(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        prompt = r.get("prompt") or r.get("question")
        if prompt is None and "messages" in r:
            prompt = r["messages"][0]["content"]
        rows.append({"id": r.get("id", f"item_{len(rows):04d}"),
                     "prompt": prompt, "source": r})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", help="LoRA adapter dir from train_sft.py")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy. Keep identical across all stages.")
    ap.add_argument("--system-prompt", default=None,
                    choices=[None, "qwen_default", "none"],
                    help="defaults to the adapter's training setting")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if --out already exists")
    ap.add_argument("--prompt-suffix", default="",
                    help="appended to every user prompt. For diagnostics only "
                         "(e.g. constraining the base model's answer length); "
                         "never use for a row of the results table, since it "
                         "changes the input rather than the model.")
    args = ap.parse_args()

    # Generations are expensive and are the input to every downstream metric,
    # so never clobber them by accident. Re-scoring is free; re-generating is not.
    if Path(args.out).exists() and not args.overwrite:
        n = sum(1 for _ in open(args.out, encoding="utf-8"))
        print(f"{args.out} already exists ({n} rows). Skipping. "
              "Pass --overwrite to regenerate.")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    system_prompt = args.system_prompt
    if args.adapter:
        cfg_path = Path(args.adapter, "training_config.json")
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            trained_with = cfg.get("system_prompt", "qwen_default")
            if system_prompt is None:
                system_prompt = trained_with
            elif system_prompt != trained_with:
                raise SystemExit(
                    f"--system-prompt {system_prompt!r} disagrees with the "
                    f"checkpoint's training setting {trained_with!r}. "
                    "Evaluating under a different prompt policy than training "
                    "measures prompting, not post-training.")
    system_prompt = system_prompt or "qwen_default"

    # Trainer's per-epoch checkpoint dirs hold the adapter but no tokenizer,
    # so fall back to the base model's.
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = read_prompts(args.prompts)
    print(f"{len(rows)} prompts from {args.prompts}")
    print(f"system_prompt={system_prompt}  temperature={args.temperature}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            texts = []
            for r in batch:
                msgs = [{"role": "user",
                         "content": r["prompt"] + args.prompt_suffix}]
                if system_prompt == "none":
                    msgs = [{"role": "system", "content": ""}] + msgs
                texts.append(tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    pad_token_id=tok.pad_token_id)
            for r, seq in zip(batch, out):
                text = tok.decode(seq[enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
                fh.write(json.dumps({
                    "id": r["id"], "prompt": r["prompt"], "response": text,
                    "ground_truth": r["source"].get("ground_truth"),
                    "metadata": r["source"].get("metadata", {}),
                    "generation": {
                        "model": args.model, "adapter": args.adapter,
                        "temperature": args.temperature,
                        "system_prompt": system_prompt,
                        "max_new_tokens": args.max_new_tokens,
                        "prompt_suffix": args.prompt_suffix},
                }, ensure_ascii=False) + "\n")
            print(f"  {min(i+args.batch_size, len(rows))}/{len(rows)}", end="\r")
    print(f"\nwrote {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
