#!/usr/bin/env python3
"""
Stage 1 (Week 1): supervised fine-tuning of Qwen2.5-3B-Instruct on the
Yoda persona + GSM8K dataset.

Design decisions, and why
-------------------------
* **Plain `transformers.Trainer`, not TRL's SFTTrainer.** TRL's API changes
  between minor versions; a hand-rolled collator is version-stable and, more
  importantly, makes the label masking auditable. `--inspect` prints exactly
  which tokens carry loss.

* **Completion-only loss.** Only the assistant turn is trained on. The prompt
  is masked to -100. Training on the user turn teaches the model to generate
  GSM8K questions, which is not the goal.

* **LoRA by default.** 3B full fine-tuning needs ~50-60 GB just for AdamW
  states; LoRA fits a 24 GB card, trains this dataset in minutes, and leaves
  the base weights untouched so the Base row of the results table and the
  later RL stages can share one download. `--full-finetune` is there if you
  want to compare.

* **One system-prompt policy, recorded in the output.** Qwen2.5's chat
  template injects a default system message when none is given. Whatever you
  choose here must also be used at evaluation time, or you are measuring
  prompting rather than post-training. The choice is written into
  `training_config.json` and `generate.py` refuses to disagree with it.

* **A validation split for checkpoint selection only.** Validation
  cross-entropy tells you when the model stops improving *on this
  distribution*. It is NOT the persona metric -- see TRAINING.md.

Usage
    python scripts/train_sft.py --output-dir outputs/sft-lora
    python scripts/train_sft.py --inspect        # show masking, then exit
"""

import argparse
import json
import os
import random
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATA = "data/combined/sft_train.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=DATA)
    p.add_argument("--output-dir", default="outputs/sft-lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=None,
                   help="default 1e-4 for LoRA, 1e-5 for full fine-tuning")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-len", type=int, default=640)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--system-prompt", default="qwen_default",
                   choices=["qwen_default", "none"],
                   help="MUST match what generate.py uses at eval time")
    p.add_argument("--load-best", action="store_true",
                   help="load the best-by-val-loss checkpoint at the end. Off by "
                        "default so the top-level save is unambiguously the LAST "
                        "epoch; the sweep evaluates every epoch anyway.")
    p.add_argument("--inspect", action="store_true",
                   help="print one fully-masked example and exit (no GPU needed)")
    return p.parse_args()


def build_messages(row, system_prompt):
    """Apply the system-prompt policy. 'qwen_default' lets the template inject
    Qwen's own default; 'none' suppresses it with an empty system turn."""
    msgs = list(row["messages"])
    if system_prompt == "none":
        msgs = [{"role": "system", "content": ""}] + msgs
    return msgs


def encode(row, tok, args):
    """Return input_ids and labels, with everything before the assistant turn
    masked out. Relies on the chat template being prefix-consistent, which
    check_formatting.py verifies for this exact model."""
    msgs = build_messages(row, args.system_prompt)
    prompt_ids = tok.apply_chat_template(
        msgs[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(msgs, tokenize=True)
    if list(full_ids[:len(prompt_ids)]) != list(prompt_ids):
        raise RuntimeError(
            "Chat template is not prefix-consistent for this model; the label "
            "masking below would be wrong. Run scripts/check_formatting.py.")
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    return {"input_ids": list(full_ids)[:args.max_len],
            "labels": labels[:args.max_len],
            "n_truncated": max(0, len(full_ids) - args.max_len)}


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    encoded = [encode(r, tok, args) for r in rows]
    truncated = sum(e["n_truncated"] > 0 for e in encoded)
    lengths = [len(e["input_ids"]) for e in encoded]
    print(f"{len(rows)} examples | max len {max(lengths)} tokens | "
          f"truncated at --max-len {args.max_len}: {truncated}")
    if truncated:
        raise SystemExit("Examples are being truncated; raise --max-len.")

    if args.inspect:
        e = encoded[0]
        trained = [t for t, l in zip(e["input_ids"], e["labels"]) if l != -100]
        masked = [t for t, l in zip(e["input_ids"], e["labels"]) if l == -100]
        print("\n--- MASKED (no loss) ---\n" + tok.decode(masked))
        print("\n--- TRAINED ON (loss) ---\n" + tok.decode(trained))
        print(f"\nloss on {len(trained)}/{len(e['input_ids'])} tokens")
        return

    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, Trainer, TrainingArguments,
                              set_seed)

    set_seed(args.seed)
    rng = random.Random(args.seed)
    idx = list(range(len(encoded)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx, train_idx = set(idx[:n_val]), idx[n_val:]
    print(f"train {len(train_idx)} | val {len(val_idx)} (checkpoint selection only)")

    class DS(Dataset):
        def __init__(self, ids): self.ids = list(ids)
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            e = encoded[self.ids[i]]
            return {"input_ids": e["input_ids"], "labels": e["labels"]}

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            k = n - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad_id] * k)
            out["labels"].append(b["labels"] + [-100] * k)
            out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * k)
        return {k: torch.tensor(v) for k, v in out.items()}

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=None)
    model.config.use_cache = False

    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()

    lr = args.lr if args.lr else (1e-5 if args.full_finetune else 1e-4)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,          # keep every epoch for the sweep
        load_best_model_at_end=args.load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=args.full_finetune,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=DS(train_idx),
                      eval_dataset=DS(val_idx), data_collator=collate)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    cfg = {"model": args.model, "data": args.data, "epochs": args.epochs,
           "lr": lr, "lora": not args.full_finetune, "lora_r": args.lora_r,
           "seed": args.seed, "system_prompt": args.system_prompt,
           "max_len": args.max_len,
           "note": "generate.py must use the same --system-prompt value."}
    Path(args.output_dir, "training_config.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    # Each epoch checkpoint must carry the same system-prompt policy, or
    # generate.py's guard cannot verify it when that epoch is evaluated.
    ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                   key=lambda c: int(c.name.split("-")[1]))
    manifest = []
    for i, c in enumerate(ckpts, start=1):
        (c / "training_config.json").write_text(
            json.dumps({**cfg, "epoch": i, "checkpoint": str(c)}, indent=2) + "\n",
            encoding="utf-8")
        manifest.append({"epoch": i, "path": str(c)})
    manifest.append({"epoch": args.epochs, "path": args.output_dir,
                     "note": "final save"})
    Path(args.output_dir, "epoch_checkpoints.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nepoch checkpoints (each independently evaluable):")
    for m in manifest:
        print(f"  epoch {m['epoch']}: {m['path']}")

    hist = [h for h in trainer.state.log_history if "eval_loss" in h]
    Path(args.output_dir, "eval_loss_history.json").write_text(
        json.dumps(hist, indent=2) + "\n", encoding="utf-8")
    print("\nvalidation loss by epoch (checkpoint selection only, NOT a persona metric):")
    for h in hist:
        print(f"  epoch {h.get('epoch'):.2f}  eval_loss {h['eval_loss']:.4f}")
    print(f"\nsaved to {args.output_dir}")


if __name__ == "__main__":
    main()
