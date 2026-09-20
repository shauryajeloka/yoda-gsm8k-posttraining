#!/usr/bin/env python3
"""
Merge the two SFT components into the training file, with a seeded shuffle,
and re-run the leakage checks over the finished artifacts.

Output
------
data/combined/sft_train.jsonl
data/FREEZE.json           sha256 of every frozen/final artifact
"""

import hashlib
import json
import random
import re
from pathlib import Path

SEED = 1337
MATH = Path("data/math/gsm8k_yoda_sft_train.jsonl")
GENERAL = Path("data/persona/general_yoda_train.jsonl")
OUT = Path("data/combined/sft_train.jsonl")

FROZEN = [
    "data/math/gsm8k_eval.jsonl",
    "data/math/gsm8k_persona_math_eval.jsonl",
    "data/persona/persona_eval_prompts.jsonl",
]
FINAL = [
    "data/math/gsm8k_yoda_sft_train.jsonl",
    "data/persona/general_yoda_train.jsonl",
    "data/combined/sft_train.jsonl",
    "data/persona/yoda_style_guide.md",
    "data/judges/yoda_persona_judge.md",
    "data/verifier/gsm8k_verifier.py",
]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def norm(t):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split())


def main():
    rows = load(MATH) + load(GENERAL)
    random.Random(SEED).shuffle(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- leakage checks over the finished artifacts --------------------
    train_prompts = {norm(r["messages"][0]["content"]) for r in rows}
    train_ids = {r["metadata"].get("source_id") for r in rows}

    failures = []
    for p in ("data/math/gsm8k_eval.jsonl",
              "data/math/gsm8k_persona_math_eval.jsonl"):
        for e in load(p):
            if norm(e["question"]) in train_prompts:
                failures.append(f"{p}: question text in SFT train: {e['id']}")
            if e["metadata"]["source_id"] in train_ids:
                failures.append(f"{p}: source_id in SFT train: {e['id']}")
    for e in load("data/persona/persona_eval_prompts.jsonl"):
        if norm(e["prompt"]) in train_prompts:
            failures.append(f"persona eval prompt in SFT train: {e['id']}")

    splits = {}
    for r in rows:
        k = (r["metadata"]["type"], r["metadata"].get("source_split"))
        splits[k] = splits.get(k, 0) + 1

    print(f"written {len(rows)} -> {OUT}  (seed={SEED})")
    print("type / source_split counts:")
    for k, v in sorted(splits.items()):
        print(f"   {k[0]:16s} split={k[1]:6s} {v}")
    print(f"leakage failures: {len(failures)}")
    for f in failures[:20]:
        print("  ", f)
    assert not failures, "LEAKAGE DETECTED -- do not train on this"

    manifest = {
        "seed": SEED,
        "frozen_evaluation_sets": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in FROZEN},
        "final_artifacts": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in FINAL},
        "raw_source": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in ("data/raw/gsm8k_train.jsonl", "data/raw/gsm8k_test.jsonl")},
        "note": ("Frozen sets must not change once training begins. If a hash "
                 "here does not match the file, the comparison across Base / "
                 "SFT / RLAIF / final stages is no longer valid."),
    }
    Path("data/FREEZE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote data/FREEZE.json")


if __name__ == "__main__":
    main()
