#!/usr/bin/env python3
"""
Deterministic split construction. Every random choice in this project happens
here, with a fixed seed, so the whole data layer is reproducible from the raw
GSM8K files.

Outputs
-------
work/sft_source_sample.jsonl       1500 GSM8K TRAIN items to be Yoda-rewritten
data/math/gsm8k_eval.jsonl         500 frozen GSM8K TEST items (STEM eval)
data/math/gsm8k_persona_math_eval.jsonl
                                   150 frozen TEST items (math + persona slice),
                                   a subset of gsm8k_eval so a single generation
                                   pass serves both metrics
"""

import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path("data/verifier")))
from gsm8k_verifier import extract_ground_truth  # noqa: E402

SEED = 1337
N_SFT = 1500
N_EVAL = 500
N_PERSONA_MATH = 150

ROOT = Path(".")
RAW = ROOT / "data" / "raw"


def load(split):
    rows = []
    with open(RAW / f"gsm8k_{split}.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            rows.append({
                "source_id": f"gsm8k_{split}_{i:05d}",
                "source_index": i,
                "question": r["question"],
                "reference_solution": r["answer"],
                "ground_truth": extract_ground_truth(r["answer"]),
            })
    return rows


def strip_calc(text):
    """Reference solution with the <<...>> calculator annotations removed."""
    return re.sub(r"<<[^<>]*>>", "", text)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    train, test = load("train"), load("test")
    assert len(train) == 7473 and len(test) == 1319

    # No question text may appear in both splits.
    overlap = {r["question"].strip() for r in train} & {r["question"].strip() for r in test}
    assert not overlap, f"train/test question overlap: {len(overlap)}"

    # ---- SFT source sample (TRAIN only) --------------------------------
    rng = random.Random(SEED)
    sft_src = rng.sample(train, N_SFT)
    sft_src.sort(key=lambda r: r["source_index"])
    for r in sft_src:
        r["reference_solution_clean"] = strip_calc(r["reference_solution"])
    write_jsonl("work/sft_source_sample.jsonl", sft_src)

    # ---- Frozen STEM eval (TEST only) ----------------------------------
    rng_eval = random.Random(SEED)
    eval_rows = rng_eval.sample(test, N_EVAL)
    eval_out = [{
        "id": r["source_id"],
        "question": r["question"],
        "ground_truth": r["ground_truth"],
        "reference_solution": strip_calc(r["reference_solution"]).strip(),
        "metadata": {
            "type": "stem_eval",
            "source": "gsm8k",
            "source_split": "test",
            "source_id": r["source_id"],
            "source_index": r["source_index"],
            "split": "test",
            "frozen": True,
            "seed": SEED,
        },
    } for r in eval_rows]
    write_jsonl("data/math/gsm8k_eval.jsonl", eval_out)

    # ---- Persona x math slice (subset of the STEM eval) ----------------
    pm = eval_out[:N_PERSONA_MATH]
    pm_out = [{
        **r,
        "metadata": {**r["metadata"],
                     "type": "persona_math_eval",
                     "subset_of": "data/math/gsm8k_eval.jsonl",
                     "score_with": ["gsm8k_verifier", "yoda_persona_judge"]},
    } for r in pm]
    write_jsonl("data/math/gsm8k_persona_math_eval.jsonl", pm_out)

    # ---- Leakage assertion ---------------------------------------------
    sft_ids = {r["source_id"] for r in sft_src}
    eval_ids = {r["id"] for r in eval_out}
    assert not (sft_ids & eval_ids)
    sft_q = {r["question"].strip() for r in sft_src}
    eval_q = {r["question"].strip() for r in eval_out}
    assert not (sft_q & eval_q), "SFT/eval question overlap"

    print(f"seed={SEED}")
    print(f"work/sft_source_sample.jsonl          {len(sft_src)}  (gsm8k train)")
    print(f"data/math/gsm8k_eval.jsonl            {len(eval_out)}  (gsm8k test, FROZEN)")
    print(f"data/math/gsm8k_persona_math_eval.jsonl {len(pm_out)}  (subset of above, FROZEN)")
    print("leakage checks: PASS (no shared ids, no shared question text)")
    for p in ["data/math/gsm8k_eval.jsonl", "data/math/gsm8k_persona_math_eval.jsonl"]:
        print(f"sha256  {sha(p)}  {p}")


if __name__ == "__main__":
    main()
