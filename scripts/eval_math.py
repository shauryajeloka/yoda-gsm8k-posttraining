#!/usr/bin/env python3
"""
STEM metric: GSM8K accuracy over generated completions.

    accuracy = n_correct / n_evaluated

Reports a Wilson confidence interval, because with n=500 a 2-point difference
between stages is inside the noise and should not be reported as a change.
Also reports which extraction rule fired, so a score that depends on the
verifier's permissive fallbacks is visible rather than hidden.

Usage
    python scripts/eval_math.py outputs/base/gsm8k_eval.jsonl \
                                outputs/sft/gsm8k_eval.jsonl
"""

import json
import math
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path("data/verifier")))
from gsm8k_verifier import verify  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def evaluate(path, strict=False):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    correct, methods, wrong = 0, Counter(), []
    detail = []
    for r in rows:
        gt = r.get("ground_truth") or r.get("metadata", {}).get("ground_truth")
        # The question is what lets the fallback tell a computed answer from a
        # quantity the problem already gave; always pass it when present.
        res = verify(r["response"], gt, strict=strict,
                     question=r.get("prompt") or r.get("question"))
        methods[res["extraction_method"]] += 1
        correct += res["correct"]
        detail.append({**res, "id": r["id"]})
        if not res["correct"] and len(wrong) < 5:
            wrong.append((r["id"], res["predicted_answer"], res["ground_truth"]))
    lo, hi = wilson(correct, len(rows))
    return {"path": path, "n": len(rows), "correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "ci95": [lo, hi], "methods": dict(methods),
            "examples_wrong": wrong, "detail": detail,
            "correct_ids": {d["id"] for d in detail if d["correct"]},
            "ids": [d["id"] for d in detail]}


def mcnemar(a_correct, b_correct):
    """
    Exact McNemar test on the PAIRED per-item outcomes.

    Every arm is scored on the same frozen problems, so the comparison is
    paired and the right question is "of the items the two arms disagree on,
    is the split lopsided?". Comparing two independent-sample Wilson intervals
    ignores the pairing: it is too conservative for detecting a real
    difference, and -- more dangerously -- overlapping intervals do NOT license
    the conclusion that two arms are equivalent. Use this for any claim about
    whether a difference is real.
    """
    b01 = len(a_correct - b_correct)      # a right, b wrong
    b10 = len(b_correct - a_correct)      # b right, a wrong
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    tail = sum(comb(n, k) for k in range(0, min(b01, b10) + 1))
    return b01, b10, min(1.0, 2 * tail / 2 ** n)


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    results = []
    print(f"{'file':44s} {'acc':>7s} {'95% CI':>16s} {'n':>5s}")
    for p in paths:
        r = evaluate(p)
        results.append(r)
        print(f"{Path(p).parent.name + '/' + Path(p).name:44s} "
              f"{r['accuracy']:7.1%} "
              f"[{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}] {r['n']:5d}")
    for r in results:
        fallback = sum(v for k, v in r["methods"].items() if k.startswith("fallback"))
        if fallback:
            print(f"\n{Path(r['path']).name}: {fallback}/{r['n']} answers came from "
                  "a fallback extraction rule; re-run with strict=True to see "
                  "the score without them.")
        print(f"{Path(r['path']).name} extraction: {r['methods']}")
    if len(results) >= 2:
        base = results[0]
        print(f"\npaired comparisons against {Path(base['path']).parent.name} "
              "(exact McNemar on the same items):")
        for r in results[1:]:
            shared = set(base["ids"]) & set(r["ids"])
            if len(shared) != len(base["ids"]):
                print(f"  WARNING: only {len(shared)} shared ids with "
                      f"{Path(r['path']).parent.name}; comparison is partial.")
            a = base["correct_ids"] & shared
            b = r["correct_ids"] & shared
            b01, b10, p = mcnemar(a, b)
            delta = (len(b) - len(a)) / len(shared) if shared else 0.0
            verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"  {Path(r['path']).parent.name:22s} {delta:+6.1%}  "
                  f"only-first={b01:3d} only-second={b10:3d}  "
                  f"p={p:.3g}  {verdict}")


if __name__ == "__main__":
    main()
