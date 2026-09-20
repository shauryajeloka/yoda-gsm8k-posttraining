#!/usr/bin/env python3
"""
Quality-control statistics for the finished dataset (assignment section 15).

Reports: example counts and proportions, prompt/response token lengths,
duplicate and near-duplicate counts, verifier re-check of every math example,
persona-hygiene counts, source-split counts, and the eval-set summaries.

Token counts use the real Qwen2.5-3B-Instruct tokenizer when transformers is
installed; otherwise a whitespace word count is reported instead, and the
output says which was used.

Usage:  python scripts/dataset_stats.py [--json out.json]
"""

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path("data/verifier")))
from gsm8k_verifier import verify  # noqa: E402

SHINGLE = 5
NEAR_DUP_J = 0.8


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        return (lambda s: len(tok.encode(s))), "Qwen2.5-3B-Instruct tokenizer"
    except Exception:
        return (lambda s: len(s.split())), "whitespace words (tokenizer unavailable)"


def shingles(text, n=SHINGLE):
    toks = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def near_duplicates(texts, limit=NEAR_DUP_J):
    """Exact and near-duplicate pairs, via a shingle inverted index."""
    exact = Counter(re.sub(r"\s+", " ", t.strip().lower()) for t in texts)
    n_exact = sum(c - 1 for c in exact.values() if c > 1)

    index, sets = defaultdict(list), [shingles(t) for t in texts]
    for i, sh in enumerate(sets):
        for s in sh:
            index[s].append(i)
    candidates = set()
    for ids in index.values():
        if len(ids) > 1:
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    candidates.add((ids[a], ids[b]))
    near = 0
    for a, b in candidates:
        u = len(sets[a] | sets[b])
        if u and len(sets[a] & sets[b]) / u >= limit:
            near += 1
    return n_exact, near, len(candidates)


def describe(name, values):
    return (f"  {name:24s} n={len(values):5d}  mean={st.mean(values):7.1f}  "
            f"median={st.median(values):6.0f}  min={min(values):5d}  "
            f"max={max(values):5d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the statistics to this path")
    args = ap.parse_args()

    count, tok_name = get_tokenizer()
    out = {"token_counter": tok_name}
    print(f"token counts measured with: {tok_name}\n")

    combined = load("data/combined/sft_train.jsonl")
    math = [r for r in combined if r["metadata"]["type"] == "persona_math"]
    gen = [r for r in combined if r["metadata"]["type"] == "persona_general"]

    print("=== COMPOSITION ===")
    print(f"  total SFT examples        {len(combined)}")
    print(f"  persona_math (GSM8K)      {len(math)}  ({len(math)/len(combined):.1%})")
    print(f"  persona_general           {len(gen)}  ({len(gen)/len(combined):.1%})")
    target = "within 75-85% target" if 0.75 <= len(math) / len(combined) <= 0.85 \
        else "OUTSIDE 75-85% target"
    print(f"  math share: {target}")
    out["composition"] = {"total": len(combined), "persona_math": len(math),
                          "persona_general": len(gen),
                          "math_share": len(math) / len(combined)}

    print("\n=== SOURCE SPLITS ===")
    splits = Counter((r["metadata"]["source"], r["metadata"]["source_split"])
                     for r in combined)
    for (src, sp), n in sorted(splits.items()):
        print(f"  source={src:9s} split={sp:6s} {n}")
    print("  gsm8k test examples in SFT train: "
          f"{sum(1 for r in combined if r['metadata']['source_split'] == 'test')}"
          "  (must be 0)")
    out["source_splits"] = {f"{s}/{p}": n for (s, p), n in splits.items()}

    print("\n=== LENGTHS ===")
    for label, rows in (("math prompt", math), ("math response", math),
                        ("general prompt", gen), ("general response", gen)):
        idx = 0 if "prompt" in label else 1
        vals = [count(r["messages"][idx]["content"]) for r in rows]
        print(describe(label, vals))
        out.setdefault("lengths", {})[label] = {
            "mean": st.mean(vals), "median": st.median(vals),
            "min": min(vals), "max": max(vals)}
    all_len = [count(r["messages"][0]["content"]) + count(r["messages"][1]["content"])
               for r in combined]
    print(describe("full example", all_len))
    out["lengths"]["full example"] = {"mean": st.mean(all_len),
                                      "max": max(all_len)}

    print("\n=== DUPLICATES ===")
    for label, rows, key in (("prompts", combined, 0), ("responses", combined, 1)):
        texts = [r["messages"][key]["content"] for r in rows]
        ex, near, cand = near_duplicates(texts)
        print(f"  {label:10s} exact duplicates: {ex:4d}   "
              f"near-duplicate pairs (Jaccard>={NEAR_DUP_J}): {near:4d}   "
              f"[{cand} candidate pairs compared]")
        out.setdefault("duplicates", {})[label] = {"exact": ex, "near": near}

    print("\n=== VERIFIER RE-CHECK (every math example) ===")
    bad, methods = [], Counter()
    for r in math:
        res = verify(r["messages"][1]["content"], r["metadata"]["ground_truth"])
        methods[res["extraction_method"]] += 1
        if not res["correct"]:
            bad.append((r["metadata"]["source_id"], res))
    print(f"  verified correct          {len(math) - len(bad)}/{len(math)}")
    print(f"  rejected by verifier      {len(bad)}")
    for sid, res in bad[:10]:
        print("   ", sid, res)
    print("  extraction methods used:", dict(methods))
    rejects = Path("work/math_rejects.jsonl")
    n_rej = sum(1 for _ in open(rejects)) if rejects.exists() else 0
    print(f"  rewrites rejected during generation and regenerated: {n_rej} "
          "outstanding at final build")
    out["verifier"] = {"correct": len(math) - len(bad), "incorrect": len(bad),
                       "extraction_methods": dict(methods)}

    print("\n=== PERSONA HYGIENE ===")
    banned = re.compile(r"\b(yoda|jedi|sith|padawan|lightsab\w*|the force|"
                        r"star wars)\b", re.IGNORECASE)
    hmm = re.compile(r"\b(hmm+|mmm+)\b", re.IGNORECASE)
    n_banned = sum(1 for r in combined if banned.search(r["messages"][1]["content"]))
    n_hmm = sum(1 for r in combined if hmm.search(r["messages"][1]["content"]))
    cues = [r["metadata"].get("inversion_cues", 0) for r in combined]
    print(f"  responses with Star Wars vocabulary   {n_banned}  (must be 0)")
    print(f"  responses containing 'hmm'/'mmm'      {n_hmm}  "
          f"({n_hmm/len(combined):.1%}; style guide budgets these as rare)")
    print(f"  inversion cues per response           mean={st.mean(cues):.2f}  "
          f"min={min(cues)}  max={max(cues)}")
    openings = Counter(" ".join(r["messages"][1]["content"].split()[:2]).lower()
                       for r in combined)
    top, n = openings.most_common(1)[0]
    print(f"  most common response opening          {top!r} "
          f"{n}/{len(combined)} ({n/len(combined):.1%})")
    out["persona_hygiene"] = {"star_wars": n_banned, "hmm": n_hmm,
                              "mean_inversion_cues": st.mean(cues),
                              "top_opening": [top, n]}

    print("\n=== CATEGORIES (general persona) ===")
    print(" ", dict(Counter(r["metadata"]["category"] for r in gen)))

    print("\n=== EVALUATION SETS (frozen) ===")
    for p in ("data/math/gsm8k_eval.jsonl",
              "data/math/gsm8k_persona_math_eval.jsonl",
              "data/persona/persona_eval_prompts.jsonl"):
        rows = load(p)
        print(f"  {p:45s} {len(rows):4d} items")
        out.setdefault("eval_sets", {})[p] = len(rows)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n",
                                   encoding="utf-8")
        print(f"\nstatistics written to {args.json}")


if __name__ == "__main__":
    main()
