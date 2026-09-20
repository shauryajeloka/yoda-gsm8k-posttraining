#!/usr/bin/env python3
"""Print a slice of the SFT source sample compactly, for authoring rewrites."""
import json, sys
start, size = int(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for l in open("work/sft_source_sample.jsonl", encoding="utf-8")]
for r in rows[start:start + size]:
    sol = " | ".join(x.strip() for x in r["reference_solution_clean"].split("####")[0].split("\n") if x.strip())
    print(f"@@{r['source_id']}")
    print(f"Q: {r['question'].strip()}")
    print(f"S: {sol}")
    print(f"A: {r['ground_truth']}")
    print()
print(f"--- shown {start}..{min(start+size, len(rows))} of {len(rows)} ---")
