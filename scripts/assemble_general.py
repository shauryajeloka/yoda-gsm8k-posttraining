#!/usr/bin/env python3
"""
Assemble the general (non-math) Yoda persona SFT set from hand-authored
conversations, and apply persona quality control.

Input format (work/general_yoda.txt), one block per example:

    @@<category>
    Q: <user prompt>
    A: <assistant response, one paragraph>

Checks
------
1. NO STAR WARS      no "Yoda", "Jedi", "the Force", etc. The voice is Yoda's;
                     the content is not about Star Wars.
2. INTERJECTIONS     at most one "hmm"/"mmm"/"yes"-style interjection per
                     response (style guide C4).
3. INVERSION         at least one inversion cue, the same floor used for the
                     math set.
4. DUPLICATES        no repeated prompts, and no repeated response openings
                     beyond a small budget, so the set is not templated.
5. LENGTH            responses must actually answer, not gesture at answering.

Output
------
data/persona/general_yoda_train.jsonl   accepted examples, chat format
work/general_rejects.jsonl              rejects with reasons
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path("scripts")))
from assemble_math_sft import BANNED, inversion_cues, MIN_CUES  # noqa: E402

SRC = Path("work/general_yoda.txt")
OUT = Path("data/persona/general_yoda_train.jsonl")
REJECTS = Path("work/general_rejects.jsonl")

INTERJECTION = re.compile(r"\b(hmm+|mmm+|yes|yeah|ah|oh)\b[,.!]", re.IGNORECASE)
MAX_INTERJECTIONS = 1
MIN_WORDS = 20
MAX_OPENING_SHARE = 0.05      # no opening bigram may start >5% of responses


def parse(path):
    blocks, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@@"):
            if cur:
                blocks.append(cur)
            cur = {"category": line[2:].strip(), "q": "", "a": ""}
        elif cur is None:
            continue
        elif line.startswith("Q:"):
            cur["q"] = line[2:].strip()
        elif line.startswith("A:"):
            cur["a"] = line[2:].strip()
        elif line.strip() and cur["a"]:
            cur["a"] += " " + line.strip()
    if cur:
        blocks.append(cur)
    return [b for b in blocks if b["q"] and b["a"]]


def main():
    blocks = parse(SRC)
    accepted, rejected = [], []
    seen_prompts, openings = set(), Counter()

    for i, b in enumerate(blocks):
        reasons = []
        q, a = b["q"], b["a"]

        key = re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()
        if key in seen_prompts:
            reasons.append("duplicate prompt")
        seen_prompts.add(key)

        if BANNED.search(a) or BANNED.search(q):
            reasons.append("banned star wars vocabulary")
        n_int = len(INTERJECTION.findall(a))
        if n_int > MAX_INTERJECTIONS:
            reasons.append(f"too many interjections ({n_int})")
        cues = inversion_cues(a)
        if cues < MIN_CUES:
            reasons.append(f"weak inversion signal ({cues} cues)")
        if len(a.split()) < MIN_WORDS:
            reasons.append(f"response too short ({len(a.split())} words)")

        if reasons:
            rejected.append({"index": i, "prompt": q, "reasons": reasons})
            continue

        openings[" ".join(a.split()[:2]).lower()] += 1
        accepted.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "metadata": {
                "type": "persona_general",
                "persona": "yoda",
                "source": "authored",
                "source_split": "none",
                "category": b["category"],
                "inversion_cues": cues,
            },
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in accepted:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(REJECTS, "w", encoding="utf-8") as fh:
        for r in rejected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"parsed    {len(blocks)}")
    print(f"accepted  {len(accepted)}  -> {OUT}")
    print(f"rejected  {len(rejected)}")
    for r in rejected[:20]:
        print("  REJECT", r["prompt"][:60], r["reasons"])

    cats = Counter(r["metadata"]["category"] for r in accepted)
    print("categories:", dict(cats))
    if accepted:
        worst, n = openings.most_common(1)[0]
        share = n / len(accepted)
        flag = "OK" if share <= MAX_OPENING_SHARE else "TOO TEMPLATED"
        print(f"most common opening: {worst!r} {n}/{len(accepted)} "
              f"({share:.1%}) -- {flag}")


if __name__ == "__main__":
    main()
