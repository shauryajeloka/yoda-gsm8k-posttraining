#!/usr/bin/env python3
"""
Build the FROZEN held-out persona evaluation prompt set.

Input:  work/persona_eval_prompts.txt   one "category<TAB>prompt" per line
Output: data/persona/persona_eval_prompts.jsonl

Leakage guard
-------------
These prompts are used to compare Base / SFT / RLAIF / final checkpoints, so
they must not appear in SFT training. Two checks run, and both must pass:

  * exact match against every training prompt (normalized) -- hard reject, and
  * near-duplicate match on content words (stopwords removed): Jaccard >= 0.8
    is a hard reject.

The threshold sits at 0.8 rather than lower because these prompts are short,
so a shared frame ("What makes someone a good X?") drives similarity up without
the answers overlapping at all. The five closest train/eval pairs are printed on
every build so that the remaining similarity is documented rather than hidden;
persona is scored on style, not on content recall, which makes a shared frame a
much weaker leakage risk here than it would be for the STEM set.

No reference responses are generated. The artifact is the prompt set; each
checkpoint generates its own completions, which the judge then scores.
"""

import hashlib
import json
import re
from pathlib import Path

SRC = Path("work/persona_eval_prompts.txt")
OUT = Path("data/persona/persona_eval_prompts.jsonl")
TRAIN = Path("data/persona/general_yoda_train.jsonl")
JACCARD_LIMIT = 0.8
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "to", "of", "in", "on", "for", "and", "or", "but", "with", "at",
    "by", "from", "as", "it", "its", "this", "that", "these", "those", "i",
    "you", "he", "she", "we", "they", "me", "my", "your", "what", "how",
    "why", "when", "where", "which", "who", "whom", "should", "would",
    "could", "can", "will", "shall", "if", "about", "so", "than", "then",
    "there", "here", "have", "has", "had", "s", "t",
}


def norm(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def content(text):
    """Tokens with stopwords removed, for a fairer similarity on short prompts."""
    toks = [t for t in norm(text) if t not in STOPWORDS]
    return set(toks) or set(norm(text))


def main():
    train_prompts = []
    with open(TRAIN, encoding="utf-8") as fh:
        for line in fh:
            train_prompts.append(json.loads(line)["messages"][0]["content"])
    train_exact = {" ".join(norm(p)) for p in train_prompts}
    train_tokens = [content(p) for p in train_prompts]
    closest = []

    rows, problems = [], []
    seen = set()
    for i, line in enumerate(SRC.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        category, prompt = line.split("\t", 1)
        key = " ".join(norm(prompt))
        if key in seen:
            problems.append(("duplicate within eval set", prompt))
            continue
        seen.add(key)
        if key in train_exact:
            problems.append(("exact overlap with SFT train", prompt))
            continue
        toks = content(prompt)
        worst = max(
            (len(toks & t) / len(toks | t), j) for j, t in enumerate(train_tokens)
        )
        closest.append((worst[0], prompt, train_prompts[worst[1]]))
        if worst[0] >= JACCARD_LIMIT:
            problems.append(
                (f"near-duplicate (J={worst[0]:.2f}) of "
                 f"{train_prompts[worst[1]]!r}", prompt))
            continue
        rows.append({
            "id": f"persona_eval_{len(rows):04d}",
            "prompt": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "metadata": {
                "type": "persona_eval",
                "category": category,
                "source": "authored",
                "source_split": "none",
                "held_out": True,
                "frozen": True,
                "score_with": ["yoda_persona_judge"],
            },
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"written  {len(rows)} -> {OUT}")
    print("categories:", dict(Counter(r["metadata"]["category"] for r in rows)))
    print(f"leakage problems: {len(problems)}")
    for reason, p in problems[:20]:
        print("  ", reason, "|", p)
    print("closest eval/train prompt pairs (content-word Jaccard):")
    for j, ev, tr in sorted(closest, reverse=True)[:5]:
        print(f"   {j:.2f}  eval={ev!r}  train={tr!r}")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
