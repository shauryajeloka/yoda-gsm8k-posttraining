#!/usr/bin/env python3
"""Retry held-out-judge items that came back unscored, identically for every arm.

On maths answers the Sonnet judge returns an API-level refusal (stop_reason
"refusal", zero output tokens) on roughly 10% of items. The prompts are
harmless GSM8K word problems, so these are spurious refusals, and they are not
random: refused answers run longer (147 vs 127 words), so excluding them
biases a mean toward shorter answers. They are also mostly, not entirely, tied
to the problem (12 of 18 shared between two arms).

This pass re-asks the judge for every unscored item, the same number of times
for every arm, and rewrites the judged file in place. Whatever still refuses is
handled downstream by comparing arms only on items scored in all of them.

    python scripts/judge_retry.py outputs/judged/*__persona_math_eval.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_persona as ep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()
    tmpl = ep.judge_prompt_template()
    for f in args.files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        missing = [i for i, r in enumerate(rows) if r.get("persona_score") is None]
        recovered = 0
        for i in missing:
            r = rows[i]
            prompt = tmpl.replace("{prompt}", r["prompt"]).replace("{response}", r["response"])
            for _ in range(args.attempts):
                try:
                    score, reason = ep.parse_score(ep.call_anthropic(prompt, args.judge_model))
                except Exception as e:                          # transient API error
                    score, reason = None, f"error: {e}"
                if score is not None:
                    r["persona_score"], r["judge_reason"] = score, reason
                    r["judge_retry"] = True
                    recovered += 1
                    break
        with open(f, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        left = sum(r.get("persona_score") is None for r in rows)
        print(f"{Path(f).name:44s} unscored {len(missing):3d} -> recovered {recovered:3d}, still unscored {left:3d}")


if __name__ == "__main__":
    main()
