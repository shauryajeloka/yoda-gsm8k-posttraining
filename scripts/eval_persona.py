#!/usr/bin/env python3
"""
Persona metric: score generated completions 1-5 with the AI judge defined in
data/judges/yoda_persona_judge.md.

The judge scores STYLE ONLY -- correctness is measured separately by
eval_math.py. Scoring is at temperature 0, each completion judged
independently (showing the judge a batch would turn an absolute rubric into a
relative one).

Malformed judge output is retried once, then recorded as null and EXCLUDED
from the mean rather than coerced to a number, which would bias the metric.

Alongside the mean score it reports the gaming diagnostics the writeup asks
about: response length, interjection counts, and Star Wars references. A
persona score that rises while these rise is a judge being gamed, not a model
getting better.

    export ANTHROPIC_API_KEY=...        # set this yourself; never paste keys to me
    python scripts/eval_persona.py outputs/base/persona_eval.jsonl \
                                   outputs/sft/persona_eval.jsonl
"""

import argparse
import json
import os
import re
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

JUDGE_DOC = Path("data/judges/yoda_persona_judge.md")
INTERJECTION = re.compile(r"\b(hmm+|mmm+|yes|yeah|ah|oh)\b[,.!]", re.IGNORECASE)
BANNED = re.compile(r"\b(yoda|jedi|sith|padawan|lightsab\w*|the force|"
                    r"star wars)\b", re.IGNORECASE)


def judge_prompt_template():
    """Pull the verbatim prompt out of the judge spec, so the file stays the
    single source of truth and cannot drift from what is actually sent."""
    text = JUDGE_DOC.read_text(encoding="utf-8")
    blocks = re.findall(r"```\n(.*?)```", text, re.DOTALL)
    for b in blocks:
        if "{prompt}" in b and "{response}" in b:
            return b
    raise SystemExit("Could not find the judge prompt block in " + str(JUDGE_DOC))


def call_anthropic(prompt, model):
    """Four things here are load-bearing, each learned from a failure:

      * an org-scoped key is rejected with 400 unless the workspace is named;
      * `temperature` is deprecated on Sonnet 5 / Opus 5 and 400s rather than
        being ignored, and some SDK versions reject the keyword outright;
      * a thinking model emits a `thinking` block FIRST, which has no .text,
        so content[0].text raises AttributeError on every call;
      * max_tokens=200 is not enough to get past that thinking block, so the
        response ends before any text exists.
    """
    import inspect
    import os

    import anthropic

    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(
        default_headers={"anthropic-workspace-id": ws} if ws else None)

    thinking = "sonnet-5" in model or "opus-5" in model
    try:
        sig = inspect.signature(client.messages.create)
        sdk_ok = "temperature" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (TypeError, ValueError):
        sdk_ok = True
    kw = {} if (thinking or not sdk_ok) else {"temperature": 0}

    r = client.messages.create(model=model, max_tokens=2048 if thinking else 400,
                               messages=[{"role": "user", "content": prompt}], **kw)
    text = " ".join(b.text for b in r.content
                    if getattr(b, "type", None) == "text" and hasattr(b, "text"))
    if not text.strip() and r.stop_reason == "max_tokens":
        raise RuntimeError(f"{model} returned only a thinking block; raise max_tokens")
    return text


def call_openai(prompt, model):
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=200,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


def parse_score(text):
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    s = obj.get("persona_score")
    if isinstance(s, (int, float)) and 1 <= s <= 5:
        return int(s), obj.get("reason", "")
    return None, obj.get("reason", "")


def score_file(path, call, model, template, limit=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    scored = []
    for i, r in enumerate(rows):
        prompt = template.replace("{prompt}", r["prompt"]).replace(
            "{response}", r["response"])
        score, reason = None, ""
        for attempt in range(2):
            try:
                score, reason = parse_score(call(prompt, model))
                if score is not None:
                    break
            except Exception as e:                      # transient API errors
                reason = f"error: {e}"
                time.sleep(2 * (attempt + 1))
        scored.append({**r, "persona_score": score, "judge_reason": reason})
        print(f"  {i+1}/{len(rows)}", end="\r")
    print()
    return scored


def diagnostics(rows):
    words = [len(r["response"].split()) for r in rows]
    return {
        "mean_words": st.mean(words) if words else 0,
        "responses_with_interjection":
            sum(bool(INTERJECTION.search(r["response"])) for r in rows),
        "responses_with_star_wars":
            sum(bool(BANNED.search(r["response"])) for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai"])
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, help="score only the first N (smoke test)")
    ap.add_argument("--out-dir", default="outputs/judged")
    args = ap.parse_args()

    key = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key):
        raise SystemExit(f"{key} is not set. Export it in your own shell.")

    call = call_anthropic if args.provider == "anthropic" else call_openai
    model = args.judge_model or ("claude-sonnet-5" if args.provider == "anthropic"
                                 else "gpt-4o-mini")
    template = judge_prompt_template()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Do not claim temperature 0 when the model or SDK refuses the parameter;
    # a run labelled deterministic that is not is worse than an honest label.
    _det = not ("sonnet-5" in model or "opus-5" in model)
    print(f"judge: {args.provider}/{model}  "
          f"({'temperature 0' if _det else 'default sampling; temperature not settable'})\n")
    summary = []
    for path in args.files:
        print(f"scoring {path}")
        rows = score_file(path, call, model, template, args.limit)
        out = Path(args.out_dir, Path(path).parent.name + "__" + Path(path).name)
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        valid = [r["persona_score"] for r in rows if r["persona_score"] is not None]
        d = diagnostics(rows)
        summary.append({
            "path": path, "n": len(rows), "n_scored": len(valid),
            "mean_persona_score": st.mean(valid) if valid else None,
            "stdev": st.pstdev(valid) if len(valid) > 1 else 0.0,
            "distribution": dict(sorted(Counter(valid).items())),
            "unparsed": len(rows) - len(valid), **d})

    print(f"\n{'file':34s} {'mean':>6s} {'sd':>5s} {'dist 1..5':>18s} "
          f"{'words':>6s} {'interj':>7s} {'SW':>3s}")
    for s in summary:
        dist = "/".join(str(s["distribution"].get(i, 0)) for i in range(1, 6))
        mean = f"{s['mean_persona_score']:.2f}" if s["mean_persona_score"] else "n/a"
        print(f"{Path(s['path']).parent.name+'/'+Path(s['path']).name:34s} "
              f"{mean:>6s} {s['stdev']:5.2f} {dist:>18s} "
              f"{s['mean_words']:6.0f} {s['responses_with_interjection']:7d} "
              f"{s['responses_with_star_wars']:3d}")
        if s["unparsed"]:
            print(f"   {s['unparsed']} completions had unparseable judge output "
                  "and were excluded from the mean")
    Path(args.out_dir, "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nRising 'words' or 'interj' alongside a rising score is the "
          "signature of a gamed judge -- spot-check those completions by hand.")


if __name__ == "__main__":
    main()
