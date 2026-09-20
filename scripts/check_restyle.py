#!/usr/bin/env python3
"""
Guard for the restyling step: did the Yoda rewrite preserve the MODEL'S
reasoning, or did it quietly substitute someone else's?

The self-distillation experiment only means what it claims if the targets
carry the base model's own chain of thought wearing a different voice. Two
ways that breaks:

  1. The rewrite drops or garbles a step, so the target teaches broken maths.
  2. The rewriter (me) silently IMPROVES the reasoning -- adds a check the
     model never made, fixes a clumsy detour, collapses two steps into one.
     Then the targets are partly Claude's reasoning, the model is being
     distilled from a different teacher than advertised, and the result no
     longer supports the claim being made about it.

The second failure is the dangerous one because it looks like good writing.

So the guard is on the intermediate VALUES, not on prose. Every result of an
arithmetic operation in the original trace must still appear in the rewrite.
That permits rephrasing, reordering within a step, and dropping LaTeX
scaffolding, while refusing a rewrite that skips a computation or invents one.

    python scripts/check_restyle.py --original outputs/base-train-cot/gsm8k_train.jsonl \
                                    --restyled work/yoda_cot.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path("data/verifier")))
from gsm8k_verifier import normalize_number, verify, _clean  # noqa: E402

# "A op B = C" -- we care about C, the value the model computed.
EQ = re.compile(r"([\d,.()][\d,.\s+\-*/x×÷()]*?)\s*=\s*(-?[\d,]*\.?\d+)")

# The base model writes its maths in LaTeX, so the plain-text equation regex
# above sees almost none of it. Without this the guard silently extracts zero
# computed values and then accepts ANY rewrite, which is worse than no guard.
_LATEX = [
    (re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}"), r"(\1)/(\2)"),
    (re.compile(r"\\(?:times|cdot)"), "*"),
    (re.compile(r"\\div"), "/"),
    (re.compile(r"\\left|\\right"), " "),
    (re.compile(r"\\[a-zA-Z]+"), " "),
    (re.compile(r"[{}$]"), " "),
]


def _delatex(text):
    out = text
    for _ in range(3):                     # nested \frac needs repeats
        for pat, repl in _LATEX:
            out = pat.sub(repl, out)
    return re.sub(r"[ \t]+", " ", out)


def computed_values(text):
    """Normalized results of every arithmetic operation in a trace.

    Only the right-hand side of an equation counts. A number that merely
    appears in the prose (a quantity restated from the question) is not
    evidence that a computation happened.
    """
    out = []
    for lhs, rhs in EQ.findall(_delatex(_clean(text))):
        if not re.search(r"[+\-*/x×÷]", lhs):
            continue                       # "x = 5" is an assignment, not a step
        n = normalize_number(rhs)
        if n is not None:
            out.append(n)
    return out


def present_numbers(text):
    return {n for n in (normalize_number(m) for m in
                        re.findall(r"-?[\d,]*\.?\d+", _delatex(_clean(text))))
            if n is not None}


def check(original, restyled, ground_truth, question=None,
          min_len_ratio=0.60, min_step_ratio=0.80):
    """Returns a list of reasons the rewrite fails. Empty list == accepted.

    Three independent floors, because each catches what the others miss:

      * final answer -- the target must still be correct;
      * computed values -- every detectable intermediate result survives;
      * length and step COUNT -- a blunt backstop. Value checking cannot see
        a step whose result is a fraction (50/60 -> 5/6 is invisible to it),
        so a rewrite can drop real reasoning and still pass. Requiring the
        rewrite to stay near the original's length and equation count closes
        that hole without needing to parse every LaTeX form.

    The length floor is the one that matters most here: the whole point of
    these targets is that they carry the base model's ~6 steps rather than
    GSM8K's ~3.5, and a rewrite that quietly compresses back down to three
    sentences would silently undo the experiment.
    """
    reasons = []

    ow, rw = len(original.split()), len(restyled.split())
    if ow and rw < min_len_ratio * ow:
        reasons.append(f"too short: {rw} words vs original {ow} "
                       f"({rw/ow:.0%}, floor {min_len_ratio:.0%})")

    os_, rs = len(computed_values(original)), len(computed_values(restyled))
    if os_ and rs < min_step_ratio * os_:
        reasons.append(f"dropped steps: {rs} equations vs original {os_} "
                       f"({rs/os_:.0%}, floor {min_step_ratio:.0%})")

    v = verify(restyled, ground_truth, question=question)
    if not v["correct"]:
        reasons.append(f"final answer wrong: got {v['predicted_answer']} "
                       f"want {v['ground_truth']}")

    want = computed_values(original)
    have = present_numbers(restyled)
    missing = [x for x in want if x not in have]
    if missing:
        reasons.append(f"dropped {len(missing)}/{len(want)} computed values: "
                       f"{missing[:6]}")

    # Invented arithmetic: a computation in the rewrite whose result appears
    # nowhere in the original trace means a step was added, not restyled.
    orig_all = present_numbers(original)
    added = [x for x in computed_values(restyled) if x not in orig_all]
    if added:
        reasons.append(f"invented {len(added)} computed values absent from the "
                       f"original trace: {added[:6]}")
    return reasons


def parse_blocks(path):
    blocks, cur, cid = {}, [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("@@"):
            if cid:
                blocks[cid] = "\n".join(cur).strip()
            cid, cur = line[2:].strip(), []
        elif cid is not None:
            cur.append(line)
    if cid:
        blocks[cid] = "\n".join(cur).strip()
    return {k: v for k, v in blocks.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="outputs/base-train-cot/gsm8k_train.jsonl")
    ap.add_argument("--restyled", required=True)
    ap.add_argument("--out", help="write accepted examples as SFT JSONL")
    ap.add_argument("--rejects", default="work/restyle_rejects.jsonl")
    ap.add_argument("--min-len-ratio", type=float, default=0.60)
    ap.add_argument("--min-step-ratio", type=float, default=0.80)
    args = ap.parse_args()

    orig = {}
    for line in open(args.original, encoding="utf-8"):
        r = json.loads(line)
        orig[r["id"]] = r

    blocks = parse_blocks(args.restyled)
    ok, bad = [], []
    for sid, text in blocks.items():
        if sid not in orig:
            bad.append({"id": sid, "reasons": ["unknown source id"]})
            continue
        o = orig[sid]
        reasons = check(o["response"], text, o.get("ground_truth"),
                        o.get("prompt"), args.min_len_ratio,
                        args.min_step_ratio)
        if reasons:
            bad.append({"id": sid, "reasons": reasons, "text": text})
        else:
            ok.append({"messages": [{"role": "user", "content": o["prompt"]},
                                    {"role": "assistant", "content": text}],
                       "metadata": {"source_id": sid, "type": "yoda_self_distill",
                                    "persona": "yoda",
                                    "ground_truth": o.get("ground_truth"),
                                    "verified": True,
                                    "steps_preserved": True}})

    print(f"parsed    {len(blocks)}")
    print(f"accepted  {len(ok)}")
    print(f"rejected  {len(bad)}")
    for b in bad[:12]:
        print(f"  REJECT {b['id']}: {b['reasons']}")
    if args.out and ok:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in ok:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"-> {args.out}")
    if bad:
        Path(args.rejects).parent.mkdir(parents=True, exist_ok=True)
        with open(args.rejects, "w", encoding="utf-8") as fh:
            for b in bad:
                fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
