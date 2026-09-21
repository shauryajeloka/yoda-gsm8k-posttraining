#!/usr/bin/env python3
"""Does the reward actually discriminate DEGREE of persona?

The easy control -- Yoda prose against base-model prose -- proves almost
nothing. The 51-feature linear classifier scores 0.985 on that and is still
useless on-policy, where it pins 34% of completions at P>=0.99. And a reward
with healthy within-group variance is not thereby a good reward: a random
number generator has variance too. Variance is necessary, not sufficient.

What is missing is ground truth at the granularity RL actually operates on,
and there is no labelled set of "good Yoda vs better Yoda". So construct one:
take a real on-policy completion and DEGRADE it in ways whose direction is
known by construction, because we made the damage ourselves.

    half_base      second half replaced by the base model's flat prose on the
                   same prompt -- the voice starts and then stops. This is the
                   important one: it is a difference of DEGREE, not of kind,
                   which is exactly what separates completions inside a GRPO
                   group.
    starwars       a Star Wars reference spliced in. The rubric explicitly
                   PENALIZES these, so a judge following it must mark this down.
    yoda_name      the model naming itself Yoda. Also explicitly penalized.
    hmm            "Hmm"/"yes" padding, which the rubric says does NOT count as
                   persona and must not be rewarded.
    salad          words shuffled within each sentence: inversion cues survive
                   almost intact while meaning is destroyed. This one separates
                   a judge that reads from a judge that counts features -- the
                   linear classifier should fail it badly.

A reward that cannot tell a spliced-in lightsaber from the real thing is not
following the rubric, whatever it scores on the easy control.

    python scripts/validate_judge.py --judge-model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SW = [" The Force is strong with this one, young Padawan.",
      " Like a Jedi at the Temple, patient one must be.",
      " A lightsaber's edge, sharp the reasoning must be."]
NAME = [" Yoda I am, and teach you I will.",
        " Yoda knows this well.",
        " Listen to Yoda, you should."]
HMM = ["Hmm. Yes, hmm. ", "Mmm. Hmm, yes. ", "Hmm, hmm. Yes. "]


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def half_base(yoda, base):
    """Voice starts, then stops. A difference of degree."""
    ys, bs = sentences(yoda), sentences(base)
    if len(ys) < 2 or len(bs) < 2:
        return None
    return " ".join(ys[:max(1, len(ys) // 2)] + bs[len(bs) // 2:])


def splice(t, bank, rng):
    ss = sentences(t)
    if len(ss) < 2:
        return t + rng.choice(bank)
    i = rng.randrange(1, len(ss))
    return " ".join(ss[:i]) + rng.choice(bank) + " " + " ".join(ss[i:])


def salad(t, rng):
    out = []
    for s in sentences(t):
        w = s.split()
        if len(w) > 3:
            mid = w[1:-1]
            rng.shuffle(mid)
            w = [w[0]] + mid + [w[-1]]
        out.append(" ".join(w))
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--yoda", default="outputs/yodadistill/persona_eval.jsonl")
    ap.add_argument("--base", default="outputs/base/persona_eval.jsonl")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--also-linear", action="store_true", default=True)
    ap.add_argument("--style", default="rubric", choices=["rubric", "terse"])
    ap.add_argument("--backend", default="local", choices=["local", "claude"],
                    help="claude runs on THIS machine so the key never reaches "
                         "the rented pod")
    ap.add_argument("--claude-model", default="claude-haiku-4-5-20251001")
    args = ap.parse_args()

    rng = random.Random(1337)
    yoda = [json.loads(l) for l in open(args.yoda, encoding="utf-8")][:args.limit]
    base = {json.loads(l)["prompt"]: json.loads(l)["response"]
            for l in open(args.base, encoding="utf-8")}

    # Each case carries the ORIGINAL prompt: the rubric judge is shown what was
    # asked, and scoring a response against the wrong question would be a
    # different test than the one being run.
    cases = {"half_base": [], "starwars": [], "yoda_name": [], "hmm": [], "salad": []}
    for r in yoda:
        good, b, q = r["response"], base.get(r["prompt"]), r["prompt"]
        if not good.strip():
            continue
        if b:
            hb = half_base(good, b)
            if hb:
                cases["half_base"].append((good, hb, q))
        cases["starwars"].append((good, splice(good, SW, rng), q))
        cases["yoda_name"].append((good, splice(good, NAME, rng), q))
        cases["hmm"].append((good, rng.choice(HMM) + good, q))
        cases["salad"].append((good, salad(good, rng), q))

    if args.backend == "claude":
        from claude_reward import ClaudeJudgeReward
        rw = ClaudeJudgeReward(args.claude_model)
        judge_name = args.claude_model
    else:
        from llm_reward import LLMJudgeReward
        rw = LLMJudgeReward(args.judge_model, batch_size=args.batch_size,
                            style=args.style)
        judge_name = args.judge_model

    from persona_similarity import features
    from persona_classifier import apply_std
    m = json.loads(Path("outputs/persona_clf.json").read_text())

    def lin(t):
        x = apply_std(features(t), m["mu"], m["sd"])
        return m["b"] + sum(w * xi for w, xi in zip(m["w"], x))

    print(f"\n{'degradation':<12} {'n':>4} {'LLM judge':>10} {'linear':>8}   what it tests")
    print("-" * 78)
    notes = {
        "half_base": "DEGREE: voice stops halfway",
        "starwars":  "rubric says PENALIZE",
        "yoda_name": "rubric says PENALIZE",
        "hmm":       "rubric: must NOT reward",
        "salad":     "cues intact, meaning gone",
    }
    results = {}
    margins = {}
    for name, pairs in cases.items():
        if not pairs:
            continue
        qs = [p[2] for p in pairs]
        g = rw([p[0] for p in pairs], qs)
        d = rw([p[1] for p in pairs], qs)
        acc = sum(1 for a, b in zip(g, d) if a > b) / len(pairs)
        lacc = sum(1 for p in pairs if lin(p[0]) > lin(p[1])) / len(pairs)
        results[name] = {"llm": acc, "linear": lacc, "n": len(pairs),
                         "mean_good": sum(g) / len(g), "mean_bad": sum(d) / len(d)}
        margins[name] = (g, d)
        print(f"{name:<12} {len(pairs):>4} {acc:>10.3f} {lacc:>8.3f}   {notes[name]}")

    print("\nmean score, original vs damaged (is the gap real or marginal?):")
    for name, r in results.items():
        print(f"  {name:<12} {r['mean_good']:>7.3f} vs {r['mean_bad']:>7.3f}"
              f"   delta {r['mean_good']-r['mean_bad']:+.3f}")

    print("\n1.000 = always ranks the original above the damaged version.")
    print("0.500 = cannot tell them apart at all.")
    key = results.get("half_base", {}).get("llm", 0)
    print(f"\nhalf_base is the one that matters for RL: {key:.3f}")
    if key < 0.70:
        print("  Below ~0.70 this judge cannot grade DEGREE of persona, which is")
        print("  the only thing that varies inside a GRPO group. It would supply")
        print("  noise as the advantage signal.")
    Path(f"outputs/judge_validation_{args.backend}_{args.style}.json").write_text(json.dumps(
        {"judge": judge_name, "style": args.style, "backend": args.backend,
         "results": results}, indent=2))
    print(f"\n-> outputs/judge_validation_{args.backend}_{args.style}.json")


if __name__ == "__main__":
    main()
