#!/usr/bin/env python3
"""Every Checkpoint-3 number, from committed artifacts, in one reproducible pass.

Written after the Week-2 tables turned out to mix persona means from a pod copy
of the classifier with paired statistics from the committed one. Everything
here is computed from files in the repo, so the report cannot drift from them.

    python scripts/analyze_rlvr.py        # prints tables, writes outputs/rlvr_summary.json

Comparisons:
  maths      exact McNemar on the 500 frozen GSM8K items (same items, paired)
  classifier paired bootstrap CI on per-item P(persona) + McNemar at 0.5
  judge      held-out Sonnet 1-5 scores, paired sign test + bootstrap CI,
             computed ONLY on items the judge scored for every arm in the
             comparison (it refuses ~10% of maths items; see judge_retry.py)
"""

import json
import random
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "data/verifier")
from gsm8k_verifier import verify  # noqa: E402
from persona_similarity import features  # noqa: E402
from persona_classifier import predict, apply_std  # noqa: E402
from claude_reward import SW_RE, NAME_RE, FILLER_RE  # noqa: E402

ARMS = {  # label -> outputs/ dir
    "base": "base-uncapped", "SFT": "yodadistill", "RLAIF": "rlaif",
    "RLVR-A (verifier)": "rlvr-verifier", "RLVR-B (combined)": "rlvr-combined",
}
# Persona generations for base predate the uncapped maths regeneration and live
# under outputs/base/ (persona answers are short, so the cap never bound them).
PERSONA_DIR = {"base": "base"}
PM_IDS = {json.loads(l)["id"] for l in open("data/math/gsm8k_persona_math_eval.jsonl")}
CLF = json.loads(Path("outputs/persona_clf.json").read_text())


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def p_clf(t):
    return predict(apply_std(features(t), CLF["mu"], CLF["sd"]), CLF["w"], CLF["b"])


def mcnemar(a, b):
    """a, b: parallel lists of booleans. Returns (only_a, only_b, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    p = sum(comb(n, k) for k in range(min(n01, n10) + 1)) * 2 / 2 ** n if n else 1.0
    return n01, n10, min(p, 1.0)


def boot_ci(d, n=4000, seed=0):
    rng = random.Random(seed)
    bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n)]


def main():
    out = {"maths": {}, "classifier": {}, "judge": {}, "diagnostics": {}, "training": {}}

    # ---------------- maths ----------------
    correct = {}
    for lab, d in ARMS.items():
        rows = {r["id"]: r for r in jl(f"outputs/{d}/gsm8k_eval.jsonl")}
        correct[lab] = {i: verify(r["response"], r["ground_truth"], question=r["prompt"])["correct"]
                        for i, r in rows.items()}
    ids = sorted(correct["base"])
    print("=== GSM8K (500 frozen items) ===")
    for lab in ARMS:
        acc = sum(correct[lab][i] for i in ids) / len(ids)
        out["maths"][lab] = {"acc": acc}
        print(f"  {lab:20s} {acc:6.1%}")
    for a, b in (("RLAIF", "RLVR-A (verifier)"), ("RLAIF", "RLVR-B (combined)"),
                 ("RLVR-A (verifier)", "RLVR-B (combined)"), ("base", "RLVR-A (verifier)")):
        o1, o2, p = mcnemar([correct[a][i] for i in ids], [correct[b][i] for i in ids])
        diff = (sum(correct[b][i] for i in ids) - sum(correct[a][i] for i in ids)) / len(ids)
        out["maths"][f"{a} -> {b}"] = {"diff": diff, "only_first": o1, "only_second": o2, "p": p}
        print(f"  {a} -> {b}: {diff:+.1%}  ({o1} lost, {o2} gained)  McNemar p={p:.3g}")

    # ---------------- classifier persona ----------------
    print("\n=== persona, style classifier (committed persona_clf.json) ===")
    cl = {}
    for lab, d in ARMS.items():
        pd = PERSONA_DIR.get(lab, d)
        g = [p_clf(r["response"]) for r in jl(f"outputs/{pd}/persona_eval.jsonl")] \
            if Path(f"outputs/{pd}/persona_eval.jsonl").exists() else None
        mrows = sorted(jl(f"outputs/{d}/gsm8k_eval.jsonl"), key=lambda r: r["id"])
        m = [p_clf(r["response"]) for r in mrows]
        cl[lab] = (g, m)
        out["classifier"][lab] = {"general": st.mean(g) if g else None, "on_maths": st.mean(m)}
        gs = f"{st.mean(g):.3f}" if g else "  —  "
        print(f"  {lab:20s} general {gs}   on-maths {st.mean(m):.3f}")
    for a, b in (("RLAIF", "RLVR-A (verifier)"), ("RLAIF", "RLVR-B (combined)"),
                 ("RLVR-A (verifier)", "RLVR-B (combined)")):
        for k, name in ((0, "general"), (1, "on_maths")):
            A, B = cl[a][k], cl[b][k]
            d = [y - x for x, y in zip(A, B)]
            lo, hi = boot_ci(d)
            _, _, p = mcnemar([x >= .5 for x in A], [y >= .5 for y in B])
            out["classifier"][f"{a} -> {b} [{name}]"] = {"diff": st.mean(d), "ci": [lo, hi], "p": p}
            print(f"  {a} -> {b} [{name}]: {st.mean(d):+.3f} CI [{lo:+.3f},{hi:+.3f}] p={p:.2g}")

    # ---------------- held-out judge ----------------
    print("\n=== persona, held-out Sonnet judge (1-5), paired on commonly-scored items ===")
    judged = {}
    for lab, d in ARMS.items():
        for kind in ("persona_eval", "persona_math_eval"):
            f = Path(f"outputs/judged/{PERSONA_DIR.get(lab, d)}__{kind}.jsonl")
            if f.exists():
                judged[(lab, kind)] = {r["prompt"]: r["persona_score"] for r in jl(f)}
    for kind in ("persona_eval", "persona_math_eval"):
        labs = [l for l in ARMS if (l, kind) in judged]
        for lab in labs:
            s = [v for v in judged[(lab, kind)].values() if v is not None]
            ref = sum(v is None for v in judged[(lab, kind)].values())
            out["judge"][f"{lab} [{kind}]"] = {"mean": st.mean(s), "n_scored": len(s), "refused": ref}
            print(f"  {kind:18s} {lab:20s} mean {st.mean(s):.2f}  (n={len(s)}, refused {ref})")
        for a, b in (("RLAIF", "RLVR-A (verifier)"), ("RLAIF", "RLVR-B (combined)"),
                     ("RLVR-A (verifier)", "RLVR-B (combined)")):
            if (a, kind) not in judged or (b, kind) not in judged:
                continue
            A, B = judged[(a, kind)], judged[(b, kind)]
            common = [q for q in A if q in B and A[q] is not None and B[q] is not None]
            d = [B[q] - A[q] for q in common]
            up, dn = sum(x > 0 for x in d), sum(x < 0 for x in d)
            _, _, p = mcnemar([x < 0 for x in d], [x > 0 for x in d])
            lo, hi = boot_ci(d)
            out["judge"][f"{a} -> {b} [{kind}]"] = {
                "diff": st.mean(d), "ci": [lo, hi], "improved": up, "worsened": dn,
                "sign_test_p": p, "n_paired": len(common)}
            print(f"    {a} -> {b}: {st.mean(d):+.2f} CI [{lo:+.2f},{hi:+.2f}]  "
                  f"up {up} / down {dn}  sign p={p:.2g}  (n={len(common)})")

    # ---------------- diagnostics ----------------
    print("\n=== gaming diagnostics (maths + general outputs) ===")
    for lab, d in ARMS.items():
        rows = jl(f"outputs/{d}/gsm8k_eval.jsonl")
        pdir = PERSONA_DIR.get(lab, d)
        pe = jl(f"outputs/{pdir}/persona_eval.jsonl") if Path(f"outputs/{pdir}/persona_eval.jsonl").exists() else []
        allr = [r["response"] for r in rows + pe]
        diag = {"maths_words": st.mean(len(r["response"].split()) for r in rows),
                "starwars_or_name": sum(bool(SW_RE.search(t) or NAME_RE.search(t)) for t in allr),
                "filler_per_resp": st.mean(len(FILLER_RE.findall(t)) for t in allr),
                "maths_cues_per100w": st.mean(features(r["response"])[0] for r in rows)}
        out["diagnostics"][lab] = diag
        print(f"  {lab:20s} words {diag['maths_words']:6.1f}  cues {diag['maths_cues_per100w']:.2f}  "
              f"SW/name {diag['starwars_or_name']}  filler {diag['filler_per_resp']:.3f}")

    # ---------------- training dynamics ----------------
    print("\n=== training dynamics (first 40 vs last 40 steps) ===")
    for lab, d in (("RLVR-A (verifier)", "rlvr-verifier-lora"), ("RLVR-B (combined)", "rlvr-combined-lora")):
        log = jl(f"outputs/{d}/rlaif_log.jsonl")
        a, b = log[:40], log[-40:]
        rec = {}
        for k in ("verifier", "persona", "style_probe", "mean_words", "kl", "mixed_groups", "truncated"):
            if k in a[0]:
                rec[k] = [st.mean(r[k] for r in a), st.mean(r[k] for r in b)]
        out["training"][lab] = rec
        print(f"  {lab}: " + "  ".join(f"{k} {v[0]:.3f}->{v[1]:.3f}" for k, v in rec.items()))

    Path("outputs/rlvr_summary.json").write_text(json.dumps(out, indent=2))
    print("\n-> outputs/rlvr_summary.json")


if __name__ == "__main__":
    main()
