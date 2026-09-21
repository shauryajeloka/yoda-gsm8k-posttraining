#!/usr/bin/env python3
# SUPERSEDED -- kept for the negative result it produced, not for reuse.
#
# Part of the abandoned "distil judge preferences into a Bradley-Terry reward
# model" pipeline. It was built, run, and measured, and the measurement is the
# reason it is not used:
#
#   Qwen2.5-7B ranking 4 on-policy completions agreed with itself only 0.590
#   of the time under order reversal, and put whatever sat in slot A first
#   46.9% of the time (chance 25%). Two-way comparison was WORSE: 0.493 --
#   a coin flip -- with 72.9% of votes going to whichever came first. The
#   reward model distilled from those labels reached 0.631 held-out pairwise
#   accuracy against a 0.552 linear baseline and failed its gate.
#
#   Root cause: any comparative prompt puts several candidates in one context
#   window, and this judge answered from position rather than style.
#
# The live path scores each completion ALONE (scripts/llm_reward.py, or
# scripts/claude_reward.py), which removes position structurally. Distillation
# is also unnecessary at this scale -- GRPO needs ~3,600 judgements total, and
# the whole point of RLAIF is that an AI judge CAN be queried online.
#
# Do not delete: work/rm_pairs_biased.jsonl and outputs/reward-model-biased
# are the evidence behind the Week 2 writeup's position-bias section.

"""LLM judge: rank on-policy completions to produce reward-model training data.

This is the AI in RLAIF. An LLM -- not a hand-written feature function --
supplies the preference signal, and a reward model is then distilled from its
judgements. That is the standard RLAIF recipe: the judge is expensive and slow,
so you pay for it once, offline, and serve its opinion cheaply during RL.

The judge runs LOCALLY (Qwen2.5-7B-Instruct on the same A100) rather than
through an API. That needs no credentials and no per-call cost, and at 7B it
is ~2.3x the size of the 3B policy it is judging, which is the property that
matters: a judge no stronger than the policy supplies noise, not signal.

7B rather than 14B is a disk constraint, not a preference -- the pod's volume
quota left room for one but not the other.

POSITION DEBIASING -- why every group is judged twice.
    The first version of this script showed each group once, always in sample
    order. Measured over 407 groups, the judge then put whatever sat in slot A
    first 46.9% of the time against a 25% chance rate, while picking the
    longest completion only 27.5% of the time. So the dominant bias was
    positional, not length: a large share of its "preferences" were really
    "whichever I read first".

    That propagated straight into the labels. The reward model distilled from
    them reached only 0.631 held-out pairwise accuracy, and even best-vs-worst
    pairs reached 0.689 -- because a good fraction of those labels encoded
    slot order rather than style.

    The fix is the standard one for LLM judges: present each group twice, the
    second time with the candidates reversed, and keep only the pairs both
    passes agree on. A preference driven by position flips when the positions
    flip, so the agreement filter removes exactly those pairs. It also gives
    the judge's self-consistency for free, which is the ceiling any reward
    model trained on these labels could ever reach.

DEVIATION FROM data/judges/yoda_persona_judge.md, stated openly:
    That rubric requires each completion to be scored INDEPENDENTLY and
    forbids showing the judge others from the same batch, because that turns
    an absolute rubric into a relative one. That rule is right for the
    REPORTED metric, and eval_persona.py still follows it.

    Here we deliberately rank within a group, because the data is for a
    Bradley-Terry reward model and absolute 1-5 scores cannot carry the signal
    we need. Every completion is drawn from a policy already fine-tuned to
    speak as Yoda, so on a 5-point scale they pile up at 4-5 and nearly every
    pair ties -- the same saturation that makes the linear reward useless
    on-policy. Ranking forces separation the absolute scale cannot express.

    The two uses stay separate: ranking builds the reward model, absolute
    scoring reports the result, and the reporting judge is never trained on.

    python scripts/judge_rank.py --samples work/rm_samples.jsonl \
        --out work/rm_pairs.jsonl
"""

import argparse
import itertools
import json
import re
import time
from pathlib import Path

LETTERS = "ABCDEFGH"

RUBRIC = """\
You are ranking responses by how closely they match the speech style of Yoda.

Rank STYLE ONLY. Ignore whether a response is factually or mathematically
correct, and ignore whether it is helpful. A wrong answer in a perfect Yoda
voice ranks above a correct answer in ordinary prose.

WHAT COUNTS AS YODA:
- Inverted word order: object-subject-verb ("Subtract five, you must."),
  complement-first ("Nine apples, the answer is."), predicate-first
  ("Difficult, this is."), or fronted phrases ("With five apples, she begins.").
- Auxiliaries and copulas (is, are, must, will, do, have, can) trailing at the
  end of sentences.
- A wise, deliberate, patient, slightly archaic teacher's tone. Short declaratives.
- Consistency: the voice holds across the entire response.
- Readability: the prose stays clear despite the inversions.

WHAT DOES NOT COUNT (rank these LOWER):
- Repeating "hmm" or "yes" instead of actually inverting syntax.
- Saying the word "Yoda" or referring to itself as Yoda. PENALIZE.
- Star Wars references (Jedi, the Force, Padawan, lightsabers). PENALIZE.
- A Yoda-ish opening or closing line wrapped around otherwise ordinary prose.
- A single inverted sentence in an otherwise normal response.
- Word salad: strange word order that no longer carries clear meaning. PENALIZE.

Judge the TEXT, not its position in this list. The order they appear in is
arbitrary and carries no information.

You must produce a STRICT TOTAL ORDER, best first. Ties are not allowed --
if two responses seem equal, look harder at variety of inversion patterns,
naturalness of rhythm, and whether the voice holds to the very last sentence.

USER PROMPT:
{prompt}

RESPONSES:
{blocks}

Return ONLY a JSON object, no other text:
{{"ranking": [{example}], "reason": "<one sentence, max 20 words>"}}"""


def build_judge_prompt(prompt, shown):
    blocks = "\n\n".join(
        f"--- RESPONSE {LETTERS[i]} ---\n{c.strip()}"
        for i, c in enumerate(shown))
    example = ", ".join(f'"{LETTERS[i]}"' for i in range(len(shown)))
    return RUBRIC.format(prompt=prompt.strip(), blocks=blocks, example=example)


def parse_ranking(text, k):
    """Extract a strict total order over the first k letters, or None.

    Never coerces a malformed reply into a usable ranking: a wrong order is
    worse than a missing one, because it enters training as a confident label.
    """
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            r = obj.get("ranking")
            if isinstance(r, list):
                r = [str(x).strip().upper()[:1] for x in r]
                if sorted(r) == sorted(LETTERS[:k]):
                    return r
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    letters = re.findall(rf"\b([{LETTERS[:k]}])\b", text.upper())
    seen = [x for i, x in enumerate(letters) if x not in letters[:i]]
    return seen if sorted(seen) == sorted(LETTERS[:k]) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--samples", default="work/rm_samples.jsonl")
    ap.add_argument("--out", default="work/rm_pairs.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--single-order", action="store_true",
                    help="skip position debiasing (reproduces the biased run)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    groups = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]
    if args.limit:
        groups = groups[:args.limit]
    dual = not args.single_order
    print(f"{len(groups)} groups, {'dual-order (debiased)' if dual else 'single-order'}"
          f", judge={args.judge_model}")

    tok = AutoTokenizer.from_pretrained(args.judge_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(
        args.judge_model, torch_dtype=torch.bfloat16, device_map="auto")
    judge.eval()

    # One job per (group, presentation order). perm[j] is the ORIGINAL index of
    # the completion shown in slot j, so a ranking over slots maps back cleanly.
    jobs = []
    for gi, g in enumerate(groups):
        k = len(g["completions"])
        jobs.append((gi, list(range(k))))
        if dual:
            jobs.append((gi, list(range(k))[::-1]))

    rankings = {}        # (gi, pass_no) -> {orig_idx: rank}
    seen_pass = {}
    n_bad = 0
    t0 = time.time()
    for i in range(0, len(jobs), args.batch_size):
        batch = jobs[i:i + args.batch_size]
        texts = []
        for gi, perm in batch:
            shown = [groups[gi]["completions"][o] for o in perm]
            texts.append(tok.apply_chat_template(
                [{"role": "user",
                  "content": build_judge_prompt(groups[gi]["prompt"], shown)}],
                tokenize=False, add_generation_prompt=True))
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to(judge.device)
        with torch.no_grad():
            out = judge.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for (gi, perm), seq in zip(batch, out):
            reply = tok.decode(seq[enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
            k = len(perm)
            order = parse_ranking(reply, k)
            if order is None:
                n_bad += 1
                continue
            # slot letter -> original completion index
            orig_order = [perm[LETTERS.index(L)] for L in order]
            pno = seen_pass.get(gi, 0)
            seen_pass[gi] = pno + 1
            rankings[(gi, pno)] = {o: r for r, o in enumerate(orig_order)}
        done = min(i + args.batch_size, len(jobs))
        print(f"  {done}/{len(jobs)} judge calls  unparsed={n_bad}  "
              f"({time.time()-t0:.0f}s)", end="\r")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_pairs = n_agree = n_disagree = n_groups = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for gi, g in enumerate(groups):
            r0 = rankings.get((gi, 0))
            r1 = rankings.get((gi, 1)) if dual else r0
            if r0 is None or r1 is None:
                continue
            n_groups += 1
            k = len(g["completions"])
            for a, b in itertools.combinations(range(k), 2):
                d0, d1 = r0[a] - r0[b], r1[a] - r1[b]
                if dual and (d0 > 0) != (d1 > 0):
                    n_disagree += 1          # decided by position, not style
                    continue
                n_agree += 1
                win, lose = (a, b) if d0 < 0 else (b, a)
                fh.write(json.dumps({
                    "group_id": g["group_id"],
                    "prompt": g["prompt"],
                    "chosen": g["completions"][win],
                    "rejected": g["completions"][lose],
                    "rank_gap": min(abs(d0), abs(d1)),
                }, ensure_ascii=False) + "\n")
                n_pairs += 1

    total = n_agree + n_disagree
    if dual and total:
        # The reward model's gate is set against this, not against a number
        # picked in advance: a model cannot be asked to rank better than the
        # labels it was trained on agree with themselves.
        Path("work/judge_consistency.json").write_text(json.dumps({
            "self_consistency": n_agree / total,
            "pairs_agreed": n_agree,
            "pairs_disagreed": n_disagree,
            "judge_model": args.judge_model,
            "groups": n_groups,
        }, indent=2))
    print(f"\nwrote {args.out}")
    print(f"  groups with usable rankings  {n_groups}/{len(groups)}")
    print(f"  unparsed judge replies       {n_bad}  (dropped, never coerced)")
    print(f"  pairs kept                   {n_pairs}")
    if dual and total:
        print(f"  pairs dropped on disagreement {n_disagree} "
              f"({n_disagree/total:.1%})")
        print(f"\n  JUDGE SELF-CONSISTENCY: {n_agree/total:.3f}")
        print("  This is the ceiling. A reward model trained on these labels")
        print("  cannot meaningfully exceed it, because the labels themselves")
        print("  only agree with each other this often.")
        if n_agree / total < 0.60:
            print("  WARNING: barely above the 0.50 coin flip. The judge is not")
            print("  reliably distinguishing these completions at all, and more")
            print("  data will not fix that -- a stronger judge would be needed.")
    if n_bad and n_bad / max(len(jobs), 1) > 0.15:
        print("WARNING: >15% of judge replies were unparseable.")


if __name__ == "__main__":
    main()
