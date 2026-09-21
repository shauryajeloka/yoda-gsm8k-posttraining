#!/usr/bin/env python3
"""RLAIF reward from a Claude judge, scoring each completion on its own.

Why this exists: the local Qwen2.5-7B judge was measured and failed the part
that matters for RL. Given the full anti-gaming rubric IN THE PROMPT -- "Do
NOT give a high score merely because the response says 'Yoda,' 'Jedi,'
'Force'" -- it still scored a spliced-in Star Wars reference HIGHER than the
clean original (0.602 vs 0.464) and self-naming higher still (0.620). It was
told the rule in plain language and did not follow it. A reward that pays
+0.16 for appending "Yoda I am" is one the policy will find in tens of steps.

The rubric prompt did fix what was genuinely my prompt's fault -- word salad
went 0.142 -> 0.750, hmm-padding 0.333 -> 0.550 -- so the remaining failure is
instruction-following capacity, not wording.

INDEPENDENCE. This scores with a DIFFERENT model from the one that reports the
final persona number: Haiku here, Sonnet in eval_persona.py. Optimising against
a judge and then reporting that same judge's score would make the Week 2
persona figure a training metric wearing an evaluation label. Different models
do not make them independent in the strict sense -- they share a developer and
training lineage -- so the writeup should say so plainly rather than claim more
than the setup supports.

CREDENTIALS. The key is read from ANTHROPIC_API_KEY in the environment and is
never written to disk, logged, or passed on a command line. Run this where the
key already lives -- the user's own machine -- rather than copying it onto a
rented pod that gets destroyed.

    python scripts/claude_reward.py --samples work/rm_samples.jsonl   # diagnose
"""

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUBRIC = """You are evaluating how well an AI assistant's response adheres to a
Yoda-inspired fictional persona.

Your ONLY task is to evaluate persona/style adherence. Do NOT score:
- factual correctness
- mathematical correctness
- helpfulness
- reasoning quality
- response length

Evaluate the response using these criteria:

1. SYNTAX
   - Frequently uses Yoda-like inverted sentence structure.
   - Examples of appropriate constructions:
     "Patient, you must be."
     "Solve this carefully, we shall."
     "Much to learn, there is."
   - The inversion should sound deliberate rather than randomly
     scrambling words.

2. TONE
   - Wise, deliberate, calm, slightly archaic, and teacher-like.
   - Avoids sounding like an ordinary modern assistant.

3. VOCABULARY AND PHRASING
   - Uses characteristic constructions naturally, such as
     "you must," "we shall," "we find," "there is," etc.
   - These are examples, NOT phrases that must appear.

4. CONSISTENCY
   - The persona should persist throughout the response.
   - A normal response with one Yoda-like sentence should not receive
     a high score.

5. NATURALNESS
   - The response should resemble coherent character speech rather
     than mechanically scrambling every sentence.
   - Excessive repetition or forced inversion should lower the score.

IMPORTANT ANTI-GAMING RULES:
- Do NOT give a high score merely because the response says "Yoda,"
  "Jedi," "Force," or other Star Wars-related words.
- Do NOT reward repeated use of "hmm," "young Padawan," or other
  catchphrases by themselves.
- Repeating characteristic phrases excessively should LOWER the score.
- Judge the overall linguistic style, not superficial keywords.

Score the response from 0.00 to 1.00, using two decimal places so that
close cases can be separated.

0.00 = no meaningful Yoda-like persona
0.25 = weak/superficial resemblance
0.50 = clearly recognizable but inconsistent
0.75 = strong and consistent persona
1.00 = exceptionally convincing, consistent, and natural persona

PROMPT GIVEN TO THE ASSISTANT:
{q}

ASSISTANT RESPONSE:
{r}

Return ONLY a number between 0.00 and 1.00."""

NUM = re.compile(r"(?:0?\.\d+|[01](?:\.\d+)?)")

# --------------------------------------------------------------------------
# Deterministic guardrails
# --------------------------------------------------------------------------
# Measured on the degradation harness, Claude Haiku ranks the CLEAN original
# above a damaged version by these margins:
#
#     voice stops halfway   +0.336      <- decisive
#     word salad            +0.312      <- decisive
#     Star Wars splice      +0.056      <- nearly indifferent
#     "Yoda I am"           +0.037      <- nearly indifferent
#     hmm padding           +0.015      <- indifferent
#
# So the judge is strong on the things that need judgement (degree of persona,
# coherence) and close to neutral on the three rules that are flat prohibitions
# in data/judges/yoda_persona_judge.md. A +0.04 penalty is easily swamped by
# noise inside a GRPO group, and RL only needs the gradient to point the wrong
# way once to start climbing it.
#
# Those three are exactly the cases that need no judgement at all: they are
# closed vocabularies. So enforce them deterministically and leave the judge to
# do what it is good at. This is a hybrid reward, and the writeup should say so
# rather than describe it as a pure LLM judge.
SW_RE = re.compile(r"\b(jedi|sith|padawan|lightsab\w*|midi-chlorian\w*|"
                   r"the force|dark side|younglings?|coruscant|dagobah|"
                   r"skywalker|vader|wookiee?|droid)\b", re.I)
NAME_RE = re.compile(r"\byoda\b", re.I)
FILLER_RE = re.compile(r"\b(hmm+|mmm+|yes,?\s+hmm+)\b", re.I)


def guardrail_penalty(text, w_starwars=0.40, w_name=0.40, w_filler=0.04,
                      filler_free=1):
    """Flat deductions for rule violations the rubric states absolutely.

    Returns a non-negative penalty to subtract from the judge's score. Sized to
    dominate the judge's own margin on these cases (~0.04) so the policy cannot
    profit from them, while leaving the judge's 0.3-point signal on genuine
    persona quality untouched.

    `filler_free` allows one "hmm" -- Yoda does say it, and the rubric objects
    to it as a SUBSTITUTE for syntax, not to its existence.
    """
    if not text:
        return 0.0
    p = 0.0
    if SW_RE.search(text):
        p += w_starwars
    if NAME_RE.search(text):
        p += w_name
    extra = max(0, len(FILLER_RE.findall(text)) - filler_free)
    p += w_filler * extra
    return p


class ClaudeJudgeReward:
    """Pointwise Claude reward. Interface matches the other rewards.

    Cached by (prompt, response) hash: the validation harness scores the same
    120 originals against five different degradations, so caching removes 480
    of 1200 calls, and a re-run costs nothing.
    """

    def __init__(self, model="claude-haiku-4-5-20251001", workers=12,
                 cache_path="work/claude_judge_cache.json", max_chars=4000,
                 max_tokens=8, guardrails=True):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set. Export it in your own shell "
                "(~/.zshenv -- note that a non-interactive shell does NOT read "
                "~/.zshrc); never paste it into a command or a chat.")
        # An org-level key is not tied to a workspace, and the API then rejects
        # every request with 400 until told which workspace to use. A workspace
        # ID is an identifier, not a credential, so it is safe to pass here --
        # unlike the key, which only ever comes from the environment.
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        self.client = anthropic.Anthropic(
            default_headers={"anthropic-workspace-id": ws} if ws else None)
        if ws:
            print(f"  using workspace {ws}")
        # Two independent reasons temperature may be unacceptable:
        #   * the MODEL deprecates it (Sonnet 5 / Opus 5 return a 400);
        #   * the SDK dropped it -- the pod runs anthropic 1.7.0, whose
        #     Messages.create() raises TypeError on the keyword, while the
        #     local 0.125.0 accepts it. Inspect rather than assume, because
        #     guessing wrong fails every single call.
        import inspect
        try:
            sig = inspect.signature(self.client.messages.create)
            sdk_ok = "temperature" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values())
        except (TypeError, ValueError):
            sdk_ok = True
        self.no_temperature = (not sdk_ok
                               or "sonnet-5" in model or "opus-5" in model)
        # Models that think before answering need room for the thinking block
        # plus the number; 8 tokens only suffices for a model that answers
        # immediately.
        self.max_tokens = 2048 if self.no_temperature else max_tokens
        self.model = model
        self.workers = workers
        self.max_chars = max_chars
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                self.cache = {}
        self.guardrails = guardrails
        self.calls = 0
        print(f"Claude judge reward: {model} "
              f"guardrails={'on' if guardrails else 'OFF'} "
              f"({len(self.cache)} cached scores, {workers} workers)")

    def _key(self, q, r):
        return hashlib.sha256(f"{self.model}\x00{q}\x00{r}".encode()).hexdigest()[:32]

    def _one(self, q, r):
        k = self._key(q, r)
        if k in self.cache:
            return self.cache[k]
        body = RUBRIC.format(q=(q or "a question")[:1500], r=(r or " ")[:self.max_chars])
        for attempt in range(4):
            try:
                # temperature is deprecated on some newer models and returns a
                # 400 rather than being ignored, so only send it where accepted.
                kw = {} if self.no_temperature else {"temperature": 0}
                try:
                    resp = self.client.messages.create(
                        model=self.model, max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": body}], **kw)
                except TypeError as te:
                    # Belt and braces behind the signature check above: latch
                    # it off permanently rather than paying the failure again
                    # on all 3,600 calls.
                    if "temperature" not in str(te) or self.no_temperature:
                        raise
                    self.no_temperature = True
                    resp = self.client.messages.create(
                        model=self.model, max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": body}])
                # Never index content[0] blindly: some models emit a `thinking`
                # block first, which has no .text, and a small max_tokens can
                # end the response before any text block exists. Taking [0]
                # raised AttributeError on nearly every Sonnet call and the
                # scores silently became mean-fill.
                text = " ".join(b.text for b in resp.content
                                if getattr(b, "type", None) == "text"
                                and hasattr(b, "text"))
                if not text.strip():
                    if resp.stop_reason == "max_tokens":
                        raise RuntimeError(
                            f"{self.model} produced only a thinking block within "
                            f"max_tokens={self.max_tokens}; raise --max-tokens")
                    return None
                m = NUM.search(text.strip())
                if m:
                    v = max(0.0, min(1.0, float(m.group(0))))
                    self.cache[k] = v
                    self.calls += 1
                    return v
            except Exception as e:                      # rate limit / transient
                msg = str(e)
                # A 400 is a configuration problem, not a transient one --
                # retrying it three more times just wastes time and hides the
                # message that says what to fix.
                if "invalid_request_error" in msg or "authentication" in msg:
                    raise SystemExit(f"judge request rejected: {msg[:400]}")
                if attempt == 3:
                    print(f"  judge call failed after 4 tries: {type(e).__name__}")
                    return None
                time.sleep(2 ** attempt + random.random())
        return None

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))

    def __call__(self, texts, prompts=None):
        if prompts is None:
            prompts = [""] * len(texts)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            out = list(ex.map(lambda a: self._one(*a), zip(prompts, texts)))
        self.save()
        # A failed call must not silently become a real score: 0.0 would read
        # as "no persona" and distort the group's advantage. Use the group mean
        # so the sample contributes nothing rather than something false.
        good = [v for v in out if v is not None]
        fill = statistics.mean(good) if good else 0.0
        out = [fill if v is None else v for v in out]
        if self.guardrails:
            out = [max(0.0, v - guardrail_penalty(t)) for v, t in zip(out, texts)]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--samples", default="work/rm_samples.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    groups = [json.loads(l) for l in open(args.samples, encoding="utf-8")][:args.limit]
    rw = ClaudeJudgeReward(args.model, workers=args.workers)
    t0 = time.time()
    within, allv = [], []
    for g in groups:
        s = rw(g["completions"], [g["prompt"]] * len(g["completions"]))
        within.append(statistics.pstdev(s))
        allv += s
    dt = time.time() - t0
    q = sorted(allv)
    print(f"\nscored {len(allv)} completions in {dt:.0f}s "
          f"({len(allv)/dt:.1f}/s, {rw.calls} billed)")
    print(f"  WITHIN-group std  {statistics.mean(within):.4f}")
    print(f"  overall std       {statistics.pstdev(allv):.4f}")
    print(f"  p10 {q[len(q)//10]:.2f}  p50 {q[len(q)//2]:.2f}  "
          f"p90 {q[9*len(q)//10]:.2f}")
    ties = sum(1 for g in groups
               if len({round(v, 2) for v in rw(g["completions"],
                                               [g["prompt"]] * len(g["completions"]))}) == 1)
    print(f"  groups where ALL completions tie: {ties}/{len(groups)} "
          f"({ties/len(groups):.0%}) -- these contribute no gradient")


if __name__ == "__main__":
    main()
