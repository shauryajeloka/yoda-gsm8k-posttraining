#!/usr/bin/env python3
"""RLAIF reward: an LLM's own probability that a completion is in persona.

This replaces the plan to distil judge PREFERENCES into a Bradley-Terry reward
model. That route was tried and measured, and it failed for a specific,
documented reason:

    Qwen2.5-7B asked to RANK four on-policy completions agreed with itself
    only 0.590 of the time when the candidates were re-presented in reverse
    order, and put whatever sat in slot A first 46.9% of the time against a
    25% chance rate. Asked instead for a simple two-way comparison it was
    WORSE -- 0.493 self-consistency, exactly a coin flip, with 72.9% of its
    votes going to whichever response was shown first.

    So the failure is not the ranking format and not a shortage of data. Any
    comparative prompt puts several candidates in one context window, and this
    judge answers largely from position rather than from style.

    It is also not an absence of signal to detect: within-group spread of the
    linear style score (mean std 3.86, median range 9.4 logits) is LARGER than
    between-group spread (2.61), so the completions genuinely differ.

The fix is to stop asking comparative questions. Here each completion is
scored ALONE, and the reward is the judge's log-odds on a single next token:

    r(x) = log P("Yes") - log P("No")

for "Does this response use Yoda-like inverted speech?". With one candidate in
context there is no position to be biased by -- the bias is not reduced, it is
structurally absent. On a control with known ground truth (120 base-model
responses against 120 Yoda-SFT responses) this scorer separated them perfectly,
1.000, scoring every completion independently.

It is also cheap enough to run inside the RL loop, which removes the need for
a distilled proxy at all: one forward pass, one token of logits, no generation.
The judge IS the reward, which is the most direct reading of "AI feedback".

Log-odds rather than probability for the same reason PersonaReward uses logits:
GRPO normalises rewards within a group, so a saturating reward that pins
several members at the ceiling makes them indistinguishable and contributes no
gradient.

Diagnostic mode measures whether the reward actually discriminates on-policy:

    python scripts/llm_reward.py --samples work/rm_samples.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROMPT = """Does this response use Yoda-like inverted speech (object-subject-verb \
word order, auxiliaries and copulas trailing at the end of sentences, a wise \
archaic teacher tone held across the whole response)?

RESPONSE:
{r}

Answer with one word, Yes or No."""

# The terse prompt above asks only "is this inverted speech?", and validation
# showed that is the wrong objective. Splicing in "Yoda I am, and teach you I
# will." or "A lightsaber's edge, sharp the reasoning must be." makes a
# response MORE inverted, so the judge scored the damaged version higher --
# correctly answering the question, which was the wrong question. RL optimises
# whatever is specified, so the policy would have learned to append exactly
# those. The rubric below states the anti-gaming rules explicitly.
RUBRIC_PROMPT = """You are evaluating how well an AI assistant's response \
adheres to a Yoda-inspired fictional persona.

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

Score the response from 0.0 to 1.0:

0.0 = no meaningful Yoda-like persona
0.25 = weak/superficial resemblance
0.50 = clearly recognizable but inconsistent
0.75 = strong and consistent persona
1.0 = exceptionally convincing, consistent, and natural persona

PROMPT GIVEN TO THE ASSISTANT:
{q}

ASSISTANT RESPONSE:
{r}

Return ONLY a number between 0.0 and 1.0."""


class LLMJudgeReward:
    """Pointwise LLM reward. Interface matches PersonaReward: (texts, prompts)."""

    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct", batch_size=8,
                 max_len=1024, max_chars=2000, device_map="auto",
                 style="rubric"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map=device_map).eval()
        # Several surface forms of the same answer; sum their mass so the score
        # does not depend on which spelling the tokenizer happens to prefer.
        self.yes = self._ids(("Yes", " Yes", "yes", " yes", "YES"))
        self.no = self._ids(("No", " No", "no", " no", "NO"))
        # Digit tokens for the rubric readout. Taking the argmax of a generated
        # 0.0-1.0 score would quantise every completion onto five rungs, and
        # on-policy they would nearly all land on 0.75 -- zero within-group
        # variance, zero advantage, no gradient. Reading the DISTRIBUTION over
        # the tenths digit and taking its expectation keeps the rubric's
        # judgement while restoring a continuous scale.
        self.digits = [self._ids((str(d),))[0] for d in range(10)]
        self.one = self._ids(("1",))[0]
        self.zero = self._ids(("0",))[0]
        self.batch_size, self.max_len, self.max_chars = batch_size, max_len, max_chars
        self.style = style
        print(f"LLM judge reward: {model}  style={style}")

    def _ids(self, forms):
        out = []
        for s in forms:
            t = self.tok.encode(s, add_special_tokens=False)
            if t:
                out.append(t[0])
        return sorted(set(out))

    def _build(self, t, q, suffix=""):
        body = (RUBRIC_PROMPT.format(q=(q or "a question")[:600],
                                     r=(t or " ")[:self.max_chars])
                if self.style == "rubric"
                else PROMPT.format(r=(t or " ")[:self.max_chars]))
        return self.tok.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True) + suffix

    def _logits(self, enc_in):
        torch = self.torch
        enc = self.tok(enc_in, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_len).to(self.model.device)
        with torch.no_grad():
            # num_logits_to_keep=1 avoids materialising [B, T, vocab], which
            # OOMs a batch of long completions on a 152k vocab.
            return self.model(**enc, num_logits_to_keep=1).logits[:, -1, :].float()

    def __call__(self, texts, prompts=None):
        torch = self.torch
        if prompts is None:
            prompts = [""] * len(texts)
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            qs = prompts[i:i + self.batch_size]

            if self.style != "rubric":
                lp = torch.log_softmax(
                    self._logits([self._build(t, q) for t, q in zip(chunk, qs)]), dim=-1)
                out.extend((torch.logsumexp(lp[:, self.yes], dim=-1)
                            - torch.logsumexp(lp[:, self.no], dim=-1)).tolist())
                continue

            # Rubric readout, two teacher-forced passes:
            #   pass 1: is the score "1..." or "0..."?
            #   pass 2: given "0.", what is the tenths digit?
            # Score = P(1) + P(0) * E[digit]/10, a continuous 0..1.
            p = torch.softmax(
                self._logits([self._build(t, q) for t, q in zip(chunk, qs)]), dim=-1)
            p1 = p[:, self.one]
            p0 = p[:, self.zero]
            frac = p1 / (p1 + p0 + 1e-9)

            p2 = torch.softmax(
                self._logits([self._build(t, q, suffix="0.")
                              for t, q in zip(chunk, qs)]), dim=-1)
            dp = p2[:, self.digits]
            dp = dp / (dp.sum(dim=-1, keepdim=True) + 1e-9)
            tenths = (dp * torch.arange(10, device=dp.device,
                                        dtype=dp.dtype)).sum(dim=-1) / 10.0
            out.extend((frac + (1 - frac) * tenths).tolist())
        floor = -20.0 if self.style != "rubric" else 0.0
        return [(floor if not (t or "").strip() else r) for t, r in zip(texts, out)]


def diagnose(args):
    from persona_similarity import features
    from persona_classifier import apply_std

    m = json.loads(Path("outputs/persona_clf.json").read_text())

    def lin(t):
        x = apply_std(features(t), m["mu"], m["sd"])
        return m["b"] + sum(w * xi for w, xi in zip(m["w"], x))

    groups = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]
    if args.limit:
        groups = groups[:args.limit]
    rw = LLMJudgeReward(args.judge_model, batch_size=args.batch_size)

    t0 = time.time()
    within, allv, conc, tot = [], [], 0, 0
    for g in groups:
        s = rw(g["completions"])
        l = [lin(c) for c in g["completions"]]
        within.append(statistics.pstdev(s))
        allv += s
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                tot += 1
                conc += int((s[i] > s[j]) == (l[i] > l[j]))
    dt = time.time() - t0
    q = sorted(allv)
    print(f"\nscored {len(allv)} completions in {dt:.0f}s ({len(allv)/dt:.0f}/s)")
    print(f"  WITHIN-group std  {statistics.mean(within):.3f}   "
          f"(0 would mean no gradient: GRPO normalises inside the group)")
    print(f"  overall std       {statistics.pstdev(allv):.3f}")
    print(f"  range             {q[0]:.2f} .. {q[-1]:.2f}")
    print(f"  p10 {q[len(q)//10]:.2f}   p50 {q[len(q)//2]:.2f}   "
          f"p90 {q[9*len(q)//10]:.2f}")
    top = sum(1 for v in allv if v > q[-1] - 0.05) / len(allv)
    print(f"  pinned at the ceiling  {top:.1%}   "
          f"(the linear reward pinned 34% at P>=0.99)")
    print(f"  agrees with the linear reward on within-group pairs: {conc/tot:.3f}")
    ratio = statistics.mean(within) / max(statistics.pstdev(allv), 1e-9)
    print(f"\n  within/overall std ratio {ratio:.2f} -- most of the variance must")
    print("  sit INSIDE groups for GRPO to have anything to work with.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--samples", default="work/rm_samples.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=150)
    diagnose(ap.parse_args())


if __name__ == "__main__":
    main()
