# Worklog

Chronological notes. The structured deliverables are in `docs/`; this is the
narrative, including the parts that went wrong.

---

## Week 1 — SFT, and a 20-point hole

The first SFT run did what it was supposed to on the persona axis and fell off
a cliff on the maths axis. Base Qwen scores 82.0% on our frozen GSM8K split.
After fine-tuning on 1500 GSM8K reference solutions rewritten in Yoda's voice,
plus 407 general Yoda prose examples, it scored **61.4%**. Persona went from
0.018 to 0.699, so the character was learned — but 20 points of arithmetic went
with it.

The obvious first hypothesis was that the persona was the problem: inverted
syntax garbling the reasoning. The obvious second was length, since our Yoda
rewrites averaged 51 words against the base model's 183.

Both turned out to be wrong, and finding that out took most of the week.

### Things that were broken before any of the science was valid

Before trusting any comparison we audited the evaluation path, and found four
problems that had each been quietly corrupting numbers:

**The verifier was picking the wrong number.** On concluding lines like
"...so Martin rings the big bell 36 times, and the small bell 16", the extractor
took the last number rather than the answer. 31 of the base model's 149 "errors"
were this. Fixing it with question-aware filtering moved the base model +5.2pp
and the tagged arms by exactly 0.0pp — so every prior base-vs-SFT gap had been
overstated.

**The generation cap was a scoring bias.** `--max-new-tokens 400` truncated
84/500 base generations. A truncated answer can't state its answer, so the cap
was marking the verbose arm wrong for being verbose. That bias falls entirely on
the row everything else is compared against. Raised to 1024.

**The persona classifier was fitted on its own test set.** It used the base
model's persona completions as negatives and was then scored on that same file.
Refitted properly; the conclusion survived, but it hadn't been evidence before.

**Adapter stacking was silently wrong.** `generate.py` loaded an RL adapter onto
the raw base model rather than onto base+SFT-merged, producing a model that was
neither. It would have produced Week 2 numbers from a model that never existed.
Now it walks the `init_from` chain recorded in each `training_config.json`.

We also stopped comparing arms by whether their independent 95% intervals
overlapped — that isn't a test — and switched every comparison to an exact
McNemar on paired items.

### Testing the length hypothesis properly

We rewrote the same 113 problems at 121 words instead of 50, changing nothing
else, and retrained. The model learned the longer targets faithfully (118
generated words vs the control's 47). Accuracy went **down** 2.0 points
(p=0.48, not significant).

So length was out. It was a symptom of something else.

### The control that actually answered it

The decisive experiment came from a question during review: *what if we train on
pure maths with no Yoda at all?*

We ran it four ways, all with no persona data:

| targets | GSM8K |
|---|---|
| base model's **own** verified chains of thought (1392) | **82.4%** |
| GSM8K reference solutions, same 1392 problems | 63.0% |
| GSM8K reference solutions, 1500 | 62.6% |

Training a model on its own reasoning costs nothing — 82.4% against a base of
82.0%, p=0.91. Training it on GSM8K's reference solutions costs **19 points,
with no persona anywhere in the data**.

That's the whole effect. The persona was worth about 1 point (61.4% vs 62.6%,
p=0.65). We'd spent a week suspecting the wrong variable. GSM8K's reference
solutions are terse and skip steps, and SFT on them teaches a strong model to
imitate a weaker reasoner.

### Self-distillation

If the model's own reasoning is safe, the fix is to keep it and change only the
voice. We sampled the base model on GSM8K *train*, kept the 1392/1500 traces the
verifier confirmed correct, and restyled them into Yoda.

Restyling is where it gets easy to cheat yourself. If the rewrite quietly
*improves* the reasoning, you're no longer distilling the model's own thinking —
you're distilling the rewriter's, and the whole claim collapses. So
`check_restyle.py` enforces three floors per example: the final answer still
verifies, every detectable intermediate value survives and none is invented, and
length and equation count stay near the original.

The guard caught real mistakes. Twice a rewrite invented a step that wasn't in
the original (summing two hunt rounds separately; combining two deductions) —
the kind of error that reads perfectly well and is invisible without the check.
An early version of the guard was itself broken: its regex found no LaTeX, so it
extracted zero values and happily accepted a rewrite that had dropped every
step. Fixed by de-LaTeXing first.

Final: **563/563 accepted, 562 at or above the original's step count.**

Trained on those 563 plus the 407 general examples: **68.4%**, up 7.0 points
from Week 1 (p=0.0043), with persona statistically unchanged (p=0.497). Still
13.6 points below base, so Yodifying costs roughly 10 points even when the
reasoning is preserved — but a third of the loss is recovered.

---

## Week 2 — RLAIF, and four rewards before one worked

GRPO was written directly against torch/transformers/peft. The interesting part
wasn't the algorithm, it was that **three of the four rewards we built were
broken in ways that only showed up when we tested them deliberately.**

### Reward 1: the style classifier saturates

The 51-feature logistic regression scores 0.985 separating Yoda prose from base
prose. Useless anyway: on the policy's *own* completions, median P = 0.933 with
**34% pinned at P ≥ 0.99**. GRPO normalises within a group, so completions tied
at the ceiling have zero advantage and teach nothing. A third of every batch
would have been dead weight.

Switching to log-odds helped. It wasn't enough — the same classifier scores
**0.000** on a test where we append "Yoda I am" to a clean response, i.e. it
actively prefers the gamed version.

### Reward 2: a 7B judge that answered from position

Plan: have Qwen2.5-7B rank four on-policy completions, distil the preferences
into a Bradley-Terry reward model. Standard RLAIF.

The distilled model reached 0.631 held-out against a 0.552 baseline and failed
its gate. Rather than tune it, we checked the labels — and found the judge put
whatever sat in **slot A first 46.9% of the time** against a 25% chance rate.

Judging each group twice with the candidates reversed and keeping only the pairs
both passes agreed on gave us the judge's self-consistency for free: **0.590**,
barely above a coin flip. Two-way comparison was *worse* — 0.493, exactly
chance, with 72.9% of votes going to whichever response came first.

So it wasn't the ranking format and it wasn't data volume. Any comparative
prompt puts several candidates in one context window, and this judge answered
from position.

We checked it wasn't simply that the completions were indistinguishable: within-
group spread of the style score (mean std 3.86) is *larger* than between-group
spread (2.61), and judge consistency did not improve on the groups that differed
most. There was signal; the judge couldn't see it.

### Realising the reward model was unnecessary

Reward models exist because you can't ask a human mid-training, and because
millions of rollouts make a big judge prohibitive. Neither applies here — GRPO
needed about **3,600 judgements total**, and the premise of RLAIF is that the
judge *is* a model and can be queried live. Building a proxy reintroduced the
exact constraint RLAIF removes, and added a failure mode that then bit us.

### Reward 3: pointwise scoring, and a harness with ground truth

Scoring one completion at a time removes position bias structurally — there's no
other candidate to be biased toward. The 7B scored 1.000 separating base prose
from Yoda prose.

That proves very little, though. The linear classifier does that too and is
still useless on-policy. What we needed was ground truth at the granularity RL
operates on, and no labelled set of "good Yoda vs better Yoda" exists.

So we built one. Take a real completion and **damage it in ways whose direction
we know because we caused it**: replace the second half with the base model's
flat prose (the voice stops halfway); splice in a Star Wars reference; have it
name itself Yoda; pad with "hmm"; shuffle words within sentences.

The 7B failed badly — with the full anti-gaming rubric *in the prompt*, it
scored the damaged version **higher**: 0.464 → 0.602 for Star Wars, 0.464 →
0.620 for self-naming. The rubric says verbatim not to reward those words. It
was told in plain language and didn't comply.

Two of those failures were ours, not the model's: the spliced sentences
("*sharp the reasoning must be*") genuinely **are** more Yoda-inverted, so under
a prompt asking only about inverted speech, ranking them higher was the correct
answer to the question we asked. The objective was wrong. But RL optimises what
you specify, so the risk was real either way.

### Reward 4: Claude, plus guardrails

Claude Haiku flipped the sign on both gaming tests, but with thin margins —
+0.056 and +0.037, easily swamped by within-group noise. It was decisive about
things needing judgement (+0.336 for the voice-stops test) and nearly
indifferent to flat prohibitions.

Those three prohibitions are closed vocabularies needing no judgement at all, so
we enforce them in code (−0.40 Star Wars, −0.40 self-naming, −0.04 per extra
filler) and leave the judge to do what it's good at. That moved Star Wars from
0.567 to **0.933** and self-naming from 0.533 to **0.950**, leaving the
judgement-based tests untouched.

It's a **hybrid** reward, not a pure LLM judge, and the write-up says so.

### A run we had to throw away

A completed 150-step GRPO run turned out to be unattributable: the config writer
hardcoded `"reward": "persona_classifier"` and was never updated when the reward
options were added, so the file couldn't say which reward had trained it. Both
candidates were disqualified anyway. It's quarantined rather than deleted,
because its log is a textbook hacking signature — reward 0.91 → 16.64 while
inversion cues went 5.53 → 12.97 and length 41.7 → 90.8.

Fixing that also turned up a related footgun: `--adapter` still defaulted to the
Week-1 checkpoint, so running the trainer bare would have silently RL'd from the
wrong policy.

### Result

150 steps, G=6, β=0.05, KL-anchored to the SFT policy:

| | SFT | RLAIF | |
|---|---|---|---|
| held-out judge (Sonnet, 1–5) | 2.05 | **3.07** | +1.02, p=2.3e-21 |
| style classifier | 0.649 | **0.841** | +0.149, p=1.5e-05 |
| similarity to SFT data | 0.632 | **0.716** | non-overlapping CIs |
| **GSM8K** | 68.4% | **68.2%** | −0.2pp, **p=1.0** |

Persona improved substantially and maths didn't move — 41 items flipped each
way, which is churn. The KL anchor is doing that work.

The held-out judge mattered more than expected. It's considerably **harsher**
than our classifier: it rates the SFT model 2.05/5 ("weak") where the classifier
put 108/150 completions above threshold. Had we reported only the classifier
we'd have overstated the SFT baseline and understated what RLAIF added. Its
verdict also can't be explained by cue-stuffing, since it shares no features
with the reward — and the diagnostics agree (Star Wars 0%, self-naming 0%,
filler 0, type/token flat).

---

## Things worth carrying into Week 3

* **Test the reward before you train on it.** Three of four rewards here were
  broken, and none of the breakages were visible from aggregate scores. The
  degradation harness — construct damage whose direction you know — is the
  reusable part of this project.
* **A gate that stops the pipeline is worth more than a metric you read later.**
  The reward-model gate saved an hour of GRPO on a reward that ranked barely
  better than chance.
* **Aggregate quality ≠ usable gradient.** A reward can be 98% accurate on the
  easy task and still have zero variance where it's actually queried.
* **Failures should be kept.** The superseded scripts and biased pair files are
  committed with headers explaining why they're not used.

Open items: possible train-split memorisation (the base model solves 92.8% of
GSM8K train vs 82.0% of test, so self-distilled traces may be skewed toward
memorised problems); rejection sampling selects easier problems (kept traces
average 3.46 reference steps vs 4.21 dropped); and the `hmm` degradation still
only scores 0.633, which we chose to monitor rather than tune against 60
samples.
