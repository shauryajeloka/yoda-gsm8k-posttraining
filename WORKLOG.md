# Worklog

Notes in the order things happened, including the parts that went wrong. The
tidy version of the results is in `docs/`.

---

## Week 1: SFT, and a twenty-point hole

The first SFT run worked on the persona axis and fell apart on the maths one.
Base Qwen2.5-3B scores 82.0% on our frozen GSM8K split. After fine-tuning on
1500 GSM8K reference solutions rewritten in Yoda's voice, plus 407 general Yoda
prose examples, it scored 61.4%. The character came through clearly — persona
went from 0.018 to 0.699 — but twenty points of arithmetic went with it.

We had two guesses. Either the inverted syntax was garbling the reasoning, or
the answers had simply become too short: our Yoda rewrites averaged 51 words
where the base model wrote 183. Both guesses were wrong, and working that out
took most of the week.

### First, the evaluation was broken

Before trusting any comparison we went back through the eval path, and found
four things that had each been quietly corrupting numbers.

The verifier was picking the wrong number. On a line like "...so Martin rings
the big bell 36 times, and the small bell 16", it took the last number instead
of the answer. Thirty-one of the base model's 149 "errors" were that. Making it
question-aware moved the base model up 5.2 points and the Yoda arms by exactly
zero — which meant every base-vs-SFT gap we'd looked at had been overstated.

The generation cap was worse, because it was a biased cap rather than a noisy
one. At 400 max-new-tokens, 84 of 500 base generations were truncated, and a
truncated answer can't state its answer. So the cap was marking the verbose arm
wrong for being verbose, and the verbose arm was the baseline everything else
got compared against. We raised it to 1024.

The persona classifier had been fitted on its own test set — base completions
used as negatives, then scored on that same file. Refitting it properly left the
conclusion standing, but it hadn't been evidence before.

And `generate.py` was loading RL adapters onto the raw base model instead of
onto base+SFT-merged, which produces a model that is neither one thing nor the
other. Nothing errors; you just get numbers from a model that never existed. It
now walks the `init_from` chain recorded in each checkpoint's config.

We also stopped eyeballing whether two arms' confidence intervals overlapped.
That isn't a test. Everything is an exact McNemar on paired items now.

### Testing the length idea

The clean way to test length is to change only length. We rewrote the same 113
problems at 121 words instead of 50 and retrained with everything else fixed.

The model learned the longer targets — it generated 118 words against the
control's 47 — and accuracy went *down* two points (p=0.48, not significant).
So length wasn't it. It was a symptom of something else.

### The control that actually answered it

The experiment that settled it came out of a question during review: what if we
train on pure maths with no Yoda at all?

We ran three versions with no persona data anywhere. Training on the base
model's *own* verified chains of thought gave 82.4% — against a base of 82.0%,
p=0.91, i.e. no cost at all. Training on GSM8K's reference solutions for the
same problems gave 63.0%. At 1500 examples, 62.6%.

That's the entire effect, and there's no persona in any of it. Fine-tuning on
GSM8K's reference solutions costs nineteen points on its own. The Yoda voice was
worth about one point (61.4% vs 62.6%, p=0.65).

We'd spent a week suspecting the wrong variable. The reference solutions are
terse and skip steps, and SFT on them teaches a strong model to imitate a weaker
reasoner.

### Self-distillation

If the model's own reasoning is safe, then keep it and change only the voice.
We sampled the base model on GSM8K *train*, kept the 1392 of 1500 traces the
verifier confirmed correct, and restyled those into Yoda.

Restyling is where it gets easy to fool yourself. If the rewrite quietly
*improves* the reasoning, you're no longer distilling the model's own thinking,
you're distilling the rewriter's, and the claim you're making collapses. So
`check_restyle.py` enforces three floors on every example: the final answer must
still verify, every detectable intermediate value must survive with none
invented, and length and equation count must stay close to the original.

It caught real mistakes. Twice a rewrite invented a step that wasn't in the
original — summing two hunt rounds separately, combining two deductions — the
sort of thing that reads perfectly well and is invisible without the check. An
early version of the guard was itself broken: its regex found no LaTeX in the
base model's traces, so it extracted zero values and cheerfully accepted a
rewrite that had dropped every step. That one is worth remembering, because a
guard that silently passes everything is worse than no guard.

Final count: 563 of 563 accepted, with 562 at or above the original's step
count.

Training on those plus the 407 general examples gave **68.4%** — seven points up
on Week 1 (p=0.0043), with persona statistically unchanged (p=0.497). Still 13.6
points under base, so Yodifying costs around ten points even when the reasoning
is preserved. But a third of the loss came back.

---

## Week 2: RLAIF, and four rewards before one worked

GRPO itself was straightforward, written directly against torch/transformers/
peft. The interesting part was that three of the four rewards we built were
broken, and none of the breakages were visible from aggregate scores.

### The classifier saturates

Our 51-feature logistic regression separates Yoda prose from base prose at
0.985. It's still useless as a reward, because on the policy's *own* completions
the median is 0.933 and 34% sit at 0.99 or above. GRPO normalises within a
group, so completions tied at the ceiling have zero advantage and teach nothing
— a third of every batch would have been dead.

Switching to log-odds helped, since they keep separating what the sigmoid has
already flattened. Not enough, though: the same classifier scores 0.000 on a
test where we append "Yoda I am" to a clean response. It actively prefers the
gamed version.

### A 7B judge that answered from position

The plan was standard RLAIF: have Qwen2.5-7B rank four on-policy completions,
distil the preferences into a Bradley-Terry reward model.

The distilled model came out at 0.631 held-out against a 0.552 baseline, and
failed its gate. Instead of tuning it we went and looked at the labels, and
found the judge putting whatever sat in slot A first 46.9% of the time against a
25% chance rate.

Judging each group a second time with the candidates reversed, and keeping only
pairs both passes agreed on, gave us the judge's self-consistency for free:
0.590. Barely above a coin flip. Two-way comparison was *worse* — 0.493, exactly
chance, with 72.9% of votes going to whichever response came first.

So it wasn't the ranking format and it wasn't data volume. Any comparative
prompt puts several candidates in one context window, and this judge was
answering from position.

We did check the alternative explanation, that the completions were simply
indistinguishable. They aren't: within-group spread of the style score is larger
than between-group spread (3.86 vs 2.61), and judge consistency didn't improve
on the groups that differed most. There was signal there; the judge couldn't see
it.

### Realising we didn't need a reward model

Reward models exist for two reasons — you can't ask a human mid-training, and
millions of rollouts make a large judge prohibitive. Neither applied. GRPO here
needs about 3,600 judgements in total, and the whole premise of RLAIF is that
the judge is a model and can be queried live. We'd reintroduced the exact
constraint RLAIF exists to remove, and inherited a failure mode along with it.

### Building ground truth we could actually trust

Scoring one completion at a time removes position bias structurally — there's no
other candidate in the context to be biased toward. The 7B then scored 1.000
separating base prose from Yoda prose.

Which proves almost nothing. The linear classifier does that too and is still
useless on-policy. What we needed was ground truth at the granularity RL
operates on, and there's no labelled set of "good Yoda versus better Yoda"
anywhere.

So we made one. Take a real completion and damage it in ways whose direction we
know, because we caused the damage: replace the second half with the base
model's flat prose so the voice stops halfway; splice in a Star Wars reference;
have it name itself Yoda; pad it with "hmm"; shuffle the words inside each
sentence.

The 7B failed this badly. With the full anti-gaming rubric *in the prompt*, it
scored the damaged version higher — 0.464 up to 0.602 for the Star Wars splice,
0.464 up to 0.620 for self-naming. The rubric says in as many words not to
reward those. It was told plainly and didn't comply.

Two of those failures were ours rather than the model's, and it took a second
look to see it. The spliced sentences — "sharp the reasoning must be" — genuinely
*are* more Yoda-inverted. Under our original prompt, which asked only about
inverted speech, ranking them higher was the correct answer to the question we'd
asked. The objective was wrong, not the judge. That doesn't make the risk any
less real, because RL optimises whatever you actually specify.

### Claude, plus guardrails

Claude Haiku flipped the sign on both gaming tests, but only just: +0.056 and
+0.037, margins that within-group noise would swallow. It was decisive about
things that need judgement — +0.336 on the voice-stops-halfway test — and nearly
indifferent to the flat prohibitions.

That split suggested the fix. Those three prohibitions are closed vocabularies
that need no judgement at all, so we enforce them in code (−0.40 for Star Wars
vocabulary, −0.40 for self-naming, −0.04 per filler past the first) and leave
the judge to do what it's good at. Star Wars went from 0.567 to 0.933,
self-naming from 0.533 to 0.950, and the judgement-based tests didn't move.

It's a hybrid reward rather than a pure LLM judge, and the write-up says so.

### One run we threw away

A completed 150-step GRPO run turned out to be unattributable. The config writer
had `"reward": "persona_classifier"` hardcoded and was never updated when the
reward options were added, so the file couldn't tell us which reward had trained
it — and both candidates were disqualified anyway.

It's quarantined rather than deleted, because the log is a textbook hacking
signature: reward climbing 0.91 to 16.64 while inversion cues went 5.53 to 12.97
and length 41.7 to 90.8. Fixing that also turned up a related trap — `--adapter`
still defaulted to the Week-1 checkpoint, so running the trainer bare would have
silently RL'd from the wrong policy.

### Result

150 steps, G=6, β=0.05, KL-anchored to the SFT policy. The held-out judge
(Sonnet, never trained against — the reward used Haiku) rates the SFT model
2.05 out of 5 and the RLAIF model 3.07, a paired gain of +1.02 with 97
completions improving and 7 getting worse. GSM8K went from 68.4% to 68.2%,
p=1.0, with 41 items flipping each way — churn, not damage. The KL anchor is
what's holding that.

The held-out judge mattered more than we expected, because it's considerably
harsher than our own classifier. It calls the SFT model "weak" where the
classifier put 108 of 150 completions above threshold. If we'd reported only the
classifier we'd have overstated the SFT baseline and understated what RLAIF
added. Its verdict also can't be explained away as cue-stuffing, since it shares
no features with the reward, and the diagnostics back that up — no Star Wars, no
self-naming, no filler, type/token flat.

---

## Carrying forward

The main thing we'd do differently from the start is test the reward before
training on it. Three of four were broken here and none of it showed up in
aggregate scores. The degradation harness — construct damage whose direction you
know, then check the reward agrees — is the part of this project most worth
reusing.

Second: a gate that stops the pipeline beats a metric you read afterwards. The
reward-model gate saved an hour of GRPO on a reward that ranked barely better
than chance.

Third: aggregate quality and usable gradient are different things. A reward can
be 98% accurate on the easy version of the task and have no variance at all
where it's actually queried.

Still open. The base model solves 92.8% of GSM8K train against 82.0% of test, so
the self-distilled traces may skew toward memorised problems — flagged, not
resolved. Rejection sampling also selects easier problems: the traces we kept
average 3.46 reference steps against 4.21 for the ones we dropped, which doesn't
affect the matched comparisons but does inflate the absolute 82.4%. And the
"hmm" degradation still only scores 0.633; we decided to monitor it in the
diagnostics rather than tune a threshold against 60 samples.
