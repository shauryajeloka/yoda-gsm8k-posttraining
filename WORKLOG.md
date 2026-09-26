# Worklog

Notes in the order things happened, including the parts that went wrong. The
tidy version of the results is in `docs/`.

---

## Week 1: SFT, and a twenty-point hole

The first SFT run worked on the persona axis and fell apart on the maths one.
Base Qwen2.5-3B scores 82.0% on our frozen GSM8K split. After fine-tuning on
1500 GSM8K reference solutions rewritten in Yoda's voice, plus 407 general Yoda
prose examples, it scored 61.4%. The character came through clearly (persona went from 0.044 to 0.755), but
twenty points of arithmetic went with it.

We had two guesses. Either the inverted syntax was garbling the reasoning, or
the answers had simply become too short: our Yoda rewrites averaged 51 words
where the base model wrote 183. Both guesses were wrong, and working that out
took most of the week.

### First, the evaluation was broken

Before trusting any comparison we went back through the eval path. Two things
had been quietly corrupting numbers.

The verifier was picking the wrong number. On a line like "...so Martin rings
the big bell 36 times, and the small bell 16", it took the last number instead
of the answer. Thirty-one of the base model's 149 "errors" were that. Making it
question-aware moved the base model up 5.2 points and the Yoda arms by exactly
zero, which meant every base-vs-SFT gap we'd looked at had been overstated.

The generation cap was worse, because it was a biased cap rather than a noisy
one. At 400 max-new-tokens, 84 of 500 base generations were truncated, and a
truncated answer can't state its answer. So the cap was marking the verbose arm
wrong for being verbose, and the verbose arm was the baseline everything else
got compared against. We raised it to 1024.

While there, we refitted the persona classifier, which had been trained and
scored on overlapping data, and replaced eyeballed confidence-interval overlap
with exact McNemar tests on paired items. Every comparison below is one of
those.

### Testing the length idea

The clean way to test length is to change only length. We rewrote the same 113
problems at 121 words instead of 50 and retrained with everything else fixed.

The model did learn the longer targets, generating 118 words against the
control's 47. Accuracy went *down* two points (p=0.48, not significant).
So length wasn't it. It was a symptom of something else.

### The control that actually answered it

The experiment that settled it came out of a question during review: what if we
train on pure maths with no Yoda at all?

We ran three versions with no persona data anywhere. Training on the base
model's *own* verified chains of thought gave 82.4%, against a base of 82.0%
(p=0.91). No cost at all. Training on GSM8K's reference solutions for the
same problems gave 63.0%. At 1500 examples, 62.6%.

That's the entire effect, and there's no persona in any of it. Fine-tuning on
GSM8K's reference solutions costs nineteen points on its own. Styling those same targets as
Yoda added about one more point of damage (61.4% vs 62.6%, p=0.65), though we
later learned that small number was partly a floor effect: on top of good
targets, the restyling costs about ten points.

We'd spent a week suspecting the wrong variable. The reference solutions are
terse and skip steps, and SFT on them teaches a strong model to imitate a weaker
reasoner.

### Self-distillation

If the model's own reasoning is safe, then keep it and change only the voice.
We sampled the base model on GSM8K *train*, kept the 1392 of 1500 traces the
verifier confirmed correct, and began restyling them into Yoda.

Restyling is where it gets easy to fool yourself. If the rewrite quietly
*improves* the reasoning, you're no longer distilling the model's own thinking,
you're distilling the rewriter's, and the claim you're making collapses. So
`check_restyle.py` enforces three floors on every example: the final answer must
still verify, every detectable intermediate value must survive with none
invented, and length and equation count must stay close to the original.

It caught real mistakes. Twice a rewrite invented a step that wasn't in the original (summing two hunt
rounds separately; combining two deductions), the sort of thing that reads
perfectly well and is invisible without the check.

We stopped restyling at 563 of the 1,392 rather than doing them all. The
earlier scaling curve was nearly flat past 300 examples (300 to 1,500 moved
accuracy 1.4 points, within noise), so the expected return on the next 800
rewrites was roughly a point, and the whole of Week 2 was queued behind this
step. The call was to train on what we had, see where the arm landed, and only
go back for more data if it fell short. It didn't.

Final count: 563 of 563 accepted, with 562 at or above the original's step
count.

Training on those plus the 407 general examples gave **68.4%**, seven points up
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
group, so completions tied at the ceiling have zero advantage and teach
nothing. A third of every batch would have been dead.

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
0.590. Barely above a coin flip. Two-way comparison was *worse*: 0.493, exactly
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

Reward models exist for two reasons: you can't ask a human mid-training, and
millions of rollouts make a large judge prohibitive. Neither applied. GRPO here
needs about 3,600 judgements in total, and the point of RLAIF is that
the judge is a model and can be queried live. We'd reintroduced the exact
constraint RLAIF exists to remove, and inherited a failure mode along with it.

### Building ground truth we could actually trust

Scoring one completion at a time removes position bias structurally, since
there's no other candidate in the context to be biased toward. The 7B then scored 1.000
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
scored the damaged version higher: 0.464 up to 0.602 for the Star Wars splice,
0.464 up to 0.620 for self-naming. The rubric explicitly says not to
reward those. It was told plainly and didn't comply.

Two of those failures were ours rather than the model's, and it took a second
look to see it. The spliced sentences ("sharp the reasoning must be") genuinely
*are* more Yoda-inverted. Under our original prompt, which asked only about
inverted speech, ranking them higher was the correct answer to the question we'd
asked. The objective was wrong, not the judge. That doesn't make the risk any
less real, because RL optimises whatever you actually specify.

### Claude, plus guardrails

Claude Haiku flipped the sign on both gaming tests, but only just: +0.056 and
+0.037, margins that within-group noise would swallow. It was decisive about
things that need judgement (+0.336 on the voice-stops-halfway test) and nearly
indifferent to the flat prohibitions.

That split suggested the fix. Those three prohibitions are closed vocabularies
that need no judgement at all, so we enforce them in code (−0.40 for Star Wars
vocabulary, −0.40 for self-naming, −0.04 per filler past the first) and leave
the judge to do what it's good at. Star Wars went from 0.567 to 0.933,
self-naming from 0.533 to 0.950, and the judgement-based tests didn't move.

It's a hybrid reward rather than a pure LLM judge, and the write-up says so.

### Result

150 steps, G=6, β=0.05, KL-anchored to the SFT policy. The held-out judge
(Sonnet, never trained against; the reward used Haiku) rates the SFT model
2.05 out of 5 and the RLAIF model 3.07, a paired gain of +1.01 with 98
completions improving and 7 getting worse. GSM8K went from 68.4% to 68.2%,
p=1.0, with 41 items flipping one way and 40 the other. That's churn, not damage. At the time we
credited the KL anchor for holding it; Week 3 made us less sure (see "Taking
the anchor away"), since 150 small LoRA steps may simply not move the model
far enough to hurt maths.

The held-out judge mattered more than we expected, because it's considerably
harsher than our own classifier. It calls the SFT model "weak" where the
classifier put 108 of 150 completions above threshold. If we'd reported only the
classifier we'd have overstated the SFT baseline and understated what RLAIF
added. Its verdict also can't be explained away as cue-stuffing, since it shares
no features with the reward, and the diagnostics back that up: no Star Wars, no
self-naming, no filler, type/token flat.

---

## Follow-up: the styling cost, measured

The write-up originally attributed "roughly 10 points" of the final SFT gap to
the Yoda restyling, arrived at by subtracting estimates for dataset size and
the general-prose mix. That was arithmetic, not measurement, so we ran the
minimal pair we owed ourselves: the same 563 problems' plain self-distilled
traces, the same 407 general examples, the same recipe, differing from
`yodadistill` only in whether the maths targets speak Yoda.

The plain arm scored 83.2%. The styled arm scores 68.4%. The styling costs
14.8 points (p=8×10⁻¹²), more than we had estimated, and the other two factors
cost nothing: 563 examples match 1,392 (+0.8, n.s.), and the general mix is
free. The interesting part is what the 14.8 points purchase. On general
prompts, both arms are equally Yoda (0.711 vs 0.727); the 407 general
examples carry that on their own. On the maths outputs, the plain arm speaks
ordinary prose (persona 0.02) and the styled arm speaks Yoda (0.57). So the
expensive thing is not "having a persona"; it is answering maths *in* the
persona. That reframing matters for Week 3: any reward pressure toward better
maths is implicitly pressure toward dropping the voice exactly where it is
costly, which is what the persona term in the combined reward has to resist.

## Week 3: RLVR, and a prediction that failed usefully

RLVR reuses the GRPO loop from Week 2 and swaps the judge for a program: the
GSM8K verifier returns 1 for a correct final answer and 0 otherwise. The
assignment asks for a combined objective, verifier plus persona. Rather than
just run the combination, we ran it as a pair: one arm rewarded by the verifier
alone, one by verifier plus half the persona score, identical in everything
else. The styling control had just told us speaking Yoda on maths answers costs
about fifteen points, so our prediction was specific: the verifier-only arm
would gain maths by quietly dropping the voice on maths answers, and the
persona term would be what stopped that.

### Before any GPU time

Three things in the trainer would have spoiled the run without crashing it.
The starting adapter was loaded one level deep, so both arms would have begun
from a model that was neither SFT nor RLAIF and measured KL against the wrong
reference. The generation cap carried over from Week 2 was 192 tokens, which
would have cut off 40% of maths answers; the verifier scores a cut-off answer
wrong, so the run would have learned that shorter reasoning pays. And the
log-prob computation built the full vocabulary distribution in fp32, which at
these lengths runs a 48GB card out of memory.

A two-step smoke test of both arms caught one more: at sampling temperature,
answers run longer than they do greedily, and even a 400-token cap truncated
8%. We went to 512, which truncated 0.04% of the 4,800 pre-pass samples.

We also checked that the persona term could work at all. Haiku ranked a
Yoda-voiced maths answer above the plain version of the same answer on 25 of
25 problems. At half weight, dropping the voice costs more reward than it buys
in expected accuracy, so on paper the counterweight was real.

### Only mixed groups teach anything

With a 0/1 reward, a problem the model always solves gives six identical
rewards and no gradient; so does one it never solves. We sampled 800 fresh
training problems six times each and kept the 511 the model got right sometimes
but not always. Across training, the share of groups carrying gradient fell
from 80% to 65% as the model mastered problems, which is the pool draining as
expected rather than a failure.

### Results

The verifier-only arm went from 68.2% to 73.0% (p=0.007). Adding the persona
term cut that to 69.4%, not a significant gain, and 3.6 points below the
verifier-only arm (p=0.041).

Then the part that overturned the prediction. On maths answers, neither arm
lost its voice: the held-out judge scored RLAIF 2.62, verifier-only 2.60 and
combined 2.59, all indistinguishable. But on general conversation, both arms
lost ground: 3.07 down to 2.83 and 2.73, both significant, even though RLVR
never sampled a general prompt. Our first reading was that the KL anchor
explained both halves. It is computed on the maths answers the policy samples,
so it should hold those in place and leave general chat free to drift. That
reading turned out to be wrong, which is the next section.

Either way, the persona term did nothing useful. Measured against the
verifier-only arm it bought no persona on either kind of prompt, and its own
Haiku score barely rose during training (0.564 to 0.579). We suspected a noisy
judge first, since the pod could not pin Haiku's temperature, and tested it:
re-scoring the same answer moves by 0.017 while real differences between
answers span 0.124. Not noise. The better explanation is that inside a group,
all six samples share a prompt and a policy, so their persona scores are nearly
identical, while the verifier swings between 0 and 1. The persona term is a
small share of the reward variance and gets a small share of the gradient. It
was too weak to steer the voice and strong enough to blunt the maths.

### Taking the anchor away

To test the anchor explanation we ran a third arm, identical to the
verifier-only one except with the KL penalty switched off. We wrote the
prediction down first: at least as much maths, and less voice on both kinds of
prompt.

The first training step matched the verifier-only arm exactly, same reward and
same answer lengths, so the two runs really did start from the same place.
After that the sampled answers diverged, as RL runs do. KL to the RLAIF policy
tracked the anchored arm step for step until about halfway, then pulled away:
over the last twenty steps it was 0.012 against 0.004.

Afterwards we measured the drift directly, as exact per-token KL to RLAIF on
each arm's own answers. Without the anchor the model had moved three times as
far on maths answers (0.0117 against 0.0040) and nearly twice as far on general
chat (0.0062 against 0.0035). Nothing we care about moved with it. Maths came
out at 74.4%, 1.4 points above the anchored arm and not significantly
different. The judge put the voice at 2.54 on maths answers and 2.79 on
general chat, within noise of the anchored arm on both.

So the anchor did restrain the model, but over 200 steps the movement it
prevented didn't show up in maths accuracy or in the voice. The same
measurement undid the other half of our story. With the anchor on, general chat
had moved about as far as maths, not further, and the general-chat voice
dropped by roughly the same amount in all three arms however far the model
moved. Our best explanation for the maths voice holding is now that a
correctness reward can barely see it. Six samples of the same problem all sound
the same, so there is no contrast between a Yoda answer and a plain one for the
verifier to reward. That is the same reason the persona term was weak. For
general chat, every arm gave back a quarter to a third of the voice RLAIF had
just built, and from final checkpoints alone we can't tell how that happens.

### Two measurement problems found along the way

Scoring the verifier-only arm, we could not reconcile its persona number with
the Week 2 table. The Week 2 means had been computed on the training pod, with
a copy of the classifier that differed from the one committed in the repo,
while the paired statistics next to them came from the committed file. The
table did not subtract to its own delta. Every conclusion survived re-scoring,
but the means changed (RLAIF 0.841 became 0.876), and we now compute every
number from committed files with one script.

The held-out judge also refused about one maths answer in ten, on harmless word
problems, and the refused answers were longer than average, so dropping them
would have tilted every mean. We retried every arm the same way (the refusals
turned out to be deterministic) and made all judge comparisons on items scored
for both arms.

## Carrying forward

The main thing we'd do differently from the start is test the reward before
training on it. Three of four were broken here and none of it showed up in
aggregate scores. The degradation harness (construct damage whose direction you know, then check
the reward agrees) is the part of this project most worth reusing.

Second: a gate that stops the pipeline beats a metric you read afterwards. The
reward-model gate saved an hour of GRPO on a reward that ranked barely better
than chance.

Third: aggregate quality and usable gradient are different things. A reward can
be 98% accurate on the easy version of the task and have no variance at all
where it's actually queried. Week 3 added a version of the same lesson: a
reward term that ranks answers perfectly can still carry almost no gradient
if the answers inside each group barely differ on it.

Fourth: test the explanation, not just the result. We had a tidy story for
why verifier-only RL kept the maths voice, that the KL anchor held it, and it
was wrong. Removing the anchor tripled the drift and changed nothing we
measured. A persona-preserving RLVR run needs general prompts back in training
with a signal that cares about the voice, either their persona reward or
distillation from the RLAIF policy, not a heavier anchor and not a heavier
persona weight on maths.

Still open from Week 3: each arm ran with a single seed, so run-to-run variance
of RL is unmeasured. The third arm showed how quickly runs separate: same seed,
identical first step, different samples from step two. The 3.6-point gap
between the verifier-only and combined arms is the claim most exposed to it.

Still open. The base model solves 92.8% of GSM8K train against 82.0% of test, so
the self-distilled traces may skew toward memorised problems. We flagged this
rather than resolved it. Rejection sampling also selects easier problems: the traces we kept
average 3.46 reference steps against 4.21 for the ones we dropped, which doesn't
affect the matched comparisons but does inflate the absolute 82.4%. And the
"hmm" degradation still only scores 0.633; we decided to monitor it in the
diagnostics rather than tune a threshold against 60 samples.
