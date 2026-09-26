# Checkpoint 2 — RLAIF

Base: `Qwen2.5-3B-Instruct` · Character: Yoda · STEM: GSM8K
Policy initialised from `outputs/sft-yodadistill` (Checkpoint 1).

---

## 1. Training code and resulting model

| what | where |
|---|---|
| GRPO trainer | [`scripts/train_rlaif.py`](../scripts/train_rlaif.py) |
| Claude reward + guardrails | [`scripts/claude_reward.py`](../scripts/claude_reward.py) |
| local-LLM reward (rejected) | [`scripts/llm_reward.py`](../scripts/llm_reward.py) |
| judge validation harness | [`scripts/validate_judge.py`](../scripts/validate_judge.py) |
| held-out AI judge | [`scripts/eval_persona.py`](../scripts/eval_persona.py) |
| pipeline | [`infra/run_rlaif_full.sh`](../infra/run_rlaif_full.sh) |
| **trained model** | `outputs/rlaif-lora/` (Git LFS) |

**Algorithm.** GRPO, written directly against torch/transformers/peft. No value
network: a group of G=6 completions is sampled per prompt and rewards are
normalised within the group.

    A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
    L   = -mean_i[ min(rho_i A_i, clip(rho_i, 1-eps, 1+eps) A_i) ] + beta * KL

**Reference model.** The KL anchor must be the SFT model, not the base. Rather
than holding a second 3B copy in memory, the SFT adapter is merged into the base
weights and a fresh LoRA is attached on top, so `model.disable_adapter()`
returns exact SFT log-probs for free.

**Hyperparameters.** 150 steps · 4 prompts/step · G=6 · β=0.05 · lr 1e-5 ·
LoRA r=16 · temperature 1.0 · 192 max new tokens.

**Reward.** Claude Haiku 4.5 scores each completion **on its own** against the
persona rubric (0.00–1.00), plus deterministic guardrails subtracting 0.40 for
Star Wars vocabulary, 0.40 for self-naming, and 0.04 per filler beyond one.
This is a hybrid reward, not a pure LLM judge; §4 explains why.

**Reward validation, on and off the eval set.** The degradation harness that
selected this reward was originally built from completions on the frozen eval
prompts, which makes the reward's design conditioned on eval items. Re-running
the same harness on completions from the disjoint *training* prompts gives the
same pass pattern, stronger: Star Wars splice 1.000, self-naming 0.983, word
salad 0.867, hmm-padding 0.750 (`outputs/judge_validation_claude_offeval.json`).
The guardrails also show **zero false positives** across all 450 real responses
in the base, SFT and RLAIF persona evals — no un-gamed output ever trips them.

---

## 2. Persona on held-out completions, scored by the AI judge

Frozen 150-prompt non-maths eval set, disjoint from all training prompts by
assertion. The RLAIF reward used **Haiku**; the numbers below are from
**Sonnet**, which was never trained against.

### Held-out AI judge (Sonnet 5, rubric in `data/judges/yoda_persona_judge.md`)

| arm | mean (1-5) | sd | 1 / 2 / 3 / 4 / 5 | words | interj | SW |
|---|---|---|---|---|---|---|
| base | 1.00 | 0.00 | 146/0/0/0/0 | 281 | 3 | 0 |
| SFT (`yodadistill`) | 2.05 | 0.86 | 43/64/32/9/0 | 43 | 0 | 0 |
| **RLAIF** | **3.07** | 0.79 | **5/26/69/47/0** | 48 | 0 | 0 |

**RLAIF vs SFT, paired on 145 items both scored: +1.02, 95% CI [+0.86, +1.19],
exact sign test p = 2.3e-21.** 97 completions improved, 7 worsened, 41 tied.

This is the most meaningful number in the checkpoint, because Sonnet shares no
features with the Haiku reward or with the style classifier. The distribution shift is the
clearest evidence: SFT put 107/148 completions at 1-2 ("no meaningful persona"
or "weak"), RLAIF puts 116/147 at 3-4 ("recognizable" or "strong and
consistent").

Note the judge is *harsher* than the style classifier, which scored SFT at
0.727 with 108/150 above threshold while Sonnet called most of those "weak".
Had we reported only the classifier, we would have overstated the SFT baseline
and understated what RLAIF added. Nine completions across the three arms
returned unparseable judge output and were excluded rather than coerced to a
number, which would have biased the mean. The judge itself is not fully
deterministic (its API does not accept a temperature); re-scoring 28 RLAIF
items gave exact agreement on 17, disagreement never exceeding one point, and a
mean shift of −0.18 — an order of magnitude below the +1.02 effect.

Diagnostics: mean words rose 43 -> 48 and interjections and Star Wars
references stayed at zero, so the gain is not length or catchphrase inflation.

Scored completions with per-item judge reasons: `outputs/judged/`.

### Style classifier (paired, same 150 prompts)

| arm | mean P(persona) | ≥0.5 |
|---|---|---|
| base | 0.044 | 1/150 |
| Week-1 SFT | 0.755 | 114/150 |
| SFT (`yodadistill`) | 0.727 | 108/150 |
| **RLAIF** | **0.876** | **137/150** |

**RLAIF vs SFT: +0.149, exact McNemar p=1.5e-05, bootstrap 95% CI
[+0.086, +0.213].**

*Revision note:* an earlier version of this table reported per-arm means from a
copy of the classifier on the training pod that differed from the committed
`outputs/persona_clf.json`, while the paired statistics beside them were
computed with the committed file — so the table's own means did not subtract
to its delta (0.841 − 0.649 ≠ 0.149). All means are now from the committed
classifier and reproduce from the repo. The paired deltas, CIs, p-values and
threshold counts were unaffected. 37 items crossed the threshold upward, 8 downward.

### Similarity to the SFT dataset

| arm | corpus | mean per-response | 95% CI |
|---|---|---|---|
| base | 0.581 | 0.475 | [0.460, 0.490] |
| Week-1 SFT | 0.858 | 0.648 | [0.623, 0.672] |
| SFT (`yodadistill`) | 0.838 | 0.632 | [0.608, 0.655] |
| **RLAIF** | **0.905** | **0.716** | **[0.696, 0.735]** |

RLAIF − base = **+0.241**, intervals non-overlapping.

---

## 3. Effect on STEM performance

Frozen 500-item GSM8K test split, greedy, 1024 new tokens, exact McNemar.

| arm | GSM8K | vs SFT |
|---|---|---|
| base | 82.0% | — |
| SFT (`yodadistill`) | 68.4% | — |
| **RLAIF** | **68.2%** | **−0.2pp, p=1.0, not significant** |

**Optimising for persona did not affect STEM performance.** 41 items went wrong
that had been right and 40 went right that had been wrong. That is churn, not
degradation. The KL anchor (β=0.05) to the SFT policy is doing this work.

### Reward-hacking diagnostics

| | SFT | RLAIF |
|---|---|---|
| Star Wars references | 0.0% | **0.0%** |
| says "Yoda" | 0.0% | **0.0%** |
| filler count | 0.00 | **0.00** |
| type/token ratio | 0.847 | 0.849 |
| mean words | 43.0 | 47.5 |
| **inversion cues /100w** | 4.96 | **7.85** |

The guardrailed patterns never appear. **Cue density rose 58% while length rose
10%**, which is the one number to treat sceptically: the style classifier in §2
is *built on* cue counts, so part of that +0.149 may be the metric measuring
itself. That is precisely why the Sonnet judge is reported alongside it: it
shares no features with the reward.

---

## 4. What did not work

Four rewards were built and measured; three were rejected on the measurements.
None of the failures were visible until we tested for them deliberately.

### 4.1 The linear style classifier saturates on-policy

A 51-feature logistic regression scores **0.985 held-out** separating Yoda
prose from base prose, and is still useless as an RL reward. On completions
from the policy itself: median P = 0.933, **34% pinned at P ≥ 0.99**, p90 =
1.000. GRPO normalises within a group, so completions tied at the ceiling have
zero advantage and contribute **no gradient**. A third of every batch was dead.

*Adjustment:* score as log-odds rather than probability (unbounded, keeps
separating what the sigmoid has flattened). It helped, but not enough: it also
scores **0.000** on the "Yoda I am" degradation test, i.e. it actively prefers
the gamed version.

### 4.2 A local 7B judge answered from POSITION, not style

Plan: have Qwen2.5-7B rank 4 on-policy completions, distil the preferences into
a Bradley-Terry reward model. Measured instead:

| | value | chance |
|---|---|---|
| self-consistency, 4-way ranking under order reversal | **0.590** | 0.500 |
| put whatever sat in slot A first | **46.9%** | 25% |
| self-consistency, 2-way comparison | **0.493** | 0.500 |
| picked whichever response was shown first | **72.9%** | 50% |

Two-way comparison was *worse* than four-way. So it was not the ranking format
and not a shortage of data: **any comparative prompt puts several candidates in
one context window, and this judge answered from position.**

Nor was it an absence of signal. Within-group spread of the style score
(mean std 3.86, median range 9.4 logits) is *larger* than between-group spread
(2.61), so the completions genuinely differ. And judge consistency did **not**
improve on the groups that differed most (0.580 high-spread vs 0.612
low-spread).

The reward model distilled from those labels reached **0.631** held-out pairwise
accuracy against a **0.552** linear baseline, and 0.943 on training pairs: it
learned the position artifact. It **failed its gate and the pipeline stopped
before GRPO**, which is what the gate was for.

*Adjustments:* judge each group twice with candidates reversed and keep only
pairs both passes agree on (this also yields self-consistency for free);
shrink the reward model (r=16 over 7 projections → r=8 attention-only) against
the 0.943/0.631 overfit gap; and make the gate **relative to measured judge
consistency** rather than an absolute 0.75, since a model cannot rank better
than its labels agree with themselves.

*Superseded code is kept, not deleted:* `scripts/build_rm_data.py`,
`judge_rank.py`, `train_reward_model.py`, `diagnose_rm.py`, plus
`work/rm_pairs_biased.jsonl` and `outputs/reward-model-biased/`. Each file
carries a header explaining why it is not used.

### 4.3 Distillation was unnecessary anyway

Reward models exist because you cannot query a human mid-training, and because
millions of rollouts make a large judge prohibitive. Neither applies here:
GRPO needs **~3,600 judgements total**, and the whole premise of RLAIF is that
the judge *is* a model and can be queried online. Building a proxy reintroduced
the exact constraint RLAIF removes, and added a failure mode that then bit us.

### 4.4 A pointwise 7B judge still could not follow the anti-gaming rules

Scoring each completion alone removes position bias *structurally*. That fixed
the mechanism and the judge scored 1.000 separating base prose from Yoda prose.

But on a controlled-degradation harness, where the correct ordering is known
because the damage is constructed, it failed the rules that matter.
Given the full anti-gaming rubric **in the prompt**:

| degradation | terse prompt | full rubric | required |
|---|---|---|---|
| voice stops halfway | 1.000 | 0.950 | ✅ |
| word salad | 0.142 | **0.750** | ✅ fixed by rubric |
| hmm padding | 0.333 | **0.550** | ✅ fixed by rubric |
| Star Wars splice | 0.258 | **0.183** | ❌ |
| "Yoda I am" | 0.117 | **0.017** | ❌ |

Mean score **rose** when the damage was added: 0.464 → 0.602 (Star Wars),
0.464 → 0.620 (self-naming). The rubric says *"Do NOT give a high score merely because the response says
'Yoda,' 'Jedi,' 'Force'"* and the model paid +0.16 for exactly that. It was
told the rule in plain language and did not follow it.

The split is instructive: where the fault was an underspecified prompt (salad,
hmm) the rubric fixed it; where it was instruction-following capacity, no
wording helped.

*Two of these failures were ours, not the model's.* The `starwars` and `yoda_name`
splices insert genuinely Yoda-inverted text (*"sharp the reasoning must be"*),
so under our original terse prompt, which asked only about inverted speech,
ranking them higher was the correct answer to the question we asked. The
objective was wrong, not the judge. But RL optimises what you specify, so the
risk was real regardless.

### 4.5 What the final reward looks like, and its remaining weakness

Claude Haiku flipped the sign on both gaming tests but with **thin margins**:

| degradation | Claude alone | **+ guardrails** | margin |
|---|---|---|---|
| Star Wars splice | 0.567 | **0.933** | +0.056 → **+0.402** |
| "Yoda I am" | 0.533 | **0.950** | +0.037 → **+0.395** |
| voice stops halfway | 0.767 | 0.767 | +0.336 |
| word salad | 0.733 | 0.733 | +0.312 |
| hmm padding | 0.500 | 0.633 | +0.055 |

Claude is decisive about things needing judgement (+0.336, +0.312) and nearly
indifferent to flat prohibitions (+0.037). A 0.04 penalty is swamped by
within-group noise. Those three rules are *closed vocabularies* needing no
judgement, so they are enforced in code and the judge does what it is good at.

**Remaining weakness, stated rather than tuned away:** `hmm` sits at 0.633
(+0.055). Rather than fit a threshold to 60 samples, filler count is tracked in
the diagnostics. It came out at 0.00, so the exploit was not taken.

Also honest: the linear classifier **beats** Claude on the degree test (0.933
vs 0.767). The hybrid is not strictly dominant.
