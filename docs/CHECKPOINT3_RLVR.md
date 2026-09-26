# Checkpoint 3 — RLVR

Base: `Qwen2.5-3B-Instruct` · Character: Yoda · STEM: GSM8K
All three arms start from the RLAIF policy (Checkpoint 2). All numbers
regenerate from committed artifacts with `python scripts/analyze_rlvr.py` and
`scripts/kl_drift.py`.

---

## 1. Design: three arms, one change each

RLVR keeps the GRPO loop from Checkpoint 2 and replaces the judge with a
program: `gsm8k_verifier.py` returns 1 if the final answer equals the ground
truth and 0 otherwise. The assignment's objective combines that with the
persona reward. We ran it as a set of minimal pairs, so that each piece of the
objective is measured rather than assumed:

| | Arm A | Arm B | Arm C |
|---|---|---|---|
| reward | verifier only | verifier + 0.5 × persona (Haiku + guardrails) | verifier only |
| KL penalty β | 0.05 | 0.05 | **0** |
| isolates | — | the persona term (A vs B) | the KL anchor (A vs C) |
| everything else | identical: start (RLAIF), prompt pool, seed, 200 steps, G=6, 4 prompts/step, lr 1e-5, 512-token cap |||

**Pre-registered predictions.** The styling control (Checkpoint 1) measured
that speaking Yoda *on maths answers* costs 14.8 points of accuracy. A reward
that sees only correctness should therefore pay the policy to drop the voice
exactly there. We predicted Arm A would gain maths and lose persona-on-maths,
and Arm B would gain less maths but hold the voice.

Arm C was added after A and B came back, to test the explanation we had given
for Arm A (that the KL anchor held the voice). Written before it ran: maths at
least Arm A's 73.0%, and voice below Arm A's on maths answers (2.60) and on
general chat (2.83).

**Why λ = 0.5.** Before training we checked the persona term could act as a
counterweight at all: Haiku ranked a Yoda-voiced maths answer above a plain
one on 25/25 paired problems (0.543 vs 0.004). At λ = 0.5, dropping the voice
costs 0.27 reward, against an expected accuracy gain of ~0.15 from doing so —
net-negative, which is the property the term needs.

### Prompt selection

GRPO learns only from groups whose rewards differ. With a 0/1 verifier, a
problem the policy always solves (or never solves) produces six identical
rewards, zero advantage and no gradient. So difficulty is defined relative to
the policy: we sampled each of 800 never-used GSM8K-train problems six times
and kept the ones solved between one and five times.

| correct out of 6 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| problems | 56 | 70 | 60 | 92 | 117 | 172 | 233 |

511 problems kept (64%). None overlap the SFT data, the self-distillation pool
or the frozen eval sets, checked by id and by normalised text.

---

## 2. Results

### Maths — frozen 500-item GSM8K, exact McNemar

| arm | GSM8K | vs RLAIF | vs base |
|---|---|---|---|
| base | 82.0% | | |
| SFT | 68.4% | | |
| RLAIF (start) | 68.2% | — | −13.8 |
| **Arm A (verifier)** | **73.0%** | **+4.8, p = 0.007** | −9.0 |
| Arm B (combined) | 69.4% | +1.2, p = 0.53 | −12.6 |
| Arm C (verifier, no KL) | 74.4% | +6.2, p = 0.0008 | −7.6 |

**Arm A vs Arm B: −3.6 points for adding the persona term, p = 0.041.**
**Arm A vs Arm C: +1.4 points for removing the KL term, p = 0.49.**

### Persona

| arm | judge, general (1–5) | judge, on maths (1–5) | classifier, general | classifier, on maths |
|---|---|---|---|---|
| RLAIF (start) | 3.07 | 2.62 | 0.876 | 0.556 |
| Arm A (verifier) | 2.83 | 2.60 | 0.837 | 0.552 |
| Arm B (combined) | 2.73 | 2.59 | 0.822 | 0.514 |
| Arm C (no KL) | 2.79 | 2.54 | 0.830 | 0.639 |

The held-out judge is Claude Sonnet, never trained against. "On maths" is the
frozen 150-item persona×maths set — a declared subset of the GSM8K eval — so it
measures whether the model still sounds like Yoda *while doing the task*.

Paired comparisons (judge: sign test on items scored for both arms; classifier:
bootstrap CI and McNemar at 0.5):

| comparison | judge, general | judge, on maths | classifier, general | classifier, on maths |
|---|---|---|---|---|
| RLAIF → Arm A | **−0.24** [−0.39, −0.09], p = 0.008 | −0.04 [−0.14, +0.06], p = 0.63 | −0.039, p = 0.17 | −0.003, p = 0.76 |
| RLAIF → Arm B | **−0.34** [−0.50, −0.18], p = 0.0006 | −0.01 [−0.12, +0.11], p = 1.0 | −0.054, p = 0.024 | −0.042, p = 0.006 |
| RLAIF → Arm C | **−0.29** [−0.44, −0.13], p = 0.003 | −0.08 [−0.18, +0.03], p = 0.22 | −0.046, p = 0.22 | +0.084, p = 2e-6 |
| Arm A → Arm B | −0.10 [−0.25, +0.03], p = 0.17 | +0.01 [−0.10, +0.12], p = 1.0 | −0.015, p = 0.44 | −0.039, p = 0.026 |
| Arm A → Arm C | −0.04 [−0.17, +0.09], p = 0.90 | −0.05 [−0.15, +0.05], p = 0.52 | −0.006, p = 1.0 | +0.087, p = 3e-8 |

Judge comparisons on maths use the 128–131 items scored for both arms (the
judge refused 7–11% of maths items; see §4). Where the judge and the classifier
disagree, the judge is the arbiter. The classifier counts inversion cues per
100 words. Arm B's answers carry fewer of them (3.09 vs 3.35), and Arm C's are
shorter (120 vs 125 words) with slightly more (3.77), so the fixed Yoda opening
line becomes a bigger share of each answer. The judge reads neither change as
a different voice.

### Where the models moved

KL to the RLAIF policy, computed exactly over the vocabulary at every token of
each arm's own greedy answers (`scripts/kl_drift.py`). Answers are
teacher-forced and scored with the adapter on (the arm) and off (RLAIF):

| arm | maths answers | general-chat answers |
|---|---|---|
| Arm A (β = 0.05) | 0.0040 | 0.0035 |
| Arm B (β = 0.05) | 0.0033 | 0.0038 |
| Arm C (β = 0) | **0.0117** | **0.0062** |

(nats per token; 95% CIs are all within ±6% of the mean. Scoring RLAIF's
answers instead of each arm's own gives the same picture.)

In the training log the two verifier-only arms have the same KL for the first
half of training (Arm C 0.0020, Arm A 0.0018 at steps 81–100). They separate
after that, reaching 0.0120 for Arm C against 0.0043 for Arm A at steps
181–200.

### Diagnostics

| | RLAIF | Arm A | Arm B | Arm C |
|---|---|---|---|---|
| maths answer length (words) | 128.5 | 124.8 | 128.7 | 120.2 |
| inversion cues / 100 words | 3.30 | 3.35 | 3.09 | 3.77 |
| Star Wars / self-naming | 0 | 0 | 0 | 0 |
| truncated at 512 during training | — | 0.1% | 0.0% | 0.06% |
| KL to RLAIF, final 40 steps (training log) | — | 0.0039 | 0.0030 | 0.0105 |
| groups carrying gradient, first → last 40 steps | — | 0.80 → 0.65 | 0.78 → 0.70 | 0.79 → 0.65 |

No reward gaming of the guardrailed kind. The verifier-only arms' answers got
shorter, by 4 words with the anchor and 8 without it. That is the direction we
watched for, and it is still small.

---

## 3. What the results say

**Arm A: the prediction was wrong.** Arm A did not drop the voice on maths
answers: the judge scores it 2.62 → 2.60 (p = 0.63), although the styling
control says the voice costs about fifteen points there. The voice eroded
instead on general prompts, 3.07 → 2.83 (−0.24, p = 0.008), which RLVR never
sampled.

Our first explanation was the KL anchor. It is computed on sampled maths
completions, so it should pin maths behaviour in place and leave general chat
free to drift. Arm C tested that by removing it.

**Arm C: the anchor was not what held the voice.** Without the KL term the
model moved much further from RLAIF: three times the drift on maths answers
(0.0117 vs 0.0040 nats per token) and 1.8 times on general chat. None of the
things we measure moved with it. Maths +1.4 (p = 0.49), voice on maths −0.05
(p = 0.52), voice on general chat −0.04 (p = 0.90). The penalty was active,
since the two KL curves only separate once it starts to bind in the second
half of training. But over 200 steps, the movement it prevented did not land
on anything the verifier or the judge can see.

That leaves two parts of the Arm A explanation standing on nothing:

1. *The maths voice held without the anchor too*, with three times the drift.
   The simpler account is that a correctness-only reward barely sees the
   voice. GRPO learns from differences inside a group of six samples, and
   those six share a prompt and a policy and speak with nearly the same voice.
   So there is almost no Yoda-versus-plain contrast for the verifier to
   reward. The 14.8-point styling cost is a difference between two separately
   trained models, not a choice the policy's own samples present. This is the
   same within-group mechanism that weakened the persona term in Arm B (below),
   seen from the other side. We did not log per-sample style inside groups, so
   this is the explanation the evidence allows, not one we measured.
2. *General chat did not drift more than maths.* With the anchor on, Arm A
   moved about equally on both (0.0035 vs 0.0040). And the general-chat voice
   fell by a similar amount in all three arms (−0.24, −0.34, −0.29), whether
   the anchor was on or off and however far the model moved. So "the anchor
   cannot see general chat" is not the cause either. What the arms share is
   maths-only RL applied on top of a general-chat voice that RLAIF had only just
   built (2.05 → 3.07 on the judge). Every arm gives back a quarter to a third
   of that gain. With only final checkpoints we cannot say whether the loss
   happens early or accumulates.

**Arm B: the persona term did not do its job.** Against Arm A, the held-out
judge finds no persona benefit anywhere: general −0.10 (p = 0.17), maths +0.01
(p = 1.0). Arm B's own Haiku persona reward barely moved during training (0.564
→ 0.579), and the term cost 3.6 points of the maths gain. We tested the obvious
explanation, a noisy judge, and rejected it: re-scoring the same answer varies
by 0.017 while real differences between answers span 0.124.

The explanation the logs support is **signal weighting inside the group**. All
six samples in a GRPO group have nearly identical persona scores, while the
verifier swings the full 0-to-1 range. In a mixed group the λ = 0.5 persona
term is a small fraction of the reward variance, so it gets a small fraction of
the gradient. It never steered the voice, but it did change the normalisation
and the direction of the updates, enough to blunt the maths learning.

**What this means for the general-chat erosion.** A heavier KL penalty is not
the lever: removing the penalty tripled the drift and did not change the voice.
The fix has to put general prompts back into training with a signal that cares
about the voice. The options are interleaving RLAIF's general prompts with
their persona reward, or distilling from the RLAIF policy on general prompts
(a per-token teacher signal, far denser than a β = 0.05 penalty). A third is
to interpolate the weights of RLAIF and Arm A after training.

---

## 4. Methodology notes and limitations

* **One seed per arm.** McNemar tests whether the eval-set difference is
  beyond item-sampling noise; it says nothing about run-to-run variance of RL
  training, which can be several points. Arm C shows how quickly runs part. It
  shares Arm A's seed, prompt pool and starting weights, and its first step
  reproduces Arm A's exactly (same reward, length and KL). The sampled
  trajectories differ from step 2, because tiny numerical differences change
  which tokens get sampled. So every between-arm difference contains run
  noise. The A-vs-B maths gap (p = 0.041) is the claim most exposed to this.
  The A-vs-C nulls mean the effect of β at this scale is smaller than that
  noise, not that it is zero.
* **No ratio clipping.** We take one gradient step per batch, so the PPO
  ratio is always 1 and clipping never activates. In Arm C the only limits on
  movement were lr 1e-5, LoRA rank 16 and gradient clipping at 1.0, which never
  bound (gradient norms stayed near 0.22). Published KL-free recipes keep
  ratio clipping, so Arm C had less protection than they do.
* **Judge refusals.** Sonnet returned an API refusal on 7–11% of maths items,
  on harmless word problems, so the refusals are spurious. They were not random
  (refused answers ran longer), so every judge comparison uses only items
  scored for both arms, after an identical retry pass for every arm.
* **General prompts were never in the RLVR prompt mix.** That was a deliberate
  choice to keep the minimal pairs minimal, and it is what exposed the drift.
* **The drift measurement is exact KL on greedy answers**, not the sampled k3
  estimate in the training log, so the two sets of numbers are not on the same
  scale. Compare arms and domains within one table.

### Bugs found by the pre-run audit, before any GPU time

Three issues that would not have crashed but would have corrupted the result:

1. The trainer loaded the starting adapter one level deep; RLAIF's adapter was
   trained on top of the merged SFT model, so both arms would have started from
   a model that was neither stage and anchored KL to the wrong reference.
2. The RLAIF generation cap (192 tokens) would have truncated 40% of maths
   answers, which the verifier scores wrong — a hidden reward for shorter
   reasoning. A smoke test showed 400 still truncated 8% at sampling
   temperature; the runs used 512 (0.04% truncated at scale).
3. The log-prob computation materialised the full vocabulary distribution in
   fp32; at RLVR lengths that exhausts a 48GB card. Replaced with an exact
   micro-batched equivalent.

---

## 5. Artifacts

| | where |
|---|---|
| trainer (verifier / combined rewards) | [`scripts/train_rlaif.py`](../scripts/train_rlaif.py) |
| difficulty pre-pass | [`scripts/rlvr_prepass.py`](../scripts/rlvr_prepass.py) |
| pipelines | [`infra/run_rlvr.sh`](../infra/run_rlvr.sh) (A, B), [`infra/run_rlvr_nokl.sh`](../infra/run_rlvr_nokl.sh) (C) |
| analysis (every number above) | [`scripts/analyze_rlvr.py`](../scripts/analyze_rlvr.py) → `outputs/rlvr_summary.json` |
| drift measurement | [`scripts/kl_drift.py`](../scripts/kl_drift.py) via [`infra/run_kl_drift.sh`](../infra/run_kl_drift.sh) → `outputs/kl_drift.json` |
| judge refusal retry | [`scripts/judge_retry.py`](../scripts/judge_retry.py) |
| models | `outputs/rlvr-verifier-lora/`, `outputs/rlvr-combined-lora/`, `outputs/rlvr-nokl-lora/` (Git LFS) |
| training logs (per step) | `outputs/rlvr-*-lora/rlaif_log.jsonl` |
