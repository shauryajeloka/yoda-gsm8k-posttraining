# Checkpoint 3 — RLVR

Base: `Qwen2.5-3B-Instruct` · Character: Yoda · STEM: GSM8K
Both arms start from the RLAIF policy (Checkpoint 2). All numbers regenerate
from committed artifacts with `python scripts/analyze_rlvr.py`.

---

## 1. Design: a two-arm ablation

RLVR keeps the GRPO loop from Checkpoint 2 and replaces the judge with a
program: `gsm8k_verifier.py` returns 1 if the final answer equals the ground
truth and 0 otherwise. The assignment's objective combines that with the
persona reward. We ran it as a minimal pair so the persona term's effect is
measured rather than assumed:

| | Arm A | Arm B |
|---|---|---|
| reward | verifier only | verifier + 0.5 × persona (Haiku + guardrails) |
| start | RLAIF | RLAIF |
| everything else | identical: prompt pool, seed, 200 steps, β=0.05, G=6, 4 prompts/step, lr 1e-5 |

**Pre-registered prediction.** The styling control (Checkpoint 1) measured
that speaking Yoda *on maths answers* costs 14.8 points of accuracy. A reward
that sees only correctness should therefore pay the policy to drop the voice
exactly there. We predicted Arm A would gain maths and lose persona-on-maths,
and Arm B would gain less maths but hold the voice.

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

**Arm A vs Arm B: −3.6 points for adding the persona term, p = 0.041.**

### Persona

| arm | judge, general (1–5) | judge, on maths (1–5) | classifier, general | classifier, on maths |
|---|---|---|---|---|
| RLAIF (start) | 3.07 | 2.62 | 0.876 | 0.556 |
| Arm A (verifier) | 2.83 | 2.60 | 0.837 | 0.552 |
| Arm B (combined) | 2.73 | 2.59 | 0.822 | 0.514 |

The held-out judge is Claude Sonnet, never trained against. "On maths" is the
frozen 150-item persona×maths set — a declared subset of the GSM8K eval — so it
measures whether the model still sounds like Yoda *while doing the task*.

Paired comparisons (judge: sign test on items scored for both arms; classifier:
bootstrap CI and McNemar at 0.5):

| comparison | judge, general | judge, on maths | classifier, general | classifier, on maths |
|---|---|---|---|---|
| RLAIF → Arm A | **−0.24** [−0.39, −0.09], p = 0.008 | −0.04 [−0.14, +0.06], p = 0.63 | −0.039, p = 0.17 | −0.003, p = 0.76 |
| RLAIF → Arm B | **−0.34** [−0.50, −0.18], p = 0.0006 | −0.01 [−0.12, +0.11], p = 1.0 | −0.054, p = 0.024 | −0.042, p = 0.006 |
| Arm A → Arm B | −0.10 [−0.25, +0.03], p = 0.17 | +0.01 [−0.10, +0.12], p = 1.0 | −0.015, p = 0.44 | −0.039, p = 0.026 |

Judge comparisons on maths use the 128–130 items scored for both arms (the
judge refused ~11% of maths items; see §4). Where the judge and the classifier
disagree — Arm A vs B on maths — the judge is the arbiter: the classifier
counts inversion cues, Arm B's answers carry fewer of them (3.09 vs 3.35 per
100 words), and the judge does not read that as a weaker voice.

### Diagnostics

| | RLAIF | Arm A | Arm B |
|---|---|---|---|
| maths answer length (words) | 128.5 | 124.8 | 128.7 |
| inversion cues / 100 words | 3.30 | 3.35 | 3.09 |
| Star Wars / self-naming | 0 | 0 | 0 |
| truncated at 512 during training | — | 0.1% | 0.0% |
| KL to RLAIF, final 40 steps | — | 0.0039 | 0.0030 |
| groups carrying gradient, first → last 40 steps | — | 0.80 → 0.65 | 0.78 → 0.70 |

No reward gaming of the guardrailed kind. Arm A's answers got slightly shorter,
which is the direction we watched for; four words is small.

---

## 3. What the results say

**The prediction was wrong, in an instructive way.** Arm A did not drop the
voice on maths answers: the judge scores it 2.62 → 2.60 (p = 0.63). The KL
anchor held the style of the completions it was computed on.

**The voice eroded where nothing was watching.** Arm A's persona on *general*
prompts fell 3.07 → 2.83 (−0.25, 95% CI [−0.40, −0.10], sign test p = 0.005).
RLVR trained only on maths prompts, so the KL penalty — computed on sampled
maths completions — constrained maths behaviour and never saw general chat.
The general-conversation voice drifted with no term pushing back. Neither the
reward nor the anchor covered it.

**Arm B shows the same pattern, slightly worse.** Its general-prompt persona
fell 3.07 → 2.73 (−0.34, p = 0.0006) and its maths persona held (2.62 → 2.59,
p = 1.0). The persona reward was computed only on maths completions, so it
protected nothing that the KL anchor wasn't already protecting.

**The persona term did not do its job.** Against Arm A, the held-out judge finds
no persona benefit anywhere — general −0.10 (p = 0.17), maths +0.01 (p = 1.0) —
and Arm B's own Haiku persona reward barely moved during training (0.564 →
0.579). Meanwhile the term cost 3.6 points of the maths gain. We tested the obvious explanation — a noisy judge — and rejected
it: re-scoring the same answer varies by 0.017 while real differences between
answers span 0.124.

The explanation the logs support is **signal weighting inside the group**. All
six samples in a GRPO group share a prompt and a policy, so their persona
scores are nearly identical, while the verifier swings the full 0-to-1 range.
In a mixed group the λ = 0.5 persona term is a small fraction of the reward
variance, so it gets a small fraction of the gradient. It never steered the
voice, but it did change the normalisation and the direction of the updates,
enough to blunt the maths learning.

---

## 4. Methodology notes and limitations

* **One seed per arm.** McNemar tests whether the eval-set difference is
  beyond item-sampling noise; it says nothing about run-to-run variance of RL
  training, which can be several points. The A-vs-B maths gap (p = 0.041) is the
  claim most exposed to this. A second seed of each arm is the natural next
  experiment (~$6).
* **Judge refusals.** Sonnet returned an API refusal on ~10% of maths items —
  harmless word problems, so spurious. Refusals were not random (refused
  answers ran longer), so every judge comparison uses only items scored for
  both arms, after an identical retry pass for every arm.
* **General prompts were never in the RLVR prompt mix.** That was a deliberate
  choice to keep the minimal pair minimal, and it is what exposed the drift. A
  production run would interleave general persona prompts.

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
| pipeline | [`infra/run_rlvr.sh`](../infra/run_rlvr.sh) |
| analysis (every number above) | [`scripts/analyze_rlvr.py`](../scripts/analyze_rlvr.py) → `outputs/rlvr_summary.json` |
| judge refusal retry | [`scripts/judge_retry.py`](../scripts/judge_retry.py) |
| models | `outputs/rlvr-verifier-lora/`, `outputs/rlvr-combined-lora/` (Git LFS) |
| training logs (per step) | `outputs/rlvr-*-lora/rlaif_log.jsonl` |
