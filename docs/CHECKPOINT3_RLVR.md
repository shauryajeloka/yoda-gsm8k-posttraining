# Checkpoint 3 — RLVR

Base: `Qwen2.5-3B-Instruct` · Character: Yoda · STEM: GSM8K
Every number here regenerates from committed artifacts with
`python scripts/analyze_rlvr.py`, `scripts/kl_drift.py` and
`scripts/merge_precision.py`.

**Read this first.** Our first three RLVR arms (A, B, C) did not start from
the RLAIF model. The trainer merged RLAIF's adapter into the bf16 weights before
attaching its own, and bf16 cannot hold an update that small: most of RLAIF was
rounded away (§3). We found this while checking a later experiment, fixed the
trainer, and reran the main arm (A2) and the no-KL ablation (C2) from RLAIF
itself. A2 is the Checkpoint 3 result. Arms A–C remain valid comparisons *with
each other*, since they share the same start, but their comparisons with RLAIF
are not. The persona-term arm (B) has not yet been rerun from the true start.

---

## 1. Headline: verifier-only RLVR from RLAIF

| | GSM8K | judge, general chat (1–5) | judge, on maths (1–5) |
|---|---|---|---|
| base | 82.0% | 1.00 | — |
| SFT | 68.4% | 2.05 | — |
| RLAIF (start) | 68.2% | 3.07 | 2.62 |
| RLAIF, regenerated on the A2 pod | 66.4% | 3.05 | — |
| **Arm A2: verifier only** | **72.6%** | **3.12** | 2.53 |

* **Maths:** +6.2 points over RLAIF regenerated on the same pod (53 items
  gained, 22 lost, exact McNemar p = 0.00045); +4.4 over the original RLAIF
  generation (p = 0.013). Still 9.4 points below base.
* **Voice on general chat:** +0.05 vs RLAIF (95% CI [−0.06, +0.15], sign test
  p = 0.43). Unchanged.
* **Voice on maths answers:** −0.09 (CI [−0.19, +0.02], p = 0.11, on the 130
  items the judge scored for both). Not significant.
* Diagnostics: maths answers 5 words shorter (123.8 vs 128.5); no Star Wars
  terms, self-naming or filler; 0.02% of samples truncated during training.

So verifier-only RLVR recovered a third to a half of the maths the persona SFT
cost, and did not measurably cost any of the voice RLAIF built. The pre-registered
prediction was that it would buy maths by dropping the voice on maths answers;
it did not.

### Design

RLVR keeps the GRPO loop from Checkpoint 2 and replaces the judge with a
program: `gsm8k_verifier.py` returns 1 if the final answer equals the ground
truth, 0 otherwise. 200 steps, 4 prompts × 6 samples per step, lr 1e-5, β = 0.05
KL to the starting policy, 512-token cap, fresh rank-16 LoRA.

**Prompt selection.** GRPO learns only from groups whose rewards differ, so a
problem the policy always or never solves gives no gradient. We sampled 800
never-used GSM8K-train problems six times each and kept the 511 solved between
one and five times. None overlap the SFT data, the self-distillation pool or
the frozen eval sets (checked by id and by normalised text).

| correct out of 6 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| problems | 56 | 70 | 60 | 92 | 117 | 172 | 233 |

(The pre-pass ran on the merged start of §3. Its maths accuracy matches RLAIF's,
68.0% vs 68.2%, so the difficulty filter is unaffected.)

---

## 2. Ablations: what each piece of the objective does

### The KL anchor, from the true start (A2 vs C2)

Arm C2 is A2 with β = 0, also started from RLAIF unmerged. It ran on an L40S
rather than an A40, so it is compared with A2 regenerated on the same L40S:
two L40S pods produced bit-identical answers (500/500), and moving A2 from A40
to L40S flipped 31 answers for a net +0.2 points.

| | β | GSM8K | judge, general | judge, on maths | maths answer length |
|---|---|---|---|---|---|
| A2 (same GPU as C2) | 0.05 | 72.8% | 3.12 | 2.53 | 123.7 words |
| C2 | 0 | 71.6% | 3.05 | 2.66 | 111.2 words |

* **Maths:** −1.2 (33 lost, 27 gained, p = 0.52). No effect.
* **Voice on general chat:** −0.07 (CI [−0.17, +0.04], p = 0.34). No effect.
* **Voice on maths answers:** +0.13 (CI [+0.04, +0.22], p = 0.008, n = 135).
  The one significant difference, and it points toward *more* voice without
  the anchor.
* **Length:** C2's maths answers are 12 words shorter; in training its
  samples went from 137 to 116 words, against 138 to 130 for A2.
* **Drift:** exact KL to RLAIF 0.0125 vs 0.0065 nats/token on maths answers,
  0.0053 vs 0.0045 on general chat.

So what β = 0.05 actually restrains over 200 steps is a slide toward shorter
answers, and with it most of the extra movement on maths. It is not what keeps
maths accuracy or the general-chat voice. The maths-voice difference is
probably the same length effect: the Yoda framing line is fixed, so a shorter
answer carries a larger share of it (inversion cues 3.94 vs 3.56 per 100 words).

The merged-start pair (A vs C, below) agrees: C's answers were shorter (120 vs
125 words), it drifted three times further, and nothing else moved.

### From the merged start (A, B, C)

These three arms share one start (RLAIF merged into bf16, §3) and differ from
Arm A in exactly one thing, so the comparisons among them are clean.

| | reward | β | GSM8K | judge, general | judge, on maths |
|---|---|---|---|---|---|
| merged start | — | — | 68.0% | 2.87 | — |
| Arm A | verifier | 0.05 | 73.0% | 2.83 | 2.60 |
| Arm B | verifier + 0.5 × persona (Haiku) | 0.05 | 69.4% | 2.73 | 2.59 |
| Arm C | verifier | 0 | 74.4% | 2.79 | 2.54 |

**The persona term (A → B) cost maths and bought no persona.** −3.6 points
(p = 0.041), general chat −0.10 (p = 0.17), maths answers +0.01 (p = 1.0). Arm
B's own Haiku persona reward barely moved in training (0.564 → 0.579). A noisy
judge does not explain it: re-scoring one answer varies by 0.017, while real
differences between answers span 0.124. The explanation the logs support is
signal weighting inside a group. All six samples of one maths problem sound
about equally Yoda, so their persona scores barely differ, while the verifier
swings from 0 to 1. The persona term is a small share of each group's reward
variance, so it gets a small share of the gradient: too weak to steer the
voice, but enough to blunt the maths. **This arm has not yet been rerun from
the true start** (B2 in `infra/run_rlvr_v2_ablations.sh`); it needs the Haiku
reward, and so an API key on the pod.

**The KL anchor (A → C)** gave the same answer as from the true start: three
times the drift (0.0117 vs 0.0040 on maths answers), no change in maths (+1.4,
p = 0.49) or voice (−0.04 and −0.05, both n.s.). The two KL curves agree until
about step 100, then C's pulls away (0.012 vs 0.004 over the last 20 steps).

**Why the maths voice holds under a style-blind reward.** The styling control
(Checkpoint 1) says the voice costs about 15 points on maths answers, so a
correctness-only reward should pay the policy to drop it. It never did, in A,
C, A2 or C2. The same within-group argument explains this: six samples of one
problem share the voice, so GRPO sees almost no Yoda-versus-plain contrast to
reward. The 15-point cost is a difference between two separately trained
models, not a choice the policy's own samples present. We did not log style
per sample inside groups, so this is the explanation the evidence allows, not
one we measured.

---

## 3. The bf16 merge, and what it did to the first result

The trainer rebuilt its starting model by merging each earlier adapter into
the weights (`merge_and_unload`) and training a fresh LoRA on top. That is
exact in real arithmetic, but the weights are bf16, which keeps 8 bits of
mantissa. RLAIF's update is about 0.05% of the size of the weights it modifies,
below the rounding step of most of them. Replaying the merge
(`scripts/merge_precision.py`):

| update merged into bf16 | entries rounded away entirely | relative error of what landed |
|---|---|---|
| SFT (rank 32, larger) | 28.9% | 0.21 |
| **RLAIF (rank 16, small)** | **84.0%** | **0.83** |

Functionally, exact per-token KL between RLAIF as trained and RLAIF merged:

| | maths answers | general-chat answers |
|---|---|---|
| RLAIF → merged RLAIF | 0.0008 | **0.0083** |
| for scale: merged start → Arm A (all of RLVR training) | 0.0040 | 0.0035 |

The damage is concentrated on general chat, which is where RLAIF trained. The
merge moved general-chat behaviour more than twice as far as Arm A's entire RLVR
run then did. The held-out judge agrees:

| general chat | judge | vs RLAIF |
|---|---|---|
| RLAIF | 3.07 | — |
| RLAIF regenerated (noise floor) | 3.05 | −0.02, p = 0.69 |
| RLAIF merged (Arms A–C start) | 2.87 | −0.21, p = 0.04 |
| Arm A | 2.83 | −0.03 from its own start, p = 0.82 |

**What this overturns.** The first version of this report said RLVR eroded the
Yoda voice on general chat (3.07 → 2.83, p = 0.008), and spent two revisions
explaining why, first with the KL anchor and then with drift. The loss was
there before RLVR ran. Arm A2, trained with the fix, keeps the general-chat
voice at 3.12.

**The fix.** `train_rlaif.py --parent-unmerged` keeps RLAIF as a frozen,
unmerged LoRA beside the new one, which is exactly how RLAIF was trained and
evaluated. The KL reference is RLAIF with the new LoRA scaled to zero.
`generate.py` and `kl_drift.py` rebuild the same stack from the checkpoint's
config. Checked before the run: KL exactly 0 at step 1, and the new LoRA kept in
fp32 as `get_peft_model` does (`add_adapter` does not, and at lr 1e-5 bf16 LoRA
weights would round most optimiser steps away).

**How we found it.** A blend experiment needed "scale 0 of an RLVR adapter" to
reproduce RLAIF, as a sanity check. It reproduced RLAIF's answers no better
than an unrelated arm did, and asking why led here.

**Checkpoint 2 has a smaller version of the same issue.** RLAIF was trained on
top of SFT merged into bf16 (29% of SFT's entries lost), but compared with SFT
evaluated unmerged. Merged SFT scores 66.6% on GSM8K (−1.8, p = 0.26, within
between-pod noise) and 0.759 on the style classifier (vs 0.727). The judge
scores it 2.07 against SFT's 2.05 (+0.03, p = 0.44). Measured from the model it
actually started from, RLAIF's gain is +1.00 (109 up, 9 down, p = 6e-23), the
same as the reported +1.01. SFT's update is large enough to survive the merge;
RLAIF's is not.

---

## 4. Blending RLAIF with Arm A2

Before the merge was found, the plan was to recover general-chat voice by
interpolating weights between RLAIF and the RLVR arm. We ran the blend on A2:
RLAIF + α × (A2's update), all generated on one pod.

| α | 0 (RLAIF) | 0.25 | 0.5 | 0.75 | 1 (A2) |
|---|---|---|---|---|---|
| GSM8K | 66.4% | 67.6% | 70.0% | 73.0% | 72.6% |
| classifier, general | 0.903 | 0.866 | 0.879 | 0.896 | 0.877 |

Maths rises roughly in proportion to α and plateaus between 0.75 and 1 (0.75 vs
1: +0.4, p = 0.88). The general-chat voice is flat across the curve, within the
classifier's regeneration noise of about 0.03, so there is no trade-off to
tune. We did not spend judge calls on the blends for that reason. Arm D (general
prompts with their own persona reward) was designed to repair a general-chat
loss that turned out not to exist, and was not run.

---

## 5. Methodology notes and limitations

* **Generation noise between pods.** Regenerating RLAIF with the same code and
  weights on a different A40 flips 43 of 500 GSM8K answers (66.4% vs 68.2%,
  p = 0.22) and moves the style classifier by 0.03. Greedy decoding in bf16
  amplifies tiny numerical differences. McNemar treats each model's answers as
  fixed, so comparisons between arms generated on different pods carry roughly
  two points of maths noise it does not see. We compare A2 with RLAIF
  regenerated on its own pod, and we treat classifier-only persona differences
  under about 0.03 as noise.
* **One seed per arm.** Arm C shares Arm A's seed, prompts and start, and its
  first step reproduces Arm A's exactly, but the sampled trajectories part from
  step 2. So every between-arm difference also contains run-to-run variance of
  RL. The A-vs-B maths gap (p = 0.041) is the claim most exposed to it.
* **B was not rerun from the true start.** Its result is a comparison with
  Arm A at the merged start. C was rerun (C2), and its conclusion carried over,
  which is some evidence B's would too, but it has not been checked.
* **Same GPU type, identical answers.** Two L40S pods generated bit-identical
  answers for A2 (500/500); A40 to L40S flipped 31. Comparisons across GPU
  types use a baseline regenerated on the same GPU.
* **No ratio clipping.** One gradient step per batch means the PPO ratio is
  always 1. In Arm C the only limits on movement were lr, LoRA rank and
  gradient clipping, which never bound (norms near 0.22).
* **Judge refusals.** Sonnet refuses 7–11% of maths items (harmless word
  problems; refused answers run longer), so every judge comparison uses only
  items scored for both arms, after an identical retry pass for every arm.
* **The drift numbers are exact KL on greedy answers**, not the sampled k3
  estimate in the training log, so the two are not on the same scale.

### Bugs found by the pre-run audit, before any GPU time

1. The trainer loaded the starting adapter one level deep, so the arms would
   have started from a model that was neither stage.
2. The RLAIF generation cap (192 tokens) would have truncated 40% of maths
   answers, which the verifier scores wrong — a hidden reward for shorter
   reasoning. The runs used 512.
3. The log-prob computation materialised the full vocabulary distribution in
   fp32, which exhausts a 48GB card at RLVR lengths.

The audit fixed the chain of adapters but not the precision of merging it,
which is the problem in §3.

---

## 6. Artifacts

| | where |
|---|---|
| trainer | [`scripts/train_rlaif.py`](../scripts/train_rlaif.py) (`--parent-unmerged`) |
| difficulty pre-pass | [`scripts/rlvr_prepass.py`](../scripts/rlvr_prepass.py) |
| pipelines | [`infra/run_rlvr_v2.sh`](../infra/run_rlvr_v2.sh) (A2, blends), [`infra/run_rlvr_v2_ablations.sh`](../infra/run_rlvr_v2_ablations.sh) (C2; B2 pending), [`infra/run_rlvr.sh`](../infra/run_rlvr.sh) (A, B), [`infra/run_rlvr_nokl.sh`](../infra/run_rlvr_nokl.sh) (C) |
| merge checks | [`scripts/merge_precision.py`](../scripts/merge_precision.py), [`infra/run_merge_checks.sh`](../infra/run_merge_checks.sh) → `outputs/merge_precision.json` |
| analysis | [`scripts/analyze_rlvr.py`](../scripts/analyze_rlvr.py) → `outputs/rlvr_summary.json`; drift: `outputs/kl_drift.json`, `outputs/kl_drift_a2.json`, `outputs/kl_drift_C2.json` |
| models | `outputs/rlvr-verifier-v2-lora/` (A2), `outputs/rlvr-{verifier,combined,nokl}-lora/` (A–C), Git LFS. C2's weights are not shipped (the LFS quota); its config, training log and every generation are. |
| training logs | `outputs/rlvr-*-lora/rlaif_log.jsonl` |
