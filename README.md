# Yoda persona + GSM8K capability: SFT → RLAIF

Post-training **Qwen2.5-3B-Instruct** to speak as **Yoda** while preserving
**GSM8K** arithmetic. Checkpoint 1 is SFT, Checkpoint 2 is RLAIF via GRPO.

Write-ups:

* [`WORKLOG.md`](WORKLOG.md) — narrative notes: what we tried, what broke, how
  we found out, what we changed
* [`docs/CHECKPOINT1_SFT.md`](docs/CHECKPOINT1_SFT.md)
* [`docs/CHECKPOINT2_RLAIF.md`](docs/CHECKPOINT2_RLAIF.md) — includes the
  required note on what did not work

---

## Headline results

All GSM8K numbers are on the same frozen 500-item test split, greedy decoding,
1024 new tokens, compared with an **exact McNemar test on paired items**.
Persona is scored on a frozen 150-prompt non-maths set.

| arm | GSM8K | vs base | persona P | similarity to SFT data |
|---|---|---|---|---|
| Base Qwen2.5-3B-Instruct | **82.0%** | — | 0.044 | 0.475 |
| SFT on GSM8K refs, Yodified (Week 1) | 61.4% | −20.6%, p=7e-17 | 0.755 | 0.648 |
| **SFT on self-distilled CoT, Yodified** | **68.4%** | −13.6%, p=7e-10 | 0.727 | 0.632 |
| **RLAIF (GRPO on the above)** | **68.2%** | −13.8%, p=4e-10 | **0.876** | **0.716** |

Scored by the **held-out** AI judge (Claude Sonnet, never trained against;
the RLAIF reward used Haiku):

| arm | persona, 1–5 | 1 / 2 / 3 / 4 / 5 |
|---|---|---|
| base | 1.00 | 146/0/0/0/0 |
| SFT | 2.05 | 43/64/32/9/0 |
| **RLAIF** | **3.07** | **5/26/69/47/0** |

RLAIF vs SFT: **+1.02 paired, 95% CI [+0.86, +1.19], sign test p=2.3e-21**
(97 improved, 7 worsened).

Two findings carry most of the story:

**1. The 20-point SFT loss is caused by WHOSE reasoning you imitate, not by
persona and not by length.** Four controls agree:

| arm | targets | GSM8K |
|---|---|---|
| `selfdistill` | the base model's own verified CoT, **no persona** | **82.4%** (p=0.91 vs base) |
| `flatref` | GSM8K reference solutions, **no persona**, same problems | 63.0% |
| `flatmath` | GSM8K reference solutions, **no persona**, 1500 | 62.6% |
| Week-1 SFT | GSM8K reference solutions, **Yodified** | 61.4% |

Training on the model's own chain of thought costs **nothing** (82.4% vs 82.0%).
Training on GSM8K's reference solutions costs **19 points**, with no persona
anywhere in the data. Yoda styling added about 1 point on top of those
already-poor targets (61.4 vs 62.6, p=0.65) — but that small number is a floor
effect. A minimal-pair control (`plain563`: the same 563 problems' plain traces,
same 407-general mix, same recipe) measured the styling cost directly:
**83.2% plain vs 68.4% styled — 14.8 points, exact McNemar p=8×10⁻¹²** — while
dataset size (563 vs 1,392: +0.8, n.s.) and the general mix cost nothing. What
the 14.8 points buy is persona *on the maths outputs themselves* (0.02 → 0.57);
persona on general prompts comes essentially free from the 407 general examples
either way. Speaking in character *while doing the task* is the expensive part.

**2. Length is a symptom, not the cause.** A matched pair rewrote the *same*
113 problems at 50.3 vs 121.4 words:

| arm | target words | generated words | GSM8K |
|---|---|---|---|
| `v1_control` | 50.3 | 47.2 | 58.8% |
| `v2_long` | 121.4 | 118.3 | 56.8% (p=0.48, **n.s.**) |

The model faithfully learned the longer targets and accuracy did not move.
Compare `v2_long` against `yodadistill`: near-identical length (121 vs 127
target, 118 vs 128 generated) but **11.6 points apart**. Roughly 8 of those
survive correcting for dataset size.

**3. RLAIF bought persona for free.** +0.149 persona (p=1.5e-05) for −0.2pp
GSM8K (p=1.0; 41 items flipped each way, pure churn).

---

## Repository layout

```
scripts/       training, generation, evaluation, reward and judge code
infra/         one experiment per shell script, each documenting its own design
data/          frozen eval sets, SFT datasets, verifier, judge rubric, style guide
  FREEZE.json  sha256 of every frozen artifact + a log of every verifier revision
outputs/       generations for every arm (the evidence behind each number)
  judged/      completions scored by the held-out Sonnet judge
work/          restyling sources and reward-model data, including failed attempts
docs/          checkpoint write-ups
```

**Model weights.** `outputs/sft-yodadistill/` (Checkpoint 1) and
`outputs/rlaif-lora/` (Checkpoint 2) ship via **Git LFS**. The other 11
ablation adapters are ~229MB each and would exceed the LFS free tier; every
number they produced is committed as JSONL under `outputs/`, so all results are
verifiable without them, and `infra/*.sh` reproduces them.

```bash
git lfs install && git clone <repo>      # weights need git-lfs
```

## Reproducing

```bash
pip install -r requirements.txt
bash infra/run_week1.sh          # SFT baseline
bash infra/run_selfdistill.sh    # the decisive no-persona controls
bash infra/run_yodadistill.sh    # the chosen SFT arm
bash infra/run_rlaif_full.sh     # RLAIF (needs ANTHROPIC_API_KEY)
```

## Evaluation discipline

Several of these were added *because* an earlier version of this repo got them
wrong; each is recorded in `data/FREEZE.json`.

* **Frozen eval sets, hash-pinned.** `data/FREEZE.json` holds sha256 for all 11
  frozen artifacts. Every arm answers the identical 500 GSM8K items, which is
  what makes the paired McNemar test valid.
* **Paired significance, not overlapping CIs.** Comparing two arms by whether
  their independent 95% intervals overlap is not a test; every comparison here
  is an exact McNemar on the same items.
* **Leak checks refuse rather than warn.** Training prompts are asserted
  disjoint from the frozen eval set, by question text and by id.
* **The judge is held out.** RLAIF optimises a Claude *Haiku* reward; the
  reported persona score comes from Claude *Sonnet*, never trained against.
  Same developer, so the independence is partial, and we say so.
* **A truncation cap is a scoring bias.** 400 max-new-tokens silently truncated
  84/500 base generations, scoring the verbose arm wrong for running long. All
  reported numbers use 1024.
