# Checkpoint 1 — SFT

Base: `Qwen2.5-3B-Instruct` · Character: Yoda · STEM: GSM8K

---

## 1. Training code and trained model

| what | where |
|---|---|
| SFT trainer (LoRA, completion-only loss) | [`scripts/train_sft.py`](../scripts/train_sft.py) |
| self-distillation sampling | [`infra/run_selfdistill.sh`](../infra/run_selfdistill.sh) |
| restyle integrity guard | [`scripts/check_restyle.py`](../scripts/check_restyle.py) |
| dataset assembly | [`scripts/assemble_math_sft.py`](../scripts/assemble_math_sft.py), [`assemble_general.py`](../scripts/assemble_general.py) |
| generation (all stages) | [`scripts/generate.py`](../scripts/generate.py) |
| pipeline | [`infra/run_yodadistill.sh`](../infra/run_yodadistill.sh) |
| **trained model** | `outputs/sft-yodadistill/` (Git LFS) |

**Recipe.** LoRA r=32, α=64, dropout 0.05, 3 epochs, lr 2e-4, max-len 1024,
completion-only loss masking via chat-template prefix consistency.

**Dataset:** `data/combined/sft_train_yodadistill.jsonl`, 970 examples:

| type | n | share | mean words |
|---|---|---|---|
| Yoda-restyled self-distilled CoT | 563 | 58.0% | 126.7 |
| general Yoda prose (non-maths) | 407 | 42.0% | — |

The maths targets are **the base model's own verified-correct chains of
thought**, restyled into Yoda. Rejection sampling over GSM8K *train* kept
1392/1500 (92.8%) traces the verifier confirmed correct; 563 were restyled.

Leak-checked against all 500 frozen eval items by question text **and** by id:
zero overlap (train split vs test split).

### Restyle integrity

Rewriting a chain of thought can silently improve it, which would mean
distilling from a different teacher than advertised. `check_restyle.py` enforces
three independent floors per example: final answer still verifies; every
detectable intermediate value survives and none is invented; length ≥60% and
equation count ≥80% of the original.

| | original base CoT | restyled | GSM8K refs (for contrast) |
|---|---|---|---|
| words | 165.2 | 126.7 | 51.4 |
| detectable computed values | 2.10 | 3.17 | 2.70 |
| restyle ≥ original step count | — | **562/563 (99.8%)** | — |

**563/563 accepted, 0 rejected.** The value ratio above 1.0 is not added
reasoning. The base model writes in LaTeX the extractor cannot parse, so 2.10
undercounts it. The real guarantee is the per-item check.

---

## 2. Similarity metric: SFT dataset vs base outputs vs fine-tuned outputs

[`scripts/persona_similarity.py`](../scripts/persona_similarity.py) scores each
model's held-out completions against the SFT dataset (1907 responses) over an
interpretable style vector (syntactic inversion cues, sentence-final
auxiliaries, fronted clauses, function-word frequencies).

| candidate | corpus similarity | mean per-response | 95% CI |
|---|---|---|---|
| **(a) base model outputs** | 0.581 | **0.475** | [0.460, 0.490] |
| **(b) fine-tuned outputs** | 0.838 | **0.632** | [0.608, 0.655] |
| Week-1 SFT (reference solutions) | 0.858 | 0.648 | [0.623, 0.672] |
| RLAIF (Checkpoint 2) | 0.905 | 0.716 | [0.696, 0.735] |

**+0.157 from base to SFT, non-overlapping intervals.** The model learned to
sound like the character.

The table above uses the Week-1 corpus as the reference so all four arms are
comparable on one scale. Scored against this model's *own* training set
(`sft_train_yodadistill.jsonl`), the ordering is unchanged and the intervals
still do not overlap: base 0.653 [0.632, 0.673] → SFT 0.730 [0.713, 0.747] →
RLAIF 0.759 [0.742, 0.773]. The base model scores higher against this
reference than against the Week-1 one for a sensible reason: the corpus is
built from its own self-distilled reasoning.

Corroborated by an independent style classifier (held-out accuracy 0.985,
AUC 0.998) on the same 150 prompts:

| arm | mean P(persona) | ≥0.5 |
|---|---|---|
| base | 0.044 | 1/150 |
| **SFT** | **0.727** | **108/150** |

Base scores essentially zero: the persona is entirely acquired, not latent.

---

## 3. STEM performance after fine-tuning

Frozen 500-item GSM8K test split, greedy, 1024 new tokens, exact McNemar on
paired items.

| arm | GSM8K | vs base |
|---|---|---|
| base | **82.0%** | — |
| Week-1 SFT (GSM8K refs, Yodified) | 61.4% | −20.6%, p=6.7e-17 |
| **SFT (self-distilled, Yodified)** | **68.4%** | −13.6%, p=6.5e-10 |

One asymmetry worth stating: fine-tuned arms end answers with an explicit tag,
so their extraction is trivial, while the base model answers in prose and 98.6%
of its answers go through the verifier's fallback extractor. Two checks say the
82.0% is not extractor generosity: a manual audit of randomly sampled fallback
extractions found every extracted number matched the model's intended answer
(the misses were genuine reasoning errors), and the `selfdistill` arm — whose
outputs are cleanly tagged, 0.2% fallback — reproduces the same level at 82.4%.

**There is degradation and it is significant.** Self-distillation recovers 7.0
points of the Week-1 loss (61.4% → 68.4%, p=0.0043) with no detectable
persona change (0.755 → 0.727, p=0.497; the CI spans −0.10 to +0.04, so a
small real drop is not excluded).

### What causes it

Four no-persona controls isolate the mechanism:

| arm | targets | persona? | GSM8K |
|---|---|---|---|
| `selfdistill` | base model's own verified CoT, 1392 | no | **82.4%** (p=0.91 vs base) |
| `flatref` | GSM8K reference solutions, same 1392 problems | no | 63.0% |
| `flatmath` | GSM8K reference solutions, 1500 | no | 62.6% |
| Week-1 SFT | GSM8K reference solutions, Yodified | yes | 61.4% |

Training on the model's **own** reasoning costs nothing. Training on GSM8K's
reference solutions costs **19 points with no persona anywhere in the data**.
Yoda styling adds ~1 point on top of these already-poor
targets (61.4 vs 62.6, p=0.65) — a floor effect. The minimal-pair control
(`plain563`: same 563 problems, plain targets, same mix and recipe) measures
the styling cost on good targets directly:

| arm | maths targets | GSM8K | persona on maths outputs |
|---|---|---|---|
| `plain563` | plain self-distilled | **83.2%** | 0.02 (1/150 ≥0.5) |
| `yodadistill` | the same traces, Yodified | 68.4% | **0.57** (57% ≥0.5) |

**Styling the maths traces costs 14.8 points (exact McNemar p=8×10⁻¹²), and
what it buys is the persona *on maths outputs*.** Dataset size is worth nothing
(`plain563` at 563 examples vs `selfdistill` at 1,392: +0.8, p=0.66) and the
general mix is worth nothing here either. Persona on *general* prompts is
carried almost entirely by the 407 general examples (0.711 vs 0.727, both 72%
≥0.5, styled maths or not).

Response shape shows the same thing: reference-trained arms collapse to short,
shallow answers while self-distilled arms match the base model:

| arm | generated words | equations |
|---|---|---|
| base | 183.0 | 2.24 |
| `selfdistill` | 181.2 | 2.35 |
| **`yodadistill`** | **127.5** | **3.45** |
| Week-1 SFT | 44.0 | 3.40 |

### Length is not the cause

A matched pair rewrote the *same* 113 problems longer, changing nothing else:

| arm | target words | generated words | GSM8K |
|---|---|---|---|
| `v1_control` | 50.3 | 47.2 | 58.8% |
| `v2_long` | 121.4 | 118.3 | 56.8% (p=0.48, **n.s.**) |

The model learned the longer targets faithfully and accuracy did not move.
`v2_long` vs `yodadistill` is the sharp comparison: near-identical length,
**11.6 points apart**, differing only in whose reasoning was restyled. About 8
points survive correcting for dataset size (113 vs 563).

### Dataset scale

| math examples | GSM8K |
|---|---|
| 146 | 57.6% |
| 300 | 61.2% |
| 800 | 60.8% |
| 1500 | 62.6% |

+5.0pp across a 10× increase (p=0.04), but nearly flat past 300. **Caveat:**
general-persona examples were pinned at 407 in every arm, so the general
*fraction* falls as maths count rises (73.6% → 21.3%) and the two effects are
confounded. The direct A/B on the general data is clean: ±407 general examples
changes GSM8K by **−1.2pp (p=0.637, n.s.)** while moving persona from 0.051 to
0.767.

---

---

## Limitations and open questions

* **Train-split memorisation is possible.** The base model solves 92.8% of
  GSM8K *train* but 82.0% of *test*, so the self-distilled traces may be drawn
  disproportionately from memorised problems. Flagged, not resolved.
* **Rejection sampling and restyling both skew easy.** Kept traces average 3.46
  reference steps vs 4.21 for dropped ones, and the 563 restyled traces are
  slightly shorter than the 829 left unstyled (165 vs 175 words). `flatref`
  uses the identical problem set, so the headline comparison is unaffected, but
  the absolute 82.4% is flattered.
* **The restyling step was not performed by a held-out model**, so its
  consistency is not independently measured; `check_restyle.py` bounds the
  damage mechanically rather than certifying the prose.
* **The shipped adapter is epoch 3; epoch 2 had marginally lower val loss**
  (1.004 vs 1.030). Every arm uses its final epoch, so comparisons are
  like-for-like, but absolute numbers may sit slightly below each arm's best.
* **The styling cost is now measured, not estimated** — the `plain563` minimal
  pair puts it at 14.8 points, all of it attributable to making the maths
  responses themselves speak in character. The measurement still bundles the
  voice with the restyler's compression of the traces (165 → 127 words); that
  split remains unseparated.
