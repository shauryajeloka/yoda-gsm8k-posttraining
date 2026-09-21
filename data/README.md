# Yoda × GSM8K post-training data

Data layer for a three-stage post-training experiment on **Qwen2.5-3B-Instruct**:
SFT → RLAIF (persona) → RLVR (verifiable math), measuring persona adherence and
STEM capability independently at every stage.

No model is trained here. This directory contains the training data, the frozen
evaluation sets, the reward/verification machinery, and the reproduction scripts.

---

## Layout

```
data/
├── persona/
│   ├── yoda_style_guide.md          7 observable traits + negative constraints
│   ├── general_yoda_train.jsonl     407 non-math Yoda conversations (SFT)
│   └── persona_eval_prompts.jsonl   150 held-out prompts   [FROZEN]
├── math/
│   ├── gsm8k_yoda_sft_train.jsonl   1500 GSM8K-train problems, Yoda solutions
│   ├── gsm8k_eval.jsonl             500 GSM8K-test problems  [FROZEN]
│   └── gsm8k_persona_math_eval.jsonl 150 of those 500, scored on both axes [FROZEN]
├── combined/
│   └── sft_train.jsonl              1907 examples, seeded shuffle
├── judges/
│   └── yoda_persona_judge.md        1–5 rubric, JSON output, RLAIF reward map
├── verifier/
│   └── gsm8k_verifier.py            deterministic answer extraction + compare
├── raw/                             official GSM8K train/test, unmodified
├── FREEZE.json                      sha256 of every frozen and final artifact
├── dataset_stats.json               machine-readable QC statistics
└── README.md
```

Scripts live in `../scripts/`. Working files (source samples, hand-authored
rewrites, reject logs) live in `../work/` and are inputs to the build, not
deliverables.

---

## Composition

| | count | share |
|---|---:|---:|
| `persona_math` (GSM8K train → Yoda solution) | 1500 | 78.7% |
| `persona_general` (non-math Yoda conversation) | 407 | 21.3% |
| **total SFT** | **1907** | |

Within the 75–85 / 15–25 target. No "normal math" (non-persona) component is
included: it would set up two conflicting targets for the same prompt type. It
stays available as a later intervention if persona SFT turns out to damage
mathematical capability.

Median example length is 117 Qwen2.5 tokens; the longest is 353, so nothing is
at risk of truncation at any reasonable context length.

---

## Data leakage discipline

This is the constraint the whole experiment rests on.

* **Training** draws only from GSM8K **train** (7,473 problems). 1,500 sampled
  with `seed=1337`.
* **STEM evaluation** draws only from GSM8K **test** (1,319 problems). 500
  sampled with the same seed; the persona×math slice is the first 150 of those,
  so one generation pass serves both metrics.
* Every record carries `source`, `source_split`, `source_id` and `source_index`
  in `metadata`, outside the `messages` list, so it never becomes training text
  and accidental leakage stays detectable after the fact.
* `build_combined.py` asserts that no evaluation question text or `source_id`
  appears in the training file, and fails the build if one does. It currently
  reports 0 failures.
* The persona eval prompts were checked against every training prompt for exact
  and near-duplicate matches (content-word Jaccard ≥ 0.8). 0 hits; the five
  closest pairs are printed on each build so the residual similarity is on the
  record rather than hidden.

GSM8K test questions were not used to generate examples, tune prompts, select
examples, or check any output.

---

## How the math data was built

Reference-first, never model-first:

```
GSM8K question
  → GSM8K reference solution (the mathematical ground truth)
  → rewrite the prose into Yoda's voice, preserving every operation
  → automatic verification
  → reject and regenerate on failure
```

Every rewrite passes five gates in `scripts/assemble_math_sft.py`:

1. **Verifier** — the extracted final answer must equal the GSM8K ground truth.
2. **Arithmetic** — every `a op b = c` written in the rewrite is evaluated and
   must hold (multi-term chains included; division tolerances match GSM8K's own
   rounding).
3. **Quantities** — every number in the rewrite must trace back to the question
   or the reference solution. Percent/decimal equivalents are allowed. Failures
   are flagged for review rather than auto-rejected.
4. **Answer tag** — the response must end with an extractable `Final answer: N`.
5. **Persona floor** — at least one inversion cue, and no Star Wars vocabulary.

Final result: **1500 / 1500 accepted, 0 rejected**, all verified through the
explicit `final_answer_tag` path (no permissive fallback was needed for any
example).

### Nine flagged examples

Nine rewrites carry quantity warnings. All were inspected; none is a
mathematical error:

* `00623`, `00880`, `02421`, `06700` — a fraction in the source written as a
  decimal in the rewrite (`1/4` → `0.25`).
* `02141`, `05184` — the source used a Unicode `¾` or omitted an intermediate
  conversion factor that the rewrite states explicitly.
* `02517`, `06539` — the source writes thousands with spaces (`$80 000`), which
  the checker reads as separate numbers.
* `05648` — the GSM8K reference solution for this item is **garbled** (it works
  with `1/6 + 1/4 + 1/12`, which are not the fractions in the question). The
  rewrite follows the question instead, and reaches the same published ground
  truth of 15.

---

## How the persona data was built

407 original non-math conversations composed for this project across ten categories
(explanation, advice, emotional, description, everyday, teaching, reflective,
casual, planning, opinion; 37–47 each). These exist so the model learns
"I always speak this way" rather than "math questions trigger a voice."

QC in `scripts/assemble_general.py`:

* no Star Wars vocabulary (0 in the final set);
* at most one interjection per response (the final set contains **zero**
  instances of "hmm"/"mmm", well inside the style guide's budget);
* the inversion floor, as above;
* no duplicate prompts;
* opening-phrase concentration tracked: the most common two-word opening
  accounts for 2.5% of the 1907 combined responses, so the set is not templated.

Exact duplicates: 0. Near-duplicate pairs (5-gram Jaccard ≥ 0.8): 0, across
both prompts and responses.

---

## Evaluation

### STEM — `gsm8k_eval.jsonl` (500) [FROZEN]

Scored by `verifier/gsm8k_verifier.py` on **answer correctness only**. Phrasing
is irrelevant: "Forty-two, the answer is." and "First we multiply… Final
answer: 42." both count as correct against `42`.

The verifier round-trips **8,792 / 8,792** GSM8K reference solutions (all of
train and test) correctly. It reports which extraction rule fired for every
result, so permissive fallbacks can be measured and disabled with `--strict`
if they ever inflate a score.

```bash
python data/verifier/gsm8k_verifier.py --selftest        # 25/25
python data/verifier/gsm8k_verifier.py --batch preds.jsonl
```

RLVR reward: `reward(response, ground_truth) -> 1.0 | 0.0`.

500 rather than the full 1,319 keeps four checkpoints × repeated runs
affordable while keeping the confidence interval tight enough to see stage
differences; the other 819 test problems stay untouched in reserve.

### Persona — `persona_eval_prompts.jsonl` (150) [FROZEN]

Fifteen prompts in each of ten categories (factual, advice, emotional,
planning, reasoning, storytelling, description, philosophical, educational,
casual). Prompts only: each checkpoint generates its own completions, which
the judge in `judges/yoda_persona_judge.md` scores 1–5 at temperature 0.

RLAIF reward: `R_persona = (persona_score - 1) / 4`.

### Persona × math — `gsm8k_persona_math_eval.jsonl` (150) [FROZEN]

A subset of the STEM eval, scored on both axes, recording
`{"math_correct": bool, "persona_score": int}` per completion. This is what
detects the failure mode where general answers sound like Yoda but the voice
evaporates the moment arithmetic starts.

---

## Checkpoint 1 similarity metric

`scripts/persona_similarity.py`. Required deliverable: a number showing that
fine-tuned outputs moved toward the character distribution.

**Choice, and why:** cosine similarity over an interpretable *style vector*
(inversion cues, sentence-final auxiliary rate, subject-final rate, fronted
clauses, sentence rhythm, register markers, and the relative frequency of 40
function words), not embedding similarity.

Sentence embeddings mostly encode topic. The held-out persona prompts are
deliberately about different subjects than the training prompts, so an
embedding cosine against the SFT corpus would largely measure topical overlap,
and would *rise* if the model drifted toward training topics, the opposite of
what should be rewarded. The persona, as defined in the style guide, is a set
of surface syntactic properties, and that is what this vector measures. An
embedding cosine is available under `--embeddings` as a secondary cross-check.

This is **not** the AI-judge score, which is a separate rubric grade used for
RLAIF and the final table.

**Validation.** Run without any model, on proxy corpora that hold content fixed
and vary only style: GSM8K reference solutions versus the Yoda rewrites of the
same problems:

```
proxy corpus                             corpus     mean             95% CI
GSM8K reference solutions (base-like)     0.629    0.352 [0.337, 0.367]
Yoda math rewrites (SFT-like)             0.888    0.638 [0.627, 0.649]
```

Same problems, same numbers, different voice, so the separation is stylistic,
and the intervals do not overlap. At Checkpoint 1, substitute real generations:

```bash
python scripts/persona_similarity.py \
    --reference data/combined/sft_train.jsonl \
    --candidate base=outputs/base_persona_eval.jsonl \
    --candidate sft=outputs/sft_persona_eval.jsonl
```

Both candidates must be generated from the **same** frozen prompt file.

---

## Reproducing

```bash
python scripts/build_splits.py          # seeded sampling; writes frozen eval sets
python scripts/assemble_math_sft.py     # gate + assemble 1500 math examples
python scripts/assemble_general.py      # gate + assemble 407 general examples
python scripts/build_combined.py        # merge, leakage assertions, FREEZE.json
python scripts/dataset_stats.py         # QC statistics (needs transformers for
                                        # real Qwen token counts)
```

All sampling uses `seed=1337`. `data/raw/` holds the official
`openai/grade-school-math` files unmodified, with their hashes in `FREEZE.json`.

---

## Freezing

`FREEZE.json` records the sha256 of every frozen and final artifact. Once
training begins, the three frozen sets must not change: not to fix a prompt,
not because a checkpoint scores badly on one. Every stage is evaluated on
identical data, or the comparison across stages means nothing.

| Model | Persona score (1–5) | GSM8K accuracy |
|---|---:|---:|
| Base Qwen2.5-3B-Instruct | — | — |
| + SFT | — | — |
| + RLAIF | — | — |
| + RLVR / combined | — | — |

Numbers come from training and evaluation, which are outside this directory.

---

## Known limitations

* The inversion detector used as a QC gate is a regex heuristic, not a parser.
  Measured on 600 GSM8K reference solutions versus these rewrites, at a
  threshold of one cue it passes 96.7% of Yoda text and 6.2% of flat prose. It
  is a **floor** that catches responses which came out as ordinary assistant
  prose; it is not a persona score, and genuine persona quality is the judge's
  job. Cue counts are stored per example in `metadata.inversion_cues` and their
  distribution is reported in the statistics (mean 2.49, min 1, max 9).
* The persona data is single-author. It is internally varied by construction
  (opening concentration 2.5%, zero near-duplicates), but it will carry one
  writer's habits, and the model will learn those habits along with the persona.
* All 1500 math rewrites resolve through an explicit `Final answer:` line. That
  is the right training target, but it means the verifier's fallback extraction
  paths are exercised only by the self-tests here — they will matter at
  evaluation time, when the base model and the RL checkpoints format their
  answers however they like.
