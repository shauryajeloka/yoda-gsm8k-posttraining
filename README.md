# Yoda × GSM8K — post-training for persona + capability

Assignment 1. Base model **Qwen2.5-3B-Instruct**, character **Yoda**, STEM task
**GSM8K** (rule-based verifier).

Three stages: SFT → RLAIF (persona reward) → RLVR (verifiable math reward),
with the last two combined into a multi-objective reward.

## Status

| | |
|---|---|
| **Week 1 — SFT** | data complete and verified; training code written, **not yet run on a GPU** |
| Week 2 — RLAIF | judge + reward mapping specified; RL loop not written |
| Week 3 — RLVR | verifier + reward function done; RL loop not written |

## Start here

* **[data/README.md](data/README.md)** — the dataset: what it contains, how it
  was built, the leakage discipline, and the QC numbers.
* **[TRAINING.md](TRAINING.md)** — runbook: how to train, what to measure with
  what, and why cross-entropy is not the persona metric.

## What's in the box

```
data/        1,907 SFT examples + 3 frozen evaluation sets + judge + verifier
scripts/     build the data (reproducible, seed 1337) and run train/generate/eval
work/        hand-authored sources; required to rebuild the data and to fit
             the persona classifier
infra/       RunPod bootstrap and one-command Week 1 / epoch sweep
outputs/     generations and metrics (the evidence behind the results table)
```

## Headline numbers so far

Data only — no model has been trained yet.

* **1,907** SFT examples: 78.7% Yoda math / 21.3% Yoda general conversation
* **1500/1500** math rewrites verified against GSM8K ground truth
* Verifier round-trips **8,792/8,792** GSM8K reference solutions
* **0** GSM8K test items in training; 0 exact or near-duplicate pairs
* Persona classifier: **0.985** held-out accuracy, **0.998** AUC on a
  content-controlled contrast

## Results

| Model | Persona score (1–5) | GSM8K accuracy |
|---|---:|---:|
| Base Qwen2.5-3B-Instruct | — | — |
| + SFT | — | — |
| + RLAIF | — | — |
| + RLVR / combined | — | — |

## Reproducing

```bash
pip install -r infra/requirements.txt
python scripts/check_formatting.py          # validate data, no GPU needed
python data/verifier/gsm8k_verifier.py --selftest
bash infra/run_week1.sh                     # needs a GPU
```

Rebuilding the dataset from `data/raw/` is byte-identical (seed 1337); hashes
of every frozen artifact are in `data/FREEZE.json`.

## Note on the trained model

LoRA adapters are gitignored — they are ~100–200 MB and reproducible from the
data plus `scripts/train_sft.py`. For the "trained SFT model" deliverable,
push the adapter to the Hugging Face Hub and link it here.
