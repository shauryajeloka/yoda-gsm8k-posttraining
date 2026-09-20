#!/usr/bin/env python3
"""
P(persona): a calibrated probability that a completion is written in the
character's voice.

This is the "probability of a Yoda-like answer" framing. It complements the
1-5 judge: the judge is a language model and can be gamed; this is a fixed
function of surface style, so the two fail in different ways. If the judge
score climbs during RLAIF while P(persona) does not, that is evidence of
reward hacking rather than genuine improvement.

How it is fitted
----------------
Logistic regression on the style features from persona_similarity.py, trained
on a *controlled* contrast: the 1500 Yoda math rewrites (positive) against the
GSM8K reference solutions for the SAME 1500 problems (negative). Same
questions, same numbers, same content -- only the voice differs. So the
classifier cannot succeed by learning the topic.

A held-out 20% split reports accuracy and AUC, so you can see whether the
classifier is real before trusting its probabilities.

IMPORTANT -- do not fit on anything you will later score. An earlier version of
this pipeline passed `--extra-negatives outputs/base/persona_eval.jsonl` and
then scored that same file, so the base model's P(persona) was measured on rows
the classifier had been trained to call negative. That is train-on-test: it
pushes the base score toward 0 and inflates the base-vs-SFT gap. `--extra-negatives` now holds out
half of that file and writes it to `--holdout-out`; score the held-out half,
never the whole file. Fitting with math-only negatives avoids the issue
entirely at the cost of domain mismatch -- both are reported in the writeup.

Usage
    python scripts/persona_classifier.py --fit outputs/persona_clf.json
    python scripts/persona_classifier.py --model outputs/persona_clf.json \
        --score base=outputs/base/persona_eval.jsonl \
        --score sft=outputs/sft/persona_eval.jsonl
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts")))
from persona_similarity import FEATURE_NAMES, features, read_responses  # noqa: E402


def standardize(X):
    n, d = len(X), len(X[0])
    mu = [sum(r[j] for r in X) / n for j in range(d)]
    sd = [max(math.sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / n), 1e-8)
          for j in range(d)]
    return mu, sd


def apply_std(x, mu, sd):
    return [(v - m) / s for v, m, s in zip(x, mu, sd)]


def fit_logreg(X, y, epochs=400, lr=0.3, l2=1e-3, seed=1337):
    d = len(X[0])
    w, b = [0.0] * d, 0.0
    idx = list(range(len(X)))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        gw, gb = [0.0] * d, 0.0
        for i in idx:
            z = b + sum(w[j] * X[i][j] for j in range(d))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = p - y[i]
            for j in range(d):
                gw[j] += e * X[i][j]
            gb += e
        n = len(X)
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b


def predict(x, w, b):
    z = b + sum(wi * xi for wi, xi in zip(w, x))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        rank_sum += sum(avg_rank for k in range(i, j) if pairs[k][1] == 1)
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def fit(out_path, extra_negatives=None, holdout_path=None):
    pos = [json.loads(l)["messages"][1]["content"]
           for l in open("data/math/gsm8k_yoda_sft_train.jsonl", encoding="utf-8")]
    neg = []
    for line in open("work/sft_source_sample.jsonl", encoding="utf-8"):
        r = json.loads(line)
        neg.append(r["reference_solution_clean"].split("####")[0].strip())
    held_out = []
    if extra_negatives:
        # Domain-matched negatives are valuable, but anything used to fit must
        # never be scored. Split the file: the first half trains, the second
        # half is written out so scoring uses rows the model has not seen.
        rows = read_responses(extra_negatives)
        rng = random.Random(1337)
        order = list(range(len(rows)))
        rng.shuffle(order)
        cut = len(order) // 2
        neg += [rows[i] for i in order[:cut]]
        held_out = [rows[i] for i in order[cut:]]
        print(f"extra negatives: {cut} used for fitting, "
              f"{len(held_out)} held out for scoring")
        if holdout_path:
            Path(holdout_path).parent.mkdir(parents=True, exist_ok=True)
            with open(holdout_path, "w", encoding="utf-8") as fh:
                for t in held_out:
                    fh.write(json.dumps({"id": f"heldout_{len(t)}",
                                         "response": t}) + "\n")
            print(f"wrote {holdout_path}")

    texts = pos + neg
    y = [1] * len(pos) + [0] * len(neg)
    X = [features(t) for t in texts]
    mu, sd = standardize(X)
    Xs = [apply_std(x, mu, sd) for x in X]

    idx = list(range(len(Xs)))
    random.Random(1337).shuffle(idx)
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    w, b = fit_logreg([Xs[i] for i in tr], [y[i] for i in tr])

    probs = [predict(Xs[i], w, b) for i in te]
    labs = [y[i] for i in te]
    acc = sum((p >= 0.5) == bool(l) for p, l in zip(probs, labs)) / len(te)
    print(f"positives {len(pos)}  negatives {len(neg)}")
    print(f"held-out accuracy {acc:.3f}   AUC {auc(probs, labs):.3f}   n={len(te)}")

    top = sorted(zip(w, FEATURE_NAMES), key=lambda t: -abs(t[0]))[:10]
    print("\nmost informative features:")
    for coef, name in top:
        print(f"  {coef:+7.3f}  {name}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"w": w, "b": b, "mu": mu, "sd": sd, "features": FEATURE_NAMES,
         "heldout_accuracy": acc, "auc": auc(probs, labs)}, indent=2),
        encoding="utf-8")
    print(f"\nsaved {out_path}")


def score(model_path, specs):
    m = json.loads(Path(model_path).read_text())
    print(f"{'corpus':20s} {'mean P(persona)':>16s} {'>=0.5':>8s} {'n':>6s}")
    for spec in specs:
        name, path = spec.split("=", 1)
        texts = read_responses(path)
        ps = [predict(apply_std(features(t), m["mu"], m["sd"]), m["w"], m["b"])
              for t in texts]
        print(f"{name:20s} {sum(ps)/len(ps):16.3f} "
              f"{sum(p >= 0.5 for p in ps)/len(ps):8.1%} {len(ps):6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", metavar="OUT")
    ap.add_argument("--extra-negatives",
                    help="domain-matched negatives; HALF are held out for scoring")
    ap.add_argument("--holdout-out",
                    help="write the held-out half of --extra-negatives here")
    ap.add_argument("--model", default="outputs/persona_clf.json")
    ap.add_argument("--score", action="append", default=[], metavar="NAME=PATH")
    args = ap.parse_args()
    if args.fit:
        fit(args.fit, args.extra_negatives, args.holdout_out)
    if args.score:
        score(args.model, args.score)
    if not args.fit and not args.score:
        ap.error("use --fit OUT and/or --score NAME=PATH")


if __name__ == "__main__":
    main()
