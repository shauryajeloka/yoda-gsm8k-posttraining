#!/usr/bin/env python3
"""
Checkpoint 1 persona-similarity metric.

WHAT THIS MEASURES
------------------
How close a corpus of model outputs is, *in style*, to the Yoda persona data
the model was fine-tuned on. It answers the Checkpoint 1 requirement:

    Similarity(base-model outputs,  SFT persona data)
      vs
    Similarity(SFT-model outputs,   SFT persona data)

If fine-tuning worked, the second number is higher, and the gap should be
large relative to the bootstrap confidence intervals printed alongside it.

This is NOT the AI-judge persona score. The judge gives a 1-5 rubric grade and
is used for RLAIF and for the final results table. This metric is a cheap,
deterministic, model-free number for the SFT checkpoint.

WHY STYLE FEATURES RATHER THAN EMBEDDINGS
-----------------------------------------
Sentence embeddings mostly encode *topic*. The held-out persona prompts are
deliberately about different subjects than the training prompts, so embedding
cosine between model outputs and the SFT corpus would largely measure
topical overlap -- and would go UP if the model started answering off-topic in
a training-like way, which is the opposite of what we want to reward.

The persona, by contrast, is defined in `yoda_style_guide.md` as a set of
observable surface properties: inverted word order, trailing auxiliaries, short
declaratives, a particular register. Those are exactly what a function-word and
syntactic-marker profile captures, and they are independent of subject matter.

So the primary metric is a cosine similarity over an interpretable style
vector. An embedding cosine is available as a secondary cross-check
(--embeddings) when sentence-transformers is installed; it is reported as
supporting evidence, not as the headline number.

FEATURES (per response, all rate-normalized so length does not dominate)
    syntactic : inversion cues, sentence-final auxiliary rate, subject-final
                rate, fronted-clause rate
    rhythm    : mean sentence length, mean word length, type-token ratio
    register  : second-person rate, modal rate, interjection rate
    lexical   : relative frequency of 40 common function words

USAGE
    # compare two sets of model outputs against the SFT persona corpus
    python scripts/persona_similarity.py \
        --reference data/combined/sft_train.jsonl \
        --candidate base=outputs/base.jsonl \
        --candidate sft=outputs/sft.jsonl

    # self-validate the metric on proxy corpora (no model needed)
    python scripts/persona_similarity.py --selftest

Candidate files are JSONL with a "response" field (or a chat "messages" list);
the reference file may be either the SFT training file or any JSONL of
responses.
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts")))
from assemble_math_sft import CUES  # noqa: E402  (the inversion cue regexes)

FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "at", "by", "from", "as", "that", "this", "it", "is", "are", "was",
    "be", "do", "does", "have", "has", "not", "no", "you", "your", "we", "they",
    "must", "will", "can", "should", "when", "then",
]
AUX = (r"must|is|are|was|were|will|do|does|did|have|has|had|can|could|should|"
       r"be|remain|remains|becomes|means")
FINAL_AUX = re.compile(r"\b(?:" + AUX + r")\s*[.!?]\s*$")
FINAL_AUX_VERB = re.compile(r"\b(?:" + AUX + r")\s+[a-z]+\s*[.!?]\s*$")
SUBJ_FINAL = re.compile(
    r"\b(?:he|she|it|they|we|you|I|[A-Z][a-z]+)(?:\s+[a-z]+){1,3}\s*[.!?]\s*$")
FRONTED = re.compile(r"^[^,.!?]{5,70},\s+\S")
SECOND_PERSON = re.compile(r"\b(you|your|yours)\b", re.IGNORECASE)
MODAL = re.compile(r"\b(must|should|will|can|could|may|might)\b", re.IGNORECASE)
INTERJECTION = re.compile(r"\b(hmm+|mmm+|yes|ah|oh)\b[,.!]", re.IGNORECASE)

FEATURE_NAMES = ([
    "inversion_cues", "final_aux", "final_aux_verb", "subject_final",
    "fronted_clause", "mean_sentence_len", "mean_word_len", "type_token_ratio",
    "second_person", "modal", "interjection",
] + [f"fw::{w}" for w in FUNCTION_WORDS])


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def features(text):
    """Interpretable style vector for one response."""
    words = re.findall(r"[A-Za-z']+", text)
    n_words = max(len(words), 1)
    sents = sentences(text) or [text]
    n_sents = len(sents)
    lower = [w.lower() for w in words]

    per100 = 100.0 / n_words
    v = [
        sum(len(c.findall(text)) for c in CUES) * per100,
        sum(bool(FINAL_AUX.search(s)) for s in sents) / n_sents,
        sum(bool(FINAL_AUX_VERB.search(s)) for s in sents) / n_sents,
        sum(bool(SUBJ_FINAL.search(s)) for s in sents) / n_sents,
        sum(bool(FRONTED.search(s)) for s in sents) / n_sents,
        n_words / n_sents / 10.0,                       # scaled to ~1
        sum(len(w) for w in words) / n_words / 5.0,     # scaled to ~1
        len(set(lower)) / n_words,
        len(SECOND_PERSON.findall(text)) * per100,
        len(MODAL.findall(text)) * per100,
        len(INTERJECTION.findall(text)) * per100,
    ]
    counts = {w: 0 for w in FUNCTION_WORDS}
    for w in lower:
        if w in counts:
            counts[w] += 1
    v.extend(counts[w] * per100 for w in FUNCTION_WORDS)
    return v


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def centroid(vectors):
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


def bootstrap_ci(values, n=2000, seed=1337, alpha=0.05):
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n)]


def read_responses(path):
    texts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "response" in r:
                texts.append(r["response"])
            elif "messages" in r:
                for m in r["messages"]:
                    if m["role"] == "assistant":
                        texts.append(m["content"])
            elif "completion" in r:
                texts.append(r["completion"])
    return texts


def score(reference_texts, candidate_texts):
    ref_vecs = [features(t) for t in reference_texts]
    ref_c = centroid(ref_vecs)
    per_response = [cosine(features(t), ref_c) for t in candidate_texts]
    corpus = cosine(centroid([features(t) for t in candidate_texts]), ref_c)
    mean = sum(per_response) / len(per_response)
    lo, hi = bootstrap_ci(per_response)
    return {"corpus_similarity": corpus, "mean_response_similarity": mean,
            "ci95": [lo, hi], "n": len(per_response)}


def selftest():
    """
    Validate that the metric separates Yoda-styled text from flat prose, using
    proxy corpora that need no model: the GSM8K reference solutions stand in
    for a non-persona ("base-like") model, and the Yoda rewrites of those same
    problems stand in for a persona-following ("SFT-like") model. Same problems,
    same content, different style -- so any separation is stylistic.
    """
    ref = [json.loads(l)["messages"][1]["content"]
           for l in open("data/persona/general_yoda_train.jsonl", encoding="utf-8")]
    yoda = [json.loads(l)["messages"][1]["content"]
            for l in open("data/math/gsm8k_yoda_sft_train.jsonl", encoding="utf-8")][:400]
    flat = []
    for line in open("work/sft_source_sample.jsonl", encoding="utf-8"):
        r = json.loads(line)
        flat.append(r["reference_solution_clean"].split("####")[0].strip())
    flat = flat[:400]

    print("Reference corpus: general_yoda_train.jsonl (persona SFT data)\n")
    print(f"{'proxy corpus':38s} {'corpus':>8s} {'mean':>8s} {'95% CI':>18s}")
    for name, texts in (("GSM8K reference solutions (base-like)", flat),
                        ("Yoda math rewrites (SFT-like)", yoda)):
        s = score(ref, texts)
        print(f"{name:38s} {s['corpus_similarity']:8.3f} "
              f"{s['mean_response_similarity']:8.3f} "
              f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
    print("\nThe two corpora answer the SAME problems, so the separation here is\n"
          "stylistic rather than topical. Non-overlapping intervals mean the\n"
          "metric can detect the shift that SFT is supposed to produce.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="data/combined/sft_train.jsonl")
    ap.add_argument("--candidate", action="append", default=[],
                    metavar="NAME=PATH", help="repeatable; e.g. base=out/base.jsonl")
    ap.add_argument("--embeddings", action="store_true",
                    help="also report the secondary embedding cosine (needs "
                         "sentence-transformers)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.candidate:
        ap.error("give at least one --candidate NAME=PATH, or use --selftest")

    ref = read_responses(args.reference)
    print(f"reference: {args.reference}  ({len(ref)} responses)\n")
    print(f"{'candidate':20s} {'corpus':>8s} {'mean':>8s} {'95% CI':>18s} {'n':>6s}")
    results = {}
    for spec in args.candidate:
        name, path = spec.split("=", 1)
        texts = read_responses(path)
        s = score(ref, texts)
        results[name] = s
        print(f"{name:20s} {s['corpus_similarity']:8.3f} "
              f"{s['mean_response_similarity']:8.3f} "
              f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] {s['n']:6d}")

    if args.embeddings:
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
            ref_emb = model.encode(ref, convert_to_tensor=True).mean(dim=0)
            print("\nsecondary embedding cosine (topic-confounded, cross-check only):")
            for spec in args.candidate:
                name, path = spec.split("=", 1)
                emb = model.encode(read_responses(path), convert_to_tensor=True)
                sim = float(util.cos_sim(emb.mean(dim=0), ref_emb))
                results[name]["embedding_cosine"] = sim
                print(f"  {name:20s} {sim:.3f}")
        except ImportError:
            print("\n(sentence-transformers not installed; embedding "
                  "cross-check skipped)")

    if len(results) >= 2:
        names = list(results)
        a, b = results[names[0]], results[names[-1]]
        gap = b["mean_response_similarity"] - a["mean_response_similarity"]
        overlap = a["ci95"][1] >= b["ci95"][0]
        print(f"\n{names[-1]} - {names[0]}: {gap:+.3f}   "
              f"confidence intervals {'OVERLAP' if overlap else 'do not overlap'}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n",
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
