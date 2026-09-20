#!/usr/bin/env python3
"""
Deterministic GSM8K answer verifier.

Two jobs:

1. EVALUATION -- score a model response against the GSM8K ground truth.
       accuracy = n_correct / n_evaluated
2. RLVR REWARD -- R_stem = 1.0 if correct else 0.0

Design constraints:

* Deterministic. No model calls, no randomness. Same input -> same output.
* Format-tolerant, value-strict. Harmless surface differences (commas, "$",
  "**", LaTeX wrappers, trailing periods, "9" vs "9.0") are normalized away.
  The numeric value itself is compared exactly.
* Persona-aware. The model under test speaks like Yoda, so the answer may be
  spelled out ("Nine apples remain.") or inverted ("Forty-two, the answer is.").
  Both are handled.
* Auditable rather than blindly permissive. Every result reports WHICH
  extraction rule fired, so a permissive fallback can be measured and, if it is
  inflating scores, disabled with strict=True.

CLI
---
    # single response
    python gsm8k_verifier.py --response "Final answer: 9" --ground-truth "9"

    # batch: JSONL with {"response": ..., "ground_truth": ...} per line
    python gsm8k_verifier.py --batch preds.jsonl

    # self-test
    python gsm8k_verifier.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Optional, Tuple

# --------------------------------------------------------------------------
# Spelled-out English numerals
# --------------------------------------------------------------------------

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}

_WORD_TOKENS = set(_UNITS) | set(_TENS) | set(_SCALES) | {"and", "negative", "minus"}

# One spelled-out number, e.g. "forty-two", "one hundred and twenty three".
_WORD_NUM_RE = re.compile(
    r"\b(?:negative\s+|minus\s+)?(?:"
    + "|".join(sorted(_WORD_TOKENS - {"and", "negative", "minus"}, key=len, reverse=True))
    + r")(?:[\s\-]+(?:and[\s\-]+)?(?:"
    + "|".join(sorted(_WORD_TOKENS - {"negative", "minus"}, key=len, reverse=True))
    + r"))*\b",
    re.IGNORECASE,
)


def words_to_number(text: str) -> Optional[float]:
    """Convert a spelled-out English numeral to a number. None if not one."""
    tokens = re.split(r"[\s\-]+", text.strip().lower())
    tokens = [t for t in tokens if t]
    if not tokens:
        return None

    negative = False
    if tokens[0] in ("negative", "minus"):
        negative = True
        tokens = tokens[1:]
    if not tokens or any(t not in _WORD_TOKENS for t in tokens):
        return None

    total, current, seen = 0, 0, False
    for tok in tokens:
        if tok == "and":
            continue
        if tok in _UNITS:
            current += _UNITS[tok]
            seen = True
        elif tok in _TENS:
            current += _TENS[tok]
            seen = True
        elif tok == "hundred":
            if current == 0:
                current = 1
            current *= 100
            seen = True
        else:  # thousand / million
            if current == 0:
                current = 1
            total += current * _SCALES[tok]
            current = 0
            seen = True
    if not seen:
        return None
    total += current
    return -float(total) if negative else float(total)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

# A bare numeric literal: 1,234.56 / -3 / .5 / 42
_NUM_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+")

_STRIP_WRAPPERS = [
    # GSM8K reference solutions carry inline calculator annotations such as
    # "<<48/2=24>>24". They duplicate the value that follows them and their
    # "=" confuses equation handling, so drop them first.
    (re.compile(r"<<[^<>]*>>"), " "),
    (re.compile(r"\\boxed\s*\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\text\s*\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\s*\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\\(|\\\)|\\\[|\\\]"), " "),
    (re.compile(r"\$\$"), " "),
]


def _clean(text: str) -> str:
    """Remove markup that carries no numeric meaning."""
    out = text
    for pat, repl in _STRIP_WRAPPERS:
        out = pat.sub(repl, out)
    out = out.replace("**", " ").replace("__", " ").replace("`", " ")
    out = re.sub(r"[ \t]+", " ", out)
    return out


def normalize_number(raw: str) -> Optional[str]:
    """
    Normalize a candidate answer string to a canonical numeric string.

    '$1,234.00' -> '1234'      '42.'  -> '42'      'forty-two' -> '42'
    '9.50'      -> '9.5'       '10.0' -> '10'      '-3'        -> '-3'

    Returns None if no unambiguous number is present.
    """
    if raw is None:
        return None
    s = _clean(str(raw)).strip()
    s = s.strip(" \t\n.:;,!?*'\"()[]{}")
    if not s:
        return None

    # Strip leading currency / trailing unit symbols.
    s = re.sub(r"^[\$€£¥]\s*", "", s)
    s = re.sub(r"\s*(%|percent|dollars?|cents?|usd)$", "", s, flags=re.IGNORECASE)
    s = s.strip()

    m = _NUM_RE.fullmatch(s.replace(" ", ""))
    if m:
        value = float(m.group(0).replace(",", ""))
    else:
        value = words_to_number(s)
        if value is None:
            # Digits embedded in a short phrase, e.g. "9 apples".
            nums = _NUM_RE.findall(s)
            if len(nums) == 1:
                value = float(nums[0].replace(",", ""))
            else:
                return None

    # A degenerate generation can emit a number with hundreds of digits (a
    # repetition loop), and float() turns that into inf -- which then raises
    # OverflowError in int(). Observed in the wild: a 977-digit run of "6".
    # Such a token is never a real answer, so reject it rather than crash.
    # This matters beyond evaluation: verify() is the RLVR reward function,
    # and a reward that raises mid-rollout kills the training run.
    if not math.isfinite(value):
        return None

    if value == int(value):
        return str(int(value))
    return repr(round(value, 6)).rstrip("0").rstrip(".")


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def extract_ground_truth(gsm8k_answer: str) -> Optional[str]:
    """Pull the reference answer out of a raw GSM8K 'answer' field ('#### 72')."""
    if "####" in gsm8k_answer:
        return normalize_number(gsm8k_answer.split("####")[-1])
    return normalize_number(gsm8k_answer)


# --------------------------------------------------------------------------
# Prediction extraction
# --------------------------------------------------------------------------

_VALUE = r"([-+]?[\$€£¥]?\s*[\d,]*\.?\d+(?:\s*%)?)"
_WORDS = r"([A-Za-z][A-Za-z\s\-]{1,40}?)"

# Rules are tried in order; the LAST match within the first matching rule wins.
# 'value_before' handles Yoda's inverted phrasing: "Forty-two, the answer is."
_RULES = [
    ("final_answer_tag", "after",
     re.compile(r"final\s+answer\s*(?:is|:|=)?\s*" + _VALUE, re.IGNORECASE)),
    ("final_answer_tag_words", "after",
     re.compile(r"final\s+answer\s*(?:is|:|=)?\s*" + _WORDS + r"\s*[\.\n]", re.IGNORECASE)),
    ("hash_delimiter", "after", re.compile(r"####\s*" + _VALUE)),
    ("boxed", "after", re.compile(r"\\boxed\s*\{\s*" + _VALUE + r"[^}]*\}")),
    ("answer_is", "after",
     re.compile(r"(?:the\s+)?answer\s*(?:is|:|=)\s*" + _VALUE, re.IGNORECASE)),
    ("answer_is_words", "after",
     re.compile(r"(?:the\s+)?answer\s*(?:is|:|=)\s*" + _WORDS + r"\s*[\.\n]", re.IGNORECASE)),
    # Inverted: "Nine apples, the answer is." / "Forty-two, it is."
    ("inverted_answer_words", "before",
     re.compile(_WORDS + r"\s*,\s*(?:the\s+answer|it)\s+(?:is|be)\b", re.IGNORECASE)),
    ("inverted_answer_digits", "before",
     re.compile(_VALUE + r"[^.\n]{0,20}?,\s*(?:the\s+answer|it)\s+(?:is|be)\b", re.IGNORECASE)),
]


def _resolve_equation_tail(text: str, m: re.Match, raw: str) -> str:
    """
    Handle answer markers followed by a whole equation rather than a value.

    GSM8K itself contains lines like
        "... the final answer is 15 gallons * 8 pints / gallon = 120 pints"
    where the captured value (15) is the first operand, not the answer. When the
    remainder of that line continues into an equation, the value after the final
    "=" is the answer.
    """
    nl = text.find("\n", m.end(1))
    tail = text[m.start(1): nl if nl != -1 else len(text)]
    if "=" not in tail:
        return raw
    nums = _NUM_RE.findall(tail.rsplit("=", 1)[1])
    return nums[0] if nums else raw


def _last(pattern: re.Pattern, text: str, group: int = 1) -> Optional[str]:
    matches = list(pattern.finditer(text))
    return matches[-1].group(group) if matches else None


def _question_numbers(question: Optional[str]) -> set:
    """Normalized numbers that appear in the problem statement."""
    if not question:
        return set()
    out = set()
    for raw in _NUM_RE.findall(_clean(question)):
        n = normalize_number(raw)
        if n is not None:
            out.add(n)
    return out


def _pick_from_last_line(line, nums, qnums):
    """
    Choose the answer among the numbers on a concluding line.

    Two competing patterns, in priority order.

    1. The line ends in an equation -- "5 - 2 = 3 slices". The answer is the
       value after the final "=", even when it also appears in the question.
       This is checked FIRST, because the operands of the equation are usually
       question quantities and the question filter below would discard the
       result and keep an operand.

    2. The line is prose that restates a given quantity after the answer:

           "Therefore, Kylar needs to pay 64 dollars for 16 glasses."
           "So, Peter can go to the movies 3 times with his $42."

       Both end with a number that was *given*, not computed, so prefer the
       last number that did not appear in the question.

    Falls back to the plain last number when every candidate is a restatement,
    or when no question was supplied.
    """
    if not nums:
        return None
    if "=" in line:
        after = [normalize_number(x) for x in _NUM_RE.findall(line.rsplit("=", 1)[1])]
        after = [n for n in after if n is not None]
        if after:
            return after[0]
    fresh = [n for n in nums if n not in qnums]
    return fresh[-1] if fresh else nums[-1]


def extract_prediction(response: str, strict: bool = False,
                       question: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Extract the model's final answer.

    Returns (normalized_answer, extraction_method). The method string is kept so
    that fallback-driven "correct" verdicts can be audited and, if they prove
    permissive, turned off with strict=True.

    `question` is optional but strongly recommended when the response carries no
    explicit answer marker: it lets the fallback discard numbers that were given
    in the problem rather than computed.
    """
    if not response or not response.strip():
        return None, "empty_response"

    text = _clean(response)

    for name, direction, pattern in _RULES:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        m = matches[-1]
        raw = m.group(1)
        if direction == "after":
            raw = _resolve_equation_tail(text, m, raw)
        norm = normalize_number(raw)
        if norm is not None:
            return norm, name
        if direction == "before" or name.endswith("_words"):
            wm = _WORD_NUM_RE.search(raw.strip())
            if wm:
                norm = normalize_number(wm.group(0))
                if norm is not None:
                    return norm, name

    if strict:
        return None, "no_explicit_answer_strict"

    # Fallback 1: a number on the last non-empty line, preferring one that was
    # not already given in the question (see _pick_from_last_line).
    qnums = _question_numbers(question)
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1]
        norms = [n for n in (normalize_number(x) for x in _NUM_RE.findall(last_line))
                 if n is not None]
        if norms:
            picked = _pick_from_last_line(last_line, norms, qnums)
            if picked is not None:
                return picked, ("fallback_last_line_number" if not qnums
                                else "fallback_last_line_number_q")
        wm = list(_WORD_NUM_RE.finditer(last_line))
        if wm:
            norm = normalize_number(wm[-1].group(0))
            if norm is not None:
                return norm, "fallback_last_line_words"

    # Fallback 2: last number anywhere. Digits only -- a global scan for
    # spelled-out numerals would match ordinary words like "one" or "two"
    # in the prose and is too permissive to allow.
    nums = _NUM_RE.findall(text)
    if nums:
        norm = normalize_number(nums[-1])
        if norm is not None:
            return norm, "fallback_last_number"

    return None, "no_number_found"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def verify(response: str, ground_truth: str, strict: bool = False,
           question: Optional[str] = None) -> dict:
    """
    Verify a model response against the ground truth.

    ground_truth may be a bare value ("72") or a raw GSM8K answer field
    ("... #### 72"). Pass `question` whenever it is available -- it is what
    lets the fallback tell a computed answer from a restated given.
    """
    gt = extract_ground_truth(ground_truth) if "####" in str(ground_truth) \
        else normalize_number(ground_truth)
    pred, method = extract_prediction(response, strict=strict, question=question)

    correct = pred is not None and gt is not None and pred == gt
    return {
        "correct": bool(correct),
        "predicted_answer": pred,
        "ground_truth": gt,
        "extraction_method": method,
    }


def reward(response: str, ground_truth: str, strict: bool = False,
           question: Optional[str] = None) -> float:
    """RLVR reward: 1.0 if the final answer is correct, else 0.0."""
    return 1.0 if verify(response, ground_truth, strict=strict,
                         question=question)["correct"] else 0.0


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

_CASES = [
    # (response, ground_truth, expected_correct)
    ("Final answer: 9", "9", True),
    ("Final answer: 9", "8", False),
    ("Nine apples remain. **Final answer: 9**", "9", True),
    ("Forty-two, the answer is.", "42", True),
    ("First we multiply... Final answer: 42.", "42", True),
    ("The answer is 1,234 dollars.", "1234", True),
    ("The answer is $1,234.00", "1234", True),
    ("\\boxed{72}", "72", True),
    ("#### 72", "72", True),
    ("Weng earned 10 dollars. Final answer: 10", "Weng earns ... #### 10", True),
    ("Final answer: 10.0", "10", True),
    ("Final answer: 0.5", "0.5", True),
    ("Final answer: -3", "-3", True),
    ("Twelve apples she now has. Final answer: 12", "12", True),
    ("Sixty dollars, the answer is.", "60", True),
    ("One hundred twenty three, the answer is.", "123", True),
    ("She sold 48 in April and 24 in May. Altogether, 72.", "72", True),
    ("Final answer: 70%", "70", True),
    ("", "5", False),
    ("I do not know.", "5", False),
    # Must NOT be lenient: an explicit wrong final answer loses, even when the
    # right number appears earlier in the reasoning.
    ("5 + 7 = 12, so 12 - 3 = 9. Final answer: 8", "9", False),
    # Last-occurrence wins (self-correction).
    ("Final answer: 8. Wait -- recheck. Final answer: 9", "9", True),
    ("Since there are 8 pints in a gallon, the final answer is 15 gallons * 8 pints / gallon = 120 pints\n#### 120", "120", True),
    ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72", "72", True),
    ("The answer is 5 + 3 = 8", "8", True),
    # A repetition loop overflows float() to inf; must not raise.
    ("The answer is " + "6" * 977, "6", False),
    ("Final answer: " + "9" * 400, "9", False),
]


# Cases that need the question to be decided. Each concluding sentence restates
# a quantity from the problem AFTER the answer, so "last number on the last
# line" picks the given, not the computed value.
_Q_CASES = [
    # (question, response, ground_truth, expected_correct)
    ("Kylar wants to buy 16 glasses. Each glass costs $5, but every second one "
     "costs 60% of the price. How much does he need to pay?",
     "Therefore, Kylar needs to pay 64 dollars for 16 glasses.", "64", True),
    ("Terry eats 2 yogurts a day. They are on sale at 4 for $5.00. How much "
     "does he spend on yogurt over 30 days?",
     "Therefore, Terry spends 75 dollars on yogurt over 30 days.", "75", True),
    ("Peter has $42. Movie tickets cost $14 each. How many times can he go?",
     "So, Peter can go to the movies 3 times with his $42.", "3", True),
    # The answer genuinely IS the last number -- the rule must not break this.
    ("She sold 48 clips in April and half as many in May. How many altogether?",
     "She sold 48 in April and 24 in May, for a total of 72.", "72", True),
    # Every candidate is a restatement -> fall back to the plain last number.
    ("He has 5 apples.", "He has 5 apples.", "5", True),
    # An equation on the final line beats the question filter: both operands
    # (5 and 2) are question quantities, and so is the answer (3).
    ("Jenny is dividing up a pizza with 12 slices. She gives 1/3 to Bill and "
     "1/4 to Mark. If Jenny eats 2 slices, how many slices are left?",
     "Slices left after Jenny eats:\n5 - 2 = 3 slices", "3", True),
]


def selftest() -> int:
    failures = 0
    for response, gt, expected in _CASES:
        result = verify(response, gt)
        if result["correct"] != expected:
            failures += 1
            print(f"FAIL  expected={expected}  got={result}  response={response!r}")
    for question, response, gt, expected in _Q_CASES:
        result = verify(response, gt, question=question)
        if result["correct"] != expected:
            failures += 1
            print(f"FAIL(q) expected={expected}  got={result}  response={response!r}")
    total = len(_CASES) + len(_Q_CASES)
    print(f"{total - failures}/{total} self-tests passed")
    return failures


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic GSM8K answer verifier")
    ap.add_argument("--response")
    ap.add_argument("--ground-truth")
    ap.add_argument("--batch", help="JSONL with 'response' and 'ground_truth' fields")
    ap.add_argument("--strict", action="store_true",
                    help="require an explicit answer marker; disable number fallbacks")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return 1 if selftest() else 0

    if args.batch:
        n = correct = 0
        methods: dict = {}
        with open(args.batch, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                res = verify(rec["response"], rec["ground_truth"], strict=args.strict)
                n += 1
                correct += res["correct"]
                methods[res["extraction_method"]] = methods.get(res["extraction_method"], 0) + 1
                print(json.dumps({**rec, **res}, ensure_ascii=False))
        acc = correct / n if n else 0.0
        print(json.dumps({"n": n, "correct": correct, "accuracy": round(acc, 4),
                          "extraction_methods": methods}), file=sys.stderr)
        return 0

    if args.response is None or args.ground_truth is None:
        ap.error("provide --response and --ground-truth, or --batch, or --selftest")

    print(json.dumps(verify(args.response, args.ground_truth, strict=args.strict),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
