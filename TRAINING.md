# Week 1 runbook — SFT + baseline evaluation

Everything below is Week 1 (SFT) only. The RLAIF and RLVR *reward* components
already exist (`data/judges/yoda_persona_judge.md`, `data/verifier/gsm8k_verifier.py`);
the RL training loops are Weeks 2–3 and are not written yet.

---

## The one decision that can invalidate the experiment

Qwen2.5's chat template injects this when no system message is given:

> `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`

If you ever put *"You are Yoda"* in the system prompt, you are measuring
**prompting**, not post-training, and the whole comparison collapses — a base
model with a good system prompt will beat a fine-tuned model without one.

So: the persona is learned into the weights, never prompted. `train_sft.py`
records its `--system-prompt` setting in `training_config.json`, and
`generate.py` **refuses to run** with a different one. Keep the default
(`qwen_default`) for both, and the Base row stays an honest out-of-the-box
baseline.

---

## Order of operations

Baseline generations must happen **before** training, on the same frozen files:

```bash
bash infra/runpod_bootstrap.sh          # once per pod
tmux new -s week1 'bash infra/run_week1.sh 2>&1 | tee outputs/week1.log'
```

`run_week1.sh` does: base generations → SFT → post-SFT generations →
all the metrics that need no API key. Judge scoring runs afterwards from
anywhere.

### Sizing (why this is not an overnight job)

The dataset is 290,432 tokens per epoch; three epochs is 871k tokens. LoRA on a
3B model at that scale is **roughly 15–25 minutes** on a single A40, and
generation for both checkpoints across all three eval sets is another ~30
minutes. Budget about an hour of GPU, i.e. **one to two dollars** on a mid-tier
card — check current RunPod pricing, these move.

A 24 GB card is enough for LoRA. Full fine-tuning needs 80 GB (AdamW states for
3B in bf16), which is why LoRA is the default.

The thing that *will* take real time is Week 3: GRPO/PPO generates many
completions per step, and that is where an overnight run earns its name.

---

## What gets measured, and with what

| Question | Instrument | Where it goes |
|---|---|---|
| Is the math right? | `eval_math.py` → verifier, exact answer match | results table |
| Does it sound like Yoda? | `eval_persona.py` → judge 1–5 | results table |
| Did SFT move it toward the character? | `persona_similarity.py` | Checkpoint 1 requirement |
| Independent style check | `persona_classifier.py` → P(persona) | judge cross-check |
| Is training done? | validation loss in `train_sft.py` | checkpoint selection only |

### Why cross-entropy is *not* the persona metric

Validation loss is the right tool for deciding when to stop training and it is
reported per epoch. It is the wrong tool for the results table:

1. **It needs reference answers we deliberately do not have.** The 150 persona
   eval prompts ship as prompts only. Writing reference Yoda answers would
   measure how well the model predicts *one author's* wording.
2. **It conflates style with content.** Low loss on a reference answer means
   the model guessed that answer's argument and vocabulary, not that it sounds
   like Yoda.
3. **It is not comparable across your four checkpoints.** SFT minimises this
   loss directly, so of course it drops. RLAIF does not optimise likelihood at
   all — after RLAIF the loss can *rise* while persona genuinely improves. A
   metric that moves in the wrong direction for one of your stages cannot
   anchor the table.

The judge score works on every checkpoint including the base model, is
independent of content, and is the same signal RLAIF optimises — which is
exactly why it also needs the independent cross-check below.

### The judge, and how to catch it being gamed

`eval_persona.py` scores at temperature 0, one completion at a time, retries a
malformed response once, then records `null` and **excludes** it from the mean
rather than coercing it to a number. It prints, alongside the score:

* mean response length
* count of responses containing "hmm"/"yes"-type interjections
* count of responses containing Star Wars vocabulary

A persona score that climbs while those climb is the reward-hacking signature
the writeup asks about. Pair it with `persona_classifier.py`: that number is a
fixed function of surface syntax and cannot be talked into a higher score, so
**judge up + P(persona) flat** is strong evidence of gaming, and **both up**
is strong evidence of real improvement.

`persona_classifier.py` is fitted on a controlled contrast — the 1500 Yoda
rewrites against the GSM8K reference solutions for the *same 1500 problems*,
so content is held fixed and only voice varies. Held-out accuracy **0.985**,
AUC **0.998**, and its most informative features are inversion cues,
subject-final sentences and trailing auxiliaries — persona markers, not topic
words.

### Answering the trade-off question properly

`gsm8k_persona_math_eval.jsonl` (150 items) is scored on **both** axes from the
**same** completions, so `{"math_correct": …, "persona_score": …}` is a paired
observation. That is what distinguishes "persona costs accuracy" from "we
generated twice under different conditions". Report the 2×2 table of
correct/incorrect × in-character/not.

Keep decoding identical across stages (greedy, `--temperature 0`, same
`--max-new-tokens`). Different sampling settings between rows would confound
everything.

---

## Expected Week 1 shape

Not predictions — the things worth noticing:

* Base persona score should be near **1**. Qwen has no reason to sound like
  Yoda, and its default system prompt says it is Qwen.
* Base GSM8K accuracy for Qwen2.5-3B-Instruct is typically in the **60–75%**
  range with greedy decoding; your number is whatever the verifier says.
* The interesting result is what SFT does to *accuracy*. The rewrites preserve
  every operation, so there is no reason for reasoning to degrade — but the
  model is now also spending capacity on voice, and the answer format changed.
  If accuracy drops, check first whether it is a **formatting** loss (the
  verifier failing to find an answer) rather than a reasoning loss:
  `eval_math.py` prints the extraction-method breakdown for exactly this.

---

## If you want to run it

I need three things from you, none of which are credentials I should handle:

1. **A pod**, and your RunPod API key exported in *your own* shell
   (`export RUNPOD_API_KEY=…`) if you want me to drive `runpodctl`. Don't paste
   keys into the chat — I'll use what the environment provides.
2. **A judge API key** the same way (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`).
3. **Confirmation of the spend**, since renting a GPU costs real money.

Alternatively: push this directory to a repo, `git clone` it on the pod, and
run the two commands above yourself.
