#!/usr/bin/env python3
"""
RLAIF for persona adherence, via GRPO, starting from the SFT checkpoint.

Why GRPO and not PPO: GRPO drops the value network. Instead of learning a
baseline, it samples a GROUP of completions for the same prompt and normalizes
their rewards within that group. With a 3B policy on one GPU that halves the
memory and removes a whole component that can be silently mistrained.

    A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

    L = -mean_i[ min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i) ]
        + beta * KL(policy || reference)

where rho_i is the importance ratio against the policy that generated the
sample. We take one gradient step per batch of generations, so rho == 1 at the
step it is used and the clipping term is inactive; it is kept because it costs
nothing and becomes load-bearing the moment you raise --inner-epochs above 1.

REFERENCE MODEL
---------------
The KL anchor must be the SFT model, not the base model. Rather than holding a
second 3B copy in memory, the SFT adapter is merged into the base weights, and
a FRESH LoRA is attached on top for the policy. Then:

    adapter enabled   -> policy
    adapter disabled  -> the SFT model exactly

so `with model.disable_adapter()` gives exact reference log-probs for free.

REWARD  (--reward-kind, default "claude")
-----------------------------------------
An LLM scores each completion ON ITS OWN, plus deterministic guardrails. Four
options exist and three of them were measured and rejected; the history is
worth keeping because each failure was invisible until tested.

  claude  (default)  Claude Haiku scores against the persona rubric, then flat
                     penalties apply for Star Wars vocabulary, self-naming and
                     filler spam. On a degradation harness with known-correct
                     orderings it ranks the clean original above a damaged one
                     0.77 (voice stops halfway), 0.73 (word salad), 0.93
                     (Star Wars splice) and 0.95 ("Yoda I am").

  llm                A local 7B doing the same job. Rejected: given the SAME
                     anti-gaming rubric it scored a spliced-in Star Wars
                     reference ABOVE the clean original (0.602 vs 0.464) and
                     self-naming higher still (0.620). Told the rule plainly,
                     it did not follow it.

  linear             The 51-feature style classifier. Rejected as the primary
                     reward: it pins 34% of on-policy completions at P>=0.99,
                     where they are indistinguishable and contribute no
                     gradient, and it scores 0.000 on the "Yoda I am" test.

  neural             Bradley-Terry model distilled from judge rankings.
                     Rejected: comparative prompts made the 7B answer from
                     POSITION (46.9% slot-A against 25% chance; 0.590
                     self-consistency under order reversal), so the labels were
                     corrupt and the distilled model failed its gate.

Why guardrails at all: Claude alone was nearly indifferent to the flat
prohibitions (+0.056 for Star Wars, +0.037 for self-naming) while being
decisive about genuine persona quality (+0.336). Those three rules need no
judgement -- they are closed vocabularies -- so they are enforced in code and
the judge is left to do what it is good at. This is a HYBRID reward, and the
writeup should describe it as one.

INDEPENDENCE: scoring uses Haiku; eval_persona.py reports with Sonnet.
Different models, same developer -- partial independence, not full. Say so.

The signature of reward hacking is the reward climbing while response length,
cue density or repetition blow up. Watch those columns, not just the reward.

Usage
    python scripts/train_rlaif.py --adapter outputs/sft-yodadistill \
        --out outputs/rlaif-lora --steps 150
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("scripts")))
from persona_similarity import features            # noqa: E402
from persona_classifier import predict, apply_std  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------

class PersonaReward:
    """Persona reward from the fitted style classifier. No API, no GPU.

    transform="logit" (default) returns the pre-sigmoid score w.x + b rather
    than the probability, and that choice matters more than it looks.

    Measured on the 150 SFT persona completions: median P(persona) is 0.965
    and 37% sit at or above 0.99. GRPO normalises rewards WITHIN a sampled
    group, so when several completions in a group are all pinned at the
    ceiling they become indistinguishable, the advantage collapses to ~0, and
    that group contributes no gradient at all. Most of the batch would be
    wasted.

    The log-odds are unbounded and keep separating completions the sigmoid has
    already flattened, so a group of four "all essentially perfect" samples
    still yields a usable ranking. Pass --reward-transform prob to get the
    saturating version back.
    """

    def __init__(self, path, transform="logit"):
        m = json.loads(Path(path).read_text())
        self.w, self.b, self.mu, self.sd = m["w"], m["b"], m["mu"], m["sd"]
        self.transform = transform

    def _score(self, text):
        x = apply_std(features(text), self.mu, self.sd)
        if self.transform == "prob":
            return predict(x, self.w, self.b)
        return self.b + sum(wi * xi for wi, xi in zip(self.w, x))

    def __call__(self, texts, prompts=None):
        floor = 0.0 if self.transform == "prob" else -20.0
        return [floor if not t.strip() else self._score(t) for t in texts]


class VerifierReward:
    """RLVR reward: a program grades correctness. No opinions to bias, no
    saturation knob to tune -- verify(completion, ground_truth) is 1.0 or 0.0.

    Sparse by construction: a group whose samples are all right or all wrong
    has zero within-group variance and contributes no gradient, which is why
    the prompt file is expected to come from scripts/rlvr_prepass.py (problems
    the CURRENT policy solves sometimes). The verifier is question-aware and
    overflow-hardened; both of those were bugs once, fixed before they could
    kill a run from inside the reward.
    """

    def __init__(self, prompt_file):
        sys.path.insert(0, str(Path("data/verifier")))
        from gsm8k_verifier import verify
        self._verify = verify
        self.gt = {}
        for line in open(prompt_file, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            q = (r.get("prompt") or r.get("question")).strip()
            self.gt[q] = str(r["ground_truth"])
        print(f"verifier reward: {len(self.gt)} problems with ground truth")

    def __call__(self, texts, prompts):
        out = []
        for t, q in zip(texts, prompts):
            gt = self.gt.get(q.strip())
            if gt is None:
                raise KeyError(f"no ground truth for prompt: {q[:80]!r}")
            ok = bool(t.strip()) and self._verify(t, gt, question=q)["correct"]
            out.append(1.0 if ok else 0.0)
        self.last_components = {"verifier": sum(out) / max(len(out), 1)}
        self.last_verifier = out
        return out


class CombinedReward:
    """The Week-3 objective: R = verifier + lambda * persona.

    The verifier is blind to style, and the styling control measured what that
    blindness is worth: dropping the voice on maths answers buys up to ~15
    points of accuracy. Pure-verifier RL is therefore expected to shed the
    persona (that is Arm A's job to demonstrate); the persona term is the
    counterweight. lambda sets the exchange rate -- at lambda=0.5 a completion
    cannot profit from de-Yodifying unless doing so actually flips it from
    wrong to right.
    """

    def __init__(self, prompt_file, claude_model, lam, workers=12):
        from claude_reward import ClaudeJudgeReward
        self.v = VerifierReward(prompt_file)
        self.p = ClaudeJudgeReward(claude_model, workers=workers)
        self.lam = lam

    def __call__(self, texts, prompts):
        rv = self.v(texts, prompts)
        rp = self.p(texts, prompts)
        self.last_components = {
            "verifier": sum(rv) / max(len(rv), 1),
            "persona": sum(rp) / max(len(rp), 1),
        }
        self.last_verifier = rv
        return [a + self.lam * b for a, b in zip(rv, rp)]


class NeuralReward:
    """Reward model distilled from the LLM judge's rankings.

    Built by scripts/train_reward_model.py with a Bradley-Terry objective, so
    its outputs are unbounded and scale-free. That needs no logit/prob choice:
    GRPO normalises within the group, and only the ORDERING inside a group
    affects the gradient.

    Scored on (prompt, completion) because that is how it was trained -- a
    completion is only good relative to what was asked.
    """

    def __init__(self, path, batch_size=16, max_len=640):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel

        cfg = json.loads(Path(path, "adapter_config.json").read_text())
        base = cfg["base_model_name_or_path"]
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        m = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=1, torch_dtype=torch.float32)
        m.config.pad_token_id = self.tok.pad_token_id
        self.model = PeftModel.from_pretrained(m, path).eval()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev)
        self.batch_size, self.max_len = batch_size, max_len
        print(f"neural reward model: {base} + {path}")

    def __call__(self, texts, prompts):
        enc_in = [self.tok.apply_chat_template(
                      [{"role": "user", "content": q},
                       {"role": "assistant", "content": t if t.strip() else " "}],
                      tokenize=False)
                  for q, t in zip(prompts, texts)]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(enc_in), self.batch_size):
                e = self.tok(enc_in[i:i + self.batch_size], return_tensors="pt",
                             padding=True, truncation=True,
                             max_length=self.max_len).to(self.dev)
                out.extend(self.model(**e).logits.squeeze(-1).float().tolist())
        # An empty completion cannot be good Yoda; floor it well below the
        # observed score range so it can never win a group by accident.
        return [(-20.0 if not t.strip() else r) for t, r in zip(texts, out)]


# --------------------------------------------------------------------------
# Prompt pool
# --------------------------------------------------------------------------

def load_prompts(path, frozen_eval):
    """General persona prompts, with the frozen eval set excluded by assertion.

    Training RLAIF on the prompts you later evaluate on would inflate the
    Week 2 result, so this refuses to run rather than warning.
    """
    import re

    def norm(t):
        return re.sub(r"\W+", " ", t.lower()).strip()

    prompts = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        p = r.get("prompt") or r["messages"][0]["content"]
        prompts.append(p)

    held = {norm(json.loads(l)["prompt"])
            for l in open(frozen_eval, encoding="utf-8") if l.strip()}
    clash = [p for p in prompts if norm(p) in held]
    if clash:
        raise SystemExit(
            f"{len(clash)} RLAIF training prompts also appear in the frozen "
            f"eval set ({frozen_eval}). Training on them would contaminate "
            f"the Week 2 comparison. First: {clash[0]!r}")
    return prompts


# --------------------------------------------------------------------------
# Log-probs
# --------------------------------------------------------------------------

def token_logprobs(model, input_ids, attention_mask, prompt_lens):
    """Per-token log-probs of the GENERATED tokens only.

    Returns (logprobs, mask) both [B, T-1]. The mask is zero on prompt tokens
    and on padding, so nothing before the assistant turn contributes to the
    loss -- the same completion-only discipline train_sft.py uses.
    """
    import torch

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    # log p(target) = logit[target] - logsumexp(logits). Equivalent to
    # gathering from log_softmax, without materialising a second [B, T, V]
    # fp32 tensor -- at RLVR lengths (~600 tok x 152k vocab) each such copy is
    # ~2GB per 6 sequences, and the old path held several for backward.
    tok_logp = (torch.gather(logits, 2, targets.unsqueeze(-1)).squeeze(-1)
                - torch.logsumexp(logits, dim=-1))

    B, Tm1 = tok_logp.shape
    idx = torch.arange(Tm1, device=input_ids.device).unsqueeze(0)
    gen_mask = (idx >= (prompt_lens.unsqueeze(1) - 1)) & \
               (attention_mask[:, 1:] > 0)
    return tok_logp, gen_mask.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    # sft-yodadistill, not sft-lora: 68.4% vs 61.4% GSM8K (+7.0pp, exact
    # McNemar p=0.0043) with persona statistically indistinguishable (p=0.497).
    # The old default pointed at the Week-1 checkpoint, so running this bare
    # would have silently RL'd from the wrong policy.
    ap.add_argument("--adapter", default="outputs/sft-yodadistill",
                    help="SFT checkpoint; becomes both the init and the KL anchor")
    ap.add_argument("--out", default="outputs/rlaif-lora")
    ap.add_argument("--prompts", default="data/persona/general_yoda_train.jsonl")
    ap.add_argument("--frozen-eval", default="data/persona/persona_eval_prompts.jsonl")
    ap.add_argument("--reward-model", default="outputs/persona_clf.json")
    ap.add_argument("--reward-kind", default="claude",
                    choices=["claude", "verifier", "combined", "llm", "linear",
                             "neural"],
                    help="llm = an LLM scores each completion directly via its "
                         "Yes/No log-odds (this is the AI feedback, and the "
                         "default). linear = the 51-feature style classifier, "
                         "which pins 34%% of on-policy completions at P>=0.99. "
                         "neural = a Bradley-Terry model distilled from judge "
                         "rankings; kept for reference but it failed its gate "
                         "because comparative prompts made the judge answer "
                         "from position (0.590 self-consistency, 0.493 for "
                         "two-way). See scripts/llm_reward.py.")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="scoring LLM for --reward-kind llm")
    # Haiku scores; eval_persona.py reports with Sonnet. Optimising against a
    # judge and then reporting that same judge would make the Week 2 persona
    # number a training metric. Same developer, so not fully independent --
    # say so in the writeup rather than overclaiming.
    ap.add_argument("--claude-model", default="claude-haiku-4-5-20251001",
                    help="scoring model for --reward-kind claude")
    ap.add_argument("--judge-workers", type=int, default=12)
    ap.add_argument("--rlvr-prompts", default="work/rlvr_prompts.jsonl",
                    help="jsonl of {prompt, ground_truth} for reward kinds "
                         "verifier/combined; produced by rlvr_prepass.py")
    ap.add_argument("--micro-batch", type=int, default=6,
                    help="sequences per forward/backward chunk; gradients "
                         "accumulate, so the update is identical to full-batch")
    ap.add_argument("--lambda-persona", type=float, default=0.5,
                    help="persona weight in the combined reward")
    ap.add_argument("--reward-transform", default="logit",
                    choices=["logit", "prob"],
                    help="logit avoids the ceiling that flattens 37%% of SFT "
                         "completions to P>=0.99 and kills their advantage")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8,
                    help="G: completions per prompt. Needs to be >1 or the "
                         "group-normalised advantage is identically zero.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature. Greedy (0) gives G identical "
                         "completions, zero reward variance and zero gradient.")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.05,
                    help="KL penalty toward the SFT model. The main knob: too "
                         "low and the policy hacks the reward, too high and "
                         "nothing moves.")
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log", default=None, help="JSONL of per-step metrics")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.reward_kind in ("verifier", "combined"):
        # Maths prompts with ground truth, leak-checked against the frozen
        # GSM8K eval set by normalised question text.
        import re as _re
        _norm = lambda t: _re.sub(r"\W+", " ", t.lower()).strip()
        held = {_norm(json.loads(l)["question"])
                for l in open("data/math/gsm8k_eval.jsonl", encoding="utf-8")}
        prompts = []
        for line in open(args.rlvr_prompts, encoding="utf-8"):
            if line.strip():
                q = (json.loads(line).get("prompt")
                     or json.loads(line).get("question")).strip()
                prompts.append(q)
        clash = [q for q in prompts if _norm(q) in held]
        if clash:
            raise SystemExit(f"{len(clash)} RLVR prompts appear in the frozen "
                             f"eval set. First: {clash[0]!r}")
    else:
        prompts = load_prompts(args.prompts, args.frozen_eval)
    print(f"{len(prompts)} training prompts (disjoint from the frozen eval set)")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    # Rebuild the full adapter stack, base-most first. An adapter trained on
    # top of a merged parent (e.g. RLAIF on SFT) must be applied to that same
    # merged parent; loading only the top adapter onto the raw base yields a
    # model that is neither stage, silently, and would also anchor KL to the
    # wrong reference. generate.py applies the identical chain at eval time.
    chain, cur, seen = [], args.adapter, set()
    while cur:
        if cur in seen:
            raise SystemExit(f"cycle in adapter init_from chain at {cur}")
        seen.add(cur); chain.append(cur)
        cfg_p = Path(cur, "training_config.json")
        cur = (json.loads(cfg_p.read_text()).get("init_from")
               if cfg_p.exists() else None)
    for a in reversed(chain):
        print(f"  merging adapter: {a}")
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    # weights are now exactly the --adapter stage

    # Fresh LoRA on top. Adapter on = policy; adapter off = SFT reference.
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.config.use_cache = True

    if args.reward_kind == "verifier":
        reward_fn = VerifierReward(args.rlvr_prompts)
    elif args.reward_kind == "combined":
        reward_fn = CombinedReward(args.rlvr_prompts, args.claude_model,
                                   args.lambda_persona, workers=args.judge_workers)
    elif args.reward_kind == "claude":
        from claude_reward import ClaudeJudgeReward
        reward_fn = ClaudeJudgeReward(args.claude_model, workers=args.judge_workers)
    elif args.reward_kind == "llm":
        from llm_reward import LLMJudgeReward
        reward_fn = LLMJudgeReward(args.judge_model)
    elif args.reward_kind == "neural":
        reward_fn = NeuralReward(args.reward_model)
    else:
        reward_fn = PersonaReward(args.reward_model, args.reward_transform)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)

    # The system-prompt policy must match SFT, or we optimise a different
    # distribution than the one we trained and will evaluate.
    cfg_path = Path(args.adapter, "training_config.json")
    sys_policy = json.loads(cfg_path.read_text()).get("system_prompt", "qwen_default") \
        if cfg_path.exists() else "qwen_default"
    print(f"system_prompt policy: {sys_policy}")

    def build(prompt):
        msgs = [{"role": "user", "content": prompt}]
        if sys_policy == "none":
            msgs = [{"role": "system", "content": ""}] + msgs
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    _probe = json.loads(Path("outputs/persona_clf.json").read_text())
    logf = open(args.log or Path(args.out, "rlaif_log.jsonl"), "w",
                encoding="utf-8")
    print(f"\n{'step':>5} {'reward':>7} {'std':>6} {'KL':>7} {'words':>6} "
          f"{'cues':>5}  (watch: reward up + words/cues up == hacking)")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = random.sample(prompts, min(args.prompts_per_step, len(prompts)))
        texts, groups = [], []
        for p in batch:
            texts.extend([build(p)] * args.group_size)
            groups.append(p)

        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = enc["input_ids"].shape[1]

        model.eval()
        with torch.no_grad():
            seq = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                pad_token_id=tok.pad_token_id)

        completions = tok.batch_decode(seq[:, prompt_len:],
                                       skip_special_tokens=True)
        completions = [c.strip() for c in completions]
        # A completion that hits the cap cannot state its answer, so the
        # verifier scores it wrong -- a hidden reward for shorter reasoning.
        # Tracked every step; the cap is sized so this stays near zero.
        gen_len = (seq[:, prompt_len:] != tok.pad_token_id).sum(1)
        trunc_frac = float((gen_len >= args.max_new_tokens).float().mean())
        rprompts = [p for p in batch for _ in range(args.group_size)]
        rewards = reward_fn(completions, rprompts)
        comps = getattr(reward_fn, "last_components", {})
        # Live voice-shedding tracker: the linear classifier is too gameable to
        # be a reward, but as a free per-step INSTRUMENT it shows the persona
        # draining in real time -- which matters most in Arm A, where nothing
        # in the reward would otherwise notice.
        style_probe = sum(
            predict(apply_std(features(c), _probe["mu"], _probe["sd"]),
                    _probe["w"], _probe["b"])
            for c in completions if c.strip()) / max(
                sum(1 for c in completions if c.strip()), 1)

        # Group-normalised advantages. Zero variance inside a group means the
        # samples are indistinguishable to the reward, so they teach nothing.
        adv = []
        r_t = torch.tensor(rewards, dtype=torch.float32)
        for g in range(len(batch)):
            sl = r_t[g * args.group_size:(g + 1) * args.group_size]
            centred = sl - sl.mean()
            denom = sl.std(unbiased=False)
            adv.append(centred / denom if denom > 1e-6 else torch.zeros_like(sl))
        adv = torch.cat(adv).to(model.device)

        attn = (seq != tok.pad_token_id).long()
        attn[:, :prompt_len] = enc["attention_mask"]
        plens = torch.full((seq.shape[0],), prompt_len, device=model.device)

        model.train()
        opt.zero_grad(set_to_none=True)
        # The loss is a mean over sequences of per-sequence token means, so it
        # decomposes exactly: L = (1/B) sum_i L_i. Running policy and reference
        # passes in micro-batches and accumulating gradients of
        # (sum of chunk L_i) / B is identical to the full-batch step, at a
        # fraction of the peak memory.
        B = seq.shape[0]
        loss_v = kl_acc = 0.0
        for c0 in range(0, B, args.micro_batch):
            sl = slice(c0, min(c0 + args.micro_batch, B))
            logp, gmask = token_logprobs(model, seq[sl], attn[sl], plens[sl])
            with torch.no_grad(), model.disable_adapter():
                ref_logp, _ = token_logprobs(model, seq[sl], attn[sl], plens[sl])
            # k3 KL estimator: non-negative, lower variance than (logp - ref).
            d = ref_logp - logp
            kl_tok = torch.exp(d) - d - 1.0
            ntok = gmask.sum(1).clamp(min=1)
            ratio = torch.exp(logp - logp.detach())     # == 1 for one inner epoch
            a = adv[sl].unsqueeze(1)
            pg = -torch.min(ratio * a,
                            torch.clamp(ratio, 1 - args.clip_eps,
                                        1 + args.clip_eps) * a)
            # At beta = 0 the KL term is left out rather than multiplied by
            # zero: 0 * inf = nan if any masked position's estimate overflows.
            tok_loss = pg if args.beta == 0 else pg + args.beta * kl_tok
            per_seq = (tok_loss * gmask).sum(1) / ntok
            chunk_loss = per_seq.sum() / B
            chunk_loss.backward()
            loss_v += float(chunk_loss)
            kl_acc += float(((kl_tok * gmask).sum(1) / ntok).sum())
        # The PRINTED loss will look absurdly small (~1e-4) and that is
        # correct, not a bug. With one inner epoch rho == 1 exactly, so the
        # policy-gradient term evaluates to -mean(A), and advantages are
        # mean-zero within every group by construction -- so its VALUE
        # cancels and only the KL term survives in the number you see. The
        # GRADIENT does not cancel: d(rho)/d(theta) = d(logp)/d(theta) != 0,
        # which recovers the usual REINFORCE estimator -A * grad log p.
        # Judge progress by reward and KL, never by the loss value.
        loss = loss_v
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0))
        skipped = not math.isfinite(gnorm)
        if skipped:
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()

        kl_v = kl_acc / B
        words = sum(len(c.split()) for c in completions) / len(completions)
        cues = sum(features(c)[0] for c in completions) / len(completions)
        rec = {"step": step, "reward": sum(rewards) / len(rewards),
               "reward_std": float(r_t.std(unbiased=False)), "kl": kl_v,
               "mean_words": words, "mean_cues": cues, "style_probe": style_probe,
               "truncated": trunc_frac, "grad_norm": gnorm, "skipped": skipped,
               "loss": float(loss), "secs": round(time.time() - t0, 1)}
        rec.update(comps)
        vs = getattr(reward_fn, "last_verifier", None)
        if vs is not None:
            # Share of groups whose VERIFIER scores vary: the part of the batch
            # carrying maths gradient. (Computed on verifier scores, not the
            # total reward, because the persona term makes every combined
            # group look mixed.) Draining toward 0 = re-filter the pool.
            G = args.group_size
            live = sum(1 for g in range(len(batch))
                       if len(set(vs[g*G:(g+1)*G])) > 1)
            rec["mixed_groups"] = live / len(batch)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if step % 5 == 0 or step == 1:
            extra = "".join(f" {k[:4]}={v:5.3f}" for k, v in comps.items())
            if "mixed_groups" in rec:
                extra += f" mix={rec['mixed_groups']:.2f}"
            print(f"{step:5d} {rec['reward']:7.3f} {rec['reward_std']:6.3f} "
                  f"{kl_v:7.4f} {words:6.1f} {cues:5.2f} "
                  f"probe={style_probe:.3f} trunc={trunc_frac:.2f}{extra}")
        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(args.out)
            Path(args.out, "training_config.json").write_text(json.dumps({
                "stage": ("rlvr" if args.reward_kind in ("verifier", "combined") else "rlaif"), "algo": "grpo", "init_from": args.adapter,
                # Record what actually ran. This was hardcoded to
                # "persona_classifier" when --reward-kind was added, which made
                # one completed 150-step run unattributable to any reward and
                # therefore unusable -- the checkpoint had to be discarded.
                "reward": args.reward_kind,
                "reward_model": (args.claude_model if args.reward_kind in ("claude", "combined")
                                 else args.judge_model if args.reward_kind == "llm"
                                 else "data/verifier/gsm8k_verifier.py"
                                 if args.reward_kind == "verifier"
                                 else args.reward_model),
                "lambda_persona": (args.lambda_persona
                                   if args.reward_kind == "combined" else None),
                "rlvr_prompts": (args.rlvr_prompts if args.reward_kind
                                 in ("verifier", "combined") else None),
                "max_new_tokens": args.max_new_tokens,
                "reward_transform": (args.reward_transform
                                     if args.reward_kind == "linear" else None),
                "beta_kl": args.beta, "group_size": args.group_size,
                "temperature": args.temperature, "lr": args.lr,
                "steps_done": step, "seed": args.seed,
                "system_prompt": sys_policy,
                "note": ("generate.py must use the same --system-prompt value. "
                         "The held-out evaluator (eval_persona.py, Sonnet) is never "
                         "used as a reward; reward=" + args.reward_kind + "."),
            }, indent=2))
    logf.close()
    print(f"\nsaved {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
