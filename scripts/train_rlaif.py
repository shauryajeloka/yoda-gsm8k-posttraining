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

REWARD
------
The fitted style classifier from scripts/persona_classifier.py -- a logistic
regression over surface style features, scored as log-odds rather than as a
probability (see PersonaReward). Deliberately NOT the LLM judge, for two
reasons:

  1. Cost. GRPO with G=8 over a few hundred steps is tens of thousands of
     completions; judging all of them is slow and expensive.
  2. Honesty. The judge is the EVALUATION metric. Optimizing the thing you
     then report measures reward hacking, not persona quality. Keeping the
     judge held out means the Week 2 comparison against SFT is credible.

The cost of that choice: the classifier is a linear model over surface
features, so it is easy to hack -- the policy can learn to stuff every
sentence with inversion cues. That is what --beta (the KL penalty) is holding
back, and what the logged diagnostics are there to catch. The signature of
hacking is the classifier reward climbing while response length, cue density
or repetition blow up. Watch those columns, not just the reward.

Usage
    python scripts/train_rlaif.py --adapter outputs/sft-lora \
        --out outputs/rlaif-lora --steps 200
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

    def __call__(self, texts):
        floor = 0.0 if self.transform == "prob" else -20.0
        return [floor if not t.strip() else self._score(t) for t in texts]


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
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)

    B, Tm1 = tok_logp.shape
    idx = torch.arange(Tm1, device=input_ids.device).unsqueeze(0)
    gen_mask = (idx >= (prompt_lens.unsqueeze(1) - 1)) & \
               (attention_mask[:, 1:] > 0)
    return tok_logp, gen_mask.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default="outputs/sft-lora",
                    help="SFT checkpoint; becomes both the init and the KL anchor")
    ap.add_argument("--out", default="outputs/rlaif-lora")
    ap.add_argument("--prompts", default="data/persona/general_yoda_train.jsonl")
    ap.add_argument("--frozen-eval", default="data/persona/persona_eval_prompts.jsonl")
    ap.add_argument("--reward-model", default="outputs/persona_clf.json")
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

    prompts = load_prompts(args.prompts, args.frozen_eval)
    print(f"{len(prompts)} training prompts (disjoint from the frozen eval set)")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading {args.model} and merging SFT adapter {args.adapter} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()      # weights are now exactly the SFT model

    # Fresh LoRA on top. Adapter on = policy; adapter off = SFT reference.
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.config.use_cache = True

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
        rewards = reward_fn(completions)

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
        logp, gmask = token_logprobs(model, seq, attn, plens)
        with torch.no_grad(), model.disable_adapter():
            ref_logp, _ = token_logprobs(model, seq, attn, plens)

        # k3 KL estimator: non-negative and lower variance than (logp - ref).
        d = ref_logp - logp
        kl_tok = torch.exp(d) - d - 1.0
        ntok = gmask.sum(1).clamp(min=1)

        ratio = torch.exp(logp - logp.detach())         # == 1 for one inner epoch
        a = adv.unsqueeze(1)
        pg = -torch.min(ratio * a,
                        torch.clamp(ratio, 1 - args.clip_eps,
                                    1 + args.clip_eps) * a)
        # The PRINTED loss will look absurdly small (~1e-4) and that is
        # correct, not a bug. With one inner epoch rho == 1 exactly, so the
        # policy-gradient term evaluates to -mean(A), and advantages are
        # mean-zero within every group by construction -- so its VALUE
        # cancels and only the KL term survives in the number you see. The
        # GRADIENT does not cancel: d(rho)/d(theta) = d(logp)/d(theta) != 0,
        # which recovers the usual REINFORCE estimator -A * grad log p.
        # Judge progress by reward and KL, never by the loss value.
        loss = (((pg + args.beta * kl_tok) * gmask).sum(1) / ntok).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        kl_v = float(((kl_tok * gmask).sum(1) / ntok).mean())
        words = sum(len(c.split()) for c in completions) / len(completions)
        cues = sum(features(c)[0] for c in completions) / len(completions)
        rec = {"step": step, "reward": sum(rewards) / len(rewards),
               "reward_std": float(r_t.std(unbiased=False)), "kl": kl_v,
               "mean_words": words, "mean_cues": cues,
               "loss": float(loss), "secs": round(time.time() - t0, 1)}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if step % 5 == 0 or step == 1:
            print(f"{step:5d} {rec['reward']:7.3f} {rec['reward_std']:6.3f} "
                  f"{kl_v:7.4f} {words:6.1f} {cues:5.2f}")
        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(args.out)
            Path(args.out, "training_config.json").write_text(json.dumps({
                "stage": "rlaif", "algo": "grpo", "init_from": args.adapter,
                "reward": "persona_classifier", "reward_model": args.reward_model,
                "reward_transform": args.reward_transform,
                "beta_kl": args.beta, "group_size": args.group_size,
                "temperature": args.temperature, "lr": args.lr,
                "steps_done": step, "seed": args.seed,
                "system_prompt": sys_policy,
                "note": "generate.py must use the same --system-prompt value. "
                        "The LLM judge was NOT used as the reward and remains "
                        "a held-out evaluator.",
            }, indent=2))
    logf.close()
    print(f"\nsaved {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
