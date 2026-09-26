#!/usr/bin/env python3
"""How much of a LoRA update survives being merged into bf16 weights?

train_rlaif.py and generate.py rebuild an RL checkpoint's parents by merging
each adapter into the base weights (merge_and_unload) in bf16. bf16 keeps 8
bits of mantissa, so adding an update much smaller than a weight's rounding
step changes nothing. This replays the same merge on CPU for every target
module (base + SFT, then + RLAIF) and compares the update that actually
landed in the weights with the update the adapter describes.

    python scripts/merge_precision.py        # -> outputs/merge_precision.json
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file

BASE = "Qwen/Qwen2.5-3B-Instruct"


def lora_deltas(adapter):
    cfg = json.loads(Path(adapter, "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(Path(adapter, "adapter_model.safetensors")))
    out = {}
    for k, a in sd.items():
        if ".lora_A." not in k:
            continue
        b = sd[k.replace(".lora_A.", ".lora_B.")]
        name = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
        # peft on GPU forms B@A in the weight dtype, then adds it in place
        out[name] = ((b.float() @ a.float()) * scale).to(torch.bfloat16)
    return out


def main():
    snap = Path(snapshot_download(BASE, allow_patterns=["*.safetensors", "*.json"]))
    shards = sorted(snap.glob("*.safetensors"))
    sft, rlaif = lora_deltas("outputs/sft-yodadistill"), lora_deltas("outputs/rlaif-lora")
    assert set(rlaif) <= set(sft), "RLAIF targets a module SFT did not"

    tot = {"sft": [0.0, 0.0, 0, 0], "rlaif": [0.0, 0.0, 0, 0]}   # err^2, delta^2, lost, n
    per_module = {}
    for shard in shards:
        with safe_open(str(shard), "pt") as f:
            for name in f.keys():
                if name not in rlaif:
                    continue
                w = f.get_tensor(name).to(torch.bfloat16)
                for stage, d in (("sft", sft[name]), ("rlaif", rlaif[name])):
                    merged = w + d                      # bf16 add, as in merge_and_unload
                    landed = merged.float() - w.float()
                    want = d.float()
                    e2 = float(((landed - want) ** 2).sum())
                    d2 = float((want ** 2).sum())
                    lost = int(((landed == 0) & (want != 0)).sum())
                    t = tot[stage]
                    t[0] += e2; t[1] += d2; t[2] += lost; t[3] += want.numel()
                    if stage == "rlaif":
                        per_module[name] = {
                            "rel_error": (e2 / d2) ** 0.5,
                            "lost_frac": lost / want.numel(),
                            "delta_over_weight": float(want.norm() / w.float().norm())}
                    w = merged
    summary = {}
    for stage, (e2, d2, lost, n) in tot.items():
        summary[stage] = {"rel_error": (e2 / d2) ** 0.5, "lost_frac": lost / n}
        print(f"{stage:6s} update: relative error after bf16 merge {summary[stage]['rel_error']:.3f}"
              f"   entries rounded away entirely {summary[stage]['lost_frac']:.1%}")
    kinds = {}
    for name, r in per_module.items():
        kinds.setdefault(name.split(".")[-2], []).append(r)
    for kind, rs in sorted(kinds.items()):
        print(f"  rlaif {kind:10s} rel error {sum(r['rel_error'] for r in rs)/len(rs):.3f}  "
              f"lost {sum(r['lost_frac'] for r in rs)/len(rs):.1%}  "
              f"|dW|/|W| {sum(r['delta_over_weight'] for r in rs)/len(rs):.4f}")
    result = {"summary": summary, "per_module": per_module}
    if torch.cuda.is_available():
        result["functional_kl"] = functional_kl()
    Path("outputs/merge_precision.json").write_text(json.dumps(result, indent=2) + "\n")
    print("-> outputs/merge_precision.json")


@torch.no_grad()
def functional_kl():
    """Exact per-token KL(RLAIF as trained || RLAIF merged into bf16), on
    RLAIF's own greedy answers: the same units as scripts/kl_drift.py, so the
    merge can be compared with how far the RLVR arms moved."""
    import statistics as st
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import kl_drift as kd          # same text encoding and bootstrap as the drift table
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token

    def build(merge_rlaif):
        m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
        m = PeftModel.from_pretrained(m, "outputs/sft-yodadistill").merge_and_unload()
        m.__dict__.pop("peft_config", None)
        m = PeftModel.from_pretrained(m, "outputs/rlaif-lora")
        return (m.merge_and_unload() if merge_rlaif else m).eval()

    trained, merged = build(False), build(True)
    out = {}
    for dom, fn in kd.DOMAINS.items():
        items = kd.encode(tok, kd.jl(f"outputs/rlaif/{fn}"))
        kls = []
        for i in range(0, len(items), 4):
            chunk = items[i:i + 4]
            L = max(len(p) + len(r) for p, r in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            resp = torch.zeros((len(chunk), L), dtype=torch.bool)
            for j, (p, r) in enumerate(chunk):
                ids[j, :len(p) + len(r)] = torch.tensor(p + r)
                att[j, :len(p) + len(r)] = 1
                resp[j, len(p) - 1:len(p) + len(r) - 1] = True
            ids, att, resp = ids.cuda(), att.cuda(), resp.cuda()
            lp = torch.log_softmax(trained(input_ids=ids, attention_mask=att).logits.float(), -1)
            lq = torch.log_softmax(merged(input_ids=ids, attention_mask=att).logits.float(), -1)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            kls += [float(kl[j][resp[j]].mean()) for j in range(len(chunk))]
        out[dom] = {"mean": st.mean(kls), "ci": kd.boot(kls), "n": len(kls)}
        print(f"merge effect, {dom:8s}: KL(trained || merged) per token {out[dom]['mean']:.5f} "
              f"CI [{out[dom]['ci'][0]:.5f}, {out[dom]['ci'][1]:.5f}]")
    return out


if __name__ == "__main__":
    main()
