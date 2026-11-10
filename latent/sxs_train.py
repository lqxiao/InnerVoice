"""SFT of the readable think/speak model on interleaved trajectories.
Sequence = chat prompt + [block_1][block_2]... ; each block tokenized separately so type boundaries are exact.
Think blocks end with "\n"; speak chunks end with "\n"; the last block ends with <|im_end|>.
Loss = CE(all response tokens; think weighted by --think_weight) + lambda * BCE(router: next-token-is-think)."""
import argparse, json, os, sys, time, math, random
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig, N_BUF
from latent.common import set_seed, JsonLogger

def encode_example(tok, ex, max_len: int, think_trunc: float = 1.0, drop_think: bool = False, fmt: str = "sxs"):
    """Returns dict of lists: ids, is_think, loss_mask, route_label, route_mask (route_label[t] = type of token t+1)."""
    prompt = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids, typ = list(p_ids), [0] * len(p_ids)
    steps = ex["steps"]
    if fmt == "cot":      # Standard CoT baseline (SxS paper): all think blocks first, then all speak blocks
        steps = [s for s in steps if s["mode"] == "think"] + [s for s in steps if s["mode"] == "speak"]
    elif fmt == "nothink":
        steps = [s for s in steps if s["mode"] == "speak"]
    if fmt == "tags":   # SxS-paper format: interleaved blocks with VISIBLE tags, all tokens are ordinary answer tokens (no router signal)
        ids_t, typ_t = list(p_ids), [0] * len(p_ids)
        for i, st in enumerate(steps):
            last = i == len(steps) - 1
            text = ("<think>\n" + st["text"].strip() + "\n</think>\n") if st["mode"] == "think" else (st["text"].strip() + ("<|im_end|>" if last else "\n"))
            t = tok(text, add_special_tokens=False)["input_ids"]; ids_t += t; typ_t += [0] * len(t)
        if len(ids_t) > max_len: return None
        L = len(ids_t); Lp = len(p_ids)
        return {"ids": ids_t, "is_think": typ_t, "loss_mask": [0] * Lp + [1] * (L - Lp), "route_label": [0] * L, "route_mask": [0] * L}
    for i, s in enumerate(steps):
        last = i == len(steps) - 1
        text = s["text"].strip()
        if s["mode"] == "think":
            if drop_think:
                continue
            if think_trunc < 1.0:
                words = text.split(); text = " ".join(words[:max(1, int(len(words) * think_trunc))])
            t = tok(text + "\n", add_special_tokens=False)["input_ids"]; typ += [1] * len(t)
        else:
            t = tok(text + ("<|im_end|>" if last else "\n"), add_special_tokens=False)["input_ids"]; typ += [0] * len(t)
        ids += t
    if len(ids) > max_len:
        return None
    L = len(ids); Lp = len(p_ids)
    loss_mask = [0] * Lp + [1] * (L - Lp)
    route_label = typ[1:] + [0]
    route_mask = [0] * (Lp - 1) + [1] * (L - Lp) + [0]   # positions whose next token is a response token
    return {"ids": ids, "is_think": typ, "loss_mask": loss_mask, "route_label": route_label, "route_mask": route_mask}

def collate(encs, pad_id):
    L = max(len(e["ids"]) for e in encs)
    def pad(key, val=0):
        return torch.tensor([e[key] + [val] * (L - len(e[key])) for e in encs])
    return {"ids": pad("ids", pad_id), "mask": torch.tensor([[1] * len(e["ids"]) + [0] * (L - len(e["ids"])) for e in encs]),
            "is_think": pad("is_think"), "loss_mask": pad("loss_mask"), "route_label": pad("route_label"), "route_mask": pad("route_mask")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--data", default="data_sxs/sxs_train.jsonl"); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr_rest", type=float, default=1e-4, help="experts, router, e_think")
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--think_weight", type=float, default=1.0); ap.add_argument("--switch_weight", type=float, default=5.0)
    ap.add_argument("--max_len", type=int, default=1536); ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_expert_ft", action="store_true", help="experts via LoRA only"); ap.add_argument("--lora_r", type=int, default=16); ap.add_argument("--no_e_think", action="store_true"); ap.add_argument("--grad_ckpt", action="store_true"); ap.add_argument("--shared_ffn", action="store_true", help="ablation: single shared FFN + gate (no experts)")
    ap.add_argument("--mix_trunc", type=float, default=0.0, help="fraction of examples with truncated think (budget robustness)")
    ap.add_argument("--n_train", type=int, default=0); ap.add_argument("--init", default=""); ap.add_argument("--format", default="sxs", choices=["sxs", "cot", "nothink", "tags"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    set_seed(a.seed); log = JsonLogger(os.path.join(a.out, "log.jsonl")); dev = "cuda"
    model = SxSModel(SxSConfig(model_name=a.model, lora_r=a.lora_r, expert_full_ft=not a.no_expert_ft, use_e_think=not a.no_e_think, grad_ckpt=a.grad_ckpt, shared_ffn=a.shared_ffn), device=dev)
    if a.init: model.load(a.init)
    tok = model.tok
    rows = [json.loads(l) for l in open(a.data)]
    if a.n_train: rows = rows[:a.n_train]
    rng = random.Random(a.seed)
    n_tr = sum(p.numel() for p in model.trainable_parameters()); print(f"trainable params {n_tr/1e6:.1f}M, examples {len(rows)}")
    lora_p, rest_p = model.param_groups()
    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr}, {"params": rest_p, "lr": a.lr_rest}], weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    model.train(); step = 0; t0 = time.time(); skipped = 0
    pbar = tqdm(total=a.steps, desc="sxs-sft")
    order = []; 
    while step < a.steps:
        opt.zero_grad(set_to_none=True); agg = {}
        for _ in range(a.grad_accum):
            encs = []
            while len(encs) < a.bs:
                if not order: order = list(range(len(rows))); rng.shuffle(order)
                ex = rows[order.pop()]
                tr = rng.uniform(0.2, 0.8) if (a.mix_trunc > 0 and rng.random() < a.mix_trunc) else 1.0
                e = encode_example(tok, ex, a.max_len, think_trunc=tr, fmt=a.format)
                if e is None: skipped += 1; continue
                encs.append(e)
            b = {k: v.to(dev) for k, v in collate(encs, tok.pad_token_id).items()}
            res = model.forward_train(b["ids"], b["mask"], b["is_think"], b["loss_mask"], b["route_label"], b["route_mask"], think_weight=a.think_weight, switch_weight=a.switch_weight)
            loss = res["loss_ce"] + a.lam * res["loss_route"]
            (loss / a.grad_accum).backward()
            for k, v in res.items():
                if torch.is_tensor(v) and v.numel() == 1 and not torch.isnan(v): agg[k] = agg.get(k, 0) + v.item() / a.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 10 == 0:
            log.log(step=step, gn=float(gn), lr=sched.get_last_lr()[0], skipped=skipped, **agg)
            pbar.set_postfix({k: f"{v:.3f}" for k, v in agg.items() if k in ("loss_ce", "ce_think", "ce_answer", "loss_route", "route_acc", "switch_acc")})
        pbar.update(1)
        if step % 500 == 0 or step == a.steps:
            model.save(os.path.join(a.out, "ckpt"))
    pbar.close()
    json.dump({"train_time_s": time.time() - t0, "skipped_too_long": skipped}, open(os.path.join(a.out, "train_summary.json"), "w"))
    print("DONE", a.out)

if __name__ == "__main__":
    main()
