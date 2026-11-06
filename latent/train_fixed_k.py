"""Phase 1: fixed-budget latent reasoning. Train with K latent steps (public output = short answer only), then evaluate.

Variants:
  --n_slots 0            pure LoRA direct-answer baseline (no latent tokens)
  --K 0                  M static soft tokens (z0), no recurrent computation (controls for extra tokens/params)
  --K 1,2,4,8            recurrent latent computation
  --distill codi         CODI-style self-distillation from an explicit-CoT teacher branch (same LoRA model)
  --eval_K a,b,c         evaluate the trained model at other step counts (test-time budget transfer)
"""
import argparse, json, os, time, math, sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.model import LatentReasoner, ModelConfig, LatentConfig
from latent.data import load_jsonl, is_correct
from latent.common import Collator, to_device, set_seed, JsonLogger, build_prompt, pad_batch


def evaluate(model, rows, K, bs=64, max_new_tokens=12, desc="eval", system=None):
    model.eval()
    tok = model.tok
    correct, per_task, preds = 0, {}, []
    timing = {}
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        prompts = [tok(build_prompt(tok, ex["question"], system), add_special_tokens=False)["input_ids"] for ex in chunk]
        p_ids, p_mask = pad_batch(prompts, tok.pad_token_id, left=True)
        p_ids, p_mask = p_ids.to(model.device), p_mask.to(model.device)
        gen = model.generate_fixed_k(p_ids, p_mask, K, max_new_tokens=max_new_tokens, timing=timing)
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for ex, t in zip(chunk, texts):
            ok = is_correct(t, ex["answer"])
            correct += ok
            d = per_task.setdefault(ex["task"], [0, 0]); d[0] += ok; d[1] += 1
            preds.append({"id": ex["id"], "task": ex["task"], "pred": t, "gold": ex["answer"], "ok": bool(ok)})
    res = {"acc": correct / max(1, len(rows)), "n": len(rows), "K": K,
           "per_task": {k: v[0] / v[1] for k, v in per_task.items()},
           "gpu_s_per_ex": timing.get("total_s", 0) / max(1, len(rows)),
           "latent_s_per_ex": timing.get("latent_s", 0) / max(1, len(rows))}
    model.train()
    return res, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n_slots", type=int, default=8)
    ap.add_argument("--ctx_layer", type=int, default=-1)
    ap.add_argument("--no_step_emb", action="store_true")
    ap.add_argument("--z_init", default="learned", choices=["learned", "last"])
    ap.add_argument("--recurrence", default="adapter", choices=["adapter", "lm"])
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--distill", default="none", choices=["none", "codi"])
    ap.add_argument("--aux_w", type=float, default=0.0, help="weight of verified intermediate-result supervision on latent slots")
    ap.add_argument("--distill_alpha", type=float, default=1.0, help="weight of teacher CoT CE")
    ap.add_argument("--distill_beta", type=float, default=1.0, help="weight of hidden-state alignment")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_latent", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_K", default="", help="comma list of extra K values to evaluate the trained model at")
    ap.add_argument("--n_train", type=int, default=0, help="subsample training set (0 = all)")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    set_seed(a.seed)
    log = JsonLogger(os.path.join(a.out, "log.jsonl"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ModelConfig(model_name=a.model, lora_r=a.lora_r,
                      latent=LatentConfig(n_slots=a.n_slots, ctx_layer=a.ctx_layer, use_step_emb=not a.no_step_emb, z_init=a.z_init, recurrence=a.recurrence))
    model = LatentReasoner(cfg, device=device)
    n_tr = sum(p.numel() for p in model.trainable_parameters())
    n_lat = sum(p.numel() for p in model.latent.parameters()) if model.latent is not None else 0
    print(f"trainable params: {n_tr/1e6:.1f}M (latent module {n_lat/1e6:.1f}M)")

    train = load_jsonl(os.path.join(a.data_dir, "train.jsonl"))
    if a.n_train:
        train = train[:a.n_train]
    val = load_jsonl(os.path.join(a.data_dir, "val.jsonl"))
    test = load_jsonl(os.path.join(a.data_dir, "test.jsonl"))
    heldout = load_jsonl(os.path.join(a.data_dir, "heldout.jsonl"))
    col = Collator(model.tok, with_cot=(a.distill == "codi"), n_slots=a.n_slots)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = DataLoader(train, batch_size=a.bs, shuffle=True, collate_fn=col, generator=g, drop_last=True)

    lora_p, rest_p = model.param_groups()
    groups = []
    if lora_p: groups.append({"params": lora_p, "lr": a.lr})
    if rest_p: groups.append({"params": rest_p, "lr": a.lr_latent})
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    model.train()
    step, t0 = 0, time.time()
    it = iter(dl)
    pbar = tqdm(total=a.steps, desc=f"K={a.K} M={a.n_slots}")
    while step < a.steps:
        opt.zero_grad(set_to_none=True)
        acc_loss, acc_ans, extra = 0.0, 0.0, {}
        for _micro in range(a.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            batch = to_device(batch, device)
            res = model.forward_fixed_k(batch["prompt_ids"], batch["prompt_mask"], batch["answer_ids"], batch["answer_mask"], a.K,
                                        return_hidden_at=batch["ans_hidden_idx"] if a.distill == "codi" else None)
            loss = res["loss"]
            if a.aux_w > 0 and res["z"] is not None:
                l_aux = model.aux_loss(res["z"], batch["inter_vals"], batch["inter_mask"])
                loss = loss + a.aux_w * l_aux; extra["aux"] = l_aux.item()
            if a.distill == "codi":
                t = model.forward_cot_teacher(batch["prompt_ids"], batch["prompt_mask"], batch["cot_ids"], batch["cot_mask"], batch["cot_hidden_idx"])
                hs, ht = res["hidden_at"].float(), t["hidden_at"].float().detach()
                # per-layer normalisation by teacher std (CODI), smooth-L1
                std = ht.std(dim=-1, keepdim=True).clamp(min=1e-3)
                l_dist = torch.nn.functional.smooth_l1_loss(hs / std, ht / std)
                loss = loss + a.distill_alpha * t["loss"] + a.distill_beta * l_dist
                extra = {"cot_ce": t["loss"].item(), "dist": l_dist.item()}
            (loss / a.grad_accum).backward()
            acc_loss += loss.item() / a.grad_accum; acc_ans += res["loss"].item() / a.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 20 == 0:
            log.log(step=step, loss=acc_loss, ans_ce=acc_ans, gn=float(gn), lr=sched.get_last_lr()[0], **extra)
            pbar.set_postfix(loss=f"{acc_ans:.3f}", **{k: f"{v:.3f}" for k, v in extra.items()})
        pbar.update(1)
        if a.eval_every and step % a.eval_every == 0 and step < a.steps:
            r, _ = evaluate(model, val, a.K, max_new_tokens=a.max_new_tokens)
            log.log(step=step, val=r); print(f"\n[step {step}] val acc={r['acc']:.3f} {r['per_task']}")
    train_time = time.time() - t0
    pbar.close()

    results = {"args": vars(a), "train_time_s": train_time, "trainable_params": n_tr, "latent_params": n_lat}
    Ks = [a.K] + [int(k) for k in a.eval_K.split(",") if k.strip() != ""]
    for K in Ks:
        for name, rows in [("val", val), ("test", test), ("heldout", heldout)]:
            r, preds = evaluate(model, rows, K, max_new_tokens=a.max_new_tokens)
            results[f"{name}@K{K}"] = r
            print(f"{name}@K{K}: acc={r['acc']:.3f} {json.dumps(r['per_task'])} gpu_s/ex={r['gpu_s_per_ex']*1000:.1f}ms latent={r['latent_s_per_ex']*1000:.2f}ms")
            if K == a.K:
                with open(os.path.join(a.out, f"preds_{name}.jsonl"), "w") as f:
                    for p in preds: f.write(json.dumps(p) + "\n")
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    if a.save:
        model.save(os.path.join(a.out, "ckpt"))
    print("DONE", a.out)


if __name__ == "__main__":
    main()
