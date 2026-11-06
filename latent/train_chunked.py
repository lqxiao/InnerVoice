"""Phase 2 training.
  --mode random : stage A, budget-robust chunked answer model (k ~ U{0..Kmax} per chunk), Answer CE only.
  --mode labels : stage B, Loss = Answer_CE + lambda * Gate_BCE with k*/labels from weak teacher labels (+ counterfactual file).
"""
import argparse, json, os, sys, time, math, random
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.model import LatentReasoner, ModelConfig, LatentConfig
from latent.data import load_jsonl
from latent.chunked import ChunkedCollator, bucket_batches
from latent.common import to_device, set_seed, JsonLogger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="checkpoint dir to initialise from (e.g. stage A)")
    ap.add_argument("--mode", default="random", choices=["random", "labels"])
    ap.add_argument("--cf_labels", default="", help="json {id: [[ok_k0..ok_Kmax] per chunk]} from label_counterfactual.py")
    ap.add_argument("--no_buffer", action="store_true", help="baseline 6: gate without buffer awareness (features zeroed, labels ignore buffer)")
    ap.add_argument("--Kmax", type=int, default=8)
    ap.add_argument("--n_slots", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gate_pos_weight", type=float, default=1.0)
    ap.add_argument("--freeze_lm", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_latent", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    set_seed(a.seed); log = JsonLogger(os.path.join(a.out, "log.jsonl"))
    device = "cuda"
    cfg = ModelConfig(model_name=a.model, lora_r=a.lora_r, latent=LatentConfig(n_slots=a.n_slots, max_steps=max(16, a.Kmax + 1)),
                      gate=(a.mode == "labels"), gate_use_buffer=not a.no_buffer)
    model = LatentReasoner(cfg, device=device)
    if a.init:
        model.load(a.init); print("loaded", a.init)
    if a.freeze_lm:
        for p in model.lm.parameters(): p.requires_grad_(False)
    train = load_jsonl(os.path.join(a.data_dir, "chunked_train.jsonl"))
    cf = json.load(open(a.cf_labels)) if a.cf_labels else None
    col = ChunkedCollator(model.tok, a.Kmax, mode=a.mode, cf_labels=cf, buffer_aware=not a.no_buffer, tau=a.tau, seed=a.seed)
    rng = random.Random(a.seed)
    lora_p, rest_p = model.param_groups()
    groups = []
    if lora_p: groups.append({"params": lora_p, "lr": a.lr})
    if rest_p: groups.append({"params": rest_p, "lr": a.lr_latent})
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    batches = bucket_batches(train, a.bs, rng); bi = 0
    model.train(); step = 0; t0 = time.time()
    pbar = tqdm(total=a.steps, desc=f"chunked/{a.mode}")
    stats = {"think": 0, "speak": 0}
    while step < a.steps:
        if bi >= len(batches):
            batches = bucket_batches(train, a.bs, rng); bi = 0
        batch = to_device(col(batches[bi]), device); bi += 1
        res = model.forward_chunked(batch["prompt_ids"], batch["prompt_mask"], batch["chunk_ids"], batch["chunk_mask"], batch["k_steps"], a.Kmax,
                                    gate_feats=batch.get("gate_feats"), gate_labels=batch.get("gate_labels"), gate_mask=batch.get("gate_mask"),
                                    gate_pos_weight=a.gate_pos_weight)
        loss = res["loss_ce"]
        extra = {}
        if "loss_gate" in res:
            loss = loss + a.lam * res["loss_gate"]; extra["gate"] = res["loss_gate"].item()
            with torch.no_grad():
                m = batch["gate_mask"].bool(); pred = (res["gate_logits"][m] > 0).float(); lab = batch["gate_labels"][m]
                extra["gate_acc"] = (pred == lab).float().mean().item()
                extra["speak_frac_lab"] = lab.mean().item(); extra["speak_frac_pred"] = pred.mean().item()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 20 == 0:
            log.log(step=step, loss=loss.item(), ce=res["loss_ce"].item(), gn=float(gn), **extra)
            pbar.set_postfix(ce=f"{res['loss_ce'].item():.3f}", **{k: f"{v:.3f}" for k, v in extra.items()})
        pbar.update(1)
    pbar.close()
    model.save(os.path.join(a.out, "ckpt"))
    json.dump({"train_time_s": time.time() - t0, "args": vars(a)}, open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)
    print("DONE", a.out)

if __name__ == "__main__":
    main()
