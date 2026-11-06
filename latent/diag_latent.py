"""Diagnostics on a trained fixed-K model: are the latent tokens actually used?
  acc(normal) vs acc(z shuffled across batch) vs acc(z = z0, i.e. K=0 path) ; ||z_K - z_0|| ; injected-token norm vs embedding norm."""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.model import LatentReasoner, ModelConfig, LatentConfig
from latent.data import load_jsonl, is_correct
from latent.common import build_prompt, pad_batch

def acc_with(model, rows, K, z_transform=None, bs=64):
    tok = model.tok; c = 0
    for i in range(0, len(rows), bs):
        ch = rows[i:i + bs]
        p = [tok(build_prompt(tok, ex["question"]), add_special_tokens=False)["input_ids"] for ex in ch]
        ids, m = pad_batch(p, tok.pad_token_id, left=True)
        g = model.generate_fixed_k(ids.cuda(), m.cuda(), K, max_new_tokens=12, z_transform=z_transform)
        for ex, t in zip(ch, tok.batch_decode(g, skip_special_tokens=True)):
            c += is_correct(t, ex["answer"])
    return c / len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n_slots", type=int, default=8); ap.add_argument("--lora_r", type=int, default=16); ap.add_argument("--ctx_layer", type=int, default=-1); ap.add_argument("--z_init", default="learned"); ap.add_argument("--recurrence", default="adapter")
    ap.add_argument("--data", default="data/test.jsonl"); ap.add_argument("--limit", type=int, default=600); ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg = ModelConfig(model_name=a.model, lora_r=a.lora_r, latent=LatentConfig(n_slots=a.n_slots, ctx_layer=a.ctx_layer, z_init=a.z_init, recurrence=a.recurrence))
    m = LatentReasoner(cfg, device="cuda"); m.load(a.ckpt); m.eval()
    rows = load_jsonl(a.data)[:a.limit]
    out = {"ckpt": a.ckpt, "K": a.K}
    with torch.no_grad():
        out["acc_normal"] = acc_with(m, rows, a.K)
        out["acc_shuffled_z"] = acc_with(m, rows, a.K, z_transform=lambda z: z[torch.randperm(z.shape[0], device=z.device)])
        out["acc_z0"] = acc_with(m, rows, 0)
        out["acc_zero_latent"] = acc_with(m, rows, a.K, z_transform=lambda z: torch.zeros_like(z))
        out["acc_noise_z"] = acc_with(m, rows, a.K, z_transform=lambda z: torch.randn_like(z))
        # state statistics on one batch
        tok = m.tok
        p = [tok(build_prompt(tok, ex["question"]), add_special_tokens=False)["input_ids"] for ex in rows[:64]]
        ids, mk = pad_batch(p, tok.pad_token_id, left=True)
        ctx, _, cache, lengths = m.encode_prompt(ids.cuda(), mk.cuda())
        if a.recurrence == "lm":
            z, states, _, _, _ = m.think_lm(ctx, mk.cuda(), a.K, cache, mk.cuda(), lengths, return_all=True)
        else:
            z, states = m.think(ctx, mk.cuda(), a.K, return_all=True)
        out["z_step_delta_norm"] = [float((states[i + 1] - states[i]).norm(dim=-1).mean()) for i in range(a.K)]
        out["z_norm"] = float(z.norm(dim=-1).mean()); out["zK_minus_z0_norm"] = float((z - states[0]).norm(dim=-1).mean())
        out["z_across_example_std"] = float(z.std(dim=0).mean())   # 0 => context independent
        lat = m.latent_tokens(z).float(); emb = m.embed.weight.float()
        out["inject_token_rms"] = float(lat.pow(2).mean().sqrt()); out["embed_rms"] = float(emb.pow(2).mean().sqrt()); out["inject_scale"] = float(m.inject_scale)
        # can the aux head decode verified intermediates from the slots?  (slot i <-> intermediate i)
        ar = [ex for ex in rows if ex.get("intermediates")]
        tot = {}; 
        for Kx in sorted(set([0, 1, 2, 4, 8, a.K])):
            hit_last, n_last, hit_slot, n_slot = 0, 0, [0] * a.n_slots, [0] * a.n_slots
            for i in range(0, len(ar), 64):
                ch = ar[i:i + 64]
                p = [tok(build_prompt(tok, ex["question"]), add_special_tokens=False)["input_ids"] for ex in ch]
                ids, mk = pad_batch(p, tok.pad_token_id, left=True)
                ctx, _, _, _ = m.encode_prompt(ids.cuda(), mk.cuda()); z = m.think(ctx, mk.cuda(), Kx)
                lg = m.aux_head(z.float())
                d = torch.stack([lg[..., j * 10:(j + 1) * 10].argmax(-1) for j in range(4)], -1); sg = lg[..., 40:42].argmax(-1)
                val = (d[..., 0] * 1000 + d[..., 1] * 100 + d[..., 2] * 10 + d[..., 3]) * torch.where(sg == 1, -1, 1)
                for b, ex in enumerate(ch):
                    inter = ex["intermediates"][:a.n_slots]
                    for si, v in enumerate(inter):
                        ok = int(val[b, si].item()) == int(v); hit_slot[si] += ok; n_slot[si] += 1
                    si = len(inter) - 1; hit_last += int(val[b, si].item()) == int(inter[-1]); n_last += 1
            tot[f"K{Kx}"] = {"final_intermediate(=answer)_acc": hit_last / max(1, n_last), "per_slot_acc": [h / n if n else None for h, n in zip(hit_slot, n_slot)]}
        out["aux_decode"] = tot
    print(json.dumps(out, indent=1))
    if a.out: json.dump(out, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
