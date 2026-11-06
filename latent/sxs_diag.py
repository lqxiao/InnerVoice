"""Router / incremental-decoding diagnostics for the SxS model.
1. teacher-forced route accuracy + switch accuracy on test trajectories (generalisation of the router)
2. incremental (KV cache, token by token, with mode switches) vs full-sequence forward on the gold trajectory: logits and router logits
3. free decoding trace: p_think around every mode switch with the surrounding text"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig, N_BUF
from latent.sxs_train import encode_example, collate

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", default="data_sxs/sxs_test_kk.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--n", type=int, default=40); ap.add_argument("--stride", type=int, default=17)
    a = ap.parse_args()
    m = SxSModel(SxSConfig(model_name=a.model), device="cuda"); m.load(a.ckpt); m.eval(); tok = m.tok
    rows = [json.loads(l) for l in open(a.data)][::a.stride][:a.n]
    # 1. teacher forced
    accs, sw_accs, n_sw = [], [], 0
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            encs = [encode_example(tok, ex, 2048) for ex in rows[i:i + 8]]; encs = [e for e in encs if e]
            b = {k: v.cuda() for k, v in collate(encs, tok.pad_token_id).items()}
            r = m.forward_train(b["ids"], b["mask"], b["is_think"], b["loss_mask"], b["route_label"], b["route_mask"])
            accs.append(r["route_acc"].item()); sw_accs.append(r["switch_acc"].item())
    print(f"[1] teacher-forced on {len(rows)} test trajectories: route_acc={sum(accs)/len(accs):.4f} switch_acc={sum(sw_accs)/len(sw_accs):.4f}")
    # 2. incremental vs full on one gold trajectory
    e = encode_example(tok, rows[0], 2048)
    ids = torch.tensor([e["ids"]], device="cuda"); typ = torch.tensor([e["is_think"]], device="cuda"); route = torch.tensor([e["route_label"]], device="cuda").bool()
    Lp = e["loss_mask"].index(1); L = ids.shape[1]
    with torch.no_grad():
        m.moe.route = route; m.moe.buf = None
        full = m.lm(inputs_embeds=m.inputs_embeds(ids, typ), attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
        full_logits = full.logits[0].float(); full_router = m.moe.router_logits[0].float().clone()
        # incremental: prefill prompt, then one token at a time with the gold route (expert of the NEXT token)
        m.moe.route = route[:, :Lp]
        out = m.lm(inputs_embeds=m.inputs_embeds(ids[:, :Lp], typ[:, :Lp]), attention_mask=torch.ones(1, Lp, dtype=torch.long, device="cuda"), use_cache=True, return_dict=True)
        cache = out.past_key_values; inc_logits = [out.logits[0, -1].float()]; inc_router = [m.moe.router_logits[0, -1].float()]
        for t in range(Lp, L - 1):
            m.moe.route = route[:, t:t + 1]
            out = m.lm(inputs_embeds=m.inputs_embeds(ids[:, t:t + 1], typ[:, t:t + 1]), attention_mask=torch.ones(1, t + 1, dtype=torch.long, device="cuda"),
                       past_key_values=cache, position_ids=torch.tensor([[t]], device="cuda"), use_cache=True, return_dict=True)
            cache = out.past_key_values; inc_logits.append(out.logits[0, -1].float()); inc_router.append(m.moe.router_logits[0, -1].float())
        inc_logits = torch.stack(inc_logits); inc_router = torch.stack(inc_router)
        d_log = (inc_logits - full_logits[Lp - 1:L - 1]).abs().max().item(); d_rt = (inc_router - full_router[Lp - 1:L - 1]).abs().max().item()
        agree = ((inc_logits.argmax(-1) == full_logits[Lp - 1:L - 1].argmax(-1)).float().mean().item())
        gold_next = ids[0, Lp:L]; tf_acc_inc = (inc_logits.argmax(-1) == gold_next).float().mean().item(); tf_acc_full = (full_logits[Lp - 1:L - 1].argmax(-1) == gold_next).float().mean().item()
        rt_inc = ((inc_router > 0) == route[0, Lp - 1:L - 1]).float().mean().item()
    print(f"[2] incremental vs full: max|dlogits|={d_log:.3f} max|drouter|={d_rt:.3f} argmax agreement={agree:.3f}; teacher-forced next-token acc full={tf_acc_full:.3f} inc={tf_acc_inc:.3f}; router acc (inc)={rt_inc:.3f}")
    # 3. free decoding trace around switches
    prompt = tok.apply_chat_template([{"role": "user", "content": rows[0]["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]], device="cuda")
    ev, th, ch, tr = m.stream(pids, lambda s: None, max_new_tokens=300, max_think_per_block=256)
    print(f"[3] free decoding: {len(th)} think blocks, {len(ch)} chunks; switches={sum(1 for i in range(1,len(tr)) if tr[i]['mode']!=tr[i-1]['mode'])}")
    print("   gold chunks:", rows[0]["chunks"]); print("   pred chunks:", ch)
    print("   p_think trace (step, mode, p):", [(t["step"], t["mode"][0], round(t["p_think"], 2)) for t in tr[:60]])
    boundary_trace(m, rows, n=3)

def boundary_trace(m, rows, n=3, max_new=300):
    tok = m.tok
    for ex in rows[:n]:
        prompt = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]], device="cuda")
        ev, th, ch, tr = m.stream(ids, lambda s: None, max_new_tokens=max_new, max_think_per_block=256)
        print("[4] ==", ex["id"], "gold:", ex["chunks"])
        for i, t in enumerate(tr):
            tid = t.get("tid"); txt = tok.decode([tid]) if tid is not None else ""
            if "\n" in txt or (i > 0 and tr[i - 1]["mode"] != t["mode"]):
                nxt = tr[i + 1] if i + 1 < len(tr) else None
                print(f"     step {i:3d} mode={t['mode'][0]} tok={txt!r:10s} p_think(next)={None if nxt is None else round(nxt['p_think'], 3)} next={None if nxt is None else nxt['mode']}")
        print("     pred chunks:", ch)

if __name__ == "__main__":
    main()
