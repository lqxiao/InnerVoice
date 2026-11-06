"""Phase 3: joint RL of the Think/Speak gate and the answer policy with custom batched rollouts.

Rollout (no grad): stochastic gate (Bernoulli(p_emit)) decides per chunk how many latent steps to take; chunk text is decoded
(greedy or sampled). A nominal clock (step_cost per latent step, GEN_COST_PER_TOKEN per decoded token, TTS audio durations)
drives buffer features and the simulator-based reward:
    R = w_acc*chunk_acc + w_all*all_correct - w_ttfc*TTFC - w_stall*P95_stall - w_gpu*gpu_time - w_prem*premature_frac
Filler (empty / unparsable chunks) is never 'meaningful' so it does not improve TTFC, and wrong early chunks are premature.
Update: GRPO-style group-normalised advantages over G rollouts per prompt; REINFORCE on gate decisions (+ optional answer tokens),
recomputed with gradient by a teacher-forced pass over the rolled-out trajectory (forward_chunked), + small BCE anchor to the
supervised labels to limit collapse.
"""
import argparse, json, os, sys, time, math, random, statistics
import torch, torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.model import LatentReasoner, ModelConfig, LatentConfig
from latent.data import load_jsonl, is_correct
from latent.chunked import ChunkedCollator, bucket_batches, parse_chunk_answers, GEN_COST_PER_TOKEN, N_BUF
from latent.simulator import AudioModel, simulate
from latent.common import to_device, set_seed, JsonLogger

NEWLINE = 198

@torch.no_grad()
def rollout(model, batch, Kmax, step_cost, audio, sample_gate=True, temperature=0.0, max_chunk_tokens=14, eps=0.0):
    """Batched stochastic streaming rollout. Returns dict with k_steps (B,N), chunk_ids (B,N,Lc), chunk_mask, gate_feats (B,N,Kmax+1,N_BUF),
    texts (B,N), events per sequence."""
    B, N, _ = batch["chunk_ids"].shape; M = model.M; dev = model.device
    tok = model.tok
    ctx, p_logits, cache, lengths = model.encode_prompt(batch["prompt_ids"], batch["prompt_mask"])
    ctx_list, mask_list = [ctx], [batch["prompt_mask"].bool()]
    last_hidden = ctx[:, -1]; full_mask = batch["prompt_mask"]; cur = lengths
    z = model.latent.init_state(B, ctx, batch["prompt_mask"]); ar = torch.arange(B, device=dev)
    clock = torch.zeros(B); play_end = torch.zeros(B)
    k_steps = torch.zeros(B, N, dtype=torch.long); feats = torch.zeros(B, N, Kmax + 1, N_BUF)
    all_tokens = [[None] * N for _ in range(B)]; events = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    for i in range(N):
        ctx_all = torch.cat(ctx_list, 1); mask_all = torch.cat(mask_list, 1)
        decided = torch.zeros(B, dtype=torch.bool, device=dev)
        z_cur = z
        for j in range(Kmax + 1):
            buf = (play_end - clock).clamp(min=0)
            f = torch.stack([buf / 10.0, torch.full((B,), j / max(1, Kmax)), torch.full((B,), i / 8.0), torch.full((B,), step_cost)], -1)
            feats[:, i, j] = f
            p = torch.sigmoid(model.gate(last_hidden, z_cur, f.to(dev)))
            if j >= Kmax:
                speak = torch.ones(B, dtype=torch.bool, device=dev)
            elif sample_gate:
                pe = p * (1 - eps) + 0.5 * eps
                speak = torch.bernoulli(pe).bool()
            else:
                speak = p >= 0.5
            newly = speak & ~decided
            k_steps[newly.cpu(), i] = j
            decided = decided | speak
            if decided.all():
                break
            z_next = model.latent.step(z_cur, ctx_all, mask_all, j)
            z_cur = torch.where(decided[:, None, None], z_cur, z_next)
            think = (~decided).cpu()
            clock[think] += step_cost
            for b in torch.nonzero(think).flatten().tolist():
                events[b].append({"type": "think", "dt": step_cost})
        z = z_cur
        # speak chunk i
        lat = model.latent_tokens(z)
        full_mask = torch.cat([full_mask, torch.ones(B, M, dtype=full_mask.dtype, device=dev)], 1)
        out = model.lm_forward(inputs_embeds=lat, attention_mask=full_mask, position_ids=model.segment_positions(cur, M), past_key_values=cache)
        cache = out.past_key_values; seg_h = [out.hidden_states[model.cfg.latent.ctx_layer]]
        cur = cur + M; nl = out.logits[:, -1]
        gen = []; done = finished.clone()
        for t in range(max_chunk_tokens):
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(nl.float() / temperature, -1), 1).squeeze(1)
            else:
                nxt = nl.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
            gen.append(nxt)
            is_end = (nxt == NEWLINE) | (nxt == model.eos_id) | (nxt == tok.eos_token_id)
            finished = finished | ((nxt == model.eos_id) | (nxt == tok.eos_token_id))
            step_mask = (~done).long()
            full_mask = torch.cat([full_mask, step_mask[:, None]], 1)
            out = model.lm_forward(input_ids=nxt[:, None], attention_mask=full_mask, position_ids=cur[:, None], past_key_values=cache)
            cache = out.past_key_values; cur = cur + step_mask
            seg_h.append(out.hidden_states[model.cfg.latent.ctx_layer]); nl = out.logits[:, -1]
            done = done | is_end
            if done.all():
                break
        G = torch.stack(gen, 1)                                  # (B, T)
        valid = (G != tok.pad_token_id)
        h = torch.cat(seg_h, 1)                                  # (B, M+T, d) ; last real hidden per seq
        n_tok = valid.sum(-1)
        seg_mask = torch.cat([torch.ones(B, M, dtype=torch.bool, device=dev), valid], 1)
        ctx_list.append(h); mask_list.append(seg_mask)
        last_hidden = h[ar, (M + n_tok - 1).clamp(min=0)]
        for b in range(B):
            ids = G[b][valid[b]].tolist(); all_tokens[b][i] = ids
            text = tok.decode(ids, skip_special_tokens=True).strip()
            dt = GEN_COST_PER_TOKEN * max(1, len(ids)); clock[b] += dt
            dur = audio.duration(text); start = max(clock[b].item(), play_end[b].item()); play_end[b] = start + dur
            events[b].append({"type": "speak", "dt": dt, "text": text, "k": int(k_steps[b, i])})
    Lc = max(1, max(len(t) for row in all_tokens for t in row))
    chunk_ids = torch.full((B, N, Lc), tok.pad_token_id, dtype=torch.long); chunk_mask = torch.zeros(B, N, Lc, dtype=torch.long)
    for b in range(B):
        for i in range(N):
            ids = all_tokens[b][i]
            if ids:
                chunk_ids[b, i, :len(ids)] = torch.tensor(ids); chunk_mask[b, i, :len(ids)] = 1
    return {"k_steps": k_steps, "chunk_ids": chunk_ids, "chunk_mask": chunk_mask, "gate_feats": feats, "events": events,
            "texts": [[tok.decode(t, skip_special_tokens=True).strip() if t else "" for t in row] for row in all_tokens]}

def reward_fn(events, texts, ex, audio, w):
    parsed = parse_chunk_answers(texts)
    oks = [bool(is_correct(parsed.get(i + 1, ""), g)) for i, g in enumerate(ex["sub_answers"])]
    si = 0
    for e in events:
        if e["type"] == "speak":
            ok = oks[si] if si < len(oks) else False
            e["ok"] = ok; e["meaningful"] = (si + 1) in parsed; si += 1
    sim = simulate(events, audio)
    chunk_acc = sum(oks) / len(oks)
    R = (w["acc"] * chunk_acc + w["all"] * float(all(oks)) - w["ttfc"] * sim.ttfc - w["stall"] * sim.p95_stall
         - w["gpu"] * sim.gpu_time - w["prem"] * (sim.premature / max(1, sim.n_chunks)))
    return R, {"chunk_acc": chunk_acc, "all": float(all(oks)), "ttfc": sim.ttfc, "p95": sim.p95_stall, "gpu": sim.gpu_time, "prem": sim.premature / max(1, sim.n_chunks), "steps": sim.hidden_steps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--init", required=True, help="stage-B checkpoint (gate trained with BCE)")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--Kmax", type=int, default=8)
    ap.add_argument("--n_slots", type=int, default=8)
    ap.add_argument("--no_buffer", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--step_cost", type=float, default=0.2)
    ap.add_argument("--step_cost_range", default="", help="e.g. 0.05,0.4 to sample per step")
    ap.add_argument("--temperature", type=float, default=0.0, help=">0 samples answer tokens too")
    ap.add_argument("--eps", type=float, default=0.05, help="gate exploration mixing")
    ap.add_argument("--w_acc", type=float, default=1.0); ap.add_argument("--w_all", type=float, default=0.5)
    ap.add_argument("--w_ttfc", type=float, default=0.1); ap.add_argument("--w_stall", type=float, default=0.3)
    ap.add_argument("--w_gpu", type=float, default=0.05); ap.add_argument("--w_prem", type=float, default=0.5)
    ap.add_argument("--answer_pg", type=float, default=0.0, help="weight of answer-token policy gradient")
    ap.add_argument("--bce_anchor", type=float, default=0.1)
    ap.add_argument("--lr_gate", type=float, default=5e-5); ap.add_argument("--lr_lm", type=float, default=1e-5)
    ap.add_argument("--freeze_lm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    set_seed(a.seed); log = JsonLogger(os.path.join(a.out, "log.jsonl")); dev = "cuda"
    cfg = ModelConfig(model_name=a.model, lora_r=16, latent=LatentConfig(n_slots=a.n_slots, max_steps=max(16, a.Kmax + 1)), gate=True, gate_use_buffer=not a.no_buffer)
    model = LatentReasoner(cfg, device=dev); model.load(a.init)
    if a.freeze_lm:
        for p in model.lm.parameters(): p.requires_grad_(False)
    train = load_jsonl(os.path.join(a.data_dir, "chunked_train.jsonl"))
    col = ChunkedCollator(model.tok, a.Kmax, mode="labels", buffer_aware=not a.no_buffer, p_random_k=0.0)
    rng = random.Random(a.seed); audio = AudioModel()
    w = {"acc": a.w_acc, "all": a.w_all, "ttfc": a.w_ttfc, "stall": a.w_stall, "gpu": a.w_gpu, "prem": a.w_prem}
    gate_p = list(model.gate.parameters()); lat_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("lm.") and not n.startswith("gate.")]
    lm_p = [p for p in model.lm.parameters() if p.requires_grad]
    groups = [{"params": gate_p, "lr": a.lr_gate}]
    if lat_p: groups.append({"params": lat_p, "lr": a.lr_lm})
    if lm_p: groups.append({"params": lm_p, "lr": a.lr_lm})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    batches = bucket_batches(train, a.prompts_per_step, rng); bi = 0
    pbar = tqdm(total=a.steps, desc="rl")
    for step in range(a.steps):
        if bi >= len(batches): batches = bucket_batches(train, a.prompts_per_step, rng); bi = 0
        exs = batches[bi]; bi += 1
        sc = a.step_cost
        if a.step_cost_range:
            lo, hi = map(float, a.step_cost_range.split(",")); sc = rng.uniform(lo, hi)
        exs_rep = [ex for ex in exs for _ in range(a.G)]
        base = col(exs_rep)                      # supervised labels for anchor (same N)
        batch = to_device(base, dev)
        model.eval()
        ro = rollout(model, batch, a.Kmax, sc, audio, sample_gate=True, temperature=a.temperature, eps=a.eps)
        model.train()
        Rs, infos = [], []
        for b, ex in enumerate(exs_rep):
            R, info = reward_fn(ro["events"][b], ro["texts"][b], ex, audio, w); Rs.append(R); infos.append(info)
        R = torch.tensor(Rs).view(len(exs), a.G)
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)).view(-1).to(dev)
        # teacher-forced recompute with grad
        res = model.forward_chunked(batch["prompt_ids"], batch["prompt_mask"], ro["chunk_ids"].to(dev), ro["chunk_mask"].to(dev), ro["k_steps"].to(dev), a.Kmax,
                                    gate_feats=ro["gate_feats"].to(dev), gate_labels=batch["gate_labels"], gate_mask=batch["gate_mask"], gate_pos_weight=1.0)
        Gl = res["gate_logits"]                                  # (B,N,Kmax+1)
        Bn, N, K1 = Gl.shape
        js = torch.arange(K1, device=dev)[None, None, :]
        ks = ro["k_steps"].to(dev)[:, :, None]
        visited = js <= ks
        action_speak = (js == ks).float()                        # 1 = speak at j==k, 0 = think at j<k
        # forced speak at j==Kmax carries no decision -> exclude
        decision_mask = visited & (js < a.Kmax)
        logp = -F.binary_cross_entropy_with_logits(Gl, action_speak, reduction="none")   # log pi(action)
        logp_seq = (logp * decision_mask).sum((1, 2))
        pg_gate = -(adv * logp_seq).mean()
        loss = pg_gate
        if a.answer_pg > 0:
            logp_tok = -res["ce_per_seq"]
            loss = loss + a.answer_pg * (-(adv * logp_tok).mean())
        if a.bce_anchor > 0 and "loss_gate" in res:
            loss = loss + a.bce_anchor * res["loss_gate"]
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        mi = {k: statistics.mean(i[k] for i in infos) for k in infos[0]}
        mi.update(step=step, R=float(R.mean()), R_std=float(R.std()), loss=loss.item(), gn=float(gn), step_cost=sc, mean_k=float(ro["k_steps"].float().mean()))
        log.log(**mi)
        if step % 10 == 0:
            pbar.set_postfix(R=f"{mi['R']:.3f}", acc=f"{mi['chunk_acc']:.3f}", ttfc=f"{mi['ttfc']:.2f}", p95=f"{mi['p95']:.2f}", k=f"{mi['mean_k']:.2f}")
        pbar.update(1)
    pbar.close()
    model.save(os.path.join(a.out, "ckpt")); print("DONE", a.out)

if __name__ == "__main__":
    main()
