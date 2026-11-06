"""InnerVoice RL (GRPO-style, on-policy REINFORCE with group baselines) over token-level disclosure decisions and answer tokens.
Rollouts: batched streaming decode with a *sampled* router (Bernoulli(sigmoid(logit))) and temperature-1 token sampling.
Reward (stage 1 + optional stage 2):
    R = final_ok + w_inter * final_ok * frac(intermediate chunks ok)
        - w_prem * (#public chunks that assert something wrong/unsupported)   # premature / unsupported statements
        - w_fmt  * (#public lines that cannot be parsed)
        - w_fco  * FCO / fco_ref                                                # stage 2: earlier first content (content-based, filler ignored)
        - w_len  * n_tokens / len_ref                                           # compute cost
Policy gradient on  sum_t log p(token_t) + w_route * sum_t log pi(decision_t), token-mean normalised, group-normalised advantages."""
import argparse, json, os, sys, time, random, re, statistics
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_eval import score_kk, score_mq, token_metrics, norm
from latent.sxs_eval_math import correct as math_correct
from latent.common import set_seed, JsonLogger

def public_chunks(tok, toks, types):
    txt = tok.decode([t for t, y in zip(toks, types) if y == 0], skip_special_tokens=True)
    return [l.strip() for l in txt.split("\n") if l.strip()]

def score_rollout(ex, chunks, tm, w):
    src = ex["source"]
    if src == "deepmath":
        final_ok = math_correct("\n".join(chunks), ex["final_answer"]); oks = [final_ok]; prem = unp = 0; inter = []
        R = float(final_ok) - (w["fco"] * tm["FCO"] / w["fco_ref"] if final_ok else 0.0) - w["len"] * tm["n_tokens"] / w["len_ref"]
        return R, {"final": float(final_ok), "chunk_acc": float(final_ok), "prem": 0, "unp": 0, "FCO": tm["FCO"], "n_tok": tm["n_tokens"], "n_think": tm["n_think"], "blocks": tm["n_speak_blocks"]}
    if src == "kk":
        oks, final_ok = score_kk(chunks, ex["chunk_checks"])
        gold = {c["name"]: c["knight"] for c in ex["chunk_checks"]}
        prem = unp = 0
        for c in chunks:
            m = re.match(r"\s*([A-Z][a-z]+) is a (knight|knave)", c)
            if not m: unp += 1
            elif m.group(1) not in gold or gold[m.group(1)] != (m.group(2) == "knight"): prem += 1
        inter = oks
    else:
        oks, final_ok = score_mq(chunks, ex["chunk_checks"])
        golds = [norm(c["answer"]) for c in ex["chunk_checks"]] + [norm(a) for c in ex["chunk_checks"] for a in c.get("aliases", [])]
        prem = sum(1 for c in chunks if not any(g and g in norm(c) for g in golds)); unp = 0
        inter = oks[:-1] if len(oks) > 1 else oks
    r_inter = (sum(inter) / len(inter)) if inter else 0.0
    R = float(final_ok) + w["inter"] * float(final_ok) * r_inter - w["prem"] * prem - w["fmt"] * unp - w["fco"] * tm["FCO"] / w["fco_ref"] - w["len"] * tm["n_tokens"] / w["len_ref"]
    return R, {"final": float(final_ok), "chunk_acc": sum(oks) / max(1, len(oks)), "prem": prem, "unp": unp, "FCO": tm["FCO"], "n_tok": tm["n_tokens"], "n_think": tm["n_think"], "blocks": tm["n_speak_blocks"]}

def build_train_batch(tok, prompt_ids_list, toks, types, pad):
    seqs, ist, lm, rl, rm = [], [], [], [], []
    for p, t, y in zip(prompt_ids_list, toks, types):
        ids = p + t; typ = [0] * len(p) + y
        Lp, L = len(p), len(ids)
        seqs.append(ids); ist.append(typ); lm.append([0] * Lp + [1] * (L - Lp))
        rl.append(typ[1:] + [0]); rm.append([0] * (Lp - 1) + [1] * (L - Lp) + [0])
    L = max(len(s) for s in seqs)
    def pad_(xs, v=0): return torch.tensor([x + [v] * (L - len(x)) for x in xs])
    return {"ids": pad_(seqs, pad), "mask": torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs]), "is_think": pad_(ist), "loss_mask": pad_(lm), "route_label": pad_(rl), "route_mask": pad_(rm)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", default="data_sxs/sxs_train.jsonl"); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--prompts", type=int, default=8); ap.add_argument("--G", type=int, default=8); ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--lr_lora", type=float, default=1e-5); ap.add_argument("--lr_rest", type=float, default=5e-5); ap.add_argument("--micro_bs", type=int, default=4); ap.add_argument("--w_route", type=float, default=1.0)
    ap.add_argument("--w_inter", type=float, default=0.5); ap.add_argument("--w_prem", type=float, default=1.0); ap.add_argument("--w_fmt", type=float, default=0.2)
    ap.add_argument("--w_fco", type=float, default=0.0); ap.add_argument("--fco_ref", type=float, default=200.0); ap.add_argument("--w_len", type=float, default=0.0); ap.add_argument("--len_ref", type=float, default=300.0)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no_expert_ft", action="store_true"); ap.add_argument("--no_e_think", action="store_true")
    ap.add_argument("--source", default="", help="restrict to kk / musique"); ap.add_argument("--save_every", type=int, default=50); ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--mask_truncated", action="store_true", help="DAPO-style overlong filtering: rollouts cut by --max_new get no gradient and do not enter the group baseline")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    set_seed(a.seed); log = JsonLogger(os.path.join(a.out, "log.jsonl")); dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SxSModel(SxSConfig(model_name=a.model, expert_full_ft=not a.no_expert_ft, use_e_think=not a.no_e_think, grad_ckpt=a.grad_ckpt), device=dev); model.load(a.ckpt); tok = model.tok
    rows = [json.loads(l) for l in open(a.data)]
    if a.source: rows = [r for r in rows if r["source"] == a.source]
    rng = random.Random(a.seed); w = {"inter": a.w_inter, "prem": a.w_prem, "fmt": a.w_fmt, "fco": a.w_fco, "fco_ref": a.fco_ref, "len": a.w_len, "len_ref": a.len_ref}
    lora_p, rest_p = model.param_groups()
    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr_lora}, {"params": rest_p, "lr": a.lr_rest}], weight_decay=0.0)
    pbar = tqdm(total=a.steps, desc="rl")
    for step in range(a.steps):
        exs = rng.sample(rows, a.prompts); exs_rep = [e for e in exs for _ in range(a.G)]
        prompts = [tok.apply_chat_template([{"role": "user", "content": e["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for e in exs_rep]
        p_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]; L = max(len(x) for x in p_ids)
        ids = torch.full((len(p_ids), L), tok.pad_token_id); mask = torch.zeros(len(p_ids), L, dtype=torch.long)
        for b, x in enumerate(p_ids): ids[b, L - len(x):] = torch.tensor(x); mask[b, L - len(x):] = 1
        model.eval(); t0 = time.time()
        toks, types = model.stream_batch(ids.to(dev), mask.to(dev), max_new_tokens=a.max_new, greedy=False, temperature=a.temperature, sample_router=True)
        t_roll = time.time() - t0
        if dev == "cuda": torch.cuda.empty_cache()
        Rs, infos = [], []
        for e, t, y in zip(exs_rep, toks, types):
            ch = public_chunks(tok, t, y); tm = token_metrics(y, t, tok, e["source"]); R, info = score_rollout(e, ch, tm, w); Rs.append(R); infos.append(info)
        finished = torch.tensor([bool(t) and t[-1] in (model.eos_id, tok.eos_token_id) for t in toks]).view(a.prompts, a.G)
        R = torch.tensor(Rs).view(a.prompts, a.G)
        if a.mask_truncated:
            # group statistics over finished rollouts only; truncated ones get zero advantage (no gradient, no penalty)
            valid = finished.float(); cnt = valid.sum(1, keepdim=True).clamp(min=1)
            mean = (R * valid).sum(1, keepdim=True) / cnt; var = (((R - mean) ** 2) * valid).sum(1, keepdim=True) / cnt
            adv = ((R - mean) / (var.sqrt() + 1e-6)) * valid
            keep = (var.sqrt().squeeze(1) > 1e-6).repeat_interleave(a.G); adv = adv.view(-1) * keep.float()
            n_trunc = int((~finished).sum())
        else:
            adv = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)
            keep = (R.std(1) > 1e-6).repeat_interleave(a.G); adv = adv.view(-1) * keep.float(); n_trunc = int((~finished).sum())
        # ---- policy gradient via teacher-forced recompute
        model.train(); opt.zero_grad(set_to_none=True); total_loss = 0.0; Bn = len(exs_rep)
        batch = build_train_batch(tok, p_ids, toks, types, tok.pad_token_id)
        for i in range(0, Bn, a.micro_bs):
            sl = slice(i, i + a.micro_bs); mb = {k: v[sl].to(dev) for k, v in batch.items()}
            if adv[sl].abs().sum() == 0: continue
            res = model.forward_train(mb["ids"], mb["mask"], mb["is_think"], mb["loss_mask"], mb["route_label"], mb["route_mask"], return_per_seq=True)
            n = res["n_resp"].clamp(min=1)
            obj = (res["logp_tok_seq"] + a.w_route * res["logp_route_seq"]) / n
            loss = -(adv[sl].to(dev) * obj).sum() / Bn
            loss.backward(); total_loss += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); opt.step()
        if dev == "cuda": torch.cuda.empty_cache()
        mi = {k: statistics.mean(x[k] for x in infos) for k in infos[0]}
        mi.update(step=step, R=float(R.mean()), R_std=float(R.std()), groups_kept=int((adv.view(a.prompts, a.G).abs().sum(1) > 0).sum()), n_truncated=n_trunc, loss=total_loss, gn=float(gn), t_rollout=t_roll, mean_len=statistics.mean(len(t) for t in toks), p95_len=sorted(len(t) for t in toks)[int(0.95 * (len(toks) - 1))])
        log.log(**mi); pbar.set_postfix(R=f"{mi['R']:.2f}", acc=f"{mi['final']:.2f}", prem=f"{mi['prem']:.2f}", FCO=f"{mi['FCO']:.0f}", think=f"{mi['n_think']:.0f}", blk=f"{mi['blocks']:.1f}"); pbar.update(1)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps: model.save(os.path.join(a.out, "ckpt"))
    pbar.close(); print("DONE", a.out)

if __name__ == "__main__":
    main()
