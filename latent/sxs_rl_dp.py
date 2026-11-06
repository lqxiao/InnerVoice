"""InnerVoice RL with rollout data-parallelism: 1 trainer GPU + N rollout workers (each a full policy copy on its own GPU).
Each step: the trainer samples prompts, workers generate G rollouts per prompt (sampled router + T=1 tokens), the trainer scores
them (same reward as sxs_rl: correctness + gated FCO, overlong masking), does the GRPO-style update, writes the trainable weights to
<out>/live/sxs.pt and workers reload them. Every --eval_every steps the workers run AIME25 Average@k and the trainer logs it.
Same reward/scoring code as latent/sxs_rl.py (imported)."""
import argparse, json, os, sys, time, random, statistics, math
import torch, torch.multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_rl import score_rollout, build_train_batch, public_chunks
from latent.sxs_eval import token_metrics
from latent.sxs_eval_math import correct as math_correct, INSTR as AIME_INSTR
from latent.common import set_seed, JsonLogger

def worker_main(rank, device, cfg_kwargs, ckpt, task_q, result_q):
    torch.cuda.set_device(device) if device.startswith("cuda") else None
    model = SxSModel(SxSConfig(**cfg_kwargs), device=device); model.load(ckpt); model.eval(); tok = model.tok
    result_q.put(("ready", rank))
    while True:
        task = task_q.get()
        if task[0] == "stop": break
        if task[0] == "weights":
            model.load(task[1]); result_q.put(("loaded", rank)); continue
        _, job_id, prompt_ids, max_new, greedy, temperature, sample_router, bs = task
        outs_t, outs_y = [], []
        for i in range(0, len(prompt_ids), bs):
            ch = prompt_ids[i:i + bs]; L = max(len(x) for x in ch)
            ids = torch.full((len(ch), L), tok.pad_token_id); mask = torch.zeros(len(ch), L, dtype=torch.long)
            for b, x in enumerate(ch): ids[b, L - len(x):] = torch.tensor(x); mask[b, L - len(x):] = 1
            with torch.no_grad():
                toks, types = model.stream_batch(ids.to(device), mask.to(device), max_new_tokens=max_new, greedy=greedy, temperature=temperature, sample_router=sample_router)
            outs_t += toks; outs_y += types
            if device.startswith("cuda"): torch.cuda.empty_cache()
        result_q.put(("gen", rank, job_id, outs_t, outs_y))

def dispatch(workers, prompt_ids, max_new, greedy, temperature, sample_router, bs):
    """Split prompt_ids across workers; returns toks, types in the original order."""
    n = len(workers); shards = [list(range(len(prompt_ids)))[i::n] for i in range(n)]
    for rank, (tq, _) in enumerate(workers):
        tq.put(("gen", rank, [prompt_ids[j] for j in shards[rank]], max_new, greedy, temperature, sample_router, bs))
    toks = [None] * len(prompt_ids); types = [None] * len(prompt_ids); got = 0
    while got < n:
        msg = RESULT_Q.get()
        if msg[0] != "gen": continue
        _, rank, _, t, y = msg
        for j, tt, yy in zip(shards[rank], t, y): toks[j] = tt; types[j] = yy
        got += 1
    return toks, types

def push_weights(model, workers, path):
    model.save(path)
    for tq, _ in workers: tq.put(("weights", path))
    got = 0
    while got < len(workers):
        if RESULT_Q.get()[0] == "loaded": got += 1

def aime_eval(workers, tok, rows, samples, max_new, bs):
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{r['prompt']}\n\n{AIME_INSTR}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts] * samples
    toks, types = dispatch(workers, ids, max_new, False, 1.0, False, bs)
    recs = []
    for i, (t, y) in enumerate(zip(toks, types)):
        r = rows[i % len(rows)]; pub = tok.decode([x for x, yy in zip(t, y) if yy == 0], skip_special_tokens=True)
        tm = token_metrics(y, t, tok, "math"); recs.append({"ok": math_correct(pub, r["answer"]), "finished": bool(t) and t[-1] in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id), **{k: tm[k] for k in ("FCO", "AIRW", "n_think", "n_speak_blocks", "n_tokens")}})
    N = len(recs)
    return {"aime_acc": sum(r["ok"] for r in recs) / N, "aime_unfinished": sum(not r["finished"] for r in recs) / N, **{f"aime_{k}": sum(r[k] for r in recs) / N for k in ("FCO", "AIRW", "n_think", "n_speak_blocks", "n_tokens")}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_workers", type=int, default=7); ap.add_argument("--steps", type=int, default=200); ap.add_argument("--prompts", type=int, default=14); ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=8000); ap.add_argument("--gen_bs", type=int, default=16); ap.add_argument("--micro_bs", type=int, default=1)
    ap.add_argument("--lr_lora", type=float, default=1e-5); ap.add_argument("--lr_rest", type=float, default=5e-5); ap.add_argument("--w_route", type=float, default=1.0)
    ap.add_argument("--w_inter", type=float, default=0.5); ap.add_argument("--w_prem", type=float, default=1.0); ap.add_argument("--w_fmt", type=float, default=0.2)
    ap.add_argument("--w_fco", type=float, default=0.1); ap.add_argument("--fco_ref", type=float, default=600.0); ap.add_argument("--w_len", type=float, default=0.0); ap.add_argument("--len_ref", type=float, default=3000.0)
    ap.add_argument("--eval_every", type=int, default=50); ap.add_argument("--aime_data", default=""); ap.add_argument("--aime_samples", type=int, default=4); ap.add_argument("--aime_max_new", type=int, default=12000)
    ap.add_argument("--no_expert_ft", action="store_true"); ap.add_argument("--no_e_think", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    set_seed(a.seed); log = JsonLogger(os.path.join(a.out, "log.jsonl")); elog = JsonLogger(os.path.join(a.out, "eval.jsonl"))
    cfg_kwargs = dict(model_name=a.model, expert_full_ft=not a.no_expert_ft, use_e_think=not a.no_e_think)
    # ---- workers
    global RESULT_Q
    ctx = mp.get_context("spawn"); RESULT_Q = ctx.Queue(); workers = []
    for r in range(a.n_workers):
        tq = ctx.Queue(); p = ctx.Process(target=worker_main, args=(r, f"cuda:{r+1}", cfg_kwargs, a.ckpt, tq, RESULT_Q), daemon=True); p.start(); workers.append((tq, p))
    # ---- trainer on cuda:0
    model = SxSModel(SxSConfig(**cfg_kwargs, grad_ckpt=True), device="cuda:0"); model.load(a.ckpt); tok = model.tok
    ready = 0
    while ready < a.n_workers:
        if RESULT_Q.get()[0] == "ready": ready += 1
    print(f"{a.n_workers} rollout workers ready", flush=True)
    rows = [json.loads(l) for l in open(a.data)]; rng = random.Random(a.seed)
    aime_rows = []
    if a.aime_data:
        from latent.sxs_eval_math import load_aime; aime_rows = load_aime(a.aime_data)
    w = {"inter": a.w_inter, "prem": a.w_prem, "fmt": a.w_fmt, "fco": a.w_fco, "fco_ref": a.fco_ref, "len": a.w_len, "len_ref": a.len_ref}
    lora_p, rest_p = model.param_groups()
    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr_lora}, {"params": rest_p, "lr": a.lr_rest}], weight_decay=0.0)
    live = os.path.join(a.out, "live")
    def run_eval(step):
        if not aime_rows: return
        t0 = time.time(); model.eval(); res = aime_eval(workers, tok, aime_rows, a.aime_samples, a.aime_max_new, a.gen_bs); res.update(step=step, t_eval=time.time() - t0)
        elog.log(**res); print(f"[eval step {step}] {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})}", flush=True)
    run_eval(0)
    for step in range(1, a.steps + 1):
        t0 = time.time(); exs = rng.sample(rows, a.prompts); exs_rep = [e for e in exs for _ in range(a.G)]
        p_ids = [tok(tok.apply_chat_template([{"role": "user", "content": e["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"] for e in exs_rep]
        toks, types = dispatch(workers, p_ids, a.max_new, False, 1.0, True, a.gen_bs); t_roll = time.time() - t0
        Rs, infos = [], []
        for e, t, y in zip(exs_rep, toks, types):
            ch = public_chunks(tok, t, y); tm = token_metrics(y, t, tok, e["source"]); R, info = score_rollout(e, ch, tm, w); Rs.append(R); infos.append(info)
        finished = torch.tensor([bool(t) and t[-1] in (model.eos_id, tok.eos_token_id) for t in toks]).view(a.prompts, a.G)
        R = torch.tensor(Rs).view(a.prompts, a.G); valid = finished.float(); cnt = valid.sum(1, keepdim=True).clamp(min=1)
        mean = (R * valid).sum(1, keepdim=True) / cnt; var = (((R - mean) ** 2) * valid).sum(1, keepdim=True) / cnt
        adv = ((R - mean) / (var.sqrt() + 1e-6)) * valid; keep = (var.sqrt().squeeze(1) > 1e-6).repeat_interleave(a.G); adv = adv.view(-1) * keep.float()
        # ---- gradient step on cuda:0
        model.train(); opt.zero_grad(set_to_none=True); total_loss = 0.0; Bn = len(exs_rep); t1 = time.time()
        batch = build_train_batch(tok, p_ids, toks, types, tok.pad_token_id)
        for i in range(0, Bn, a.micro_bs):
            sl = slice(i, i + a.micro_bs)
            if adv[sl].abs().sum() == 0: continue
            mb = {k: v[sl].to("cuda:0") for k, v in batch.items()}
            res = model.forward_train(mb["ids"], mb["mask"], mb["is_think"], mb["loss_mask"], mb["route_label"], mb["route_mask"], return_per_seq=True)
            obj = (res["logp_tok_seq"] + a.w_route * res["logp_route_seq"]) / res["n_resp"].clamp(min=1)
            loss = -(adv[sl].to("cuda:0") * obj).sum() / Bn; loss.backward(); total_loss += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0); opt.step(); torch.cuda.empty_cache(); t_train = time.time() - t1
        t2 = time.time(); push_weights(model, workers, live); t_sync = time.time() - t2
        mi = {k: statistics.mean(x[k] for x in infos) for k in infos[0]}
        mi.update(step=step, R=float(R.mean()), groups_kept=int((adv.view(a.prompts, a.G).abs().sum(1) > 0).sum()), n_truncated=int((~finished).sum()), loss=total_loss, gn=float(gn),
                  t_rollout=t_roll, t_train=t_train, t_sync=t_sync, mean_len=statistics.mean(len(t) for t in toks), p95_len=sorted(len(t) for t in toks)[int(0.95 * (len(toks) - 1))])
        log.log(**mi); print(f"step {step}: R={mi['R']:.3f} acc={mi['final']:.3f} FCO={mi['FCO']:.0f} think={mi['n_think']:.0f} blocks={mi['blocks']:.2f} trunc={mi['n_truncated']} groups={mi['groups_kept']}/{a.prompts} | roll {t_roll:.0f}s train {t_train:.0f}s sync {t_sync:.0f}s", flush=True)
        if step % a.eval_every == 0:
            model.save(os.path.join(a.out, f"ckpt_step{step}")); run_eval(step)
    model.save(os.path.join(a.out, "ckpt"))
    for tq, p in workers: tq.put(("stop",))
    print("DONE", a.out, flush=True)

if __name__ == "__main__":
    main()
