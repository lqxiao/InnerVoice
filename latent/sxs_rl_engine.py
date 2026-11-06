"""InnerVoice RL, Scenario B: vLLM rollout engines + multi-GPU trainer (torchrun).

    engines : latent/vllm_server.py (one docker container per GPU, scripts/launch_engines.sh) serving the merged InnerVoice model
    trainer : this file under torchrun --nproc_per_node=W. Rank 0 drives the engines; every rank holds a full policy copy
              (base + LoRA + experts/router), trains on its shard of the rollouts, grads are all-reduced (sum) -> identical updates.
    per step: sample P prompts -> engines generate G rollouts each (T=1, cap --max_new) -> label pass (router argmax recomputed
              on the trainer, no lm_head) -> rewards (same score_rollout as sxs_rl) -> GRPO-style advantages with overlong masking
              -> policy gradient on token + decision log-probs -> rank 0 folds LoRA into merged safetensors -> engines hot-reload.
    every --eval_every steps: AIME25 Average@k through the engines (T=1, --aime_max_new), metrics in <out>/eval.jsonl, ckpt saved.
"""
import argparse, json, os, sys, time, random, statistics, threading, urllib.request
import torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_rl import score_rollout, build_train_batch, public_chunks
from latent.sxs_eval import token_metrics
from latent.sxs_eval_math import correct as math_correct, INSTR as AIME_INSTR, load_aime
from latent.sxs_export import write_weights
from latent.common import set_seed, JsonLogger


def post(url, obj, timeout=10800):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())


class Engines:
    def __init__(self, urls, eos_ids, im_end):
        self.urls = urls; self.eos = set(eos_ids); self.im_end = im_end

    def wait_ready(self, timeout=3600):
        t0 = time.time()
        for u in self.urls:
            while True:
                try: urllib.request.urlopen(u + "/health", timeout=10).read(); break
                except Exception:
                    if time.time() - t0 > timeout: raise RuntimeError(f"engine {u} not ready")
                    time.sleep(10)

    def _par(self, fn):
        errs = [None] * len(self.urls)
        def run(i):
            try: fn(i)
            except Exception as e: errs[i] = e
        ts = [threading.Thread(target=run, args=(i,)) for i in range(len(self.urls))]
        for t in ts: t.start()
        for t in ts: t.join()
        for e in errs:
            if e is not None: raise e

    def generate(self, prompt_ids, n, temperature, max_tokens):
        """Returns (toks, finished) prompt-major: prompt 0 samples 0..n-1, prompt 1, ... Finished rollouts end with <|im_end|>."""
        k = len(self.urls); shards = [list(range(len(prompt_ids)))[i::k] for i in range(k)]; res = [None] * len(prompt_ids)
        def run(i):
            if not shards[i]: return
            r = post(self.urls[i] + "/generate", {"prompt_token_ids": [prompt_ids[j] for j in shards[i]], "n": n, "temperature": temperature, "max_tokens": max_tokens})
            for j, outs in zip(shards[i], r["outputs"]): res[j] = outs
        self._par(run)
        toks, fin = [], []
        for outs in res:
            assert len(outs) == n
            for o in outs:
                ids = o["token_ids"]; f = o["finish_reason"] == "stop"
                if f and (not ids or ids[-1] not in self.eos): ids.append(self.im_end)
                toks.append(ids); fin.append(f)
        return toks, fin

    def reload(self, path):
        out = [None] * len(self.urls)
        def run(i): out[i] = post(self.urls[i] + "/reload", {"path": path})["loaded"]
        self._par(run); return out


def label_types(model, tok, p_ids, toks, idxs, bs, dev):
    """Router argmax labels for sequences idxs (prompt + rollout), recomputed on the trainer. Returns {i: types}."""
    out = {}
    order = sorted(idxs, key=lambda i: -(len(p_ids[i]) + len(toks[i])))
    for s in range(0, len(order), bs):
        chunk = order[s:s + bs]; seqs = [p_ids[i] + toks[i] for i in chunk]; L = max(len(x) for x in seqs)
        ids = torch.full((len(chunk), L), tok.pad_token_id); mask = torch.zeros(len(chunk), L, dtype=torch.long)
        for b, x in enumerate(seqs): ids[b, :len(x)] = torch.tensor(x); mask[b, :len(x)] = 1
        route = model.infer_route(ids.to(dev), mask.to(dev)).long().cpu()
        for b, i in enumerate(chunk):
            Lp, n = len(p_ids[i]), len(toks[i]); out[i] = route[b, Lp - 1:Lp - 1 + n].tolist()
    return out


def bcast(obj, src=0):
    box = [obj]; dist.broadcast_object_list(box, src=src); return box[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--live", required=True, help="exported model dir served by the engines; rewritten every step")
    ap.add_argument("--engines", required=True, help="comma-separated base URLs, e.g. http://127.0.0.1:8001,http://127.0.0.1:8002")
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--prompts", type=int, default=32); ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=8000); ap.add_argument("--micro_bs", type=int, default=1); ap.add_argument("--label_bs", type=int, default=8)
    ap.add_argument("--lr_lora", type=float, default=1e-5); ap.add_argument("--lr_rest", type=float, default=5e-5); ap.add_argument("--w_route", type=float, default=1.0)
    ap.add_argument("--w_inter", type=float, default=0.5); ap.add_argument("--w_prem", type=float, default=1.0); ap.add_argument("--w_fmt", type=float, default=0.2)
    ap.add_argument("--w_fco", type=float, default=0.1); ap.add_argument("--fco_ref", type=float, default=600.0); ap.add_argument("--w_len", type=float, default=0.0); ap.add_argument("--len_ref", type=float, default=3000.0)
    ap.add_argument("--eval_every", type=int, default=50); ap.add_argument("--aime_data", default=""); ap.add_argument("--aime_samples", type=int, default=4); ap.add_argument("--aime_max_new", type=int, default=12000)
    ap.add_argument("--no_expert_ft", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no_eval0", action="store_true")
    a = ap.parse_args()
    dist.init_process_group("nccl"); rank = dist.get_rank(); W = dist.get_world_size(); local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local); dev = f"cuda:{local}"; is0 = rank == 0
    if is0:
        os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
        log = JsonLogger(os.path.join(a.out, "log.jsonl")); elog = JsonLogger(os.path.join(a.out, "eval.jsonl"))
    set_seed(a.seed)
    model = SxSModel(SxSConfig(model_name=a.model, expert_full_ft=not a.no_expert_ft, use_e_think=False, grad_ckpt=True), device=dev); model.load(a.ckpt); tok = model.tok
    eos = [model.eos_id, tok.eos_token_id]
    engines = Engines(a.engines.split(","), eos, model.eos_id) if is0 else None
    rows = [json.loads(l) for l in open(a.data)]; rng = random.Random(a.seed)
    aime_rows = load_aime(a.aime_data) if a.aime_data else []
    w = {"inter": a.w_inter, "prem": a.w_prem, "fmt": a.w_fmt, "fco": a.w_fco, "fco_ref": a.fco_ref, "len": a.w_len, "len_ref": a.len_ref}
    lora_p, rest_p = model.param_groups(); params = model.trainable_parameters()
    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr_lora}, {"params": rest_p, "lr": a.lr_rest}], weight_decay=0.0)
    if is0: engines.wait_ready(); print(f"{len(engines.urls)} engines ready; trainer world={W}", flush=True)
    dist.barrier()

    def labels_all(p_ids, toks):
        model.eval(); idxs = list(range(len(toks)))[rank::W]
        mine = label_types(model, tok, p_ids, toks, idxs, a.label_bs, dev); torch.cuda.empty_cache()
        allp = [None] * W; dist.all_gather_object(allp, mine); merged = {}
        for d in allp: merged.update(d)
        return [merged[i] for i in range(len(toks))]

    def run_eval(step):
        if not aime_rows: return
        t0 = time.time(); payload = None
        if is0:
            ids = [tok(tok.apply_chat_template([{"role": "user", "content": f"{r['prompt']}\n\n{AIME_INSTR}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"] for r in aime_rows]
            toks, fin = engines.generate(ids, a.aime_samples, 1.0, a.aime_max_new); payload = (ids, toks, fin)
        ids, toks, fin = bcast(payload); p_ids = [ids[i // a.aime_samples] for i in range(len(toks))]
        types = labels_all(p_ids, toks)
        if is0:
            recs = []
            for i, (t, y) in enumerate(zip(toks, types)):
                r = aime_rows[i // a.aime_samples]; pub = tok.decode([x for x, yy in zip(t, y) if yy == 0], skip_special_tokens=True); tm = token_metrics(y, t, tok, "math")
                recs.append({"ok": math_correct(pub, r["answer"]), "finished": fin[i], **{k: tm[k] for k in ("FCO", "AIRW", "n_think", "n_speak_blocks", "n_tokens")}})
            N = len(recs); res = {"step": step, "aime_acc": sum(r["ok"] for r in recs) / N, "aime_unfinished": sum(not r["finished"] for r in recs) / N, "n": N,
                                  **{f"aime_{k}": sum(r[k] for r in recs) / N for k in ("FCO", "AIRW", "n_think", "n_speak_blocks", "n_tokens")}, "t_eval": time.time() - t0}
            elog.log(**res); print(f"[eval step {step}] " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
        dist.barrier()

    if not a.no_eval0: run_eval(0)
    for step in range(1, a.steps + 1):
        t0 = time.time(); payload = None
        if is0:
            exs = rng.sample(rows, a.prompts); exs_rep = [e for e in exs for _ in range(a.G)]
            pr = [tok(tok.apply_chat_template([{"role": "user", "content": e["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"] for e in exs]
            toks, fin = engines.generate(pr, a.G, 1.0, a.max_new); payload = (exs_rep, pr, toks, fin)
        exs_rep, pr, toks, fin = bcast(payload); p_ids = [pr[i // a.G] for i in range(len(toks))]; t_roll = time.time() - t0
        t1 = time.time(); types = labels_all(p_ids, toks); t_label = time.time() - t1
        # ---- rewards + advantages (rank 0, broadcast)
        adv = infos = None
        if is0:
            Rs, infos = [], []
            for e, t, y in zip(exs_rep, toks, types):
                ch = public_chunks(tok, t, y); tm = token_metrics(y, t, tok, e["source"]); R, info = score_rollout(e, ch, tm, w); Rs.append(R); infos.append(info)
            R = torch.tensor(Rs).view(a.prompts, a.G); finished = torch.tensor(fin).view(a.prompts, a.G); valid = finished.float(); cnt = valid.sum(1, keepdim=True).clamp(min=1)
            mean = (R * valid).sum(1, keepdim=True) / cnt; var = (((R - mean) ** 2) * valid).sum(1, keepdim=True) / cnt
            adv = ((R - mean) / (var.sqrt() + 1e-6)) * valid; keep = (var.sqrt().squeeze(1) > 1e-6).repeat_interleave(a.G); adv = (adv.view(-1) * keep.float()).tolist()
        adv = torch.tensor(bcast(adv))
        # ---- policy gradient on this rank's shard; grads summed across ranks
        t2 = time.time(); model.train(); opt.zero_grad(set_to_none=True); Bn = len(toks); loss_sum = 0.0
        mine = [i for i in list(range(Bn))[rank::W] if abs(adv[i]) > 0]; mine.sort(key=lambda i: -(len(p_ids[i]) + len(toks[i])))
        for s in range(0, len(mine), a.micro_bs):
            sl = mine[s:s + a.micro_bs]
            batch = build_train_batch(tok, [p_ids[i] for i in sl], [toks[i] for i in sl], [types[i] for i in sl], tok.pad_token_id)
            mb = {k: v.to(dev) for k, v in batch.items()}
            res = model.forward_train(mb["ids"], mb["mask"], mb["is_think"], mb["loss_mask"], mb["route_label"], mb["route_mask"], return_per_seq=True)
            obj = (res["logp_tok_seq"] + a.w_route * res["logp_route_seq"]) / res["n_resp"].clamp(min=1)
            loss = -(adv[sl].to(dev) * obj).sum() / Bn; loss.backward(); loss_sum += loss.item()
        for p in params:
            if p.grad is None: p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); t_train = time.time() - t2
        # ---- weight sync: fold LoRA -> live dir -> engines reload
        t3 = time.time()
        if is0:
            write_weights(model, a.live); engines.reload(a.live)
        dist.barrier(); t_sync = time.time() - t3
        if is0:
            mi = {k: statistics.mean(x[k] for x in infos) for k in infos[0]}
            mi.update(step=step, R=float(R.mean()), groups_kept=int((adv.view(a.prompts, a.G).abs().sum(1) > 0).sum()), n_truncated=int((~finished).sum()), loss=loss_sum, gn=float(gn),
                      t_rollout=t_roll, t_label=t_label, t_train=t_train, t_sync=t_sync, mean_len=statistics.mean(len(t) for t in toks), p95_len=sorted(len(t) for t in toks)[int(0.95 * (len(toks) - 1))])
            log.log(**mi)
            print(f"step {step}: R={mi['R']:.3f} acc={mi['final']:.3f} FCO={mi['FCO']:.0f} think={mi['n_think']:.0f} blocks={mi['blocks']:.2f} trunc={mi['n_truncated']} groups={mi['groups_kept']}/{a.prompts} "
                  f"| roll {t_roll:.0f}s label {t_label:.0f}s train {t_train:.0f}s sync {t_sync:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            if is0: model.save(os.path.join(a.out, f"ckpt_step{step}"))
            run_eval(step)
    if is0: model.save(os.path.join(a.out, "ckpt")); print("DONE", a.out, flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
