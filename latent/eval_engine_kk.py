"""K&K / MuSiQue evaluation of an InnerVoice checkpoint through live engines (greedy + Average@k), labels recovered with the ckpt.
    python latent/eval_engine_kk.py --model Qwen/Qwen3-1.7B --ckpt <dir with sxs.pt> --engines http://127.0.0.1:8001 \
        --data data_sxs/sxs_test_kk.jsonl,data_sxs/sxs_test_mq.jsonl --stride 4 --limit 150 --samples 8 --out results/rl/<run>/kk_mq_avg8.json [--route_sample]"""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_rl_engine import Engines, label_types
from latent.sxs_rl import disclosure_stats, public_chunks
from latent.sxs_eval import token_metrics, score_kk, score_mq
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--engines", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--stride", type=int, default=4); ap.add_argument("--limit", type=int, default=150); ap.add_argument("--samples", type=int, default=8); ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--out", required=True); ap.add_argument("--route_sample", action="store_true")
a = ap.parse_args()
m = SxSModel(SxSConfig(model_name=a.model, use_e_think=False, route_sample=a.route_sample), device="cuda"); m.load(a.ckpt); m.eval(); tok = m.tok
eng = Engines(a.engines.split(","), [m.eos_id, tok.eos_token_id], m.eos_id); eng.wait_ready(); summ = {"model": a.model, "ckpt": a.ckpt, "samples": a.samples, "route_sample": a.route_sample, "sets": {}}
for f in a.data.split(","):
    rows = [json.loads(l) for l in open(f)][::a.stride][:a.limit]; name = os.path.basename(f).replace(".jsonl", "")
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"] for r in rows]
    t0 = time.time(); tg, fg = eng.generate(ids, 1, 0.0, a.max_new); ts, fs = eng.generate(ids, a.samples, 1.0, a.max_new)
    lg = label_types(m, tok, ids, tg, list(range(len(tg))), 16, "cuda"); p_s = [ids[i // a.samples] for i in range(len(ts))]; ls_ = label_types(m, tok, p_s, ts, list(range(len(ts))), 16, "cuda")
    def rec(r, t, y, fin):
        ch = public_chunks(tok, t, y); tm = token_metrics(y, t, tok, r["source"]); ds = disclosure_stats(tok, t, y, r["prompt"])
        oks, ok = (score_kk(ch, r["chunk_checks"]) if r["source"] == "kk" else score_mq(ch, r["chunk_checks"]))
        return {"ok": float(ok), "chunk_acc": sum(oks) / max(1, len(oks)), "finished": fin, "n_short": ds["n_short"], **{k: tm[k] for k in ("FCO", "AIRW", "ARI", "n_think", "n_speak_blocks", "n_tokens", "ttft_norm")}}
    G = [rec(r, t, lg[i], fg[i]) for i, (r, t) in enumerate(zip(rows, tg))]; S = [rec(rows[i // a.samples], t, ls_[i], fs[i]) for i, t in enumerate(ts)]
    mean = lambda L, k: sum(x[k] for x in L) / max(1, len(L))
    summ["sets"][name] = {"n": len(rows), "acc_greedy": mean(G, "ok"), f"acc_avg{a.samples}": mean(S, "ok"), "chunk_acc_greedy": mean(G, "chunk_acc"), "chunk_acc_sampled": mean(S, "chunk_acc"),
                          **{f"{k}_greedy": mean(G, k) for k in ("FCO", "AIRW", "ARI", "n_think", "n_speak_blocks", "n_tokens", "ttft_norm", "n_short")}, "t": time.time() - t0}
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ["sets"][name].items()}), flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(summ, open(a.out, "w"), indent=1)
