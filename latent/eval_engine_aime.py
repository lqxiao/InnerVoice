"""AIME25 Average@k through live InnerVoice engines (serving <live>), router labels recovered on a GPU with the matching sxs.pt ckpt.
    python latent/eval_engine_aime.py --ckpt results/rl/<run>/ckpt --engines http://127.0.0.1:8001,... --aime <parquet> --samples 8 --out results/rl/<run>/aime25_avg8.json [--route_sample]"""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_rl_engine import Engines, label_types
from latent.sxs_rl import disclosure_stats
from latent.sxs_eval_math import load_aime, INSTR, correct
from latent.sxs_eval import token_metrics
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--engines", required=True); ap.add_argument("--aime", required=True)
ap.add_argument("--samples", type=int, default=8); ap.add_argument("--max_new", type=int, default=12000); ap.add_argument("--out", required=True); ap.add_argument("--route_sample", action="store_true"); ap.add_argument("--label_bs", type=int, default=8)
a = ap.parse_args()
m = SxSModel(SxSConfig(model_name=a.model, use_e_think=False, route_sample=a.route_sample), device="cuda"); m.load(a.ckpt); m.eval(); tok = m.tok
rows = load_aime(a.aime); eng = Engines(a.engines.split(","), [m.eos_id, tok.eos_token_id], m.eos_id); eng.wait_ready()
ids = [tok(tok.apply_chat_template([{"role": "user", "content": f"{r['prompt']}\n\n{INSTR}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"] for r in rows]
t0 = time.time(); toks, fin = eng.generate(ids, a.samples, 1.0, a.max_new); t_gen = time.time() - t0
p_ids = [ids[i // a.samples] for i in range(len(toks))]; lab = label_types(m, tok, p_ids, toks, list(range(len(toks))), a.label_bs, "cuda"); types = [lab[i] for i in range(len(toks))]
recs = []
for i, (t, y) in enumerate(zip(toks, types)):
    r = rows[i // a.samples]; pub = tok.decode([x for x, yy in zip(t, y) if yy == 0], skip_special_tokens=True); tm = token_metrics(y, t, tok, "math"); ds = disclosure_stats(tok, t, y, r["prompt"])
    recs.append({"id": r["id"], "ok": correct(pub, r["answer"]), "finished": fin[i], "unsup_frac": ds["n_unsup"] / max(1, ds["n_judged"]), "n_short": ds["n_short"], **{k: tm[k] for k in ("ARI", "ABO", "AIRW", "FCO", "n_think", "n_speak", "n_speak_blocks", "n_tokens")}})
N = len(recs); summ = {"model": a.model, "ckpt": a.ckpt, "samples": a.samples, "n": len(rows), "acc": sum(r["ok"] for r in recs) / N, "unfinished": sum(not r["finished"] for r in recs) / N,
        **{k: sum(r[k] for r in recs) / N for k in ("ARI", "ABO", "AIRW", "FCO", "n_think", "n_speak", "n_speak_blocks", "n_tokens", "unsup_frac", "n_short")}, "route_sample": a.route_sample, "t_gen": t_gen}
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(summ, open(a.out, "w"), indent=1); json.dump(recs, open(a.out.replace(".json", "_records.json"), "w"))
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}))
