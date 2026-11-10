"""AIME25 Average@k for the BASE Qwen3 model in native thinking mode through a plain vLLM engine (latent/vllm_server.py serving the HF
model dir). Public text = everything after </think>; unfinished = no </think> within the cap (counts as wrong).
    python latent/eval_engine_base.py --model_dir <hf snapshot dir> --engines http://127.0.0.1:8001 --aime <parquet> --samples 8 --out results/sxs_eval/base_4B/aime25/summary_base_think_avg8_engine.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from latent.sxs_rl_engine import Engines
from latent.sxs_eval_math import load_aime, correct
ap = argparse.ArgumentParser(); ap.add_argument("--model_dir", required=True); ap.add_argument("--engines", required=True); ap.add_argument("--aime", required=True)
ap.add_argument("--samples", type=int, default=8); ap.add_argument("--max_new", type=int, default=12000); ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_dir); im_end = tok.convert_tokens_to_ids("<|im_end|>"); tclose = tok.convert_tokens_to_ids("</think>")
rows = load_aime(a.aime); eng = Engines(a.engines.split(","), [im_end, tok.eos_token_id], im_end); eng.wait_ready()
ids = [tok(tok.apply_chat_template([{"role": "user", "content": f"{r['prompt']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}], tokenize=False, add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)["input_ids"] for r in rows]
t0 = time.time(); toks, fin = eng.generate(ids, a.samples, 1.0, a.max_new); t_gen = time.time() - t0
recs = []
for i, t in enumerate(toks):
    r = rows[i // a.samples]
    if tclose in t: k = t.index(tclose); think, pub = t[:k], t[k + 1:]
    else: think, pub = t, []
    pub_txt = tok.decode(pub, skip_special_tokens=True)
    recs.append({"id": r["id"], "ok": correct(pub_txt, r["answer"]) if pub else False, "finished": (tclose in t) and fin[i], "n_think": len(think), "n_speak": len(pub), "n_tokens": len(t), "FCO": len(think) + 1})
N = len(recs); summ = {"model": a.model_dir, "mode": "base thinking (vLLM)", "samples": a.samples, "n": len(rows), "acc": sum(r["ok"] for r in recs) / N, "unfinished": sum(not r["finished"] for r in recs) / N,
        **{k: sum(r[k] for r in recs) / N for k in ("n_think", "n_speak", "n_tokens", "FCO")}, "acc_among_finished": (sum(r["ok"] for r in recs if r["finished"]) / max(1, sum(r["finished"] for r in recs))), "t_gen": t_gen}
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(summ, open(a.out, "w"), indent=1); json.dump(recs, open(a.out.replace(".json", "_records.json"), "w"))
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}))
