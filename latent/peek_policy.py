"""Peek at the current RL policy: generate one AIME problem through a live engine and label it with a saved sxs.pt checkpoint.
    python latent/peek_policy.py --ckpt results/rl/<run>/ckpt_stepN --url http://127.0.0.1:8001 --aime <parquet> --idx 3 --max_new 12000"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_rl_engine import Engines, label_types
from latent.sxs_eval_math import load_aime, INSTR, correct
from latent.sxs_eval import token_metrics
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--url", default="http://127.0.0.1:8001")
ap.add_argument("--aime", required=True); ap.add_argument("--idx", type=int, default=0); ap.add_argument("--max_new", type=int, default=12000); ap.add_argument("--show", type=int, default=900)
a = ap.parse_args()
m = SxSModel(SxSConfig(model_name=a.model, use_e_think=False), device="cuda"); m.load(a.ckpt); m.eval(); tok = m.tok
r = load_aime(a.aime)[a.idx]
ids = tok(tok.apply_chat_template([{"role": "user", "content": f"{r['prompt']}\n\n{INSTR}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
eng = Engines([a.url], [m.eos_id, tok.eos_token_id], m.eos_id); toks, fin = eng.generate([ids], 1, 1.0, a.max_new)
y = label_types(m, tok, [ids], toks, [0], 1, "cuda")[0]; t = toks[0]
tm = token_metrics(y, t, tok, "math"); pub = tok.decode([x for x, yy in zip(t, y) if yy == 0], skip_special_tokens=True)
print(f"gold={r['answer']} correct={correct(pub, r['answer'])} finished={fin[0]} n_tok={len(t)} think={tm['n_think']} FCO={tm['FCO']} blocks={tm['n_speak_blocks']}")
print("=== interleaved (first tokens), [S]=speak [T]=think ===")
cur, mode, out = [], None, []
for x, yy in zip(t[:a.show], y[:a.show]):
    if mode is not None and yy != mode: out.append(("[S] " if mode == 0 else "[T] ") + tok.decode(cur).replace("\n", "\\n")); cur = []
    cur.append(x); mode = yy
out.append(("[S] " if mode == 0 else "[T] ") + tok.decode(cur).replace("\n", "\\n"))
print("\n".join(out)); print("=== PUBLIC stream ==="); print(pub[:1500])
