"""Peek at a saved InnerVoice policy with HF decode only (no engine): sample one problem, print interleaved think/speak.
    python latent/peek_hf.py --ckpt results/rl/<run>/ckpt_step50 --aime <parquet> --idx 2 [--pool data_math/rl_pool.jsonl --idx 7] --max_new 1200"""
import argparse, os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_eval_math import load_aime, INSTR, correct
from latent.sxs_eval import token_metrics
from latent.sxs_rl import disclosure_stats, segments
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--aime", default=""); ap.add_argument("--pool", default="")
ap.add_argument("--idx", type=int, default=0); ap.add_argument("--max_new", type=int, default=1200); ap.add_argument("--greedy", action="store_true"); ap.add_argument("--show", type=int, default=600)
a = ap.parse_args()
m = SxSModel(SxSConfig(model_name=a.model, use_e_think=False), device="cuda"); m.load(a.ckpt); m.eval(); tok = m.tok
if a.pool:
    r = [json.loads(l) for l in open(a.pool)][a.idx]; prompt = r["prompt"]; gold = r["final_answer"]
else:
    r = load_aime(a.aime)[a.idx]; prompt = f"{r['prompt']}\n\n{INSTR}"; gold = r["answer"]
ids = tok(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
toks, types = m.stream_batch(ids, torch.ones_like(ids), max_new_tokens=a.max_new, greedy=a.greedy, temperature=1.0)
t, y = toks[0], types[0]; tm = token_metrics(y, t, tok, "math"); pub = tok.decode([x for x, yy in zip(t, y) if yy == 0], skip_special_tokens=True)
ds = disclosure_stats(tok, t, y, prompt)
print(f"gold={gold} correct={correct(pub, gold)} finished={bool(t) and t[-1] in (m.eos_id, tok.eos_token_id)} n_tok={len(t)} think={tm['n_think']} FCO={tm['FCO']} blocks={tm['n_speak_blocks']} unsup={ds}")
print("=== interleaved ===")
for mode, text in segments(tok, t[:a.show], y[:a.show]): print(("[T] " if mode else "[S] ") + text.replace("\n", "\\n")[:400])
