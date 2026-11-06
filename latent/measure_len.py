"""Measure the SFT policy's completion-length distribution on RL training prompts (to set the RL max generation length)."""
import argparse, json, random, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_eval_math import correct
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=64); ap.add_argument("--max_new", type=int, default=12000); ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
a = ap.parse_args()
m = SxSModel(SxSConfig(model_name=a.model), device="cuda"); m.load(a.ckpt); m.eval(); tok = m.tok
rows = random.Random(1).sample([json.loads(l) for l in open(a.data)], a.n)
prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]; L = max(len(x) for x in enc)
ids = torch.full((len(enc), L), tok.pad_token_id); mask = torch.zeros(len(enc), L, dtype=torch.long)
for b, x in enumerate(enc): ids[b, L - len(x):] = torch.tensor(x); mask[b, L - len(x):] = 1
lens, fin, oks, thinks = [], [], [], []
for i in range(0, len(rows), 16):
    toks, types = m.stream_batch(ids[i:i+16].cuda(), mask[i:i+16].cuda(), max_new_tokens=a.max_new, greedy=False, temperature=1.0, sample_router=True)
    for r, t, y in zip(rows[i:i+16], toks, types):
        lens.append(len(t)); fin.append(bool(t) and t[-1] in (m.eos_id, tok.eos_token_id)); thinks.append(sum(y))
        pub = tok.decode([x for x, yy in zip(t, y) if yy == 0], skip_special_tokens=True); oks.append(correct(pub, r["final_answer"]))
import numpy as np; l = np.array(lens)
print(json.dumps({"n": len(lens), "finished": float(np.mean(fin)), "acc": float(np.mean(oks)), "len_p50": int(np.percentile(l, 50)), "len_p90": int(np.percentile(l, 90)), "len_p95": int(np.percentile(l, 95)), "len_max": int(l.max()), "think_p50": int(np.percentile(thinks, 50)), "acc_if_len_le_6000": float(np.mean([o for o, x in zip(oks, lens) if x <= 6000] or [0])), "frac_le_4000": float(np.mean(l <= 4000)), "frac_le_6000": float(np.mean(l <= 6000)), "frac_le_8000": float(np.mean(l <= 8000))}))
