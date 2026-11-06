"""Check that the vLLM InnerVoice engine reproduces the HF SxSModel: greedy tokens (prefix agreement) and router labels
(trainer-side infer_route vs. labels produced during HF streaming decode).
    python latent/test_engine_equiv.py --model Qwen/Qwen3-1.7B --ckpt results/sxs/noethink_1.7B/ckpt --url http://127.0.0.1:8001"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_rl_engine import Engines, label_types

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--url", default="http://127.0.0.1:8001")
ap.add_argument("--data", default="data_sxs/sxs_test.jsonl"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--max_new", type=int, default=300)
a = ap.parse_args()
dev = "cuda"
m = SxSModel(SxSConfig(model_name=a.model, use_e_think=False), device=dev); m.load(a.ckpt); m.eval(); tok = m.tok
rows = [json.loads(l) for l in open(a.data)][:a.n]
p_ids = [tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"] for r in rows]
L = max(len(x) for x in p_ids); ids = torch.full((a.n, L), tok.pad_token_id); mask = torch.zeros(a.n, L, dtype=torch.long)
for b, x in enumerate(p_ids): ids[b, L - len(x):] = torch.tensor(x); mask[b, L - len(x):] = 1
t0 = time.time(); hf_toks, hf_types = m.stream_batch(ids.to(dev), mask.to(dev), max_new_tokens=a.max_new, greedy=True); t_hf = time.time() - t0
eng = Engines([a.url], [m.eos_id, tok.eos_token_id], m.eos_id); eng.wait_ready()
t0 = time.time(); en_toks, fin = eng.generate(p_ids, 1, 0.0, a.max_new); t_en = time.time() - t0
print(f"HF {t_hf:.1f}s  engine {t_en:.1f}s")
# 1) token agreement
for b in range(a.n):
    h, e = hf_toks[b], en_toks[b]; k = 0
    while k < min(len(h), len(e)) and h[k] == e[k]: k += 1
    print(f"[{b}] hf_len={len(h)} engine_len={len(e)} common_prefix={k} identical={h == e}")
# 2) label agreement: recompute labels for the HF sequences and compare with the labels HF produced while decoding
lab = label_types(m, tok, p_ids, hf_toks, list(range(a.n)), 8, dev)
tot = mis = 0
for b in range(a.n):
    y = hf_types[b]; z = lab[b]; tot += len(y); mis += sum(int(u != v) for u, v in zip(y, z))
print(f"label mismatch (infer_route vs HF decode labels): {mis}/{tot}")
# 3) engine sample with labels
lab_e = label_types(m, tok, p_ids, en_toks, [0], 8, dev)[0]
th = tok.decode([t for t, y in zip(en_toks[0], lab_e) if y == 1], skip_special_tokens=True); pub = tok.decode([t for t, y in zip(en_toks[0], lab_e) if y == 0], skip_special_tokens=True)
print("--- engine sample 0 THINK:", th[:400].replace("\n", " | ")); print("--- engine sample 0 SPEAK:", pub[:400].replace("\n", " | "))
