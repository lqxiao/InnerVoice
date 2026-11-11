"""Apple-style evaluation (Pass@1 greedy + Average@k, format rate, normalised TTFT) of a plain policy served by engines, under a given
prompt template. Works for the untrained base (Direct / CoT / zero-shot interleave baselines) and for exported RL checkpoints.
    python latent/eval_engine_apple.py --model Qwen/Qwen3-1.7B --engines http://127.0.0.1:8041 --template direct \
        --data data_apple/test_kk.jsonl,data_apple/test_mq.jsonl --samples 8 --max_new 2048 --out results/rl/apple/base_direct_1.7B.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from latent.sxs_rl_engine import Engines
from latent.apple_rl import TEMPLATES, TASK_HINT, parse_tags, token_types_from_tags, public_lines, score_final
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--engines", required=True); ap.add_argument("--template", default="interleave", choices=list(TEMPLATES))
ap.add_argument("--data", required=True); ap.add_argument("--samples", type=int, default=8); ap.add_argument("--max_new", type=int, default=2048); ap.add_argument("--temperature", type=float, default=0.8); ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model); im_end = tok.convert_tokens_to_ids("<|im_end|>")
eng = Engines(a.engines.split(","), [im_end, tok.eos_token_id], im_end); eng.wait_ready()
def build(e):
    txt = tok.apply_chat_template([{"role": "system", "content": TEMPLATES[a.template] + TASK_HINT.get(e["source"], "")}, {"role": "user", "content": e["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(txt.replace("<think>\n\n</think>\n\n", ""), add_special_tokens=False)["input_ids"]
summ = {"model": a.model, "template": a.template, "samples": a.samples, "temperature": a.temperature, "sets": {}}
for f in a.data.split(","):
    rows = [json.loads(l) for l in open(f)]; name = os.path.basename(f).replace(".jsonl", ""); ids = [build(r) for r in rows]
    t0 = time.time(); tg, fg = eng.generate(ids, 1, 0.0, a.max_new); ts, fs = eng.generate(ids, a.samples, a.temperature, a.max_new)
    def rec(r, t, fin):
        segs, fmt = parse_tags(tok.decode(t, skip_special_tokens=True)); ok, oks, _ = score_final(r, public_lines(segs)); _, first = token_types_from_tags(tok, t)
        inter = oks[:-1] if r["source"] == "musique" else oks
        return {"n": r.get("n"), "ok": float(ok), "format_ok": float(fmt), "finished": fin, "ttft_norm": first / max(1, len(t)), "n_answer_segs": float(sum(1 for m, _ in segs if m == "answer")), "n_tokens": len(t), "inter_frac": (sum(inter) / len(inter)) if inter else 0.0}
    G = [rec(r, t, f) for r, t, f in zip(rows, tg, fg)]; S = [rec(rows[i // a.samples], t, f) for i, (t, f) in enumerate(zip(ts, fs))]
    mean = lambda L, k: sum(x[k] for x in L) / max(1, len(L))
    by_n = {str(n): mean([x for x in G if x["n"] == n], "ok") for n in sorted(set(x["n"] for x in G))}
    summ["sets"][name] = {"n": len(rows), "acc_greedy": mean(G, "ok"), f"acc_avg{a.samples}": mean(S, "ok"), "format_greedy": mean(G, "format_ok"), "format_sampled": mean(S, "format_ok"), "unfinished_greedy": 1 - mean(G, "finished"),
                          **{f"{k}_greedy": mean(G, k) for k in ("ttft_norm", "n_answer_segs", "n_tokens", "inter_frac")}, "acc_greedy_by_n": by_n, "t": time.time() - t0}
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ["sets"][name].items()}), flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(summ, open(a.out, "w"), indent=1)
