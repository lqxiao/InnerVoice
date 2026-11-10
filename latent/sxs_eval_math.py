"""AIME25 evaluation under the SxS protocol for our model (router-driven batched streaming decode) and for the base model.
Accuracy = last \\boxed{} in the PUBLIC stream vs gold (integer normalisation); Average@k with temperature 1.0 (paper: 16);
token metrics ARI/ABO/AIRW/FCO over the generated sequence (think tokens = router-think / <think> span)."""
import argparse, json, os, sys, re, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_eval import token_metrics
from latent.sxs_math_data import last_boxed, norm_ans

INSTR = "Solve the problem. Reason privately; publicly present the solution in short paragraphs as it becomes certain, and end with 'Final answer: \\boxed{<answer>}'."

def load_aime(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pylist()
    return [{"id": f"aime25-{r['problem_idx']}", "prompt": r["problem"], "answer": str(r["answer"]).strip()} for r in t]

def correct(public_text, gold):
    b = last_boxed(public_text)
    if b is None:
        m = re.findall(r"Final answer:\s*(-?\d+)", public_text); b = m[-1] if m else None
    if b is None: return False
    try: return int(norm_ans(b).replace(",", "")) == int(gold)
    except Exception: return norm_ans(b) == norm_ans(gold)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--ckpt", default="", help="empty = base model"); ap.add_argument("--thinking", type=int, default=1, help="base model only")
    ap.add_argument("--data", default="/scratch/liqiangx/extdata/aime25/data/train-00000-of-00001.parquet"); ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=4); ap.add_argument("--max_new_tokens", type=int, default=12000); ap.add_argument("--bs", type=int, default=30)
    ap.add_argument("--no_expert_ft", action="store_true"); ap.add_argument("--no_e_think", action="store_true"); ap.add_argument("--tag", default=""); ap.add_argument("--shared_ffn", action="store_true")
    ap.add_argument("--tags_format", action="store_true", help="checkpoint trained with --format tags: decode with the answer expert only and split think/speak by <think> tags")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = load_aime(a.data)
    recs = []
    if a.ckpt:
        from latent.sxs_model import SxSModel, SxSConfig
        model = SxSModel(SxSConfig(model_name=a.model, expert_full_ft=not a.no_expert_ft, use_e_think=not a.no_e_think, shared_ffn=a.shared_ffn), device="cuda"); model.load(a.ckpt); model.eval(); tok = model.tok
        prompts = [tok.apply_chat_template([{"role": "user", "content": f"{r['prompt']}\n\n{INSTR}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
        enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        for si in range(a.samples):
            for i in range(0, len(rows), a.bs):
                ch = enc[i:i + a.bs]; L = max(len(x) for x in ch)
                ids = torch.full((len(ch), L), tok.pad_token_id); mask = torch.zeros(len(ch), L, dtype=torch.long)
                for b, x in enumerate(ch): ids[b, L - len(x):] = torch.tensor(x); mask[b, L - len(x):] = 1
                if a.tags_format:
                    model.moe.route = None; model.moe.sample_route = False
                    import types as _t
                    orig = model.moe.forward
                    def fwd_answer_only(x, _m=model.moe):
                        _m.router_logits = torch.full(x.shape[:2], -10.0, device=x.device); _m.last_route = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device); return _m.answer(x)
                    model.moe.forward = fwd_answer_only
                    toks, types = model.stream_batch(ids.cuda(), mask.cuda(), max_new_tokens=a.max_new_tokens, greedy=(a.samples == 1))
                    model.moe.forward = orig
                    to, tc = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
                    new_types = []
                    for t in toks:
                        inside, ty = False, []
                        for x in t:
                            if x == to: inside = True; ty.append(1); continue
                            if x == tc: inside = False; ty.append(1); continue
                            ty.append(1 if inside else 0)
                        new_types.append(ty)
                    types = new_types
                else:
                    toks, types = model.stream_batch(ids.cuda(), mask.cuda(), max_new_tokens=a.max_new_tokens, greedy=(a.samples == 1))
                for b, r in enumerate(rows[i:i + a.bs]):
                    pub = tok.decode([t for t, y in zip(toks[b], types[b]) if y == 0], skip_special_tokens=True)
                    priv = tok.decode([t for t, y in zip(toks[b], types[b]) if y == 1], skip_special_tokens=True)
                    tm = token_metrics(types[b], toks[b], tok, "math")
                    recs.append({"id": r["id"], "sample": si, "ok": correct(pub, r["answer"]), "gold": r["answer"], "public": pub[-600:], "private_tail": priv[-300:], "tok": tm, "finished": (toks[b][-1] in (model.eos_id, tok.eos_token_id)) if toks[b] else False})
                print(f"sample {si} batch {i}: acc so far {sum(x['ok'] for x in recs)/len(recs):.3f}", flush=True)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, padding_side="left"); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        think_open, think_close = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
        prompts = [tok.apply_chat_template([{"role": "user", "content": f"{r['prompt']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}], tokenize=False, add_generation_prompt=True, enable_thinking=bool(a.thinking)) for r in rows]
        for si in range(a.samples):
            for i in range(0, len(rows), a.bs):
                enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=a.samples > 1, temperature=1.0 if a.samples > 1 else None, top_p=1.0 if a.samples > 1 else None, pad_token_id=tok.pad_token_id)
                for b, r in enumerate(rows[i:i + a.bs]):
                    ids = g[b, enc["input_ids"].shape[1]:].tolist(); ids = ids[:ids.index(tok.pad_token_id)] if tok.pad_token_id in ids else ids
                    types, inside = [], False
                    for t in ids:
                        if t == think_open: inside = True; types.append(1); continue
                        if t == think_close: inside = False; types.append(1); continue
                        types.append(1 if inside else 0)
                    txt = tok.decode(ids, skip_special_tokens=True); pub = txt.split("</think>")[-1] if "</think>" in txt else ("" if a.thinking else txt)
                    finished = (not a.thinking) or ("</think>" in txt)
                    recs.append({"id": r["id"], "sample": si, "ok": finished and correct(pub, r["answer"]), "gold": r["answer"], "public": pub[-600:], "tok": token_metrics(types, ids, tok, "math"), "finished": finished})
                print(f"sample {si} batch {i}: acc so far {sum(x['ok'] for x in recs)/len(recs):.3f}", flush=True)
    N = len(recs)
    agg = {"model": a.model, "ckpt": a.ckpt, "samples": a.samples, "n": len(rows), "acc": sum(r["ok"] for r in recs) / N, "unfinished": sum(not r["finished"] for r in recs) / N,
           **{k: sum(r["tok"][k] for r in recs) / N for k in ("ARI", "ABO", "AIRW", "FCO", "n_tokens", "n_think", "n_speak", "n_speak_blocks")}}
    print(json.dumps(agg, indent=1))
    tag = a.tag or (("sxs" if a.ckpt else f"base_{'think' if a.thinking else 'nothink'}") + f"_avg{a.samples}")
    json.dump(agg, open(os.path.join(a.out, f"summary_{tag}.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"records_{tag}.jsonl"), "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
