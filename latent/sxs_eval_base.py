"""'Base Model' rows of the SxS protocol: unmodified Qwen3 in thinking mode (<think>...</think> = think tokens, rest = speak) and
in no-thinking mode. Same scoring and token-level metrics as sxs_eval (ARI / ABO / AIRW / FCO), Average@k with temperature 1.0 when --samples>1."""
import argparse, json, os, sys, re, torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from latent.sxs_eval import score_kk, score_mq, token_metrics, norm, meaningful_text
from latent.sxs_data import KK_INSTR, MQ_INSTR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=3000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--samples", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    rows = [json.loads(l) for l in open(a.data)][::a.stride]
    if a.limit: rows = rows[:a.limit]
    think_open, think_close = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    recs = []
    for i in tqdm(range(0, len(rows), a.bs)):
        chunk = rows[i:i + a.bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=bool(a.thinking)) for ex in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        for si in range(a.samples):
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=a.samples > 1, temperature=1.0 if a.samples > 1 else None, top_p=1.0 if a.samples > 1 else None, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for ex, g in zip(chunk, gen):
                ids = g.tolist(); ids = ids[:ids.index(tok.pad_token_id)] if tok.pad_token_id in ids else ids
                # token types: inside <think>..</think> => think ; tags themselves counted as think
                types, in_think = [], False
                for t in ids:
                    if t == think_open: in_think = True; types.append(1); continue
                    if t == think_close: in_think = False; types.append(1); continue
                    types.append(1 if in_think else 0)
                txt = tok.decode(ids, skip_special_tokens=True)
                finished_think = (not a.thinking) or ("</think>" in txt) or (think_close in ids)
                public = txt.split("</think>")[-1] if "</think>" in txt else ("" if a.thinking else txt)
                chunks = [l.strip() for l in public.split("\n") if l.strip()]
                oks, final_ok = (score_kk if ex["source"] == "kk" else score_mq)(chunks, ex["chunk_checks"])
                if not finished_think: oks = [False] * len(oks); final_ok = False
                tm = token_metrics(types, ids, tok, ex["source"])
                recs.append({"id": ex["id"], "source": ex["source"], "sample": si, "oks": oks, "final_ok": final_ok, "chunks": chunks[:12], "finished_think": finished_think, "tok": tm})
    N = len(recs)
    agg = {"model": a.model, "thinking": a.thinking, "n": len(rows), "samples": a.samples,
           "final_acc": sum(r["final_ok"] for r in recs) / N, "all_correct": sum(all(r["oks"]) for r in recs) / N,
           "chunk_acc": sum(sum(r["oks"]) for r in recs) / max(1, sum(len(r["oks"]) for r in recs)), "unfinished_think": sum(not r["finished_think"] for r in recs) / N,
           **{k: sum(r["tok"][k] for r in recs) / N for k in ("ARI", "ABO", "AIRW", "FCO", "n_tokens", "n_think", "n_speak", "n_speak_blocks")}}
    print(json.dumps(agg, indent=1))
    tag = f"base_{'think' if a.thinking else 'nothink'}" + (f"_avg{a.samples}" if a.samples > 1 else "")
    json.dump(agg, open(os.path.join(a.out, f"summary_{tag}.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"records_{tag}.jsonl"), "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
