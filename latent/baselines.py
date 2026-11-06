"""Baselines 1 & 2: base Qwen3 (no LoRA) answering directly (no thinking) or with full explicit CoT (thinking mode).
Measures accuracy, generated tokens, wall time, time-to-first-answer-token."""
import argparse, json, os, sys, time
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from latent.data import load_jsonl, is_correct

SYS_DIRECT = "Answer with only the final answer in the form 'Answer: <value>'. Do not explain."
SYS_COT = "Think step by step, then finish with a final line of the form 'Answer: <value>'."

def run(model, tok, rows, thinking: bool, bs: int, max_new: int, device):
    res, n_tok, t_total = [], 0, 0.0
    for i in tqdm(range(0, len(rows), bs), desc=f"thinking={thinking}"):
        chunk = rows[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "system", "content": SYS_COT if thinking else SYS_DIRECT}, {"role": "user", "content": ex["question"]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for ex in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        t_total += dt
        gen = out[:, enc["input_ids"].shape[1]:]
        for ex, g in zip(chunk, gen):
            ids = g.tolist()
            ids = ids[:ids.index(tok.pad_token_id)] if tok.pad_token_id in ids else ids
            txt = tok.decode(ids, skip_special_tokens=True)
            # final answer = text after </think> if present
            ans_part = txt.split("</think>")[-1] if "</think>" in txt else txt
            # if thinking truncated (no </think>), count as wrong
            ok = is_correct(ans_part, ex["answer"]) and ("</think>" in txt or not thinking)
            n_think = ids.index(tok.convert_tokens_to_ids("</think>")) if thinking and tok.convert_tokens_to_ids("</think>") in ids else (len(ids) if thinking else 0)
            n_tok += len(ids)
            res.append({"id": ex["id"], "task": ex["task"], "ok": bool(ok), "gold": ex["answer"], "pred": ans_part.strip()[-80:], "n_tokens": len(ids), "n_think_tokens": n_think, "sec_per_batch": dt})
    acc = sum(r["ok"] for r in res) / len(res)
    per_task = {}
    for r in res:
        d = per_task.setdefault(r["task"], [0, 0]); d[0] += r["ok"]; d[1] += 1
    return {"acc": acc, "per_task": {k: v[0] / v[1] for k, v in per_task.items()}, "n": len(res),
            "avg_tokens": n_tok / len(res), "avg_think_tokens": sum(r["n_think_tokens"] for r in res) / len(res),
            "gpu_s_per_ex": t_total / len(res)}, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="test,heldout")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_new_cot", type=int, default=1536)
    ap.add_argument("--modes", default="direct,cot")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    results = {}
    for split in a.splits.split(","):
        rows = load_jsonl(os.path.join(a.data_dir, f"{split}.jsonl"))
        if a.limit: rows = rows[:a.limit]
        for mode in a.modes.split(","):
            thinking = mode == "cot"
            r, preds = run(model, tok, rows, thinking, a.bs, a.max_new_cot if thinking else 16, device)
            results[f"{split}/{mode}"] = r
            print(split, mode, json.dumps(r))
            with open(os.path.join(a.out, f"preds_{split}_{mode}.jsonl"), "w") as f:
                for p in preds: f.write(json.dumps(p) + "\n")
            json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
