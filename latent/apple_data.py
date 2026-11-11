"""Apple interleaved-RL (Xie et al. 2025) data splits from our converted K&K / MuSiQue records.
K&K: 3/4/5 people, 900 train + 100 test each. MuSiQue: 3-hop (900 train / 100 test) and 4-hop (all we have: 360 train / 85 test),
context capped at ~1000 tokens by dropping paragraphs not cited by the reasoning steps. Prompts = bare puzzle / context + question
(our SxS instructions stripped); gold = chunk_checks (intermediate + final ground truth).
    python latent/apple_data.py --out data_apple"""
import argparse, json, os, re, random
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--train", default="data_sxs/sxs_train.jsonl"); ap.add_argument("--test_kk", default="data_sxs/sxs_test_kk.jsonl"); ap.add_argument("--test_mq", default="data_sxs/sxs_test_mq.jsonl")
ap.add_argument("--out", default="data_apple"); ap.add_argument("--ctx_tokens", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = random.Random(a.seed); tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

def kk_prompt(r):
    p = r["prompt"]; i = p.find("So who is a knight and who is a knave?")
    return (p[: i + len("So who is a knight and who is a knave?")] if i >= 0 else p.split("\n\n")[0]).strip()

def mq_prompt(r, cap):
    p = r["prompt"]; q = re.search(r"\n\nQuestion: (.*?)\n\n", p, re.S); question = q.group(1).strip() if q else ""
    ctx = p[: q.start()] if q else p
    paras = re.split(r"\n\n(?=\[\d+\] )", ctx.strip()); cited = set(int(x) for x in re.findall(r"\[(\d+)\]", json.dumps(r.get("steps", ""))))
    def n_tok(ps): return len(tok("\n\n".join(ps) + "\n\nQuestion: " + question)["input_ids"])
    while n_tok(paras) > cap:
        drop = [i for i, pp in enumerate(paras) if int(re.match(r"\[(\d+)\]", pp).group(1)) not in cited]
        if not drop: break
        paras.pop(drop[-1])
    return "\n\n".join(paras) + "\n\nQuestion: " + question

def emit(rows, path):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(path, len(rows))

train = [json.loads(l) for l in open(a.train)]
kk_tr = [r for r in train if r["source"] == "kk"]; mq_tr = [r for r in train if r["source"] == "musique"]
out_tr = []
for n in (3, 4, 5):
    rs = [r for r in kk_tr if len(r["chunk_checks"]) == n]; rng.shuffle(rs)
    out_tr += [{"id": r["id"], "source": "kk", "n": n, "prompt": kk_prompt(r), "chunk_checks": r["chunk_checks"]} for r in rs[:900]]
for h, cap_n in ((3, 900), (4, 900)):
    rs = [r for r in mq_tr if len(r["chunk_checks"]) == h]; rng.shuffle(rs)
    out_tr += [{"id": r["id"], "source": "musique", "n": h, "prompt": mq_prompt(r, a.ctx_tokens), "chunk_checks": r["chunk_checks"]} for r in rs[:cap_n]]
rng.shuffle(out_tr); emit(out_tr, f"{a.out}/train.jsonl")
kk_te = [json.loads(l) for l in open(a.test_kk)]; mq_te = [json.loads(l) for l in open(a.test_mq)]
emit([{"id": r["id"], "source": "kk", "n": len(r["chunk_checks"]), "prompt": kk_prompt(r), "chunk_checks": r["chunk_checks"]} for n in (3, 4, 5) for r in [x for x in kk_te if len(x["chunk_checks"]) == n][:100]], f"{a.out}/test_kk.jsonl")
emit([{"id": r["id"], "source": "musique", "n": len(r["chunk_checks"]), "prompt": mq_prompt(r, a.ctx_tokens), "chunk_checks": r["chunk_checks"]} for h in (3, 4) for r in [x for x in mq_te if len(x["chunk_checks"]) == h][:100]], f"{a.out}/test_mq.jsonl")
emit([{"id": r["id"], "source": "kk", "n": len(r["chunk_checks"]), "prompt": kk_prompt(r), "chunk_checks": r["chunk_checks"]} for n in (6, 7, 8) for r in [x for x in kk_te if len(x["chunk_checks"]) == n][:100]], f"{a.out}/test_kk_hard.jsonl")
