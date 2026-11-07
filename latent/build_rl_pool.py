"""RL prompt pool: DeepMath problems (answers only) NOT used in any SFT set, difficulty >= --min_diff.
    python latent/build_rl_pool.py --out data_math/rl_pool.jsonl"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--src", default="data_math/dm_sxs_train.jsonl"); ap.add_argument("--exclude", nargs="*", default=["data_math/self/sxs_q32high_train.jsonl", "data_math/dm_sxs_dev.jsonl"])
ap.add_argument("--min_diff", type=float, default=5); ap.add_argument("--out", required=True)
a = ap.parse_args()
ban = set()
for f in a.exclude: ban |= {json.loads(l)["prompt"] for l in open(f)}
seen, n = set(), 0
with open(a.out, "w") as fo:
    for l in open(a.src):
        r = json.loads(l)
        if r["prompt"] in ban or r["prompt"] in seen or float(r.get("difficulty", 0)) < a.min_diff: continue
        seen.add(r["prompt"]); n += 1
        fo.write(json.dumps({k: r[k] for k in ("id", "source", "prompt", "final_answer", "chunk_checks", "topic", "difficulty")}) + "\n")
print("wrote", n, "problems ->", a.out)
