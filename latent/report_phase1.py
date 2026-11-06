"""Aggregate Phase-1 results into markdown tables."""
import json, glob, os, sys, argparse
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--baselines", default="")
    a = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*", "results.json"))):
        name = os.path.basename(os.path.dirname(f)); runs[name] = json.load(open(f))
    rows = []
    tasks = ["arith_chain", "word_prob", "graph_reach", "even_sum", "arith_chain_deep"]
    print(f"## Phase 1 results: {a.dir}\n")
    print("| run | train K | test acc | " + " | ".join(tasks[:3]) + " | heldout acc | even_sum | arith_deep | ms/ex | latent ms/ex | train min |")
    print("|---|---|---|" + "---|" * 3 + "---|---|---|---|---|---|")
    for name, r in runs.items():
        K = r["args"]["K"]; t = r.get(f"test@K{K}"); h = r.get(f"heldout@K{K}")
        if not t: continue
        pt, ph = t["per_task"], h["per_task"]
        print(f"| {name} | {K} | {t['acc']:.3f} | " + " | ".join(f"{pt.get(x, float('nan')):.3f}" for x in tasks[:3]) +
              f" | {h['acc']:.3f} | {ph.get('even_sum', float('nan')):.3f} | {ph.get('arith_chain_deep', float('nan')):.3f} | {t['gpu_s_per_ex']*1000:.1f} | {t['latent_s_per_ex']*1000:.2f} | {r['train_time_s']/60:.1f} |")
    print("\n### Test-time budget transfer (test acc when evaluated at K' != train K)\n")
    Ks = [0, 1, 2, 4, 8]
    print("| run \\ eval K | " + " | ".join(map(str, Ks)) + " |"); print("|---|" + "---|" * len(Ks))
    for name, r in runs.items():
        vals = [r.get(f"test@K{k}", {}).get("acc") for k in Ks]
        if sum(v is not None for v in vals) <= 1: continue
        print(f"| {name} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")
    if a.baselines and os.path.exists(a.baselines):
        b = json.load(open(a.baselines))
        print("\n### Base-model baselines (no LoRA)\n")
        print("| mode | split | acc | per task | avg tokens | ms/ex |"); print("|---|---|---|---|---|---|")
        for k, v in b.items():
            print(f"| {k.split('/')[1]} | {k.split('/')[0]} | {v['acc']:.3f} | {json.dumps({t: round(x, 3) for t, x in v['per_task'].items()})} | {v['avg_tokens']:.0f} | {v['gpu_s_per_ex']*1000:.0f} |")

if __name__ == "__main__":
    main()
