"""SxS-protocol results table: one row per method, per test set.
Rows: Base (think), Base (no-think), SFT no-think, SFT Standard CoT, SFT SxS (+ablations). Metrics: final acc (greedy / Average@4),
all-chunks-correct, ARI, ABO, AIRW, FCO (token indices, paper proxies), think tokens, speak blocks, plus simulator TTFC/P95 for ours."""
import json, glob, os, argparse

def load(f):
    return json.load(open(f)) if os.path.exists(f) else None

def row(name, g, s=None):
    if g is None: return f"| {name} | – |" + " – |" * 9
    acc_s = f"{s['final_acc']:.3f}" if s else "–"
    ttfc = f"{g['mean_ttfc']:.2f} / {g['p95_stall']:.2f}" if "mean_ttfc" in g else "–"
    return (f"| {name} | {g['final_acc']:.3f} | {acc_s} | {g.get('all_correct', float('nan')):.3f} | {g['ARI']:.0f} | {g['ABO']:.0f} | {g['AIRW']:.0f} | {g['FCO']:.0f} | "
            f"{g.get('n_think', g.get('mean_think_tokens', float('nan'))):.0f} | {g['n_speak_blocks']:.2f} | {ttfc} |")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results"); a = ap.parse_args()
    for split in ("sxs_test_kk", "sxs_test_mq"):
        print(f"\n### {split}  (150 examples, stride 4; token-index metrics follow the SxS paper: ARI / ABO / AIRW; FCO = first-content onset)\n")
        print("| method | final acc (greedy) | final acc (Avg@4, T=1) | all chunks ok | ARI | ABO | AIRW | FCO | think tok | speak blocks | sim TTFC s / P95 pause s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for size in ("1.7B", "4B"):
            b = f"{a.root}/sxs/base_{size}/{split}"
            print(row(f"Qwen3-{size} base, thinking mode", load(f"{b}/summary_base_think.json")))
            print(row(f"Qwen3-{size} base, no thinking", load(f"{b}/summary_base_nothink.json")))
        for name, label in [("nothink_1.7B", "SFT no-think (1.7B)"), ("cot_1.7B", "SFT Standard CoT, think-then-speak (1.7B)"),
                            ("main_1.7B", "InnerVoice SFT (1.7B)"), ("noethink_1.7B", "  ablation: no e_think input embedding (1.7B)"),
                            ("trunc_1.7B", "  ablation: 30% truncated-think mix (1.7B)"),
                            ("nothink_loraexp_1.7B", "SFT no-think, LoRA experts (1.7B)"), ("cot_loraexp_1.7B", "SFT Standard CoT, LoRA experts (1.7B)"),
                            ("loraexp_1.7B", "InnerVoice SFT, LoRA experts (1.7B)"),
                            ("nothink_4B", "SFT no-think (4B)"), ("cot_4B", "SFT Standard CoT, think-then-speak (4B)"), ("main_4B", "InnerVoice SFT (4B)")]:
            d = f"{a.root}/sxs_eval/{name}/{split}"
            print(row(label, load(f"{d}/summary_gate_tc0.03.json"), load(f"{d}/summary_gate_tc0.03_avg4.json")))
        g = f"{a.root}/sxs_eval/main_1.7B_grid/{split}"
        if glob.glob(f"{g}/summary_*.json"):
            print(f"\n**Budget / policy grid (InnerVoice 1.7B, greedy, simulated 0.03 s per think token)**\n")
            print("| policy | final acc | all chunks ok | think tok | ARI | AIRW | FCO | sim TTFC s | P50 / P95 pause s | GPU s | premature |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            order = ["nothink", "budget-32", "budget-64", "budget-128", "gate", "fixed-64", "stitch-64"]
            rows = {json.load(open(f))["policy"]: json.load(open(f)) for f in glob.glob(f"{g}/summary_*.json")}
            m = load(f"{a.root}/sxs_eval/main_1.7B/{split}/summary_gate_tc0.03.json")
            if m: rows["gate"] = m
            for p in order:
                r = rows.get(p)
                if r: print(f"| {p} | {r['final_acc']:.3f} | {r['all_correct']:.3f} | {r['mean_think_tokens']:.0f} | {r['ARI']:.0f} | {r['AIRW']:.0f} | {r['FCO']:.0f} | {r['mean_ttfc']:.2f} | {r['p50_stall']:.2f} / {r['p95_stall']:.2f} | {r['mean_gpu_time']:.2f} | {r['premature_rate']:.3f} |")

if __name__ == "__main__":
    main()
