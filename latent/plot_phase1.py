"""Accuracy vs latent steps K (test / per task), one panel per model size. Static PNG for the report."""
import json, glob, os, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed categorical order (validated default palette, see dataviz skill references/palette.md)
COLORS = {"test (all)": "#1F5FD6", "arith_chain": "#B8541A", "word_prob": "#2A8C4A", "graph_reach": "#7B4FB5"}

def load(dirpath, prefix=""):
    pts = {}
    for f in sorted(glob.glob(os.path.join(dirpath, f"{prefix}K*_s*", "results.json"))):
        name = os.path.basename(os.path.dirname(f))
        if not name.startswith(prefix + "K") or not name[len(prefix) + 1:].split("_")[0].isdigit(): continue
        r = json.load(open(f)); K = r["args"]["K"]; t = r[f"test@K{K}"]
        pts.setdefault(K, []).append((t["acc"], t["per_task"]))
    return pts

def panel(ax, pts, title, lora_only=None, cot=None):
    Ks = sorted(pts)
    series = {"test (all)": [sum(p[0] for p in pts[k]) / len(pts[k]) for k in Ks]}
    for task in ["arith_chain", "word_prob", "graph_reach"]:
        series[task] = [sum(p[1][task] for p in pts[k]) / len(pts[k]) for k in Ks]
    # end labels with simple collision avoidance
    ends = sorted([(ys[-1], name) for name, ys in series.items()])
    placed = []
    for y, name in ends:
        yy = y
        if placed and yy - placed[-1] < 0.035: yy = placed[-1] + 0.035
        placed.append(yy)
        ax.annotate(name, (len(Ks) - 1, y), xytext=(6, (yy - y) * 300), textcoords="offset points", va="center", fontsize=8, color="#444")
    for name, ys in series.items():
        ax.plot(range(len(Ks)), ys, color=COLORS[name], lw=2, marker="o", ms=5, label=name)
    if lora_only is not None:
        ax.axhline(lora_only, color="#888", lw=1, ls="--"); ax.annotate("LoRA only, no latent", (0, lora_only), xytext=(0, 4), textcoords="offset points", fontsize=8, color="#666")
    if cot is not None:
        ax.axhline(cot, color="#888", lw=1, ls=":"); ax.annotate("base model, explicit CoT", (0, cot), xytext=(0, 4), textcoords="offset points", fontsize=8, color="#666")
    ax.set_xticks(range(len(Ks))); ax.set_xticklabels([str(k) for k in Ks]); ax.set_xlabel("latent steps K (train = eval)")
    ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=11, loc="left"); ax.grid(axis="y", color="#eee"); ax.spines[["top", "right"]].set_visible(False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results"); ap.add_argument("--out", default="results/phase1_acc_vs_K.png"); a = ap.parse_args()
    panels = []
    for tag, title in [("p1_1.7B", "Qwen3-1.7B, adapter recurrence"), ("p1_1.7B", "Qwen3-1.7B, LM recurrence (COCONUT-style)"), ("p1_4B", "Qwen3-4B, adapter recurrence")]:
        prefix = "lm_" if "LM" in title else ""
        d = os.path.join(a.root, tag)
        pts = load(d, prefix)
        if not pts: continue
        lo = glob.glob(os.path.join(d, "lora_only_s0", "results.json")); lo = json.load(open(lo[0]))["test@K0"]["acc"] if lo else None
        bl = glob.glob(os.path.join(a.root, f"baselines_{tag.split('_')[1]}", "results.json")); cot = json.load(open(bl[0])).get("test/cot", {}).get("acc") if bl else None
        panels.append((pts, title, lo, cot))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4), squeeze=False)
    for ax, (pts, title, lo, cot) in zip(axes[0], panels): panel(ax, pts, title, lo, cot)
    axes[0][0].set_ylabel("exact-match accuracy (test)"); axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Fixed-budget latent reasoning: accuracy does not change with K", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(a.out, dpi=150); print("saved", a.out)
