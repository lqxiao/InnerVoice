"""Per-size breakdown of K&K accuracy (2..8 people) and per-hop MuSiQue accuracy from records files."""
import json, glob, os, re, argparse, collections
ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results"); a = ap.parse_args()
def load(f): return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
rows = {"Qwen3-1.7B base think": f"{a.root}/sxs/base_1.7B/sxs_test_kk/records_base_think.jsonl",
        "Qwen3-4B base think": f"{a.root}/sxs/base_4B/sxs_test_kk/records_base_think.jsonl",
        "SFT no-think 1.7B": f"{a.root}/sxs_eval/nothink_1.7B/sxs_test_kk/records_gate_tc0.03.jsonl",
        "SFT CoT 1.7B": f"{a.root}/sxs_eval/cot_1.7B/sxs_test_kk/records_gate_tc0.03.jsonl",
        "SFT SxS 1.7B": f"{a.root}/sxs_eval/main_1.7B/sxs_test_kk/records_gate_tc0.03.jsonl",
        "SFT SxS 1.7B (LoRA experts)": f"{a.root}/sxs_eval/loraexp_1.7B/sxs_test_kk/records_gate_tc0.03.jsonl",
        "SFT SxS 4B": f"{a.root}/sxs_eval/main_4B/sxs_test_kk/records_gate_tc0.03.jsonl"}
sizes = list(range(2, 9))
print("### K&K all-verdicts-correct by number of people (greedy)\n")
print("| method | " + " | ".join(f"n={s}" for s in sizes) + " | mean think tok (n=2 / n=7) |"); print("|---|" + "---|" * (len(sizes) + 1))
for name, f in rows.items():
    recs = load(f)
    if not recs: continue
    by = collections.defaultdict(list); tk = collections.defaultdict(list)
    for r in recs:
        m = re.search(r"people(\d)", r["id"]); n = int(m.group(1))
        by[n].append(all(r["oks"]) if r["oks"] else False)
        tk[n].append(r.get("n_think_tokens", r.get("tok", {}).get("n_think", 0)))
    cells = [f"{sum(by[s])/len(by[s]):.2f} ({len(by[s])})" if by[s] else "–" for s in sizes]
    print(f"| {name} | " + " | ".join(cells) + f" | {sum(tk[2])/max(1,len(tk[2])):.0f} / {sum(tk[7])/max(1,len(tk[7])):.0f} |")
print("\n### MuSiQue final-answer accuracy by hops (greedy)\n")
rows_mq = {k: v.replace("sxs_test_kk", "sxs_test_mq") for k, v in rows.items()}
hops = [2, 3, 4]
print("| method | " + " | ".join(f"{h}-hop" for h in hops) + " | intermediate chunk acc |"); print("|---|" + "---|" * (len(hops) + 1))
for name, f in rows_mq.items():
    recs = load(f)
    if not recs: continue
    by = collections.defaultdict(list); inter_ok = inter_n = 0
    for r in recs:
        h = int(r["id"].split("hop")[0].split("-")[-1]); by[h].append(r["final_ok"])
        inter_ok += sum(r["oks"][:-1]); inter_n += len(r["oks"][:-1])
    print(f"| {name} | " + " | ".join(f"{sum(by[h])/len(by[h]):.2f} ({len(by[h])})" if by[h] else "–" for h in hops) + f" | {inter_ok/max(1,inter_n):.2f} |")
