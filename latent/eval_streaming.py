"""Phase 2/4 evaluation: stream chunked answers under different Think/Speak policies and score with the playback simulator.
Policies: gate (threshold), fixed-k (baseline 3 / STITCH-like fixed thinking per chunk), nothink (k=0), oracle (k* from counterfactual file).
Reports accuracy (all chunks / per chunk), TTFC, P50/P95 inter-chunk pause, GPU time, hidden steps, premature commitment rate."""
import argparse, json, os, sys, time, statistics
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.model import LatentReasoner, ModelConfig, LatentConfig
from latent.data import load_jsonl, is_correct
from latent.chunked import parse_chunk_answers, weak_kstar, combine_kstar
from latent.simulator import AudioModel, simulate, aggregate
from latent.common import build_prompt, pad_batch

def make_policy(name: str, Kmax: int, thr: float = 0.5, kstar=None, fixed_k: int = 0):
    if name == "gate":
        return lambda i, j, p, f: (p is None) or (p >= thr)
    if name == "fixed":
        return lambda i, j, p, f: j >= fixed_k
    if name == "oracle":
        return lambda i, j, p, f: j >= (kstar[i] if i < len(kstar) else 0)
    if name == "stitch":   # think only while audio is buffered (free time), never stall; fixed max k
        return lambda i, j, p, f: not (j < fixed_k and f["buf"] >= f["step_cost"])
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gate", action="store_true", help="checkpoint has a gate head")
    ap.add_argument("--no_buffer", action="store_true")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="chunked_test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--Kmax", type=int, default=8)
    ap.add_argument("--n_slots", type=int, default=8)
    ap.add_argument("--policy", default="gate", choices=["gate", "fixed", "oracle", "stitch"])
    ap.add_argument("--fixed_k", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--step_cost", type=float, default=0.0, help="simulated per-latent-step compute cost (0 = measured)")
    ap.add_argument("--cf_labels", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = ModelConfig(model_name=a.model, lora_r=16, latent=LatentConfig(n_slots=a.n_slots, max_steps=max(16, a.Kmax + 1)), gate=a.gate, gate_use_buffer=not a.no_buffer)
    model = LatentReasoner(cfg, device="cuda"); model.load(a.ckpt); model.eval()
    tok = model.tok
    rows = load_jsonl(os.path.join(a.data_dir, f"{a.split}.jsonl"))
    if a.limit: rows = rows[:a.limit]
    cf = json.load(open(a.cf_labels)) if a.cf_labels else {}
    audio = AudioModel()
    sims, recs = [], []
    n_chunks_total, n_chunks_ok, n_all_ok, n_prem, n_emitted = 0, 0, 0, 0, 0
    gate_trace_all = []
    for ex in tqdm(rows, desc=f"{a.policy}"):
        kstar = combine_kstar(weak_kstar(ex, a.Kmax), cf.get(ex["id"]), a.Kmax)
        pol = make_policy(a.policy, a.Kmax, a.thr, kstar, a.fixed_k)
        ids = tok(build_prompt(tok, ex["question"]), add_special_tokens=False)["input_ids"]
        p_ids, p_mask = pad_batch([ids], tok.pad_token_id, left=True)
        ev, chunks, trace = model.stream_chunked(p_ids.cuda(), p_mask.cuda(), a.Kmax, pol, max_chunks=ex["n_sub"] + 1, max_chunk_tokens=16,
                                                 newline_ids=[198], step_cost=a.step_cost, audio=audio)
        parsed = parse_chunk_answers(chunks)
        oks = []
        for i, gold in enumerate(ex["sub_answers"]):
            oks.append(bool(is_correct(parsed.get(i + 1, ""), gold)))
        # annotate speak events with correctness (premature/unsupported = wrong chunk emitted)
        si = 0
        for e in ev:
            if e["type"] == "speak":
                e["ok"] = oks[si] if si < len(oks) else False; e["meaningful"] = bool(e["text"].strip()); si += 1
        sim = simulate(ev, audio)
        sims.append(sim)
        n_chunks_total += len(oks); n_chunks_ok += sum(oks); n_all_ok += all(oks); n_prem += sim.premature; n_emitted += sim.n_chunks
        for t in trace: t["id"] = ex["id"]; t["sub_task"] = ex["sub_tasks"][t["chunk"]] if t["chunk"] < len(ex["sub_tasks"]) else None
        gate_trace_all.extend(trace)
        recs.append({"id": ex["id"], "chunks": chunks, "oks": oks, "k_per_chunk": [e["k"] for e in ev if e["type"] == "speak"], "sim": sim.as_dict(), "sub_tasks": ex["sub_tasks"]})
    agg = aggregate(sims)
    agg.update({"policy": a.policy, "fixed_k": a.fixed_k, "thr": a.thr, "step_cost": a.step_cost, "n": len(rows),
                "chunk_acc": n_chunks_ok / max(1, n_chunks_total), "all_correct_acc": n_all_ok / len(rows),
                "premature_rate": n_prem / max(1, n_emitted), "mean_k": statistics.mean([k for r in recs for k in r["k_per_chunk"]] or [0])})
    # per sub-task accuracy and mean k
    pt = {}
    for r in recs:
        for i, (t, ok) in enumerate(zip(r["sub_tasks"], r["oks"])):
            d = pt.setdefault(t, {"ok": 0, "n": 0, "k": 0.0}); d["ok"] += ok; d["n"] += 1; d["k"] += (r["k_per_chunk"][i] if i < len(r["k_per_chunk"]) else 0)
    agg["per_subtask"] = {t: {"acc": d["ok"] / d["n"], "mean_k": d["k"] / d["n"], "n": d["n"]} for t, d in pt.items()}
    print(json.dumps(agg, indent=1))
    tag = f"{a.policy}" + (f"_k{a.fixed_k}" if a.policy in ("fixed", "stitch") else "") + f"_sc{a.step_cost}"
    json.dump(agg, open(os.path.join(a.out, f"summary_{tag}.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"records_{tag}.jsonl"), "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, f"gate_trace_{tag}.jsonl"), "w") as f:
        for t in gate_trace_all: f.write(json.dumps(t) + "\n")

if __name__ == "__main__":
    main()
