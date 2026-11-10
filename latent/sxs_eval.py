"""Streaming inference + evaluation for the readable think/speak model.
Think tokens and answer tokens are kept in SEPARATE streams: think text is private (logged to records only), answer chunks are
public (scored, and fed to the playback simulator as speech). Policies:
  gate         router decides (p_think >= 0.5), with a per-block think cap
  nothink      never think
  budget-N     router decides but at most N think tokens per block
  fixed-N      exactly N think tokens before every chunk (STITCH/fixed-budget style)
  stitch-N     think (up to N tokens per block) only while audio is buffered; otherwise answer
Metrics: chunk accuracy, final/all-correct accuracy, TTFC, P50/P95 pause, GPU time, think tokens, premature rate."""
import argparse, json, os, sys, re, string, statistics, collections
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.simulator import AudioModel, simulate, aggregate

def norm(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation)); s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def score_kk(chunks, checks):
    verdict = {}
    for c in chunks:
        m = re.match(r"\s*([A-Z][a-z]+) is a (knight|knave)", c)
        if m and m.group(1) not in verdict: verdict[m.group(1)] = m.group(2) == "knight"
    oks = [verdict.get(ch["name"]) == ch["knight"] for ch in checks]
    return oks, all(oks) and len(verdict) == len(checks)

def score_mq(chunks, checks):
    oks = []
    final_ok = False
    for ch in checks:
        if ch["type"] == "final":
            golds = [ch["answer"]] + list(ch.get("aliases", []))
            fin = [c for c in chunks if c.lower().startswith("final answer")]
            pred = fin[-1].split(":", 1)[-1] if fin else (chunks[-1] if chunks else "")
            final_ok = any(norm(pred) == norm(g) for g in golds) or any(norm(g) and norm(g) in norm(pred) for g in golds)
            oks.append(final_ok)
        else:
            oks.append(any(norm(ch["answer"]) in norm(c) for c in chunks))
    return oks, final_ok

STOPLIST = ("okay", "ok", "sure", "let me", "let's", "alright", "well", "hmm", "so,", "first,", "i will", "i'll", "we need", "the question")

def meaningful_text(text: str, source: str) -> bool:
    t = text.strip().lower()
    if not t or any(t.startswith(w) for w in STOPLIST):
        return False
    if source == "kk":
        return re.search(r"\b[a-z]+ is a (knight|knave)", t) is not None
    if source == "convfill":
        return len(t.split()) >= 4 and re.search(r"\d|\b(you|your|the|a|an)\b [a-z]+ (is|are|was|should|need|can|will|has|have)\b", t) is not None
    return len(t.split()) >= 2

def token_metrics(types, tids, tok, source):
    """types[i] in {0 speak, 1 think} for generated token i (0-indexed). Paper metrics (1-indexed positions):
    ARI = mean index of speak tokens; ABO = mean onset index of speak blocks; AIRW = mean length of the think span preceding
    each speak block; FCO = onset index of the first speak block with task-relevant content (len+1 if none)."""
    n = len(types)
    speak_idx = [i + 1 for i, t in enumerate(types) if t == 0]
    blocks = []   # (onset, end) 1-indexed inclusive
    i = 0
    while i < n:
        if types[i] == 0:
            j = i
            while j + 1 < n and types[j + 1] == 0: j += 1
            blocks.append((i + 1, j + 1)); i = j + 1
        else:
            i += 1
    waits = []
    prev_end = 0
    for (on, en) in blocks:
        waits.append(on - 1 - prev_end); prev_end = en
    fco = n + 1
    for (on, en) in blocks:
        txt = tok.decode([tids[k] for k in range(on - 1, en) if tids[k] is not None], skip_special_tokens=True)
        if meaningful_text(txt, source):
            fco = on; break
    return {"ARI": sum(speak_idx) / len(speak_idx) if speak_idx else float(n + 1), "ABO": sum(b[0] for b in blocks) / len(blocks) if blocks else float(n + 1),
            "AIRW": sum(waits) / len(waits) if waits else float(n), "FCO": float(fco), "n_tokens": n, "n_think": sum(types), "n_speak": n - sum(types), "n_speak_blocks": len(blocks)}

def token_f1(a, b):
    ta, tb = norm(a).split(), norm(b).split()
    if not ta or not tb: return 0.0
    import collections
    ca, cb = collections.Counter(ta), collections.Counter(tb); ov = sum((ca & cb).values())
    if ov == 0: return 0.0
    p, r = ov / len(ta), ov / len(tb); return 2 * p * r / (p + r)

def score_convfill(chunks, checks, thr=0.5):
    """chunk i 'ok' if its token-F1 with the i-th reference phrase >= thr (order-aligned); final_ok = mean F1 >= thr."""
    refs = [c["answer"] for c in checks]
    f1s = [token_f1(chunks[i], refs[i]) if i < len(chunks) else 0.0 for i in range(len(refs))]
    oks = [f >= thr for f in f1s]
    return oks, (sum(f1s) / max(1, len(f1s))) >= thr

SCORERS = {"kk": None, "musique": None, "convfill": score_convfill}

def make_policy(name):
    if name == "gate": return lambda s: None, 256
    if name == "nothink": return lambda s: "answer", 0
    kind, n = name.split("-"); n = int(n)
    if kind == "budget": return (lambda s: None), n
    if kind == "fixed": return (lambda s: "think" if s["n_think_block"] < n else "answer"), n
    if kind == "stitch": return (lambda s: "think" if (s["n_think_block"] < n and s["buf"] > 0.05) else "answer"), n
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--policy", default="gate"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think_cost", type=float, default=0.0, help="simulated seconds per think token (0 = measured)")
    ap.add_argument("--max_new_tokens", type=int, default=600); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--samples", type=int, default=1, help="k samples per example (Average@k); >1 uses temperature sampling"); ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--no_expert_ft", action="store_true"); ap.add_argument("--no_e_think", action="store_true"); ap.add_argument("--shared_ffn", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    model = SxSModel(SxSConfig(model_name=a.model, expert_full_ft=not a.no_expert_ft, use_e_think=not a.no_e_think, shared_ffn=a.shared_ffn), device="cuda"); model.load(a.ckpt); model.eval()
    tok = model.tok; audio = AudioModel()
    rows = [json.loads(l) for l in open(a.data)]
    rows = rows[::a.stride]
    if a.limit: rows = rows[:a.limit]
    policy, cap = make_policy(a.policy)
    sims, recs = [], []; n_chunk_ok = n_chunk = n_all = n_final = 0; think_tok_total = 0
    tm_all = []
    for ex in tqdm(rows, desc=a.policy):
        prompt = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]], device="cuda")
        for si in range(a.samples):
            events, think_blocks, chunks, trace = model.stream(ids, policy, max_new_tokens=a.max_new_tokens, max_think_per_block=cap if cap > 0 else 1,
                                                               audio=audio, think_cost=a.think_cost or None, greedy=(a.samples == 1))
            oks, final_ok = (score_kk if ex["source"] == "kk" else score_mq if ex["source"] == "musique" else score_convfill)(chunks, ex["chunk_checks"])
            for e in events:
                if e["type"] == "speak":
                    if ex["source"] == "convfill":
                        ok = any(token_f1(e["text"], c["answer"]) >= 0.5 for c in ex["chunk_checks"])
                    elif ex["source"] == "kk":
                        m = re.match(r"\s*([A-Z][a-z]+) is a (knight|knave)", e["text"])
                        ok = bool(m) and any(ch["name"] == m.group(1) and ch["knight"] == (m.group(2) == "knight") for ch in ex["chunk_checks"])
                    else:
                        ok = any(norm(ch["answer"]) and norm(ch["answer"]) in norm(e["text"]) for ch in ex["chunk_checks"])
                    e["ok"] = ok; e["meaningful"] = meaningful_text(e["text"], ex["source"])
            sim = simulate(events, audio); sims.append(sim)
            types = [1 if t["mode"] == "think" else 0 for t in trace]; tids = [t.get("tid") for t in trace]
            tm = token_metrics(types, tids, tok, ex["source"]); tm_all.append(tm)
            n_think = tm["n_think"]; think_tok_total += n_think
            n_chunk_ok += sum(oks); n_chunk += len(oks); n_all += all(oks); n_final += final_ok
            n_sw = sum(1 for i in range(1, len(trace)) if trace[i]["mode"] != trace[i - 1]["mode"])
            recs.append({"id": ex["id"], "sample": si, "source": ex["source"], "think": think_blocks, "chunks": chunks, "oks": oks, "final_ok": final_ok, "n_think_tokens": n_think,
                         "n_switches": n_sw, "n_think_blocks": len(think_blocks), "sim": sim.as_dict(), "tok": tm})
    N = len(recs)
    agg = aggregate(sims)
    agg.update({"policy": a.policy, "n": len(rows), "samples": a.samples, "chunk_acc": n_chunk_ok / max(1, n_chunk), "all_correct": n_all / N, "final_acc": n_final / N,
                "mean_think_tokens": think_tok_total / N, "mean_switches": sum(r["n_switches"] for r in recs) / N, "mean_think_blocks": sum(r["n_think_blocks"] for r in recs) / N,
                **{k: sum(t[k] for t in tm_all) / N for k in ("ARI", "ABO", "AIRW", "FCO", "n_tokens", "n_speak_blocks")}, "premature_rate": sum(s.premature for s in sims) / max(1, sum(s.n_chunks for s in sims)), "think_cost": a.think_cost})
    for src in ("kk", "musique"):
        sub = [r for r in recs if r["source"] == src]
        if sub: agg[f"{src}_final_acc"] = sum(r["final_ok"] for r in sub) / len(sub); agg[f"{src}_chunk_acc"] = sum(sum(r["oks"]) for r in sub) / max(1, sum(len(r["oks"]) for r in sub))
    print(json.dumps(agg, indent=1))
    tag = f"{a.policy}_tc{a.think_cost}" + (f"_avg{a.samples}" if a.samples > 1 else "")
    json.dump(agg, open(os.path.join(a.out, f"summary_{tag}.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"records_{tag}.jsonl"), "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
