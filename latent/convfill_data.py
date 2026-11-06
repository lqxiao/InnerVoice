"""ConvFill (Talker/Reasoner) -> single-model interleaved think/speak trajectories.
Per turn: thoughts[i] (knowledge chunk or <sil>) aligned with response[i] (spoken phrase).
  thoughts[i] != <sil> : think = thoughts[i] ; speak = response[i]
  thoughts[i] == <sil> : speak = response[i]  (filler / grounded small talk while nothing is known yet)  -> flagged is_filler
Prompt = conversation history (previous user turns and the full spoken responses) + current user turn."""
import json, os, random, argparse, glob

SYS = "You are a voice assistant. Think privately when you need to work something out; speak in short natural phrases, one per line."

def convert_file(path, domain):
    out = []
    for ci, line in enumerate(open(path)):
        row = json.loads(line); hist = []
        for ti, t in enumerate(row["conversation"]):
            th, rs = t["thoughts"], t["response"]
            if len(th) != len(rs) or not rs: continue
            steps, flags = [], []
            for a, b in zip(th, rs):
                a, b = str(a).strip(), str(b).strip()
                if not b: continue
                if a and a != "<sil>":
                    steps.append({"mode": "think", "text": a})
                steps.append({"mode": "speak", "text": b}); flags.append(a == "<sil>" or not a)
            if not steps: continue
            prompt = "\n".join(hist + [f"User: {t['user']}"]) if hist else f"User: {t['user']}"
            out.append({"id": f"cf-{domain}-{ci}-{ti}", "source": "convfill", "domain": domain, "prompt": prompt, "steps": steps,
                        "chunks": [s["text"] for s in steps if s["mode"] == "speak"], "chunk_is_filler": flags,
                        "chunk_checks": [{"type": "ref", "answer": s["text"]} for s in steps if s["mode"] == "speak"],
                        "n_turn": ti, "conv": f"{domain}-{ci}"})
            hist.append(f"User: {t['user']}"); hist.append("Assistant: " + " ".join(str(x) for x in rs))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev_frac", type=float, default=0.05); ap.add_argument("--test_frac", type=float, default=0.05)
    a = ap.parse_args(); rows = []
    for f in sorted(glob.glob(f"{a.dir}/data/conversations_*.jsonl")):
        dom = os.path.basename(f)[len("conversations_"):-6]; rows += convert_file(f, dom)
    convs = sorted({r["conv"] for r in rows}); rng = random.Random(a.seed); rng.shuffle(convs)
    n = len(convs); dev = set(convs[:int(n * a.dev_frac)]); test = set(convs[int(n * a.dev_frac):int(n * (a.dev_frac + a.test_frac))])
    split = {"train": [r for r in rows if r["conv"] not in dev | test], "dev": [r for r in rows if r["conv"] in dev], "test": [r for r in rows if r["conv"] in test]}
    os.makedirs(a.out, exist_ok=True)
    for k, rs in split.items():
        rng.shuffle(rs)
        with open(f"{a.out}/convfill_{k}.jsonl", "w") as fo:
            for r in rs: fo.write(json.dumps(r) + "\n")
        nf = sum(sum(r["chunk_is_filler"]) for r in rs); nc = sum(len(r["chunks"]) for r in rs)
        tw = sum(len(s["text"].split()) for r in rs for s in r["steps"] if s["mode"] == "think")
        print(f"{k}: turns={len(rs)} chunks/turn={nc/len(rs):.2f} filler_frac={nf/nc:.2f} think_words/turn={tw/len(rs):.1f} prompt_words={sum(len(r['prompt'].split()) for r in rs)/len(rs):.0f}")
    print(json.dumps(split["train"][0], indent=1)[:1200])

if __name__ == "__main__":
    main()
