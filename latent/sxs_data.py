"""Convert Knights&Knaves and MuSiQue into SxS-style interleaved think/speak trajectories.

Unified record:
  {"id", "source", "prompt", "steps": [{"mode": "think"|"speak", "text"}], "chunks": [gold public chunks],
   "chunk_checks": [...], "final_answer", "aliases"}
Entailment alignment (programmatic stand-in for the LLM checker of SxS, Sec. 3.1):
  K&K   : the verdict for person P is disclosable right after the LAST reasoning step that mentions P (later steps never revisit P);
          reasoning steps between two disclosures are merged into one think block.
  MuSiQue: the answer of hop i is disclosable after the reasoning step for hop i (which cites the supporting passage).
"""
import json, random, re, os, argparse
from typing import List, Dict

KK_INSTR = ("Reason privately. As soon as you are certain about a person, state it publicly on its own line as "
            "'<Name> is a knight.' or '<Name> is a knave.' Cover every person.")
MQ_INSTR = ("Answer the question using the passages. Reason privately; publicly state each intermediate finding on its own line, "
            "and finish with a line 'Final answer: <answer>'.")

def convert_kk(row: Dict, idx: str) -> Dict:
    names, sol, steps = row["names"], row["solution"], row["cot_repeat_steps"]
    last = {}
    for i, s in enumerate(steps):
        for p in names:
            if re.search(rf"\b{re.escape(p)}\b", s):
                last[p] = i
    for p in names:
        last.setdefault(p, len(steps) - 1)
    order = sorted(names, key=lambda p: (last[p], names.index(p)))
    traj, cursor = [], 0
    for p in order:
        end = last[p] + 1
        if end > cursor:
            traj.append({"mode": "think", "text": " ".join(steps[cursor:end])}); cursor = end
        traj.append({"mode": "speak", "text": f"{p} is a {'knight' if sol[names.index(p)] else 'knave'}."})
    prompt = row["quiz"].strip() + "\n\n" + KK_INSTR
    return {"id": f"kk-{idx}", "source": "kk", "prompt": prompt, "steps": traj,
            "chunks": [s["text"] for s in traj if s["mode"] == "speak"],
            "chunk_checks": [{"type": "kk", "name": p, "knight": bool(sol[names.index(p)])} for p in order],
            "final_answer": row["solution_text"], "aliases": [], "n_people": len(names)}

def _natural(q: str) -> str:
    if ">>" in q:
        x, rel = [t.strip() for t in q.split(">>", 1)]
        return f"What is the {rel} of {x}?"
    return q

def _statement(q: str, a: str) -> str:
    if ">>" in q:
        x, rel = [t.strip() for t in q.split(">>", 1)]
        return f"The {rel} of {x} is {a}."
    return f"{q} {a}."

def convert_musique(row: Dict, rng: random.Random, n_distractors: int = 4) -> Dict:
    dec = row["question_decomposition"]
    answers = [d["answer"] for d in dec]
    resolved = []
    for d in dec:
        q = d["question"]
        for k in range(len(dec), 0, -1):
            q = q.replace(f"#{k}", answers[k - 1])
        resolved.append(q)
    sup = [p for p in row["paragraphs"] if p["is_supporting"]]
    dis = [p for p in row["paragraphs"] if not p["is_supporting"]]
    paras = sup + rng.sample(dis, min(n_distractors, len(dis))); rng.shuffle(paras)
    context = "\n\n".join(f"[{i+1}] {p['title']}: {p['paragraph_text']}" for i, p in enumerate(paras))
    by_idx = {p["idx"]: p for p in row["paragraphs"]}
    traj, checks = [], []
    for i, (d, q) in enumerate(zip(dec, resolved)):
        para = by_idx[d["paragraph_support_idx"]]; pi = paras.index(para) + 1
        think = f"Sub-question {i+1}: {_natural(q)} Passage [{pi}] ({para['title']}) is relevant; it indicates the answer is {d['answer']}."
        if i == len(dec) - 1:
            speak = f"Final answer: {row['answer']}"
            checks.append({"type": "final", "answer": row["answer"], "aliases": row.get("answer_aliases", [])})
        else:
            speak = _statement(q, d["answer"])
            checks.append({"type": "contains", "answer": d["answer"]})
        traj += [{"mode": "think", "text": think}, {"mode": "speak", "text": speak}]
    prompt = f"{context}\n\nQuestion: {row['question']}\n\n{MQ_INSTR}"
    return {"id": f"mq-{row['id']}", "source": "musique", "prompt": prompt, "steps": traj,
            "chunks": [s["text"] for s in traj if s["mode"] == "speak"], "chunk_checks": checks,
            "final_answer": row["answer"], "aliases": row.get("answer_aliases", []), "n_hops": len(dec)}

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]

def save(rows, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kk_dir", required=True); ap.add_argument("--mq_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mq_train", type=int, default=6000); ap.add_argument("--mq_dev", type=int, default=300); ap.add_argument("--mq_test", type=int, default=500)
    ap.add_argument("--kk_dev", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    kk_train = []
    for f in sorted(os.listdir(f"{a.kk_dir}/train")):
        for i, r in enumerate(load_jsonl(f"{a.kk_dir}/train/{f}")): kk_train.append(convert_kk(r, f"{f[:-6]}-{i}"))
    kk_test = []
    for f in sorted(os.listdir(f"{a.kk_dir}/test")):
        for i, r in enumerate(load_jsonl(f"{a.kk_dir}/test/{f}")): kk_test.append(convert_kk(r, f"test-{f[:-6]}-{i}"))
    rng.shuffle(kk_train); kk_dev, kk_train = kk_train[:a.kk_dev], kk_train[a.kk_dev:]
    mq_tr = [r for r in load_jsonl(f"{a.mq_dir}/musique_ans_v1.0_train.jsonl") if r.get("answerable", True)]
    mq_dv = [r for r in load_jsonl(f"{a.mq_dir}/musique_ans_v1.0_dev.jsonl") if r.get("answerable", True)]
    rng.shuffle(mq_tr); rng.shuffle(mq_dv)
    mq_train = [convert_musique(r, rng) for r in mq_tr[:a.mq_train]]
    mq_dev = [convert_musique(r, rng) for r in mq_dv[:a.mq_dev]]
    mq_test = [convert_musique(r, rng) for r in mq_dv[a.mq_dev:a.mq_dev + a.mq_test]]
    train = kk_train + mq_train; rng.shuffle(train)
    save(train, f"{a.out}/sxs_train.jsonl"); save(kk_dev + mq_dev, f"{a.out}/sxs_dev.jsonl")
    save(kk_test, f"{a.out}/sxs_test_kk.jsonl"); save(mq_test, f"{a.out}/sxs_test_mq.jsonl")
    for name, rows in [("train", train), ("dev", kk_dev + mq_dev), ("test_kk", kk_test), ("test_mq", mq_test)]:
        nt = [sum(len(s["text"].split()) for s in r["steps"] if s["mode"] == "think") for r in rows]
        nc = [len(r["chunks"]) for r in rows]
        print(f"{name}: n={len(rows)}  mean think words={sum(nt)/len(nt):.1f}  mean chunks={sum(nc)/len(nc):.2f}  prompt words={sum(len(r['prompt'].split()) for r in rows)/len(rows):.0f}")
    print(json.dumps(train[0], indent=1)[:1500]); print(json.dumps([r for r in train if r["source"] == "musique"][0], indent=1)[:2500])

if __name__ == "__main__":
    main()
