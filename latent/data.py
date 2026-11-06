"""Synthetic reasoning tasks with programmatically verified answers.

Each example: {task, question, answer, cot, n_steps, intermediates}
Task families
  arith_chain  : "Start with 23. Multiply by 4. Add 17. ..."   (depth 3-5 train, 6-7 unseen-depth)
  word_prob    : templated multi-step word problems (2-4 steps)
  graph_reach  : ProsQA-style "Every X is a Y" DAG reachability, yes/no, path length >= 2
  even_sum     : held-out family (never trained): sum of even numbers in a list
Multi-chunk (Phase 2): several sub-questions of mixed difficulty in one prompt,
                       answer = one line per sub-question.
"""
import json, random, re, os
from typing import List, Dict

OPS = ["+", "-", "*"]

def _fmt_op(op, b):
    return {"+": f"Add {b}.", "-": f"Subtract {b}.", "*": f"Multiply by {b}."}[op]

def gen_arith_chain(rng: random.Random, depth: int) -> Dict:
    while True:
        a = rng.randint(1, 50)
        v = a
        steps, inter, cot = [], [], []
        ok = True
        for _ in range(depth):
            op = rng.choice(OPS)
            b = rng.randint(2, 9) if op == "*" else rng.randint(1, 99)
            nv = eval(f"{v}{op}{b}")
            if abs(nv) > 9999:
                ok = False
                break
            steps.append(_fmt_op(op, b))
            cot.append(f"{v}{op}{b}={nv}")
            v = nv
            inter.append(v)
        if ok:
            break
    q = f"Start with {a}. " + " ".join(steps) + " What is the final number?"
    return dict(task="arith_chain", question=q, answer=str(v), cot="; ".join(cot) + ".",
                n_steps=depth, intermediates=inter)

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy",
         "Mallory", "Oscar", "Peggy", "Trent", "Victor", "Walter", "Liu", "Chen", "Priya", "Omar"]
ITEMS = ["apples", "marbles", "stickers", "coins", "books", "pencils", "cards", "shells", "stamps", "cookies"]

def gen_word_prob(rng: random.Random, n_steps: int) -> Dict:
    """Chain of n_steps simple operations expressed as a story; answer non-negative int."""
    while True:
        n1, n2 = rng.sample(NAMES, 2)
        it = rng.choice(ITEMS)
        a = rng.randint(5, 60)
        v = a
        sents = [f"{n1} has {a} {it}."]
        cot, inter = [], []
        ok = True
        for _ in range(n_steps):
            kind = rng.choice(["buy_packs", "give", "receive", "double", "share_half", "lose_frac"])
            if kind == "buy_packs":
                p, c = rng.randint(2, 6), rng.randint(2, 9)
                sents.append(f"{n1} buys {p} packs with {c} {it} each.")
                nv = v + p * c; cot.append(f"{v}+{p}*{c}={nv}")
            elif kind == "give":
                d = rng.randint(1, max(1, v - 1))
                sents.append(f"{n1} gives {d} {it} to {n2}.")
                nv = v - d; cot.append(f"{v}-{d}={nv}")
            elif kind == "receive":
                d = rng.randint(1, 40)
                sents.append(f"{n2} gives {n1} {d} more {it}.")
                nv = v + d; cot.append(f"{v}+{d}={nv}")
            elif kind == "double":
                sents.append(f"{n1} then doubles the number of {it} by trading.")
                nv = v * 2; cot.append(f"{v}*2={nv}")
            elif kind == "share_half":
                if v % 2 != 0:
                    ok = False; break
                sents.append(f"{n1} gives half of the {it} to {n2}.")
                nv = v // 2; cot.append(f"{v}/2={nv}")
            else:  # lose a third
                if v % 3 != 0:
                    ok = False; break
                sents.append(f"{n1} loses a third of the {it}.")
                nv = v - v // 3; cot.append(f"{v}-{v}/3={nv}")
            if nv < 0 or nv > 9999:
                ok = False; break
            v = nv; inter.append(v)
        if ok:
            break
    q = " ".join(sents) + f" How many {it} does {n1} have now?"
    return dict(task="word_prob", question=q, answer=str(v), cot="; ".join(cot) + ".",
                n_steps=n_steps, intermediates=inter)

SYLL = ["wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus", "vumpus", "impus",
        "jompus", "lempus", "grimpus", "brimpus", "shumpus", "sterpus", "lorpus", "gorpus", "felpus",
        "quimpus", "bompus", "kerpus", "harpus", "zilpus", "worpus"]

def gen_graph_reach(rng: random.Random, n_nodes: int, want_yes: bool) -> Dict:
    """Random DAG over concept names. Question: is every <root> a <target>?  Path length >= 2 for yes."""
    for _ in range(200):
        names = rng.sample(SYLL, n_nodes)
        order = list(range(n_nodes))
        edges = set()
        # chain + random extra edges, respecting topological order
        for i in range(n_nodes - 1):
            j = rng.randint(i + 1, min(n_nodes - 1, i + 3))
            edges.add((i, j))
        for _ in range(n_nodes):
            i, j = sorted(rng.sample(range(n_nodes), 2))
            edges.add((i, j))
        adj = {i: set() for i in range(n_nodes)}
        for i, j in edges:
            adj[i].add(j)
        def reach(s):
            seen, st = set(), [s]
            while st:
                u = st.pop()
                for w in adj[u]:
                    if w not in seen:
                        seen.add(w); st.append(w)
            return seen
        def dist(s, t):
            from collections import deque
            dq, d = deque([s]), {s: 0}
            while dq:
                u = dq.popleft()
                for w in adj[u]:
                    if w not in d:
                        d[w] = d[u] + 1; dq.append(w)
            return d.get(t)
        src = rng.randint(0, n_nodes // 2)
        R = reach(src)
        if want_yes:
            cands = [t for t in R if dist(src, t) >= 2]
            if not cands:
                continue
            tgt = rng.choice(cands)
        else:
            cands = [t for t in range(n_nodes) if t not in R and t != src and t > src]
            if not cands:
                continue
            tgt = rng.choice(cands)
        facts = [f"Every {names[i]} is a {names[j]}." for i, j in sorted(edges)]
        rng.shuffle(facts)
        q = " ".join(facts) + f" Is every {names[src]} a {names[tgt]}? Answer Yes or No."
        if want_yes:
            # reconstruct a path for the cot
            from collections import deque
            dq, par = deque([src]), {src: None}
            while dq:
                u = dq.popleft()
                for w in adj[u]:
                    if w not in par:
                        par[w] = u; dq.append(w)
            path, u = [], tgt
            while u is not None:
                path.append(names[u]); u = par[u]
            path.reverse()
            cot = " -> ".join(path) + ", so yes."
            n_steps = len(path) - 1
        else:
            cot = f"{names[src]} leads to " + ", ".join(names[t] for t in sorted(R)) + f"; {names[tgt]} is not among them, so no."
            n_steps = len(R)
        return dict(task="graph_reach", question=q, answer="Yes" if want_yes else "No", cot=cot,
                    n_steps=n_steps, intermediates=[])
    raise RuntimeError("graph gen failed")

def gen_even_sum(rng: random.Random, n: int) -> Dict:
    xs = [rng.randint(10, 99) for _ in range(n)]
    ev = [x for x in xs if x % 2 == 0]
    q = "Here is a list of numbers: " + ", ".join(map(str, xs)) + ". What is the sum of the even numbers in the list?"
    return dict(task="even_sum", question=q, answer=str(sum(ev)), cot="Even numbers: " + "+".join(map(str, ev)) + f"={sum(ev)}.",
                n_steps=len(ev), intermediates=[])

# ------------------------------------------------------------------ easy items (for mixed-difficulty chunked prompts)
def gen_easy(rng: random.Random) -> Dict:
    kind = rng.choice(["add", "sub", "mul", "fact"])
    if kind == "add":
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        return dict(task="easy", question=f"What is {a} plus {b}?", answer=str(a + b), cot=f"{a}+{b}={a+b}.", n_steps=1, intermediates=[])
    if kind == "sub":
        a, b = rng.randint(20, 99), rng.randint(1, 19)
        return dict(task="easy", question=f"What is {a} minus {b}?", answer=str(a - b), cot=f"{a}-{b}={a-b}.", n_steps=1, intermediates=[])
    if kind == "mul":
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        return dict(task="easy", question=f"What is {a} times {b}?", answer=str(a * b), cot=f"{a}*{b}={a*b}.", n_steps=1, intermediates=[])
    n = rng.randint(3, 9)
    return dict(task="easy", question=f"How many sides does a polygon with {n} vertices have?", answer=str(n), cot=f"A polygon has as many sides as vertices: {n}.", n_steps=0, intermediates=[])

# ------------------------------------------------------------------ dataset builders
def gen_single(rng: random.Random, task: str, split: str) -> Dict:
    if task == "arith_chain":
        return gen_arith_chain(rng, rng.choice([3, 4, 5]))
    if task == "arith_chain_deep":   # unseen depth
        ex = gen_arith_chain(rng, rng.choice([6, 7])); ex["task"] = "arith_chain_deep"; return ex
    if task == "word_prob":
        return gen_word_prob(rng, rng.choice([2, 3, 4]))
    if task == "graph_reach":
        return gen_graph_reach(rng, rng.randint(8, 12), rng.random() < 0.5)
    if task == "even_sum":
        return gen_even_sum(rng, rng.randint(6, 8))
    if task == "easy":
        return gen_easy(rng)
    raise ValueError(task)

TRAIN_TASKS = ["arith_chain", "word_prob", "graph_reach"]
HELDOUT_TASKS = ["even_sum", "arith_chain_deep"]

def build_split(seed: int, n_per_task: int, tasks: List[str], split: str) -> List[Dict]:
    rng = random.Random(seed)
    out = []
    seen = set()
    for t in tasks:
        k = 0
        while k < n_per_task:
            ex = gen_single(rng, t, split)
            if ex["question"] in seen:
                continue
            seen.add(ex["question"]); ex["id"] = f"{t}-{split}-{k}"; out.append(ex); k += 1
    rng.shuffle(out)
    return out

def gen_chunked(rng: random.Random, n_sub: int) -> Dict:
    """Prompt with n_sub sub-questions of mixed difficulty; answer = one line per sub-question."""
    subs = []
    for i in range(n_sub):
        if rng.random() < 0.45:
            subs.append(gen_easy(rng))
        else:
            subs.append(gen_single(rng, rng.choice(TRAIN_TASKS), "train"))
    q = "Answer each question on its own line, in the format 'A<i>: <answer>'.\n" + "\n".join(f"Q{i+1}: {s['question']}" for i, s in enumerate(subs))
    chunks = [f"A{i+1}: {s['answer']}" for i, s in enumerate(subs)]
    return dict(task="chunked", question=q, chunks=chunks, answer="\n".join(chunks),
                sub_tasks=[s["task"] for s in subs], sub_answers=[s["answer"] for s in subs],
                sub_cots=[s["cot"] for s in subs], sub_questions=[s["question"] for s in subs], n_sub=n_sub)

def build_chunked(seed: int, n: int, n_sub_choices=(2, 3, 4)) -> List[Dict]:
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        ex = gen_chunked(rng, rng.choice(n_sub_choices))
        if ex["question"] in seen:
            continue
        seen.add(ex["question"]); ex["id"] = f"chunked-{seed}-{len(out)}"; out.append(ex)
    return out

def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def normalize_answer(s: str) -> str:
    s = s.strip()
    m = re.search(r"answer\s*:\s*(.+)", s, flags=re.I)
    if m:
        s = m.group(1)
    s = s.strip().strip(".").strip()
    m = re.search(r"-?\d[\d,]*", s)
    if m:
        return m.group(0).replace(",", "")
    s = s.lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    return s

def is_correct(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--n_chunked_train", type=int, default=6000)
    ap.add_argument("--n_chunked_test", type=int, default=400)
    a = ap.parse_args()
    save_jsonl(build_split(1, a.n_train, TRAIN_TASKS, "train"), f"{a.out}/train.jsonl")
    save_jsonl(build_split(2, a.n_val, TRAIN_TASKS, "val"), f"{a.out}/val.jsonl")
    save_jsonl(build_split(3, a.n_test, TRAIN_TASKS, "test"), f"{a.out}/test.jsonl")
    save_jsonl(build_split(4, a.n_test, HELDOUT_TASKS, "heldout"), f"{a.out}/heldout.jsonl")
    save_jsonl(build_chunked(5, a.n_chunked_train), f"{a.out}/chunked_train.jsonl")
    save_jsonl(build_chunked(6, a.n_chunked_test), f"{a.out}/chunked_test.jsonl")
    for f in ["train", "val", "test", "heldout", "chunked_train", "chunked_test"]:
        rows = load_jsonl(f"{a.out}/{f}.jsonl"); print(f, len(rows)); print("  ", json.dumps(rows[0])[:400])
