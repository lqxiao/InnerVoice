"""DeepMath -> SxS-style interleaved trajectories (scaled-down replication of the paper's data pipeline).
Stages:
  select  : subset of DeepMath problems (short final answers, bounded reasoning length) -> jsonl {id, question, final_answer, reasoning}
  gen     : synthesise a reader-facing multi-paragraph solution with Qwen3 (no thinking), keep only those whose boxed answer matches
  align   : entailment alignment (programmatic proxy): answer paragraph j becomes disclosable at the smallest reasoning prefix that
            contains >=60% of its numeric/LaTeX anchors (final-answer paragraph: the boxed value); monotone in j. Emits steps[].
"""
import argparse, json, os, re, random, sys
try:
    import pyarrow.parquet as pq
except ImportError:  # only needed for the select stage
    pq = None

BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

def last_boxed(s):
    m = BOX.findall(s); return m[-1].strip() if m else None

def norm_ans(a):
    if a is None: return None
    a = a.strip().replace(" ", "").replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac").replace("^{\\circ}", "").replace("^\\circ", "")
    a = a.rstrip(".")
    return a

def select(a):
    rows = []
    for f in sorted(os.listdir(a.deepmath_dir + "/data")):
        if not f.endswith(".parquet"): continue
        t = pq.read_table(f"{a.deepmath_dir}/data/{f}").to_pylist()
        for i, r in enumerate(t):
            fa = r["final_answer"].strip()
            if len(fa) > 25 or "\\text" in fa: continue
            sol = r["r1_solution_1"]
            nw = len(sol.split())
            if nw < 150 or nw > a.max_reason_words: continue
            if norm_ans(last_boxed(sol)) != norm_ans(fa): continue
            rows.append({"id": f"dm-{f.split('-')[1]}-{i}", "question": r["question"], "final_answer": fa, "difficulty": r["difficulty"], "topic": r["topic"], "reasoning": sol})
    random.Random(a.seed).shuffle(rows); rows = rows[:a.n]
    with open(a.out, "w") as fo:
        for r in rows: fo.write(json.dumps(r) + "\n")
    print("selected", len(rows))

GEN_SYS = ("You write clean, reader-facing solutions. Given a problem, a correct worked reasoning, and the correct final answer, write the solution "
           "a reader should see: 3 to 6 short paragraphs separated by blank lines, each paragraph stating one concrete step or intermediate result "
           "(with its value), no exploratory musings, no 'let me'/'wait'. The LAST paragraph must be exactly: Final answer: \\boxed{<answer>}")

def gen(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    rows = [json.loads(l) for l in open(a.inp)]
    if a.shard_n > 1: rows = rows[a.shard_i::a.shard_n]
    out = open(a.out, "w"); kept = 0
    for i in range(0, len(rows), a.bs):
        ch = rows[i:i + a.bs]
        msgs = []
        for r in ch:
            reason = " ".join(r["reasoning"].split()[-a.reason_words:])
            u = f"Problem:\n{r['question']}\n\nWorked reasoning (may be messy):\n{reason}\n\nCorrect final answer: {r['final_answer']}\n\nWrite the reader-facing solution now."
            msgs.append(tok.apply_chat_template([{"role": "system", "content": GEN_SYS}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
        enc = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=6000).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(ch, texts):
            t = t.strip()
            if norm_ans(last_boxed(t)) != norm_ans(r["final_answer"]): continue
            paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
            if len(paras) < 2: continue
            r2 = dict(r); r2["answer_paras"] = paras; out.write(json.dumps(r2) + "\n"); kept += 1
        print(f"{i+len(ch)}/{len(rows)} kept={kept}", flush=True)
    out.close(); print("kept", kept)

ANCH = re.compile(r"-?\d+(?:\.\d+)?|\\frac\{[^{}]+\}\{[^{}]+\}|\\sqrt\{[^{}]+\}|\\pi|\\infty")

def anchors(s):
    return set(x for x in ANCH.findall(s) if not (x.isdigit() and len(x) == 1 and x in "012"))   # drop trivial 0/1/2

def split_reasoning(text, min_words=40):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged = []
    for p in paras:
        if merged and len(merged[-1].split()) < min_words: merged[-1] += "\n\n" + p
        else: merged.append(p)
    return merged

def split(a):
    """Use the R1 solution's own structure: text before '**Final Answer**' = private reasoning, text after = reader-facing answer paragraphs."""
    rows = [json.loads(l) for l in open(a.inp)]
    out = open(a.out, "w"); kept = 0; n_par = 0
    for r in rows:
        txt = r["reasoning"]
        if "</think>" in txt:                      # R1 format: <exploratory reasoning> ... **Final Answer** \boxed{} </think> <clean solution>
            reason, ans = txt.rsplit("</think>", 1)
            reason = reason.split("**Final Answer**")[0] if "**Final Answer**" in reason else reason
        elif "**Final Answer**" in txt:
            reason, ans = txt.rsplit("**Final Answer**", 1)
        else:
            continue
        reason = reason.strip()
        if len(reason.split()) < 100: continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", ans.strip()) if p.strip()]
        merged = []
        for p in paras:   # attach display-math / very short fragments to the preceding paragraph
            if merged and (p.startswith("\\[") or p.startswith("$$") or len(p.split()) < 6 or re.fullmatch(r"\\\[.*\\\]", p, re.S)):
                merged[-1] += "\n\n" + p
            else:
                merged.append(p)
        if norm_ans(last_boxed(" ".join(merged))) != norm_ans(r["final_answer"]): continue
        if len(merged) < 2: continue
        r2 = dict(r); r2["reasoning"] = reason; r2["answer_paras"] = merged; out.write(json.dumps(r2) + "\n"); kept += 1; n_par += len(merged)
    out.close(); print(f"split kept {kept}/{len(rows)}  mean answer paragraphs {n_par/max(1,kept):.2f}")

def selfgen(a):
    """Self-distillation traces: the base Qwen3 in thinking mode solves the problem; keep samples whose boxed answer is correct.
    reasoning = <think> content ; answer_paras = paragraphs of the post-think solution."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    rows = [json.loads(l) for l in open(a.inp)]
    if a.shard_n > 1: rows = rows[a.shard_i::a.shard_n]
    out = open(a.out, "w"); kept = 0; done = 0
    for i in range(0, len(rows), a.bs):
        ch = rows[i:i + a.bs]
        msgs = [tok.apply_chat_template([{"role": "user", "content": f"{r['question']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in ch]
        enc = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.6, top_p=0.95, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(ch, texts):
            done += 1
            if "</think>" not in t: continue
            think, ans = t.split("</think>", 1); think = think.replace("<think>", "").strip(); ans = ans.strip()
            if norm_ans(last_boxed(ans)) != norm_ans(r["final_answer"]): continue
            paras = [p.strip() for p in re.split(r"\n\s*\n", ans) if p.strip()]
            merged = []
            for p in paras:
                if merged and (p.startswith("\\[") or p.startswith("$$") or len(p.split()) < 6): merged[-1] += "\n\n" + p
                else: merged.append(p)
            if len(merged) < 2 or len(think.split()) < 80: continue
            r2 = {k: r[k] for k in ("id", "question", "final_answer", "difficulty", "topic")}; r2["reasoning"] = think; r2["answer_paras"] = merged
            out.write(json.dumps(r2) + "\n"); kept += 1
        print(f"{done}/{len(rows)} kept={kept}", flush=True)
    out.close(); print("kept", kept)

def align(a):
    rows = [json.loads(l) for l in open(a.inp)]
    out = open(a.out, "w"); stats = {"n": 0, "chunks": 0, "think_blocks": 0, "cover": 0.0}
    for r in rows:
        R = split_reasoning(r["reasoning"]); A = r["answer_paras"]
        if len(R) < 2: continue
        prefix_anch = []; acc = set()
        for p in R: acc = acc | anchors(p); prefix_anch.append(set(acc))
        fa = norm_ans(r["final_answer"])
        idx = []; prev = 0
        for j, para in enumerate(A):
            is_final = j == len(A) - 1
            an = anchors(para)
            def ok(i):
                if is_final: return (fa in {norm_ans(x) for x in prefix_anch[i]}) or (fa in norm_ans(" ".join(R[:i + 1])))
                if not an: return True
                return len(an & prefix_anch[i]) >= 0.6 * len(an)
            i = prev
            while i < len(R) and not ok(i): i += 1
            if i >= len(R): i = len(R) - 1
            idx.append(i); prev = i
        steps = []; cursor = 0
        for j, para in enumerate(A):
            end = idx[j] + 1
            if end > cursor:
                steps.append({"mode": "think", "text": "\n\n".join(R[cursor:end])}); cursor = end
            steps.append({"mode": "speak", "text": para})
        nb = sum(1 for s in steps if s["mode"] == "think")
        rec = {"id": r["id"], "source": "deepmath", "prompt": r["question"], "steps": steps, "chunks": A, "final_answer": r["final_answer"],
               "chunk_checks": [{"type": "final", "answer": r["final_answer"], "aliases": []}], "topic": r["topic"], "difficulty": r["difficulty"]}
        out.write(json.dumps(rec) + "\n"); stats["n"] += 1; stats["chunks"] += len(A); stats["think_blocks"] += nb; stats["cover"] += cursor / len(R)
    out.close()
    n = max(1, stats["n"]); print(f"aligned {stats['n']} | mean chunks {stats['chunks']/n:.2f} | mean think blocks {stats['think_blocks']/n:.2f} | reasoning coverage {stats['cover']/n:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["select", "gen", "split", "align", "selfgen"])
    ap.add_argument("--deepmath_dir", default="/scratch/liqiangx/extdata/deepmath"); ap.add_argument("--inp", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12000); ap.add_argument("--max_reason_words", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--max_new", type=int, default=450)
    ap.add_argument("--reason_words", type=int, default=1500); ap.add_argument("--shard_i", type=int, default=0); ap.add_argument("--shard_n", type=int, default=1)
    a = ap.parse_args(); {"select": select, "gen": gen, "split": split, "align": align, "selfgen": selfgen}[a.stage](a)
