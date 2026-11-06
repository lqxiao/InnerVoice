"""Teacher traces from Bedrock (default: Claude Fable 5.1) for DeepMath problems, in the SxS paper's Reasoning + Answer format.
The teacher is asked for two sections: <reasoning> (exploratory, first-person working, may backtrack) and <answer> (clean
reader-facing solution in 3-7 short paragraphs ending with Final answer: \\boxed{...}). Kept only if the boxed answer matches.
If Bedrock also returns a reasoningContent block (model's own thinking), it is prepended to the reasoning section.
Output schema = selfgen: {id, question, final_answer, difficulty, topic, reasoning, answer_paras}. Resumable (skips done ids)."""
import argparse, json, os, sys, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.config import Config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_math_data import last_boxed, norm_ans

SYS = ("You are an expert mathematician producing training data. Respond with EXACTLY two sections.\n"
       "<reasoning>\nYour private working: explore, compute, check, and if needed backtrack, in first person, several paragraphs separated by blank lines. "
       "Do not state the final answer as settled until you have verified it.\n</reasoning>\n"
       "<answer>\nThe reader-facing solution: 3 to 7 short paragraphs separated by blank lines, each stating one step or intermediate result with its value, "
       "no exploration, no 'let me'. The last paragraph must be exactly: Final answer: \\boxed{<answer>}\n</answer>")

_client = None; _lock = threading.Lock()
def client(region):
    global _client
    with _lock:
        if _client is None:
            _client = boto3.client("bedrock-runtime", region_name=region, config=Config(read_timeout=600, connect_timeout=20, retries={"max_attempts": 6, "mode": "adaptive"}))
    return _client

def call(r, a):
    c = client(a.region)
    for attempt in range(6):
        try:
            kw = dict(modelId=a.model, system=[{"text": SYS + ("\n" + a.sys_extra if a.sys_extra else "")}], messages=[{"role": "user", "content": [{"text": f"Problem:\n{r['question']}"}]}], inferenceConfig=({"maxTokens": a.max_tokens} if a.temperature < 0 else {"maxTokens": a.max_tokens, "temperature": a.temperature}))
            if a.extra_json: kw["additionalModelRequestFields"] = json.loads(a.extra_json)
            elif a.effort: kw["additionalModelRequestFields"] = {"thinking": {"type": "adaptive"}, "output_config": {"effort": a.effort}}
            resp = c.converse(**kw)
            parts = resp["output"]["message"]["content"]
            text = "".join(p.get("text", "") for p in parts)
            native = "".join(p["reasoningContent"].get("reasoningText", {}).get("text", "") for p in parts if "reasoningContent" in p)
            return text, native, resp.get("usage", {})
        except Exception as e:
            msg = str(e)
            if "Throttling" in msg or "TooMany" in msg or "ServiceUnavailable" in msg or "ModelNotReady" in msg: time.sleep(3 * (attempt + 1) + 2 * attempt ** 2)
            else: time.sleep(2 * (attempt + 1))
            last = msg
    return None, None, {"error": last}

def parse(text, native=""):
    """<reasoning> tag if present, else the model's native reasoningContent (Qwen3 on Bedrock thinks natively and only emits <answer>)."""
    m1 = re.search(r"<reasoning>(.*?)</reasoning>", text, re.S); m2 = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    reason = m1.group(1).strip() if m1 else (native or "").strip()
    if not reason or not m2: return None, None
    ans = m2.group(1).strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", ans) if p.strip()]
    merged = []
    for p in paras:
        if merged and (p.startswith("\\[") or p.startswith("$$") or len(p.split()) < 6): merged[-1] += "\n\n" + p
        else: merged.append(p)
    return reason, merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data_math/dm_self_src.jsonl"); ap.add_argument("--out", default="data_math/self/gen_fable.jsonl")
    ap.add_argument("--model", default="us.anthropic.claude-fable-5-1"); ap.add_argument("--region", default="us-west-2"); ap.add_argument("--effort", default="high")
    ap.add_argument("--max_tokens", type=int, default=9000); ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--extra_json", default="", help="additionalModelRequestFields as JSON (overrides --effort)"); ap.add_argument("--sys_extra", default="", help="appended to the system prompt"); ap.add_argument("--min_reason_words", type=int, default=60)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp)]
    if a.limit: rows = rows[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out): done.add(json.loads(l)["id"])
    todo = [r for r in rows if r["id"] not in done]; print(f"todo {len(todo)} (done {len(done)})", flush=True)
    kept = n = 0; usage_tot = {"in": 0, "out": 0}; t0 = time.time()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out = open(a.out, "a"); lock = threading.Lock(); bad = open(a.out + ".rejected.jsonl", "a")
    def work(r):
        text, native, usage = call(r, a)
        if text is None: return r, None, usage
        reason, paras = parse(text, native)
        return r, (reason, paras, native, text), usage
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for f in as_completed(futs):
            r, res, usage = f.result(); n += 1
            usage_tot["in"] += usage.get("inputTokens", 0); usage_tot["out"] += usage.get("outputTokens", 0)
            ok = False
            if res is not None:
                reason, paras, native, text = res
                if reason and paras and len(paras) >= 2 and len(reason.split()) >= a.min_reason_words and norm_ans(last_boxed(" ".join(paras))) == norm_ans(r["final_answer"]):
                    rec = {k: r[k] for k in ("id", "question", "final_answer", "difficulty", "topic")}
                    rec["reasoning"] = reason if (not native or native.strip() in reason) else (native.strip() + "\n\n" + reason); rec["answer_paras"] = paras; rec["teacher"] = a.model
                    with lock: out.write(json.dumps(rec) + "\n"); out.flush()
                    ok = True
                else:
                    with lock: bad.write(json.dumps({"id": r["id"], "final_answer": r["final_answer"], "text": (text or "")[-800:]}) + "\n")
            kept += ok
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0; print(f"{n}/{len(todo)} kept={kept} ({kept/max(1,n):.2f}) tokens in/out={usage_tot['in']}/{usage_tot['out']} {el/60:.1f} min", flush=True)
    print("DONE kept", kept, "of", n)

if __name__ == "__main__":
    main()
