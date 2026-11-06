"""LLM-judge evaluation for ConvFill outputs (no gold answers). Judge = Qwen3 (no thinking), batched, scoring logits of a single
digit token so it is cheap. Per turn, given the conversation context, the reference Reasoner knowledge chunks (gold thoughts) and the
model's PUBLIC phrases, the judge rates 1-5:
  faithfulness : public phrases do not assert facts absent from / contradicting the reference knowledge
  coherence    : phrases form a fluent, natural spoken reply to the user
  filler_ok    : filler (phrases carrying no knowledge) is grounded and not repetitive
Also rule-based: filler_rate (phrases with no content words from the reference thoughts), mean phrase length."""
import argparse, json, os, sys, re, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer

DIMS = {"faithfulness": "Do the assistant's spoken phrases avoid asserting facts that are absent from or contradict the KNOWLEDGE? (5 = fully grounded, 1 = invents or contradicts)",
        "coherence": "Do the phrases together form a fluent, natural, on-topic spoken reply to the user's last message? (5 = very natural, 1 = incoherent)",
        "filler_ok": "Are phrases that carry no knowledge (filler) brief, grounded in the conversation and non-repetitive? (5 = yes or no filler, 1 = long/repetitive/generic filler)"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="Qwen/Qwen3-4B"); ap.add_argument("--records", required=True, nargs="+"); ap.add_argument("--test", default="data_convfill/convfill_test.jsonl")
    ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left"); model = AutoModelForCausalLM.from_pretrained(a.judge, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(1, 6)]
    test = {json.loads(l)["id"]: json.loads(l) for l in open(a.test)}
    results = {}
    for rec_file in a.records:
        recs = [json.loads(l) for l in open(rec_file)]
        scores = {d: [] for d in DIMS}; filler_rates = []; plen = []
        for i in range(0, len(recs), a.bs):
            ch = recs[i:i + a.bs]
            for dim, q in DIMS.items():
                prompts = []
                for r in ch:
                    ex = test[r["id"]]; know = [s["text"] for s in ex["steps"] if s["mode"] == "think"]
                    ctx = ex["prompt"][-1500:]
                    u = (f"CONVERSATION (last user message at the end):\n{ctx}\n\nKNOWLEDGE available to the assistant:\n" + ("\n".join("- " + k for k in know) or "- (none)") +
                         "\n\nASSISTANT SPOKEN PHRASES:\n" + "\n".join("- " + c for c in r["chunks"]) + f"\n\nQuestion: {q}\nAnswer with a single digit 1-5.")
                    prompts.append(tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
                enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=3000).to("cuda")
                with torch.no_grad(): logits = model(**enc).logits[:, -1, digit_ids].float()
                probs = torch.softmax(logits, -1); expv = (probs * torch.arange(1, 6, device=probs.device)).sum(-1)
                scores[dim] += expv.tolist()
            for r in ch:
                ex = test[r["id"]]; kw = set(w.lower() for s in ex["steps"] if s["mode"] == "think" for w in re.findall(r"[a-zA-Z]{4,}", s["text"]))
                fill = [c for c in r["chunks"] if not (kw & set(w.lower() for w in re.findall(r"[a-zA-Z]{4,}", c)))]
                filler_rates.append(len(fill) / max(1, len(r["chunks"]))); plen += [len(c.split()) for c in r["chunks"]]
        results[rec_file] = {d: sum(v) / len(v) for d, v in scores.items()} | {"filler_rate": sum(filler_rates) / len(filler_rates), "mean_phrase_words": sum(plen) / max(1, len(plen)), "n": len(recs)}
        print(rec_file, json.dumps(results[rec_file], indent=None))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(results, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
