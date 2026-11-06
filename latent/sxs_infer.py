"""Interactive / single-question inference: prints the PRIVATE think stream and the PUBLIC answer stream separately,
with the router's decisions and the simulated playback timeline.
  python latent/sxs_infer.py --ckpt results/sxs/main_1.7B/ckpt --question "..." [--policy gate|nothink|budget-64|fixed-32|stitch-64]
"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from latent.sxs_eval import make_policy
from latent.simulator import AudioModel, simulate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--question", default=""); ap.add_argument("--jsonl", default="", help="take the prompt of record --idx from this file")
    ap.add_argument("--idx", type=int, default=0); ap.add_argument("--policy", default="gate"); ap.add_argument("--think_cost", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=600); ap.add_argument("--no_expert_ft", action="store_true")
    a = ap.parse_args()
    model = SxSModel(SxSConfig(model_name=a.model, expert_full_ft=not a.no_expert_ft), device="cuda"); model.load(a.ckpt); model.eval()
    tok = model.tok
    if a.jsonl:
        ex = [json.loads(l) for l in open(a.jsonl)][a.idx]; q = ex["prompt"]; print("GOLD chunks:", ex["chunks"])
    else:
        q = a.question
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]], device="cuda")
    policy, cap = make_policy(a.policy)
    events, think_blocks, chunks, trace = model.stream(ids, policy, max_new_tokens=a.max_new_tokens, max_think_per_block=cap or 1,
                                                       audio=AudioModel(), think_cost=a.think_cost or None)
    print("\n================ PRIVATE (think) ================")
    for i, t in enumerate(think_blocks): print(f"[think block {i+1}] {t.strip()}")
    print("\n================ PUBLIC (answer, sent to TTS) ================")
    for i, c in enumerate(chunks): print(f"[chunk {i+1}] {c}")
    print("\n================ interleaving ================")
    ti = ci = 0
    for e in events:
        if e["type"] == "speak" and e.get("text"):
            print(f"  SPEAK {e['text']!r}")
        elif e.get("kind") == "prefill":
            print(f"  prefill {e['dt']*1000:.0f} ms")
    sim = simulate(events, AudioModel())
    d = sim.as_dict(); d["think_tokens"] = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in think_blocks)
    print("\nrouter decisions (first 20):", [(t["step"], t["mode"], round(t["p_think"], 2)) for t in trace[:20]])
    print("playback simulation:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}))

if __name__ == "__main__":
    main()
