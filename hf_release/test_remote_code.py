"""Check the transformers remote-code class against the training-time SxSModel on the same exported dir (greedy, K&K prompts)."""
import json, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from latent.sxs_model import SxSModel, SxSConfig
exp, base, ckpt, data = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained(exp)
hf = AutoModelForCausalLM.from_pretrained(exp, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
print(type(hf).__name__, "router dtype:", hf.moe.router[0].weight.dtype)
ref = SxSModel(SxSConfig(model_name=base, use_e_think=False), device="cuda"); ref.load(ckpt); ref.eval()
rows = [json.loads(l) for l in open(data)][:4]; ok = 0
for r in rows:
    o = hf.generate_streams(tok, r["prompt"], max_new_tokens=300)
    ids = tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    t, y = ref.stream_batch(ids, torch.ones_like(ids), max_new_tokens=300, greedy=True)
    t0 = [x for x in t[0] if x not in (ref.eos_id, tok.eos_token_id)]; y0 = y[0][:len(t0)]
    same = (o["tokens"] == t0 and o["types"] == y0); ok += same
    print(f"len {len(o['tokens'])} vs {len(t0)} identical_tokens={o['tokens'] == t0} identical_types={o['types'] == y0}")
print("THINK:", o["think"][:200].replace("\n", " | ")); print("SPEAK:", o["speak"][:200].replace("\n", " | ")); print(f"{ok}/{len(rows)} identical")
