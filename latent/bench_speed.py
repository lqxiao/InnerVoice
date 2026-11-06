"""Inference-speed benchmark: InnerVoice (router + two-expert last layer) vs vanilla Qwen3 (same LoRA-free base) at batch 1 and batch 16."""
import time, json, sys, os, torch, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def bench_vanilla(model, tok, prompts, new_tokens):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(): out = model.generate(**enc, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); return time.perf_counter() - t0

def bench_innervoice(model, tok, prompts, new_tokens):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    # router-driven batched decode, forced to run exactly new_tokens steps (ignore EOS) for a fair count
    B, L = enc["input_ids"].shape; mask = enc["attention_mask"].clone(); pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    model.moe.route = None; model.moe.buf = None
    with torch.no_grad():
        out = model.lm(inputs_embeds=model.inputs_embeds(enc["input_ids"], torch.zeros_like(enc["input_ids"])), attention_mask=mask, position_ids=pos, use_cache=True, return_dict=True)
        cache = out.past_key_values; cur = mask.long().sum(-1); rl = model.moe.router_logits[:, -1]; ll = out.logits[:, -1]
        for _ in range(new_tokens - 1):
            mt = rl > 0; nxt = ll.argmax(-1); mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device="cuda")], 1)
            out = model.lm(inputs_embeds=model.inputs_embeds(nxt[:, None], mt.long()[:, None]), attention_mask=mask, past_key_values=cache, position_ids=cur[:, None], use_cache=True, return_dict=True)
            cache = out.past_key_values; cur = cur + 1; rl = model.moe.router_logits[:, -1]; ll = out.logits[:, -1]
    torch.cuda.synchronize(); return time.perf_counter() - t0

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--ckpt", default="results/sxs/main_1.7B/ckpt"); ap.add_argument("--new", type=int, default=200)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
q = "Every lorpus is a harpus. Every harpus is a kerpus. Every felpus is a brimpus. Every brimpus is a lempus. Every lempus is a kerpus. Is every felpus a kerpus? Answer Yes or No."
prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
res = {}
van = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
for B in (1, 16):
    bench_vanilla(van, tok, [prompt] * B, 8)
    t = bench_vanilla(van, tok, [prompt] * B, a.new); res[f"vanilla_B{B}"] = {"s": t, "ms_per_step": 1000 * t / a.new, "tok_per_s": B * a.new / t}
mem_v = torch.cuda.max_memory_allocated() / 2**30; del van; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
iv = SxSModel(SxSConfig(model_name=a.model), device="cuda"); iv.load(a.ckpt); iv.eval()
n_extra = sum(p.numel() for p in iv.moe.think.parameters()) + sum(p.numel() for p in iv.moe.router.parameters()) + iv.e_think.numel()
n_total = sum(p.numel() for p in iv.parameters())
for B in (1, 16):
    bench_innervoice(iv, tok, [prompt] * B, 8)
    t = bench_innervoice(iv, tok, [prompt] * B, a.new); res[f"innervoice_B{B}"] = {"s": t, "ms_per_step": 1000 * t / a.new, "tok_per_s": B * a.new / t}
mem_i = torch.cuda.max_memory_allocated() / 2**30
iv.merge_lora()
for B in (1, 16):
    bench_innervoice(iv, tok, [prompt] * B, 8)
    t = bench_innervoice(iv, tok, [prompt] * B, a.new); res[f"innervoice_merged_B{B}"] = {"s": t, "ms_per_step": 1000 * t / a.new, "tok_per_s": B * a.new / t}
res["params"] = {"total": n_total, "extra_think_expert_router_ethink": n_extra, "extra_frac": n_extra / n_total}
res["peak_mem_GB"] = {"vanilla": mem_v, "innervoice": mem_i}
print(json.dumps(res, indent=1))
