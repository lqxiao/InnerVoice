"""Export an InnerVoice checkpoint (SxSModel: base + LoRA + last-layer experts/router) as a merged HF-style safetensors directory
loadable by the vLLM plugin (latent/vllm_plugin). LoRA is folded into the base weights (incl. the LoRA inside both experts).

    python latent/sxs_export.py --model Qwen/Qwen3-4B --ckpt results/sxs/q32high_sxs_4B_noethink/ckpt --out /path/export_dir
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig

ARCH = "InnerVoiceQwen3ForCausalLM"


def merged_state_dict(model: SxSModel):
    """HF-named merged weights: LoRA folded in (fp32 accumulate), e_think and LoRA factors dropped, router kept fp32."""
    from peft.tuners.lora import LoraLayer
    lm = model.lm
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    sd = {}
    with torch.no_grad():
        for name, mod in base.named_modules():
            if isinstance(mod, LoraLayer):
                W = mod.base_layer.weight.data.float()
                for ad in mod.active_adapters:
                    if ad not in mod.lora_A: continue
                    W = W + (mod.lora_B[ad].weight.float() @ mod.lora_A[ad].weight.float()) * mod.scaling[ad]
                sd[f"{name}.weight"] = W.to(mod.base_layer.weight.dtype).contiguous().cpu()
                if getattr(mod.base_layer, "bias", None) is not None:
                    sd[f"{name}.bias"] = mod.base_layer.bias.data.contiguous().cpu()
        tie = getattr(base.config, "tie_word_embeddings", False)
        for name, p in base.named_parameters():
            if "lora_" in name or ".base_layer." in name or name in sd: continue
            if tie and name == "lm_head.weight": continue
            sd[name] = p.data.contiguous().cpu()
    return sd


def write_weights(model: SxSModel, out_dir: str):
    """Atomically (re)write <out_dir>/model.safetensors."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, ".model.safetensors.tmp"); final = os.path.join(out_dir, "model.safetensors")
    save_file(merged_state_dict(model), tmp, metadata={"format": "pt"}); os.replace(tmp, final)
    return final


def write_config(model: SxSModel, out_dir: str):
    from transformers import AutoConfig, GenerationConfig
    os.makedirs(out_dir, exist_ok=True)
    cfg = AutoConfig.from_pretrained(model.cfg.model_name)
    cfg.architectures = [ARCH]; cfg.router_hidden = model.cfg.router_hidden
    cfg.save_pretrained(out_dir); model.tok.save_pretrained(out_dir)
    try: GenerationConfig.from_pretrained(model.cfg.model_name).save_pretrained(out_dir)
    except Exception as e: print("no generation_config:", e)


def export(model: SxSModel, out_dir: str):
    assert not model.cfg.use_e_think, "vLLM export requires a checkpoint trained with --no_e_think"
    write_config(model, out_dir); return write_weights(model, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no_expert_ft", action="store_true"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    m = SxSModel(SxSConfig(model_name=a.model, expert_full_ft=not a.no_expert_ft, use_e_think=False), device=a.device); m.load(a.ckpt)
    t0 = time.time(); f = export(m, a.out)
    print(f"wrote {f} ({os.path.getsize(f) / 1e9:.2f} GB) in {time.time() - t0:.1f}s")
