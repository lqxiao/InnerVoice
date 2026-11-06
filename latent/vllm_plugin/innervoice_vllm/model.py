"""InnerVoice for vLLM: Qwen3ForCausalLM whose LAST decoder-layer MLP is a two-expert MoE (think / answer) selected per token by a
disclosure router. The decision for token t+1 is a pure function of the normed hidden state h_t entering the last-layer MLP, so no
sequence state is needed and the standard vLLM scheduler / KV cache / CUDA graphs work unchanged. The input-side e_think embedding
is NOT supported here (checkpoints must be trained with --no_e_think; ablation showed it is unnecessary).

Weight names (HF-style, produced by latent/sxs_export.py):
    model.layers.<last>.mlp.{think,answer}.{gate_proj,up_proj,down_proj}.weight   (bf16)
    model.layers.<last>.mlp.router.{0,2}.{weight,bias}                             (fp32; router input = [h_t, 4 zero buffer feats])
"""
import torch, torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3MLP

N_BUF = 4


class TwoExpertMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, quant_config, prefix, router_hidden):
        super().__init__()
        self.answer = Qwen3MLP(hidden_size, intermediate_size, hidden_act, quant_config=quant_config, prefix=f"{prefix}.answer")
        self.think = Qwen3MLP(hidden_size, intermediate_size, hidden_act, quant_config=quant_config, prefix=f"{prefix}.think")
        self.router = nn.Sequential(nn.Linear(hidden_size + N_BUF, router_hidden), nn.GELU(), nn.Linear(router_hidden, 1)).float()

    def forward(self, x):
        xf = x.float()
        logits = self.router(torch.cat([xf, xf.new_zeros(*xf.shape[:-1], N_BUF)], -1)).squeeze(-1)
        route = logits > 0                                   # True -> think expert produces the next token
        return torch.where(route[..., None], self.think(x), self.answer(x))


class InnerVoiceQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        cfg = vllm_config.model_config.hf_config
        layers = self.model.layers
        n = len(layers) - 1
        layers[n].mlp = TwoExpertMLP(cfg.hidden_size, cfg.intermediate_size, cfg.hidden_act, vllm_config.quant_config,
                                     prefix=f"{prefix}model.layers.{n}.mlp", router_hidden=getattr(cfg, "router_hidden", 256))
