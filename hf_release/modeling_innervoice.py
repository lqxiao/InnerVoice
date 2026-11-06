"""InnerVoice: Qwen3 whose LAST decoder-layer FFN is a two-expert MoE (inner "think" expert / outer "answer" expert) chosen per token
by a disclosure router. Think tokens are readable text but are meant for the model itself (never spoken / sent to TTS); answer tokens
are the public stream. The router decision at position t (from the normed hidden state entering the last-layer MLP) selects the expert
that produces token t+1, so decoding is a standard autoregressive loop plus one bit per token.

    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, torch_dtype="bfloat16").cuda()
    tok = AutoTokenizer.from_pretrained(REPO)
    out = m.generate_streams(tok, "Is 2^10 larger than 1000?", max_new_tokens=512)
    print(out["think"]); print(out["speak"])

Plain `generate()` also works but mixes both streams in one text; use `generate_streams` (or read `m.last_route`) to separate them.
"""
import copy
import torch, torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3Model, Qwen3PreTrainedModel

N_BUF = 4  # router side features (audio-buffer state); zeros for text use. Kept for weight compatibility.


class TwoExpertMLP(nn.Module):
    def __init__(self, config: Qwen3Config, router_hidden: int = 256):
        super().__init__()
        self.answer = Qwen3MLP(config)
        self.think = Qwen3MLP(config)
        self.router = nn.Sequential(nn.Linear(config.hidden_size + N_BUF, router_hidden), nn.GELU(), nn.Linear(router_hidden, 1))
        self.route = None            # optional (B,L) bool override (teacher forcing)
        self.router_logits = None    # (B,L) logits of the last forward
        self.last_route = None       # (B,L) bool: True = next token is a think token

    def forward(self, x):
        xf = x.float()
        logits = self.router(torch.cat([xf, xf.new_zeros(*xf.shape[:-1], N_BUF)], -1).to(self.router[0].weight.dtype)).squeeze(-1).float()
        self.router_logits = logits
        route = self.route if self.route is not None else logits > 0
        self.last_route = route
        return torch.where(route[..., None], self.think(x), self.answer(x))


class InnerVoiceQwen3ForCausalLM(Qwen3ForCausalLM):
    _keep_in_fp32_modules = ["router"]

    def __init__(self, config):
        # same body as Qwen3ForCausalLM.__init__, but the last-layer MLP is swapped BEFORE post_init (which validates _keep_in_fp32_modules)
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.layers[-1].mlp = TwoExpertMLP(config, getattr(config, "router_hidden", 256))
        self.post_init()

    @property
    def moe(self):
        return self.model.layers[-1].mlp

    @torch.no_grad()
    def generate_streams(self, tokenizer, prompt, max_new_tokens: int = 1024, greedy: bool = True, temperature: float = 1.0, messages=None):
        """Batch-1 decode that returns the two streams separately. `prompt` is a user message (or pass `messages`)."""
        msgs = messages or [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        eos = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id}
        out = self(input_ids=ids, use_cache=True); past = out.past_key_values
        toks, types = [], []
        for _ in range(max_new_tokens):
            is_think = bool(self.moe.last_route[0, -1])
            logits = out.logits[:, -1].float()
            nxt = logits.argmax(-1) if greedy else torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(1)
            t = int(nxt)
            if t in eos: break
            toks.append(t); types.append(int(is_think))
            out = self(input_ids=nxt[:, None], past_key_values=past, use_cache=True); past = out.past_key_values
        return {"tokens": toks, "types": types,
                "think": tokenizer.decode([t for t, y in zip(toks, types) if y == 1], skip_special_tokens=True),
                "speak": tokenizer.decode([t for t, y in zip(toks, types) if y == 0], skip_special_tokens=True),
                "interleaved": [(("think" if y else "speak"), tokenizer.decode([t])) for t, y in zip(toks, types)]}
