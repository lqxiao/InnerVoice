"""Readable think/speak model: Qwen3 + LoRA, last-layer two-expert MoE (think / answer) selected by a router,
think-type input embedding e_think. Both experts are initialised from the pretrained last-layer MLP.

Position t: router(h_t, buffer_t) -> p(next token is THINK). The selected expert produces the logits for token t+1.
Think tokens are readable text but are never sent to TTS; their input embedding is embed(tok) + e_think.
"""
import copy, time
from dataclasses import dataclass
from typing import Optional, List, Callable
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

N_BUF = 4  # remaining audio s /10, think tokens so far in this block /64, chunks emitted /8, per-token think cost s


def route_hash_u(ids, pos):
    """Deterministic pseudo-random u in [0,1) from (token id, position) — integer hashing only, so the inference engine and the
    trainer recover identical values on any hardware. Used for reproducible *sampled* routing: route = sigmoid(logit) > u."""
    a = (pos.long() * 1000003 + ids.long() * 7919 + 12345) % 4294967296
    h = (a * 40503) % 4294967296
    h = ((h ^ (h >> 15)) * 40503) % 4294967296
    return h.float() / 4294967296.0


class TwoExpertMLP(nn.Module):
    def __init__(self, mlp: nn.Module, d: int, router_hidden: int = 256, full_ft: bool = True, shared: bool = False):
        super().__init__()
        self.answer = mlp
        self.think = mlp if shared else copy.deepcopy(mlp)   # shared=True: ablation 'single FFN + disclosure gate' (gate only labels visibility)
        if full_ft:
            for p in list(self.answer.parameters()) + list(self.think.parameters()):
                p.requires_grad_(True)
        self.router = nn.Sequential(nn.Linear(d + N_BUF, router_hidden), nn.GELU(), nn.Linear(router_hidden, 1))
        nn.init.normal_(self.router[-1].weight, std=0.01); nn.init.constant_(self.router[-1].bias, -1.0)   # start biased to answer
        self.route: Optional[torch.Tensor] = None      # (B,L) bool teacher-forced routing (True = think expert)
        self.buf: Optional[torch.Tensor] = None        # (B,L,N_BUF)
        self.router_logits: Optional[torch.Tensor] = None
        self.sample_route: bool = False                # RL exploration: sample the mode from sigmoid(logit) instead of argmax
        self.sample_hash: bool = False                 # reproducible sampling: route = sigmoid(logit) > route_hash_u(token, position)
        self.aux = None                                # (ids, pos), each (B,L), set by the caller before the forward when sample_hash
        self.last_route: Optional[torch.Tensor] = None # (B,L) bool route actually used in the last forward

    def forward(self, x):
        B, L, _ = x.shape
        buf = self.buf if self.buf is not None else x.new_zeros(B, L, N_BUF)
        logits = self.router(torch.cat([x.float(), buf.float().to(x.device)], -1)).squeeze(-1)
        self.router_logits = logits
        if self.route is not None:
            route = self.route
        elif self.sample_route:
            route = torch.bernoulli(torch.sigmoid(logits.detach())).bool()
        elif self.sample_hash:
            ids, pos = self.aux
            route = torch.sigmoid(logits.detach()) > route_hash_u(ids, pos).to(logits.device)
        else:
            route = logits > 0
        self.last_route = route
        return torch.where(route[..., None], self.think(x), self.answer(x))


@dataclass
class SxSConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    lora_r: int = 16
    lora_alpha: int = 32
    expert_full_ft: bool = True
    dtype: str = "bfloat16"
    router_hidden: int = 256
    use_e_think: bool = True
    grad_ckpt: bool = False
    route_sample: bool = False      # hash-sampled routing at decode time (engine + trainer label pass); False = argmax
    shared_ffn: bool = False        # ablation: one shared FFN (no think/answer experts), gate only decides visibility


class SxSModel(nn.Module):
    def __init__(self, cfg: SxSConfig, device="cuda", base_model=None, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        self.tok = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        base = base_model or AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype, attn_implementation="sdpa")
        base.config.use_cache = True
        d = base.config.hidden_size; self.d = d
        if cfg.lora_r > 0:
            lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.lm = get_peft_model(base, lcfg)
            layers = self.lm.base_model.model.model.layers
        else:
            for p in base.parameters(): p.requires_grad_(False)
            self.lm = base; layers = base.model.layers
        if cfg.grad_ckpt:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(base, "enable_input_require_grads"): base.enable_input_require_grads()
        last = layers[-1]
        last.mlp = TwoExpertMLP(last.mlp, d, cfg.router_hidden, full_ft=cfg.expert_full_ft, shared=cfg.shared_ffn)
        self.moe = last.mlp; self.moe.sample_hash = cfg.route_sample
        self.e_think = nn.Parameter(torch.randn(d) * 0.02)
        self.to(device); self.device = device
        self.eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.newline_id = self.tok.convert_tokens_to_ids("Ċ")

    @property
    def embed(self):
        return self.lm.get_input_embeddings()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self):
        lora = [p for n, p in self.named_parameters() if p.requires_grad and "lora_" in n]
        rest = [p for n, p in self.named_parameters() if p.requires_grad and "lora_" not in n]
        return lora, rest

    def inputs_embeds(self, ids, is_think):
        e = self.embed(ids)
        if not self.cfg.use_e_think:
            return e
        return e + is_think.to(e.dtype)[..., None] * self.e_think.to(e.dtype)

    # ------------------------------------------------------------------ training (teacher forced, right padded full sequence)
    def forward_train(self, ids, mask, is_think, loss_mask, route_label, route_mask, buf=None, think_weight: float = 1.0, switch_weight: float = 1.0, return_per_seq: bool = False):
        """ids/mask/is_think/loss_mask/route_label/route_mask: (B,L). route_label[t] = 1 if token t+1 is a think token.
        loss_mask: 1 on response tokens (targets). Returns dict(loss_ce, loss_route, ...)."""
        B, L = ids.shape
        pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
        self.moe.route = route_label.bool(); self.moe.buf = buf
        out = self.lm(inputs_embeds=self.inputs_embeds(ids, is_think), attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
        # NOTE: keep self.moe.route = route_label until the caller's backward has run. With gradient checkpointing the last layer is
        # re-executed during backward and must see the same teacher-forced route (clearing it here made the recompute fall back to
        # argmax / hash routing: wrong expert on label-disagreeing positions, and a crash when aux is unset). Decode paths reset it.
        logits = out.logits[:, :-1].float(); tgt = ids[:, 1:]; lm_ = loss_mask[:, 1:].float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(B, L - 1)
        w = lm_ * torch.where(is_think[:, 1:].bool(), torch.full_like(lm_, think_weight), torch.ones_like(lm_))
        loss_ce = (ce * w).sum() / w.sum().clamp(min=1)
        ce_think = (ce * lm_ * is_think[:, 1:]).sum() / (lm_ * is_think[:, 1:]).sum().clamp(min=1)
        ce_ans = (ce * lm_ * (1 - is_think[:, 1:])).sum() / (lm_ * (1 - is_think[:, 1:])).sum().clamp(min=1)
        rl = self.moe.router_logits
        bce = F.binary_cross_entropy_with_logits(rl, route_label.float(), reduction="none")
        is_switch = (route_label.bool() != is_think.bool()).float()          # next-token type differs from current token type
        rw = route_mask.float() * (1.0 + (switch_weight - 1.0) * is_switch)
        loss_route = (bce * rw).sum() / rw.sum().clamp(min=1)
        with torch.no_grad():
            pred = (rl > 0).float(); m = route_mask.bool()
            acc = (pred[m] == route_label.float()[m]).float().mean()
            # switch-point accuracy: positions where label differs from the current token type
            sw = m & (route_label.bool() != is_think.bool())
            acc_switch = (pred[sw] == route_label.float()[sw]).float().mean() if sw.any() else torch.tensor(float("nan"))
        res = {"loss_ce": loss_ce, "loss_route": loss_route, "ce_think": ce_think, "ce_answer": ce_ans, "route_acc": acc, "switch_acc": acc_switch,
               "think_frac_label": route_label.float()[m].mean()}
        if return_per_seq:
            res["logp_tok_seq"] = -(ce * lm_).sum(-1)                        # (B,) sum log p(token) over response tokens
            res["logp_route_seq"] = -(bce * route_mask.float()).sum(-1)      # (B,) sum log pi(disclosure decision) over response positions
            res["n_resp"] = lm_.sum(-1)
            res["logp_tok"] = -ce                                            # (B,L-1) per-token log p(target); mask with loss_mask[:,1:]
            res["logp_route"] = -bce                                         # (B,L) per-position log pi(taken decision); mask with route_mask
        return res

    # ------------------------------------------------------------------ streaming generation (batch 1)
    @torch.no_grad()
    def stream(self, prompt_ids, policy: Callable, max_new_tokens: int = 400, max_think_per_block: int = 256, audio=None,
               think_cost: Optional[float] = None, greedy: bool = True, record: bool = True, use_buf: bool = False):
        """policy(state) -> 'think' | 'answer' | None (None = follow router). state has router p_think, think tokens in block,
        chunks emitted, buffer remaining seconds. Returns events (for the simulator), think_blocks, chunks, trace."""
        dev = prompt_ids.device
        sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
        ids = prompt_ids; L = ids.shape[1]
        is_think = torch.zeros_like(ids)
        mask = torch.ones_like(ids)
        self.moe.route = None
        events, chunks, think_blocks, trace = [], [], [], []
        t_compute, play_end = 0.0, 0.0
        n_think_block, n_chunks = 0, 0
        cur_chunk, cur_think = [], []
        sync(); t0 = time.perf_counter()
        def bfeat(buf_rem, n_tb, n_ch):   # router buffer features; zeros unless the model was trained with them
            if not use_buf:
                return torch.zeros(1, 1, N_BUF, device=dev)
            return torch.tensor([[[buf_rem / 10, n_tb / 64, n_ch / 8, think_cost or 0.0]]], device=dev)
        self.moe.buf = bfeat(0.0, 0, 0).expand(1, L, N_BUF).contiguous()
        self.moe.route = torch.zeros(1, L, dtype=torch.bool, device=dev)   # prompt positions: irrelevant (not predicted) except last -> set below
        out = self.lm(inputs_embeds=self.inputs_embeds(ids, is_think), attention_mask=mask, use_cache=True, return_dict=True)
        cache = out.past_key_values
        sync(); dt = time.perf_counter() - t0; t_compute += dt
        events.append({"type": "think", "dt": dt, "kind": "prefill"})
        # the prompt pass used route=answer at the last position; re-decide below with the router and recompute that position if needed
        p_think = torch.sigmoid(self.moe.router_logits[0, -1]).item()
        prev_mode = "answer"
        last_logits = out.logits[:, -1]
        cur_len = L
        for step in range(max_new_tokens):
            buf_rem = max(0.0, play_end - t_compute)
            state = {"p_think": p_think, "n_think_block": n_think_block, "n_chunks": n_chunks, "buf": buf_rem, "prev_mode": prev_mode}
            mode = policy(state)
            if mode is None:
                mode = "think" if p_think >= 0.5 else "answer"
            if n_think_block >= max_think_per_block:
                mode = "answer"
            if record:
                trace.append({"step": step, "p_think": p_think, "mode": mode, "buf": buf_rem, "n_chunks": n_chunks})
            _trace_entry = trace[-1] if record else None
            if (mode == "think") != (prev_mode == "think"):
                # the last position's logits were computed with the previous expert; recompute that position with the chosen expert
                # (cheap: one position through the last layer would be ideal; we re-run the last token with the cache truncated by 1)
                cache.crop(cur_len - 1)
                last_tok = ids[:, -1:]; last_is_think = is_think[:, -1:]
                self.moe.route = torch.tensor([[mode == "think"]], device=dev)
                self.moe.buf = bfeat(buf_rem, n_think_block, n_chunks)
                out = self.lm(inputs_embeds=self.inputs_embeds(last_tok, last_is_think), attention_mask=mask, past_key_values=cache,
                              position_ids=torch.tensor([[cur_len - 1]], device=dev), use_cache=True, return_dict=True)
                cache = out.past_key_values; last_logits = out.logits[:, -1]
            nxt = last_logits.argmax(-1) if greedy else torch.multinomial(torch.softmax(last_logits.float(), -1), 1).squeeze(1)
            tid = int(nxt.item())
            if _trace_entry is not None: _trace_entry["tid"] = tid
            sync(); ts = time.perf_counter()
            # bookkeeping of the emitted token
            if mode == "think":
                cur_think.append(tid); n_think_block += 1
            else:
                if tid not in (self.eos_id, self.tok.eos_token_id):
                    cur_chunk.append(tid)
            if tid in (self.eos_id, self.tok.eos_token_id):
                break
            # feed the token back
            mask = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype, device=dev)], 1)
            ids = torch.cat([ids, nxt[:, None]], 1); is_think = torch.cat([is_think, torch.tensor([[int(mode == "think")]], device=dev)], 1)
            self.moe.route = torch.tensor([[mode == "think"]], device=dev)   # provisional: same mode continues; corrected above if it switches
            self.moe.buf = bfeat(buf_rem, n_think_block, n_chunks)
            out = self.lm(inputs_embeds=self.inputs_embeds(nxt[:, None], is_think[:, -1:]), attention_mask=mask, past_key_values=cache,
                          position_ids=torch.tensor([[cur_len]], device=dev), use_cache=True, return_dict=True)
            cache = out.past_key_values; last_logits = out.logits[:, -1]; cur_len += 1
            p_think = torch.sigmoid(self.moe.router_logits[0, -1]).item()
            sync(); dt = time.perf_counter() - ts
            dt_eff = think_cost if (mode == "think" and think_cost) else dt
            t_compute += dt_eff
            if mode == "think":
                events.append({"type": "think", "dt": dt_eff, "dt_measured": dt})
                if prev_mode == "answer" and cur_chunk:   # answer chunk interrupted by thinking without newline -> close it
                    text = self.tok.decode(cur_chunk, skip_special_tokens=True).strip(); chunks.append(text); cur_chunk = []
                    dur = audio.duration(text) if audio else 0.0; start = max(t_compute, play_end); play_end = start + dur; n_chunks += 1
                    events.append({"type": "speak", "dt": 0.0, "text": text})
            else:
                if prev_mode == "think" and cur_think:
                    think_blocks.append(self.tok.decode(cur_think, skip_special_tokens=True)); cur_think = []; n_think_block = 0
                if "\n" in self.tok.decode([tid]):
                    text = self.tok.decode(cur_chunk, skip_special_tokens=True).strip(); cur_chunk = []
                    if text:
                        dur = audio.duration(text) if audio else 0.0; start = max(t_compute, play_end); play_end = start + dur; n_chunks += 1
                        events.append({"type": "speak", "dt": dt, "text": text}); chunks.append(text)
                else:
                    events.append({"type": "think", "dt": dt, "kind": "decode_answer"})   # compute time of answer tokens before the chunk is complete
            prev_mode = mode
        if cur_think:
            think_blocks.append(self.tok.decode(cur_think, skip_special_tokens=True))
        if cur_chunk:
            text = self.tok.decode(cur_chunk, skip_special_tokens=True).strip()
            if text:
                chunks.append(text); events.append({"type": "speak", "dt": 0.0, "text": text})
        self.moe.route = None; self.moe.buf = None
        return events, think_blocks, chunks, trace

    @torch.no_grad()
    def stream_batch(self, prompt_ids, prompt_mask, max_new_tokens: int = 4096, greedy: bool = True, temperature: float = 1.0, sample_router: bool = False):
        """Batched streaming decode where the ROUTER decides the mode of every next token (pure 'gate' policy; no decode-time
        overrides, so no recompute is ever needed). Left-padded prompts. Returns per-sequence token ids and types (1 = think)."""
        B, L = prompt_ids.shape; dev = prompt_ids.device
        mask = prompt_mask.clone(); pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
        self.moe.route = None; self.moe.buf = None; self.moe.sample_route = sample_router; self.moe.aux = (prompt_ids, pos)
        out = self.lm(inputs_embeds=self.inputs_embeds(prompt_ids, torch.zeros_like(prompt_ids)), attention_mask=mask, position_ids=pos, use_cache=True, return_dict=True)
        cache = out.past_key_values; cur = mask.long().sum(-1)
        router_logit = self.moe.router_logits[:, -1]; last_logits = out.logits[:, -1]
        toks = [[] for _ in range(B)]; types = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for step in range(max_new_tokens):
            mode_think = self.moe.last_route[:, -1]              # mode chosen inside the last forward (argmax, or sampled when sample_router)
            nxt = last_logits.argmax(-1) if greedy else torch.multinomial(torch.softmax(last_logits.float() / temperature, -1), 1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, self.tok.pad_token_id), nxt)
            nl, ml, dl = nxt.tolist(), mode_think.tolist(), done.tolist()
            for b in range(B):
                if not dl[b]: toks[b].append(nl[b]); types[b].append(int(ml[b]))
            done = done | (nxt == self.eos_id) | (nxt == self.tok.eos_token_id)
            if done.all(): break
            mask = torch.cat([mask, (~done).long()[:, None]], 1); self.moe.aux = (nxt[:, None], cur[:, None])
            out = self.lm(inputs_embeds=self.inputs_embeds(nxt[:, None], mode_think.long()[:, None]), attention_mask=mask, past_key_values=cache,
                          position_ids=cur[:, None], use_cache=True, return_dict=True)
            cache = out.past_key_values; cur = cur + 1
            router_logit = self.moe.router_logits[:, -1]; last_logits = out.logits[:, -1]
        self.moe.sample_route = False; self.moe.aux = None
        return toks, types

    @torch.no_grad()
    def infer_route(self, ids, mask):
        """Recover the router's own (argmax) disclosure decisions for a full right-padded sequence: runs the transformer body only
        (no lm_head logits) and returns (B,L) bool with route[b,t] = 'token t+1 is a think token'. Used to label engine rollouts;
        requires use_e_think=False (the input embedding must not depend on the labels)."""
        assert not self.cfg.use_e_think, "infer_route needs a --no_e_think model"
        pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
        self.moe.route = None; self.moe.buf = None; self.moe.sample_route = False; self.moe.aux = (ids, pos)
        body = (self.lm.get_base_model() if hasattr(self.lm, "get_base_model") else self.lm).model
        body(inputs_embeds=self.inputs_embeds(ids, torch.zeros_like(ids)), attention_mask=mask, position_ids=pos, use_cache=False)
        self.moe.aux = None
        return self.moe.last_route

    def merge_lora(self):
        """Fold LoRA weights into the base for inference (fewer kernel launches per step). Keeps the two-expert MoE and router intact."""
        if self.cfg.lora_r > 0 and hasattr(self.lm, "merge_and_unload"):
            self.lm = self.lm.merge_and_unload()
            layers = self.lm.model.layers
            assert isinstance(layers[-1].mlp, TwoExpertMLP); self.moe = layers[-1].mlp
        return self

    def save(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        sd = {k: v for k, v in self.state_dict().items() if any(t in k for t in ("lora_", "layers.27.mlp", "layers.35.mlp", "e_think", ".mlp.router", ".mlp.think", ".mlp.answer"))}
        # keep everything trainable + router/experts regardless of layer index
        for k, v in self.state_dict().items():
            if "mlp.think" in k or "mlp.answer" in k or "mlp.router" in k: sd[k] = v
        torch.save({"sd": sd, "cfg": self.cfg}, os.path.join(path, "sxs.pt"))

    def load(self, path):
        import os
        ck = torch.load(os.path.join(path, "sxs.pt"), map_location=self.device, weights_only=False)
        missing, unexpected = self.load_state_dict(ck["sd"], strict=False)
        assert not unexpected, unexpected[:5]
        return self
