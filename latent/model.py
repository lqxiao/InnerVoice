"""Qwen3 + LoRA + recurrent latent reasoning module + Think/Speak gate.

Layout of one chunk:   [prompt / history (KV cache)] [M latent soft tokens] [public answer chunk]
Latent recurrence:     z_{k+1} = LayerNorm(z_k + F(z_k, context))      (weights shared across steps)
Answer conditioning:   z_K is projected into M soft input embeddings that sit between the context
                       and the answer; answer tokens attend to them causally (plain KV-cache positions).
Gate:                  p_emit = sigmoid(Gate(context, latent_state, buffer_state))
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from peft import LoraConfig, get_peft_model


@dataclass
class LatentConfig:
    n_slots: int = 8          # M latent vectors; 0 disables the module (pure LoRA baseline)
    n_heads: int = 8
    mlp_ratio: int = 4
    max_steps: int = 16
    ctx_layer: int = -1       # which LM hidden_states entry feeds F (-1 = final, post-norm)
    use_step_emb: bool = True
    dropout: float = 0.0
    z_init: str = "learned"   # "learned": constant z0 ; "last": z0 = LN(W h_last + slot_emb) from the context's final hidden state
    recurrence: str = "adapter"  # "adapter": small F (self-attn + cross-attn + MLP); "lm": F = the LM itself (COCONUT-style):
                                 #   z_{k+1} = LN(z_k + W h_k) with h_k = LM hidden states at the latent positions fed as soft tokens


class LatentModule(nn.Module):
    """F(z, ctx) = self-attn over slots + cross-attn to context + MLP (pre-norm); then LN(z + F)."""

    def __init__(self, d: int, cfg: LatentConfig):
        super().__init__()
        self.cfg = cfg
        M = cfg.n_slots
        self.z0 = nn.Parameter(torch.randn(M, d) * 0.02)
        self.ln_ctx = nn.LayerNorm(d)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_ratio * d), nn.GELU(), nn.Linear(cfg.mlp_ratio * d, d))
        self.step_emb = nn.Embedding(cfg.max_steps, d) if cfg.use_step_emb else None
        self.ln_out = nn.LayerNorm(d)
        if cfg.z_init == "last":
            self.init_proj = nn.Linear(d, d)
            self.init_ln = nn.LayerNorm(d)
        if cfg.recurrence == "lm":
            self.lm_ln = nn.LayerNorm(d)
            self.lm_proj = nn.Linear(d, d)
            nn.init.normal_(self.lm_proj.weight, std=0.02); nn.init.zeros_(self.lm_proj.bias)
        # near-zero init of the residual branches -> step 0 state ~ z0 at init, stable start
        for lin in (self.mlp[-1], self.self_attn.out_proj, self.cross_attn.out_proj):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        if self.step_emb is not None:
            nn.init.normal_(self.step_emb.weight, std=0.02)

    def init_state(self, B: int, ctx: Optional[torch.Tensor] = None, ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cfg.z_init == "last" and ctx is not None:
            # last valid context position per example (works for left padding and for right-padded histories)
            idx = (ctx_mask.long().cumsum(-1).argmax(-1)) if ctx_mask is not None else torch.full((ctx.shape[0],), ctx.shape[1] - 1, device=ctx.device)
            h_last = ctx[torch.arange(ctx.shape[0], device=ctx.device), idx].float()
            h = self.init_proj(self.init_ln(h_last))[:, None, :] + self.z0[None]
            return self.ln_out(h)
        return self.ln_out(self.z0).unsqueeze(0).expand(B, -1, -1)

    def step(self, z: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor, k: int) -> torch.Tensor:
        """One recurrent update. z: (B,M,d) fp32; ctx: (B,L,d); ctx_mask: (B,L) bool (True = valid)."""
        ctx = self.ln_ctx(ctx.float())
        h = z
        if self.step_emb is not None:
            h = h + self.step_emb.weight[min(k, self.cfg.max_steps - 1)]
        a = self.ln1(h)
        h = h + self.self_attn(a, a, a, need_weights=False)[0]
        h = h + self.cross_attn(self.ln2(h), ctx, ctx, key_padding_mask=~ctx_mask, need_weights=False)[0]
        h = h + self.mlp(self.ln3(h))
        f = h - z                      # F(z, ctx)
        return self.ln_out(z + f)      # z_next = LayerNorm(z + F(z, ctx))

    def step_lm(self, z: torch.Tensor, h: torch.Tensor, k: int) -> torch.Tensor:
        """Recurrence through the LM: h = LM hidden states (B,M,d) produced by feeding the latent tokens of z."""
        f = self.lm_proj(self.lm_ln(h.float()))
        if self.step_emb is not None:
            f = f + self.step_emb.weight[min(k, self.cfg.max_steps - 1)]
        return self.ln_out(z + f)

    def run(self, z: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor, K: int, k0: int = 0) -> torch.Tensor:
        for k in range(K):
            z = self.step(z, ctx, ctx_mask, k0 + k)
        return z


class GateHead(nn.Module):
    """p_emit = sigmoid(MLP([ctx_last, mean(z), buffer_feats]))."""
    N_BUF = 4  # remaining_audio_sec/10, steps_taken/max_steps, chunks_emitted/8, buffer_empty flag

    def __init__(self, d: int, hidden: int = 512, use_buffer: bool = True):
        super().__init__()
        self.use_buffer = use_buffer
        self.ln_c = nn.LayerNorm(d); self.ln_z = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(2 * d + self.N_BUF, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, ctx_last: torch.Tensor, z: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
        if not self.use_buffer:
            buf = torch.zeros_like(buf)
        x = torch.cat([self.ln_c(ctx_last.float()), self.ln_z(z.float().mean(1)), buf.float()], dim=-1)
        return self.net(x).squeeze(-1)  # logits


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    latent: LatentConfig = field(default_factory=LatentConfig)
    gate: bool = False
    gate_use_buffer: bool = True
    dtype: str = "bfloat16"
    attn_impl: str = "sdpa"


class LatentReasoner(nn.Module):
    def __init__(self, cfg: ModelConfig, device="cuda", base_model: Optional[nn.Module] = None, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        self.tok = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        base = base_model or AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype, attn_implementation=cfg.attn_impl)
        base.config.use_cache = True
        if cfg.lora_r > 0:
            lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                              target_modules=list(cfg.lora_targets), bias="none", task_type="CAUSAL_LM")
            self.lm = get_peft_model(base, lcfg)
        else:
            for p in base.parameters():
                p.requires_grad_(False)
            self.lm = base
        d = base.config.hidden_size
        self.d = d
        self.M = cfg.latent.n_slots
        self.latent = LatentModule(d, cfg.latent) if self.M > 0 else None
        if self.M > 0:
            self.inject = nn.Linear(d, d)
            nn.init.eye_(self.inject.weight); nn.init.zeros_(self.inject.bias)
            emb_rms = self.embed.weight.detach().float().pow(2).mean().sqrt().item()
            self.inject_scale = nn.Parameter(torch.tensor(emb_rms))
            # auxiliary supervision of latent slots with verified intermediate results: 4 decimal digits + sign
            self.aux_head = nn.Linear(d, 4 * 10 + 2)
        self.gate = GateHead(d, use_buffer=cfg.gate_use_buffer) if cfg.gate else None
        self.to(device)
        self.device = device
        self.eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")

    # ------------------------------------------------------------------ helpers
    @property
    def embed(self):
        return self.lm.get_input_embeddings()

    @property
    def lm_dtype(self):
        return self.embed.weight.dtype

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self):
        lora = [p for n, p in self.lm.named_parameters() if p.requires_grad]
        rest = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("lm.")]
        return lora, rest

    def latent_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent state (B,M,d) fp32 -> soft input embeddings (B,M,d) in LM dtype."""
        return (self.inject(z) * self.inject_scale).to(self.lm_dtype)

    def lm_forward(self, *, input_ids=None, inputs_embeds=None, attention_mask, position_ids, past_key_values=None, logits_to_keep=0):
        return self.lm(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                       position_ids=position_ids, past_key_values=past_key_values, use_cache=True,
                       output_hidden_states=True, logits_to_keep=logits_to_keep, return_dict=True)

    # ------------------------------------------------------------------ context / prompt pass
    def encode_prompt(self, prompt_ids, prompt_mask):
        """Left-padded prompt. Returns (ctx hidden (B,L,d), prompt logits at last position (B,1,V), cache, lengths)."""
        pos = (prompt_mask.long().cumsum(-1) - 1).clamp(min=0)
        out = self.lm_forward(input_ids=prompt_ids, attention_mask=prompt_mask, position_ids=pos, logits_to_keep=1)
        ctx = out.hidden_states[self.cfg.latent.ctx_layer]
        return ctx, out.logits, out.past_key_values, prompt_mask.long().sum(-1)

    def segment_positions(self, lengths: torch.Tensor, n: int) -> torch.Tensor:
        return lengths[:, None] + torch.arange(n, device=lengths.device)[None, :]

    def think(self, ctx, ctx_mask, K: int, z: Optional[torch.Tensor] = None, k0: int = 0, return_all=False):
        """Run K latent steps. Returns z_K (fp32, (B,M,d)) and optionally all intermediate states."""
        B = ctx.shape[0]
        if z is None:
            z = self.latent.init_state(B, ctx, ctx_mask)
        states = [z]
        for k in range(K):
            z = self.latent.step(z, ctx, ctx_mask.bool(), k0 + k)
            states.append(z)
        return (z, states) if return_all else z

    def aux_loss(self, z: torch.Tensor, inter_vals: torch.Tensor, inter_mask: torch.Tensor) -> torch.Tensor:
        """z (B,M,d) fp32; inter_vals (B,M) ints (slot i <-> i-th intermediate result); inter_mask (B,M)."""
        logits = self.aux_head(z.float())                               # (B,M,42)
        v = inter_vals.abs().clamp(max=9999)
        digits = torch.stack([(v // 1000) % 10, (v // 100) % 10, (v // 10) % 10, v % 10], -1)   # (B,M,4)
        sign = (inter_vals < 0).long()
        l = 0.0
        for i in range(4):
            l = l + F.cross_entropy(logits[..., i * 10:(i + 1) * 10].reshape(-1, 10), digits[..., i].reshape(-1), reduction="none")
        l = l + F.cross_entropy(logits[..., 40:42].reshape(-1, 2), sign.reshape(-1), reduction="none")
        l = l.view(z.shape[0], z.shape[1])
        return (l * inter_mask).sum() / inter_mask.sum().clamp(min=1)

    def think_lm(self, ctx, ctx_mask, K: int, cache, full_mask, cur, z=None, return_all=False):
        """COCONUT-style recurrence: K passes of the M latent tokens through the LM (with KV cache). Each pass appends M positions
        to the cache (visible to later tokens). Returns z_K and the extended (cache, full_mask, cur)."""
        B = ctx.shape[0]; M = self.M; dev = ctx.device
        if z is None:
            z = self.latent.init_state(B, ctx, ctx_mask)
        states = [z]
        for k in range(K):
            emb = self.latent_tokens(z)
            full_mask = torch.cat([full_mask, torch.ones(B, M, dtype=full_mask.dtype, device=dev)], 1)
            out = self.lm_forward(inputs_embeds=emb, attention_mask=full_mask, position_ids=self.segment_positions(cur, M), past_key_values=cache, logits_to_keep=1)
            cache = out.past_key_values
            h = out.hidden_states[self.cfg.latent.ctx_layer][:, -M:]
            z = self.latent.step_lm(z, h, k); states.append(z); cur = cur + M
        return (z, states, cache, full_mask, cur) if return_all else (z, cache, full_mask, cur)

    # ------------------------------------------------------------------ Phase-1 training forward (single chunk, fixed K)
    def forward_fixed_k(self, prompt_ids, prompt_mask, answer_ids, answer_mask, K: int, return_hidden_at: Optional[torch.Tensor] = None):
        """Teacher-forced loss on the answer, with K latent steps between prompt and answer.
        return_hidden_at: (B,) index into answer segment whose hidden states (all layers) are returned (for distillation)."""
        B, La = answer_ids.shape
        ctx, p_logits, cache, lengths = self.encode_prompt(prompt_ids, prompt_mask)
        M = self.M
        ans_emb = self.embed(answer_ids)
        full_mask = prompt_mask
        if M > 0:
            if self.cfg.latent.recurrence == "lm":
                z, cache, full_mask, lengths = self.think_lm(ctx, prompt_mask, K, cache, full_mask, lengths)
            else:
                z = self.think(ctx, prompt_mask, K)
            seg_emb = torch.cat([self.latent_tokens(z), ans_emb], dim=1)
        else:
            z = None
            seg_emb = ans_emb
        n = seg_emb.shape[1]
        full_mask = torch.cat([full_mask, torch.ones(B, M, dtype=prompt_mask.dtype, device=prompt_mask.device), answer_mask], dim=1)
        pos = self.segment_positions(lengths, n)
        out = self.lm_forward(inputs_embeds=seg_emb, attention_mask=full_mask, position_ids=pos, past_key_values=cache)
        logits = torch.cat([p_logits, out.logits], dim=1)[:, M:M + La]  # logits predicting answer_ids[:, j]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), answer_ids.reshape(-1), reduction="none").view(B, La)
        loss = (loss * answer_mask).sum() / answer_mask.sum().clamp(min=1)
        res = {"loss": loss, "logits": logits, "z": z}
        if return_hidden_at is not None:
            idx = (M + return_hidden_at).clamp(max=n - 1)
            res["hidden_at"] = torch.stack([h[torch.arange(B), idx] for h in out.hidden_states], dim=1)  # (B, n_layers+1, d)
        return res

    def forward_cot_teacher(self, prompt_ids, prompt_mask, cot_ids, cot_mask, hidden_at: torch.Tensor):
        """Teacher branch (CODI-style self-distillation): explicit CoT then answer; returns CE + hidden states at `hidden_at`."""
        B, Lc = cot_ids.shape
        ctx, p_logits, cache, lengths = self.encode_prompt(prompt_ids, prompt_mask)
        full_mask = torch.cat([prompt_mask, cot_mask], dim=1)
        pos = self.segment_positions(lengths, Lc)
        out = self.lm_forward(input_ids=cot_ids, attention_mask=full_mask, position_ids=pos, past_key_values=cache)
        logits = torch.cat([p_logits, out.logits], dim=1)[:, :Lc]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), cot_ids.reshape(-1), reduction="none").view(B, Lc)
        loss = (loss * cot_mask).sum() / cot_mask.sum().clamp(min=1)
        hid = torch.stack([h[torch.arange(B), hidden_at.clamp(max=Lc - 1)] for h in out.hidden_states], dim=1)
        return {"loss": loss, "hidden_at": hid}

    # ------------------------------------------------------------------ generation (single chunk)
    @torch.no_grad()
    def generate_fixed_k(self, prompt_ids, prompt_mask, K: int, max_new_tokens: int = 16, stop_ids: Optional[List[int]] = None, timing: Optional[dict] = None, z_transform=None):
        """Greedy decoding with K latent steps. Returns token ids (B, T) right-padded with pad id."""
        stop_ids = set(stop_ids or []) | {self.eos_id, self.tok.eos_token_id}
        B = prompt_ids.shape[0]
        dev = prompt_ids.device
        t0 = time.perf_counter()
        ctx, p_logits, cache, lengths = self.encode_prompt(prompt_ids, prompt_mask)
        full_mask = prompt_mask
        if self.M > 0:
            t1 = time.perf_counter()
            if self.cfg.latent.recurrence == "lm":
                z, cache, full_mask, lengths = self.think_lm(ctx, prompt_mask, K, cache, full_mask, lengths)
            else:
                z = self.think(ctx, prompt_mask, K)
            if z_transform is not None:
                z = z_transform(z)
            lat = self.latent_tokens(z)
            if timing is not None:
                torch.cuda.synchronize() if dev.type == "cuda" else None
                timing["latent_s"] = timing.get("latent_s", 0) + (time.perf_counter() - t1)
            full_mask = torch.cat([full_mask, torch.ones(B, self.M, dtype=full_mask.dtype, device=dev)], 1)
            out = self.lm_forward(inputs_embeds=lat, attention_mask=full_mask, position_ids=self.segment_positions(lengths, self.M), past_key_values=cache, logits_to_keep=1)
            next_logits = out.logits[:, -1]
            cache = out.past_key_values
            cur = lengths + self.M
        else:
            next_logits = p_logits[:, -1]
            cur = lengths
        gen = []
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(max_new_tokens):
            nxt = next_logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.tok.pad_token_id), nxt)
            gen.append(nxt)
            done = done | torch.tensor([int(x) in stop_ids for x in nxt.tolist()], device=dev)
            if done.all():
                break
            full_mask = torch.cat([full_mask, torch.ones(B, 1, dtype=full_mask.dtype, device=dev)], 1)
            out = self.lm_forward(input_ids=nxt[:, None], attention_mask=full_mask, position_ids=cur[:, None], past_key_values=cache, logits_to_keep=1)
            next_logits = out.logits[:, -1]
            cache = out.past_key_values
            cur = cur + 1
        if timing is not None:
            torch.cuda.synchronize() if dev.type == "cuda" else None
            timing["total_s"] = timing.get("total_s", 0) + (time.perf_counter() - t0)
        return torch.stack(gen, 1)

    # ------------------------------------------------------------------ Phase-2: multi-chunk forward with per-chunk latent budgets + gate
    def forward_chunked(self, prompt_ids, prompt_mask, chunk_ids, chunk_mask, k_steps, Kmax: int,
                        gate_feats=None, gate_labels=None, gate_mask=None, gate_pos_weight: float = 1.0):
        """prompt: left-padded. chunk_ids/mask: (B,N,Lc) right-padded per chunk. k_steps: (B,N) latent steps before chunk i.
        gate_feats: (B,N,Kmax+1,N_BUF) buffer features at decision (i,j); gate_labels/mask: (B,N,Kmax+1) (1 = Speak).
        Latent state carries across chunks; context for F = prompt + all previous segments.
        Returns dict(loss_ce, loss_gate, gate_logits (B,N,Kmax+1), n_tokens)."""
        B, N, Lc = chunk_ids.shape
        M = self.M; assert M > 0
        dev = prompt_ids.device
        ctx, p_logits, cache, lengths = self.encode_prompt(prompt_ids, prompt_mask)
        ctx_list, mask_list = [ctx], [prompt_mask.bool()]
        last_hidden = ctx[:, -1]                       # left padding -> last position is the last real token
        prev_logits = p_logits[:, -1]
        full_mask = prompt_mask
        cur = lengths
        z = self.latent.init_state(B, ctx, prompt_mask)
        ce_sum, ce_cnt = 0.0, 0
        ce_seq = torch.zeros(B, device=dev)
        gate_logits_all = []
        ar = torch.arange(B, device=dev)
        for i in range(N):
            ctx_all = torch.cat(ctx_list, 1); mask_all = torch.cat(mask_list, 1)
            states = [z]
            for j in range(Kmax):
                states.append(self.latent.step(states[-1], ctx_all, mask_all, j))
            Z = torch.stack(states, 1)                                   # (B, Kmax+1, M, d)
            if self.gate is not None and gate_feats is not None:
                gl = self.gate(last_hidden[:, None].expand(-1, Kmax + 1, -1).reshape(B * (Kmax + 1), -1),
                               Z.reshape(B * (Kmax + 1), M, -1), gate_feats[:, i].reshape(B * (Kmax + 1), -1)).view(B, Kmax + 1)
                gate_logits_all.append(gl)
            z_sel = Z[ar, k_steps[:, i]]                                 # (B, M, d)
            cm = chunk_mask[:, i]
            seg = torch.cat([self.latent_tokens(z_sel), self.embed(chunk_ids[:, i])], 1)
            seg_mask = torch.cat([torch.ones(B, M, dtype=cm.dtype, device=dev), cm], 1)
            full_mask = torch.cat([full_mask, seg_mask], 1)
            pos = self.segment_positions(cur, M + Lc)
            out = self.lm_forward(inputs_embeds=seg, attention_mask=full_mask, position_ids=pos, past_key_values=cache)
            cache = out.past_key_values
            logits = torch.cat([prev_logits[:, None], out.logits], 1)[:, M:M + Lc]
            ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), chunk_ids[:, i].reshape(-1), reduction="none").view(B, Lc)
            ce_sum = ce_sum + (ce * cm).sum(); ce_cnt = ce_cnt + cm.sum(); ce_seq = ce_seq + (ce * cm).sum(-1)
            h = out.hidden_states[self.cfg.latent.ctx_layer]
            ctx_list.append(h); mask_list.append(seg_mask.bool())
            lc = cm.sum(-1)
            last_hidden = h[ar, (M + lc - 1).clamp(min=0)]
            prev_logits = out.logits[ar, (M + lc - 1).clamp(min=0)]
            cur = cur + M + lc
            z = z_sel
        res = {"loss_ce": ce_sum / ce_cnt.clamp(min=1), "n_tokens": ce_cnt, "ce_per_seq": ce_seq}
        if gate_logits_all:
            G = torch.stack(gate_logits_all, 1)                          # (B,N,Kmax+1)
            res["gate_logits"] = G
            if gate_labels is not None:
                pw = torch.tensor(gate_pos_weight, device=dev)
                bce = F.binary_cross_entropy_with_logits(G, gate_labels.float(), pos_weight=pw, reduction="none")
                res["loss_gate"] = (bce * gate_mask).sum() / gate_mask.sum().clamp(min=1)
        return res

    @torch.no_grad()
    def stream_chunked(self, prompt_ids, prompt_mask, Kmax: int, policy, max_chunks: int = 8, max_chunk_tokens: int = 16,
                       newline_ids=None, step_cost: float = 0.0, audio=None, record_gate: bool = True):
        """Batch size 1 streaming decode with Think/Speak decisions.
        policy(i, j, p_emit, buf_feats) -> True (speak) / False (think). p_emit from the gate head (None if no gate).
        Buffer features follow the simulator's clock: compute time measured with cuda.synchronize + per-step `step_cost` override.
        Returns list of events for the simulator and the list of chunk texts."""
        assert prompt_ids.shape[0] == 1
        dev = prompt_ids.device
        newline_ids = set(newline_ids or [])
        sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
        events, chunks, gate_trace = [], [], []
        sync(); t0 = time.perf_counter()
        ctx, p_logits, cache, lengths = self.encode_prompt(prompt_ids, prompt_mask)
        sync(); t_prefill = time.perf_counter() - t0
        events.append({"type": "think", "dt": t_prefill, "kind": "prefill"})
        ctx_list, mask_list = [ctx], [prompt_mask.bool()]
        last_hidden = ctx[:, -1]; prev_logits = p_logits[:, -1]
        full_mask = prompt_mask; cur = lengths
        z = self.latent.init_state(1, ctx, prompt_mask)
        M = self.M
        # buffer clock
        t_compute = t_prefill; play_end = 0.0
        finished = False
        for i in range(max_chunks):
            ctx_all = torch.cat(ctx_list, 1); mask_all = torch.cat(mask_list, 1)
            j = 0
            while True:
                buf_rem = max(0.0, play_end - t_compute)
                feats = torch.tensor([[buf_rem / 10.0, j / max(1, Kmax), i / 8.0, step_cost]], device=dev)
                p_emit = torch.sigmoid(self.gate(last_hidden, z, feats)).item() if self.gate is not None else None
                speak = True if j >= Kmax else policy(i, j, p_emit, {"buf": buf_rem, "step_cost": step_cost})
                if record_gate:
                    gate_trace.append({"chunk": i, "j": j, "p_emit": p_emit, "buf": buf_rem, "speak": bool(speak)})
                if speak:
                    break
                sync(); ts = time.perf_counter()
                z = self.latent.step(z, ctx_all, mask_all, j)
                sync(); dt = time.perf_counter() - ts
                dt_eff = step_cost if step_cost > 0 else dt
                events.append({"type": "think", "dt": dt_eff, "dt_measured": dt})
                t_compute += dt_eff; j += 1
            # speak: inject latents and decode one chunk
            sync(); ts = time.perf_counter()
            lat = self.latent_tokens(z)
            full_mask = torch.cat([full_mask, torch.ones(1, M, dtype=full_mask.dtype, device=dev)], 1)
            out = self.lm_forward(inputs_embeds=lat, attention_mask=full_mask, position_ids=self.segment_positions(cur, M), past_key_values=cache)
            cache = out.past_key_values
            seg_hidden = [out.hidden_states[self.cfg.latent.ctx_layer]]
            cur = cur + M
            next_logits = out.logits[:, -1]
            toks = []
            for t in range(max_chunk_tokens):
                nxt = next_logits.argmax(-1)
                tid = int(nxt.item())
                toks.append(tid)
                full_mask = torch.cat([full_mask, torch.ones(1, 1, dtype=full_mask.dtype, device=dev)], 1)
                out = self.lm_forward(input_ids=nxt[:, None], attention_mask=full_mask, position_ids=cur[:, None], past_key_values=cache)
                cache = out.past_key_values; cur = cur + 1
                seg_hidden.append(out.hidden_states[self.cfg.latent.ctx_layer])
                next_logits = out.logits[:, -1]
                if tid in (self.eos_id, self.tok.eos_token_id):
                    finished = True; break
                if tid in newline_ids:
                    break
            sync(); dt = time.perf_counter() - ts
            text = self.tok.decode(toks, skip_special_tokens=True).strip()
            h = torch.cat(seg_hidden, 1)
            ctx_list.append(h); mask_list.append(torch.ones(1, h.shape[1], dtype=torch.bool, device=dev))
            last_hidden = h[:, -1]; prev_logits = next_logits
            t_compute += dt
            dur = audio.duration(text) if audio is not None else 0.0
            start = max(t_compute, play_end); play_end = start + dur
            events.append({"type": "speak", "dt": dt, "text": text, "k": j})
            chunks.append(text)
            if finished or not text:
                break
        return events, chunks, gate_trace

    # ------------------------------------------------------------------ checkpointing (only trainable parts)
    def save(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        if self.cfg.lora_r > 0:
            self.lm.save_pretrained(os.path.join(path, "lora"))
        extra = {k: v for k, v in self.state_dict().items() if not k.startswith("lm.")}
        torch.save({"extra": extra, "cfg": self.cfg}, os.path.join(path, "extra.pt"))

    def load(self, path: str):
        import os
        if self.cfg.lora_r > 0:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            sd = load_file(os.path.join(path, "lora", "adapter_model.safetensors"))
            set_peft_model_state_dict(self.lm, sd)
        ck = torch.load(os.path.join(path, "extra.pt"), map_location=self.device, weights_only=False)
        missing, unexpected = self.load_state_dict(ck["extra"], strict=False)
        unexpected = [k for k in unexpected if not k.startswith("lm.")]
        assert not unexpected, unexpected
        return self
