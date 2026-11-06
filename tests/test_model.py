"""Gradient, causal-mask, KV-cache and latent-influence tests on a tiny random Qwen3 (runs on CPU in fp32)."""
import os, sys, torch, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
from latent.model import LatentReasoner, ModelConfig, LatentConfig

torch.manual_seed(0)
TOK_NAME = os.environ.get("TEST_TOKENIZER", "Qwen/Qwen3-1.7B")

def tiny(n_slots=4, lora_r=4, gate=False):
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    base = Qwen3ForCausalLM(cfg).float()
    tok = AutoTokenizer.from_pretrained(TOK_NAME)
    mc = ModelConfig(model_name=TOK_NAME, lora_r=lora_r, dtype="float32", latent=LatentConfig(n_slots=n_slots, n_heads=4), gate=gate)
    return LatentReasoner(mc, device="cpu", base_model=base, tokenizer=tok)

def batch(B=2, Lp=7, La=5, left_pad=True):
    p = torch.randint(10, 1000, (B, Lp)); pm = torch.ones(B, Lp, dtype=torch.long)
    if left_pad:
        pm[0, :3] = 0
    a = torch.randint(10, 1000, (B, La)); am = torch.ones(B, La, dtype=torch.long); am[1, -2:] = 0
    return p, pm, a, am

def test_gradients_reach_latent_and_lora():
    m = tiny(); p, pm, a, am = batch()
    res = m.forward_fixed_k(p, pm, a, am, K=3)
    res["loss"].backward()
    lat_g = [n for n, q in m.latent.named_parameters() if q.grad is not None and q.grad.abs().sum() > 0]
    assert any("cross_attn" in n for n in lat_g), "no gradient into cross-attention of latent module"
    assert any("mlp" in n for n in lat_g), "no gradient into latent MLP"
    assert m.latent.z0.grad is not None and m.latent.z0.grad.abs().sum() > 0
    assert m.inject.weight.grad.abs().sum() > 0
    lora_g = [n for n, q in m.lm.named_parameters() if q.requires_grad and q.grad is not None and q.grad.abs().sum() > 0]
    assert len(lora_g) > 0, "no gradient into LoRA"
    base_frozen = [n for n, q in m.lm.named_parameters() if "lora_" not in n and q.requires_grad]
    assert not base_frozen

def test_latent_influences_logits():
    m = tiny(); p, pm, a, am = batch()
    with torch.no_grad():
        r1 = m.forward_fixed_k(p, pm, a, am, K=2)
        # perturb latent output by hooking z0 (changes z -> should change answer logits)
        m.latent.z0.add_(torch.randn_like(m.latent.z0) * 5.0)
        r2 = m.forward_fixed_k(p, pm, a, am, K=2)
    assert (r1["logits"] - r2["logits"]).abs().max() > 1e-3
    # and different K produce different logits (recurrence does something once weights are non-zero)
    for lin in (m.latent.mlp[-1], m.latent.cross_attn.out_proj):
        torch.nn.init.normal_(lin.weight, std=0.1)
    with torch.no_grad():
        r3 = m.forward_fixed_k(p, pm, a, am, K=1); r4 = m.forward_fixed_k(p, pm, a, am, K=4)
    assert (r3["logits"] - r4["logits"]).abs().max() > 1e-3

def test_causal_mask_future_answer_tokens_do_not_leak():
    m = tiny(); p, pm, a, am = batch(left_pad=False)
    with torch.no_grad():
        r1 = m.forward_fixed_k(p, pm, a, am, K=2)
        a2 = a.clone(); a2[:, -1] = (a2[:, -1] + 17) % 1000  # change last answer token
        r2 = m.forward_fixed_k(p, pm, a2, am, K=2)
    # logits predicting tokens 0..La-1 depend on tokens < j only; changing token La-1 may change nothing except (nothing)
    assert torch.allclose(r1["logits"][:, :-1], r2["logits"][:, :-1], atol=1e-4), "future answer token leaked into earlier logits"
    # prompt hidden states must not depend on latents / answers (prompt pass happens before injection)
    ctx1, *_ = m.encode_prompt(p, pm)
    m.latent.z0.data.add_(1.0)
    ctx2, *_ = m.encode_prompt(p, pm)
    assert torch.allclose(ctx1, ctx2)

def test_left_padding_matches_unpadded():
    m = tiny(); torch.manual_seed(1)
    p = torch.randint(10, 1000, (1, 6)); a = torch.randint(10, 1000, (1, 4))
    pm, am = torch.ones(1, 6, dtype=torch.long), torch.ones(1, 4, dtype=torch.long)
    with torch.no_grad():
        r_ref = m.forward_fixed_k(p, pm, a, am, K=2)
        p_pad = torch.cat([torch.full((1, 3), m.tok.pad_token_id), p], 1); pm_pad = torch.cat([torch.zeros(1, 3, dtype=torch.long), pm], 1)
        r_pad = m.forward_fixed_k(p_pad, pm_pad, a, am, K=2)
    assert torch.allclose(r_ref["logits"], r_pad["logits"], atol=1e-3), (r_ref["logits"] - r_pad["logits"]).abs().max()

def test_kv_cache_generation_matches_full_forward():
    """Greedy tokens from the incremental (cached) decoder must equal argmax of a full teacher-forced pass over the same tokens."""
    m = tiny(); torch.manual_seed(2)
    for lin in (m.latent.mlp[-1], m.latent.cross_attn.out_proj):
        torch.nn.init.normal_(lin.weight, std=0.1)
    p, pm, _, _ = batch(B=2, Lp=8)
    with torch.no_grad():
        gen = m.generate_fixed_k(p, pm, K=3, max_new_tokens=5, stop_ids=[-1])
        am = torch.ones_like(gen)
        r = m.forward_fixed_k(p, pm, gen, am, K=3)
        pred = r["logits"].argmax(-1)
    # position j logits predict gen[:, j]; incremental argmax must match for all j
    assert torch.equal(pred, gen), (pred, gen)

def test_no_latent_baseline_runs():
    m = tiny(n_slots=0); p, pm, a, am = batch()
    res = m.forward_fixed_k(p, pm, a, am, K=0)
    res["loss"].backward()
    assert res["logits"].shape[1] == a.shape[1]
    with torch.no_grad():
        gen = m.generate_fixed_k(p, pm, K=0, max_new_tokens=3, stop_ids=[-1])
    assert gen.shape == (2, 3)

def test_k0_static_soft_tokens_context_independent():
    m = tiny(); p, pm, a, am = batch(left_pad=False)
    with torch.no_grad():
        z_a = m.think(*m.encode_prompt(p, pm)[0:1], pm, 0)
        p2 = torch.randint(10, 1000, p.shape)
        z_b = m.think(m.encode_prompt(p2, pm)[0], pm, 0)
        assert torch.allclose(z_a, z_b)  # K=0: no context read
        z_c = m.think(m.encode_prompt(p2, pm)[0], pm, 1)
        for lin in (m.latent.cross_attn.out_proj,):
            torch.nn.init.normal_(lin.weight, std=0.1)
        z_d = m.think(m.encode_prompt(p, pm)[0], pm, 1); z_e = m.think(m.encode_prompt(p2, pm)[0], pm, 1)
        assert not torch.allclose(z_d, z_e)  # K>=1 reads context

def test_gate_head():
    m = tiny(gate=True); p, pm, _, _ = batch()
    ctx, _, _, L = m.encode_prompt(p, pm)
    z = m.think(ctx, pm, 2)
    buf = torch.zeros(2, 4)
    logit = m.gate(ctx[:, -1], z, buf)
    assert logit.shape == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor([0., 1.]))
    loss.backward()
    assert m.gate.net[0].weight.grad.abs().sum() > 0

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))

# ---------------------------------------------------------------- Phase 2 pieces
def test_forward_chunked_and_stream():
    import sys as _s; _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from latent.chunked import ChunkedCollator, make_decisions
    from latent.simulator import AudioModel, simulate
    m = tiny(gate=True)
    exs = [{"id": "a", "n_sub": 2, "question": "Q1: x\nQ2: y", "chunks": ["A1: 4", "A2: Yes"], "sub_tasks": ["easy", "graph_reach"],
            "sub_cots": ["4", "a->b"], "sub_answers": ["4", "Yes"]},
           {"id": "b", "n_sub": 2, "question": "Q1: longer question here\nQ2: y", "chunks": ["A1: 1234", "A2: No"], "sub_tasks": ["arith_chain", "easy"],
            "sub_cots": ["1;2;3", "x"], "sub_answers": ["1234", "No"]}]
    col = ChunkedCollator(m.tok, Kmax=3, mode="labels")
    b = col(exs)
    assert b["gate_labels"].shape == (2, 2, 4) and b["gate_feats"].shape == (2, 2, 4, 4)
    # every chunk has exactly one Speak label within mask
    assert torch.all((b["gate_labels"] * b["gate_mask"]).sum(-1) == 1)
    res = m.forward_chunked(b["prompt_ids"], b["prompt_mask"], b["chunk_ids"], b["chunk_mask"], b["k_steps"], 3,
                            gate_feats=b["gate_feats"], gate_labels=b["gate_labels"], gate_mask=b["gate_mask"])
    (res["loss_ce"] + res["loss_gate"]).backward()
    assert m.gate.net[0].weight.grad.abs().sum() > 0 and m.latent.z0.grad.abs().sum() > 0
    # chunk-level causality: chunk 0 logits must not depend on chunk 1 tokens
    with torch.no_grad():
        ids2 = b["chunk_ids"].clone(); ids2[:, 1, 0] = 77
        r1 = m.forward_chunked(b["prompt_ids"], b["prompt_mask"], b["chunk_ids"], b["chunk_mask"], b["k_steps"], 3)
        r2 = m.forward_chunked(b["prompt_ids"], b["prompt_mask"], ids2, b["chunk_mask"], b["k_steps"], 3)
    # (loss differs because chunk 1 changed, but we can check chunk-0 by zeroing chunk-1 mask)
    cm0 = b["chunk_mask"].clone(); cm0[:, 1] = 0
    with torch.no_grad():
        r1 = m.forward_chunked(b["prompt_ids"], b["prompt_mask"], b["chunk_ids"], cm0, b["k_steps"], 3)
        r2 = m.forward_chunked(b["prompt_ids"], b["prompt_mask"], ids2, cm0, b["k_steps"], 3)
    assert torch.allclose(r1["loss_ce"], r2["loss_ce"], atol=1e-5)
    # streaming decode with a fixed-k policy produces events the simulator accepts
    ev, chunks, trace = m.stream_chunked(b["prompt_ids"][:1], b["prompt_mask"][:1], Kmax=3, policy=lambda i, j, p, f: j >= 2,
                                         max_chunks=2, max_chunk_tokens=3, newline_ids=[198], step_cost=0.1, audio=AudioModel())
    assert sum(e["type"] == "think" for e in ev) >= 4  # prefill + 2 steps per chunk
    assert all(t["p_emit"] is not None for t in trace)
    r = simulate(ev); assert r.hidden_steps >= 4 and r.n_chunks == len(chunks)

def test_make_decisions_buffer_awareness():
    import sys as _s; _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from latent.chunked import make_decisions
    from latent.simulator import AudioModel
    texts = ["A1: 42", "A2: Yes"]
    # no audio buffered yet and expensive steps -> buffer-aware rule speaks early on chunk 0; non-aware thinks k* steps
    f, l, mk, k_aware = make_decisions([4, 4], texts, 8, step_cost=1.0, audio=AudioModel(), tau=0.3, buffer_aware=True)
    f, l, mk, k_blind = make_decisions([4, 4], texts, 8, step_cost=1.0, audio=AudioModel(), tau=0.3, buffer_aware=False)
    assert k_aware[0] < k_blind[0] == 4
    # cheap steps -> both think the full k*
    f, l, mk, k_cheap = make_decisions([4, 4], texts, 8, step_cost=0.01, audio=AudioModel(), tau=0.3, buffer_aware=True)
    assert k_cheap == [4, 4]

def test_z_init_last_is_context_dependent():
    from latent.model import LatentReasoner, ModelConfig, LatentConfig
    from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    m = LatentReasoner(ModelConfig(model_name=TOK_NAME, lora_r=4, dtype="float32", latent=LatentConfig(n_slots=4, n_heads=4, z_init="last")),
                       device="cpu", base_model=Qwen3ForCausalLM(cfg).float(), tokenizer=AutoTokenizer.from_pretrained(TOK_NAME))
    p, pm, a, am = batch(left_pad=True)
    with torch.no_grad():
        z1 = m.think(m.encode_prompt(p, pm)[0], pm, 0)
        p2 = p.clone(); p2[:, -1] = (p2[:, -1] + 5) % 1000
        z2 = m.think(m.encode_prompt(p2, pm)[0], pm, 0)
    assert not torch.allclose(z1, z2)
    # left-padding invariance of the init
    with torch.no_grad():
        z_ref = m.think(m.encode_prompt(p[1:], pm[1:])[0], pm[1:], 0)
        pp = torch.cat([torch.full((1, 3), m.tok.pad_token_id), p[1:]], 1); mm = torch.cat([torch.zeros(1, 3, dtype=torch.long), pm[1:]], 1)
        z_pad = m.think(m.encode_prompt(pp, mm)[0], mm, 0)
    assert torch.allclose(z_ref, z_pad, atol=1e-3)
    res = m.forward_fixed_k(p, pm, a, am, K=2); res["loss"].backward(); assert m.latent.init_proj.weight.grad.abs().sum() > 0

def _tiny_lm_rec(K_slots=4):
    from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    return LatentReasoner(ModelConfig(model_name=TOK_NAME, lora_r=4, dtype="float32", latent=LatentConfig(n_slots=K_slots, n_heads=4, recurrence="lm")),
                          device="cpu", base_model=Qwen3ForCausalLM(cfg).float(), tokenizer=AutoTokenizer.from_pretrained(TOK_NAME))

def test_lm_recurrence_gradients_kv_and_padding():
    m = _tiny_lm_rec(); p, pm, a, am = batch()
    res = m.forward_fixed_k(p, pm, a, am, K=3); res["loss"].backward()
    assert m.latent.lm_proj.weight.grad.abs().sum() > 0 and m.latent.z0.grad.abs().sum() > 0
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for n, q in m.lm.named_parameters() if "lora_" in n)
    # K changes logits; context changes z (recurrence through the LM is context dependent)
    with torch.no_grad():
        r1 = m.forward_fixed_k(p, pm, a, am, K=1); r3 = m.forward_fixed_k(p, pm, a, am, K=3)
        assert (r1["logits"] - r3["logits"]).abs().max() > 1e-4
        assert (r3["z"][0] - r3["z"][1]).abs().max() > 1e-4
    # left-padding invariance
    torch.manual_seed(3); p1 = torch.randint(10, 1000, (1, 6)); a1 = torch.randint(10, 1000, (1, 4))
    pm1, am1 = torch.ones(1, 6, dtype=torch.long), torch.ones(1, 4, dtype=torch.long)
    with torch.no_grad():
        r_ref = m.forward_fixed_k(p1, pm1, a1, am1, K=2)
        pp = torch.cat([torch.full((1, 3), m.tok.pad_token_id), p1], 1); mm = torch.cat([torch.zeros(1, 3, dtype=torch.long), pm1], 1)
        r_pad = m.forward_fixed_k(pp, mm, a1, am1, K=2)
    assert torch.allclose(r_ref["logits"], r_pad["logits"], atol=1e-3)
    # kv-cache generation == teacher forced argmax
    with torch.no_grad():
        gen = m.generate_fixed_k(p, pm, K=2, max_new_tokens=4, stop_ids=[-1])
        r = m.forward_fixed_k(p, pm, gen, torch.ones_like(gen), K=2)
    assert torch.equal(r["logits"].argmax(-1), gen)
    # causality: prompt pass unaffected; answer logits at j don't see tokens > j
    with torch.no_grad():
        a2 = a.clone(); a2[:, -1] = (a2[:, -1] + 9) % 1000
        r1 = m.forward_fixed_k(p, pm, a, am, K=2); r2 = m.forward_fixed_k(p, pm, a2, am, K=2)
    assert torch.allclose(r1["logits"][:, :-1], r2["logits"][:, :-1], atol=1e-4)
