import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
from latent.sxs_model import SxSModel, SxSConfig, N_BUF
TOK = os.environ.get("TEST_TOKENIZER", "Qwen/Qwen3-1.7B")

def tiny():
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    return SxSModel(SxSConfig(model_name=TOK, lora_r=4, dtype="float32"), device="cpu", base_model=Qwen3ForCausalLM(cfg).float(),
                    tokenizer=AutoTokenizer.from_pretrained(TOK))

def test_experts_identical_at_init_and_router_biased_to_answer():
    m = tiny()
    x = torch.randn(2, 5, 64)
    m.moe.route = torch.zeros(2, 5, dtype=torch.bool); a = m.moe(x)
    m.moe.route = torch.ones(2, 5, dtype=torch.bool); t = m.moe(x); m.moe.route = None
    assert torch.allclose(a, t)
    assert (m.moe.router_logits < 0).all()

def test_train_forward_grads_and_routing_effect():
    m = tiny()
    B, L = 2, 12
    ids = torch.randint(10, 1000, (B, L)); mask = torch.ones(B, L, dtype=torch.long)
    is_think = torch.zeros(B, L, dtype=torch.long); is_think[:, 5:9] = 1
    loss_mask = torch.zeros(B, L, dtype=torch.long); loss_mask[:, 4:] = 1
    route_label = torch.zeros(B, L, dtype=torch.long); route_label[:, 4:8] = 1   # token t+1 is think for t in 4..7
    route_mask = torch.zeros(B, L, dtype=torch.long); route_mask[:, 3:L - 1] = 1
    res = m.forward_train(ids, mask, is_think, loss_mask, route_label, route_mask)
    (res["loss_ce"] + res["loss_route"]).backward()
    assert m.e_think.grad.abs().sum() > 0
    assert m.moe.router[0].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.moe.think.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.moe.answer.parameters())
    # e_think changes logits at think positions only (causally: at and after)
    with torch.no_grad():
        m.moe.route = route_label.bool()
        o1 = m.lm(inputs_embeds=m.inputs_embeds(ids, is_think), attention_mask=mask).logits
        o2 = m.lm(inputs_embeds=m.inputs_embeds(ids, torch.zeros_like(is_think)), attention_mask=mask).logits
        m.moe.route = None
    assert torch.allclose(o1[:, :5], o2[:, :5], atol=1e-5) and (o1[:, 5:] - o2[:, 5:]).abs().max() > 1e-5

def test_stream_runs_and_respects_policy():
    m = tiny(); m.eval()
    ids = torch.randint(10, 1000, (1, 7))
    ev, th, ch, tr = m.stream(ids, policy=lambda s: "think" if s["n_think_block"] < 3 and s["n_chunks"] == 0 else "answer", max_new_tokens=8)
    assert len(tr) == 8 or any(e.get("kind") != "prefill" for e in ev)
    assert tr[0]["mode"] == "think" and tr[3]["mode"] == "answer"
    ev2, th2, ch2, tr2 = m.stream(ids, policy=lambda s: "answer", max_new_tokens=5)
    assert all(t["mode"] == "answer" for t in tr2) and th2 == []
    ev3, th3, ch3, tr3 = m.stream(ids, policy=lambda s: None, max_new_tokens=5)   # router decides
    assert len(tr3) >= 1

def test_stream_returns_every_newline_chunk():
    """chunks returned == speak events with text (regression: intermediate chunks were dropped)."""
    m = tiny(); m.eval()
    ids = torch.randint(10, 1000, (1, 7))
    ev, th, ch, tr = m.stream(ids, policy=lambda s: "answer", max_new_tokens=40)
    assert ch == [e["text"] for e in ev if e["type"] == "speak" and e.get("text")]

def test_grad_ckpt_training_step():
    from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    m = SxSModel(SxSConfig(model_name=TOK, lora_r=4, dtype="float32", grad_ckpt=True), device="cpu", base_model=Qwen3ForCausalLM(cfg).float(),
                 tokenizer=AutoTokenizer.from_pretrained(TOK)); m.train()
    B, L = 2, 10
    ids = torch.randint(10, 1000, (B, L)); mask = torch.ones(B, L, dtype=torch.long); is_think = torch.zeros(B, L, dtype=torch.long); is_think[:, 4:7] = 1
    loss_mask = torch.zeros(B, L, dtype=torch.long); loss_mask[:, 3:] = 1; route_label = torch.zeros(B, L, dtype=torch.long); route_label[:, 3:6] = 1
    route_mask = torch.zeros(B, L, dtype=torch.long); route_mask[:, 2:L - 1] = 1
    res = m.forward_train(ids, mask, is_think, loss_mask, route_label, route_mask); (res["loss_ce"] + res["loss_route"]).backward()
    assert m.moe.router[0].weight.grad.abs().sum() > 0 and any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.lm.named_parameters() if "lora_" in n)

def test_stream_batch_matches_sequential_router_decode():
    m = tiny(); m.eval()
    # make the router non-trivial
    with torch.no_grad(): m.moe.router[-1].bias.fill_(0.0); m.moe.router[-1].weight.normal_(std=0.5)
    p1 = torch.randint(10, 1000, (1, 6)); p2 = torch.randint(10, 1000, (1, 9))
    ev1, th1, ch1, tr1 = m.stream(p1, lambda s: None, max_new_tokens=8, max_think_per_block=10**6)
    ev2, th2, ch2, tr2 = m.stream(p2, lambda s: None, max_new_tokens=8, max_think_per_block=10**6)
    pad = m.tok.pad_token_id
    ids = torch.full((2, 9), pad); ids[0, 3:] = p1[0]; ids[1] = p2[0]
    mask = torch.zeros(2, 9, dtype=torch.long); mask[0, 3:] = 1; mask[1] = 1
    toks, types = m.stream_batch(ids, mask, max_new_tokens=8)
    assert toks[0] == [t["tid"] for t in tr1] and toks[1] == [t["tid"] for t in tr2]
    assert types[0] == [int(t["mode"] == "think") for t in tr1] and types[1] == [int(t["mode"] == "think") for t in tr2]

def test_rl_step_runs_on_tiny(tmp_path):
    import json, subprocess, sys as _s
    from latent.sxs_rl import score_rollout, build_train_batch, public_chunks
    from latent.sxs_eval import token_metrics
    m = tiny(); m.eval()
    ex = {"source": "kk", "prompt": "Who is a knight?", "chunk_checks": [{"type": "kk", "name": "Zoey", "knight": True}, {"type": "kk", "name": "Oliver", "knight": False}]}
    ids = torch.randint(10, 1000, (2, 6)); mask = torch.ones(2, 6, dtype=torch.long)
    toks, types = m.stream_batch(ids, mask, max_new_tokens=6, greedy=False, sample_router=True)
    assert all(len(t) == len(y) for t, y in zip(toks, types))
    w = {"inter": .5, "prem": 1, "fmt": .2, "fco": 0, "fco_ref": 200, "len": 0, "len_ref": 300}
    R, info = score_rollout(ex, ["Zoey is a knight.", "Oliver is a knight.", "garbage"], token_metrics(types[0], toks[0], m.tok, "kk"), w)
    assert info["prem"] == 1 and info["unp"] == 1 and info["final"] == 0.0
    b = build_train_batch(m.tok, [ids[0].tolist(), ids[1].tolist()], toks, types, m.tok.pad_token_id)
    m.train(); res = m.forward_train(b["ids"], b["mask"], b["is_think"], b["loss_mask"], b["route_label"], b["route_mask"], return_per_seq=True)
    assert res["logp_tok_seq"].shape == (2,) and res["logp_route_seq"].shape == (2,)
    (-(res["logp_tok_seq"] + res["logp_route_seq"]) / res["n_resp"].clamp(min=1)).sum().backward()
    assert m.moe.router[0].weight.grad is not None
