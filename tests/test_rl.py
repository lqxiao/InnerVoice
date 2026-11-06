import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_model import tiny
from latent.chunked import ChunkedCollator
from latent.rl_gate import rollout, reward_fn
from latent.simulator import AudioModel

def test_rollout_and_reward():
    m = tiny(gate=True); m.eval()
    exs = [{"id": "a", "n_sub": 2, "question": "Q1: x\nQ2: y", "chunks": ["A1: 4", "A2: Yes"], "sub_tasks": ["easy", "graph_reach"],
            "sub_cots": ["4", "a->b"], "sub_answers": ["4", "Yes"]}] * 3
    col = ChunkedCollator(m.tok, Kmax=3, mode="labels")
    b = col(exs)
    ro = rollout(m, b, 3, 0.2, AudioModel(), sample_gate=True, temperature=1.0)
    assert ro["k_steps"].shape == (3, 2) and ro["chunk_ids"].shape[:2] == (3, 2)
    assert ro["gate_feats"].shape == (3, 2, 4, 4)
    w = {"acc": 1, "all": .5, "ttfc": .1, "stall": .3, "gpu": .05, "prem": .5}
    R, info = reward_fn(ro["events"][0], ro["texts"][0], exs[0], AudioModel(), w)
    assert isinstance(R, float) and "ttfc" in info
    # recompute pass accepts rollout tensors
    res = m.forward_chunked(b["prompt_ids"], b["prompt_mask"], ro["chunk_ids"], ro["chunk_mask"], ro["k_steps"], 3,
                            gate_feats=ro["gate_feats"], gate_labels=b["gate_labels"], gate_mask=b["gate_mask"])
    assert res["gate_logits"].shape == (3, 2, 4) and res["ce_per_seq"].shape == (3,)
    (res["ce_per_seq"].sum() + res["gate_logits"].sum()).backward()

def test_chunk_prefix_parsing_for_scoring():
    from latent.chunked import parse_chunk_answers
    from latent.data import is_correct
    assert parse_chunk_answers(["A2: 42"]) == {2: "42"}
    assert is_correct(parse_chunk_answers(["A2: 42"]).get(2, ""), "42")
    assert not is_correct(parse_chunk_answers(["A2: 42"]).get(1, ""), "2")   # wrong index is not credited
