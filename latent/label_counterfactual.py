"""Counterfactual budget experiments: for each chunk of each chunked example, with ground-truth previous chunks,
decode chunk i with k = 0..Kmax latent steps and record correctness -> cf[id][i] = [ok_k0, ..., ok_Kmax].
Also writes an aggregate accuracy-vs-k table per sub-task (how much extra thinking buys)."""
import argparse, json, os, sys, time
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.model import LatentReasoner, ModelConfig, LatentConfig
from latent.data import load_jsonl, is_correct
from latent.chunked import ChunkedCollator, chunk_texts, parse_chunk_answers
from latent.common import to_device

@torch.no_grad()
def decode_chunk_at_budgets(model, batch, i, Kmax, max_tokens, newline_ids):
    """Teacher-force chunks < i, then for every k in 0..Kmax decode chunk i greedily. Returns texts[k][b]."""
    B = batch["prompt_ids"].shape[0]; M = model.M; dev = model.device
    ctx, p_logits, cache, lengths = model.encode_prompt(batch["prompt_ids"], batch["prompt_mask"])
    ctx_list, mask_list = [ctx], [batch["prompt_mask"].bool()]
    prev_logits = p_logits[:, -1]; full_mask = batch["prompt_mask"]; cur = lengths
    z = model.latent.init_state(B, ctx, batch["prompt_mask"]); ar = torch.arange(B, device=dev)
    k_prev = batch["k_steps"]  # budgets used for the teacher-forced previous chunks
    for t in range(i):
        ctx_all = torch.cat(ctx_list, 1); mask_all = torch.cat(mask_list, 1)
        states = [z]
        for j in range(Kmax):
            states.append(model.latent.step(states[-1], ctx_all, mask_all, j))
        z = torch.stack(states, 1)[ar, k_prev[:, t]]
        cm = batch["chunk_mask"][:, t]; Lc = cm.shape[1]
        seg = torch.cat([model.latent_tokens(z), model.embed(batch["chunk_ids"][:, t])], 1)
        seg_mask = torch.cat([torch.ones(B, M, dtype=cm.dtype, device=dev), cm], 1)
        full_mask = torch.cat([full_mask, seg_mask], 1)
        out = model.lm_forward(inputs_embeds=seg, attention_mask=full_mask, position_ids=model.segment_positions(cur, M + Lc), past_key_values=cache)
        cache = out.past_key_values
        h = out.hidden_states[model.cfg.latent.ctx_layer]; ctx_list.append(h); mask_list.append(seg_mask.bool())
        lc = cm.sum(-1); cur = cur + M + lc
    ctx_all = torch.cat(ctx_list, 1); mask_all = torch.cat(mask_list, 1)
    states = [z]
    for j in range(Kmax):
        states.append(model.latent.step(states[-1], ctx_all, mask_all, j))
    # expand batch over k: (Kmax+1)*B sequences sharing the same cache (replicated)
    texts = []
    for k in range(Kmax + 1):
        import copy
        cache_k = copy.deepcopy(cache)
        zk = states[k]
        fm = torch.cat([full_mask, torch.ones(B, M, dtype=full_mask.dtype, device=dev)], 1)
        out = model.lm_forward(inputs_embeds=model.latent_tokens(zk), attention_mask=fm, position_ids=model.segment_positions(cur, M), past_key_values=cache_k, logits_to_keep=1)
        cache_k = out.past_key_values; nl = out.logits[:, -1]; c = cur + M
        gen = []; done = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(max_tokens):
            nxt = nl.argmax(-1); nxt = torch.where(done, torch.full_like(nxt, model.tok.pad_token_id), nxt); gen.append(nxt)
            done = done | torch.tensor([int(x) in newline_ids or int(x) in (model.eos_id, model.tok.eos_token_id) for x in nxt.tolist()], device=dev)
            if done.all(): break
            fm = torch.cat([fm, torch.ones(B, 1, dtype=fm.dtype, device=dev)], 1)
            out = model.lm_forward(input_ids=nxt[:, None], attention_mask=fm, position_ids=c[:, None], past_key_values=cache_k, logits_to_keep=1)
            cache_k = out.past_key_values; nl = out.logits[:, -1]; c = c + 1
        texts.append(model.tok.batch_decode(torch.stack(gen, 1), skip_special_tokens=True))
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="chunked_train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--Kmax", type=int, default=8)
    ap.add_argument("--n_slots", type=int, default=8)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = ModelConfig(model_name=a.model, lora_r=16, latent=LatentConfig(n_slots=a.n_slots, max_steps=max(16, a.Kmax + 1)))
    model = LatentReasoner(cfg, device="cuda"); model.load(a.ckpt); model.eval()
    tok = model.tok
    newline_ids = {i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])} if False else {tok.convert_tokens_to_ids("Ċ")}
    rows = load_jsonl(os.path.join(a.data_dir, f"{a.split}.jsonl"))
    if a.limit: rows = rows[:a.limit]
    col = ChunkedCollator(tok, a.Kmax, mode="random")
    by_n = {}
    for r in rows: by_n.setdefault(r["n_sub"], []).append(r)
    cf = {}; agg = {}
    for n, rs in by_n.items():
        for s in tqdm(range(0, len(rs), a.bs), desc=f"n_sub={n}"):
            chunk = rs[s:s + a.bs]
            batch = to_device(col(chunk), "cuda")
            # teacher-forced previous chunks use the full budget (k=Kmax) so their latent state is 'as good as it gets'
            batch["k_steps"] = torch.full_like(batch["k_steps"], a.Kmax)
            for i in range(n):
                texts = decode_chunk_at_budgets(model, batch, i, a.Kmax, 12, newline_ids)
                for b, ex in enumerate(chunk):
                    def _val(t):  # strip the 'A<i>:' chunk prefix before comparing; a chunk with a wrong/missing index is wrong
                        pa = parse_chunk_answers([t]); return pa.get(i + 1, "")
                    oks = [bool(is_correct(_val(texts[k][b]), ex["sub_answers"][i])) for k in range(a.Kmax + 1)]
                    cf.setdefault(ex["id"], [None] * n)[i] = oks
                    d = agg.setdefault(ex["sub_tasks"][i], [0] * (a.Kmax + 1) + [0])
                    for k in range(a.Kmax + 1): d[k] += oks[k]
                    d[-1] += 1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(cf, open(a.out, "w"))
    table = {t: {"n": d[-1], "acc_vs_k": [d[k] / d[-1] for k in range(a.Kmax + 1)]} for t, d in agg.items()}
    json.dump(table, open(a.out.replace(".json", "_table.json"), "w"), indent=1)
    print(json.dumps(table, indent=1))

if __name__ == "__main__":
    main()
