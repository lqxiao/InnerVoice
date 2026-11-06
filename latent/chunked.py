"""Phase 2 data plumbing: chunked trajectories, Think/Speak decision labels, buffer-state features."""
import random, json, re
from typing import List, Dict, Optional
import torch
from latent.common import build_prompt, pad_batch
from latent.simulator import AudioModel

N_BUF = 4  # [buffer_remaining/10, j/Kmax, chunk_idx/8, step_cost]
GEN_COST_PER_TOKEN = 0.03  # seconds, nominal decode cost used when simulating labels

def chunk_texts(ex: Dict) -> List[str]:
    ch = list(ex["chunks"])
    return [c + "\n" for c in ch[:-1]] + [ch[-1] + "<|im_end|>"]

def weak_kstar(ex: Dict, Kmax: int) -> List[int]:
    """Weak labels from the (programmatically verified) teacher reasoning trajectory length."""
    out = []
    for t, cot in zip(ex["sub_tasks"], ex["sub_cots"]):
        if t == "easy":
            out.append(0)
        elif t == "graph_reach":
            out.append(min(Kmax, 4))
        else:
            out.append(min(Kmax, cot.count(";") + 1))
    return out

def combine_kstar(weak: List[int], cf: Optional[List[List[bool]]], Kmax: int) -> List[int]:
    """cf[i][k] = chunk i correct with k latent steps (counterfactual budget experiment). k* = smallest sufficient k;
    if no budget makes it correct fall back to the weak label."""
    if cf is None:
        return weak
    out = []
    for i, w in enumerate(weak):
        ks = [k for k, ok in enumerate(cf[i]) if ok]
        out.append(ks[0] if ks else w)
    return out

def make_decisions(kstar: List[int], texts: List[str], Kmax: int, step_cost: float, audio: AudioModel,
                   tau: float = 0.3, buffer_aware: bool = True, n_tok: Optional[List[int]] = None):
    """Simulate the labelling policy forward in time to obtain per-decision-point buffer features + labels.
    Think iff (j < k*) and [buffer-aware: remaining required thinking time (k*-j)*step_cost <= buffer + tau].
    Returns feats (N,Kmax+1,N_BUF), labels (N,Kmax+1) (1 = Speak), mask (N,Kmax+1), k_used (N)."""
    N = len(kstar)
    feats = torch.zeros(N, Kmax + 1, N_BUF); labels = torch.zeros(N, Kmax + 1); mask = torch.zeros(N, Kmax + 1)
    b = 0.0; k_used = []
    for i in range(N):
        j = 0
        while True:
            feats[i, j] = torch.tensor([b / 10.0, j / max(1, Kmax), i / 8.0, step_cost])
            if j >= Kmax:
                speak = True
            elif buffer_aware:
                speak = not (j < kstar[i] and (kstar[i] - j) * step_cost <= b + tau)
            else:
                speak = not (j < kstar[i])
            labels[i, j] = float(speak); mask[i, j] = 1.0
            if speak:
                break
            b = max(0.0, b - step_cost); j += 1
        k_used.append(j)
        gen = GEN_COST_PER_TOKEN * (n_tok[i] if n_tok else 6)
        b = max(0.0, b - gen) + audio.duration(texts[i])
    return feats, labels, mask, k_used

class ChunkedCollator:
    """Batches must share n_sub (use bucket_batches). Modes:
       random  : k_steps ~ U{0..Kmax} per chunk (budget-robust answer training), no gate targets
       labels  : k_steps/gate targets from k* (weak + counterfactual) via make_decisions"""
    def __init__(self, tok, Kmax: int, mode: str = "random", cf_labels: Optional[Dict] = None, step_costs=(0.05, 0.1, 0.2, 0.4),
                 audio: AudioModel = AudioModel(), buffer_aware: bool = True, tau: float = 0.3, p_random_k: float = 0.25, seed: int = 0):
        self.tok, self.Kmax, self.mode, self.cf, self.step_costs = tok, Kmax, mode, cf_labels or {}, step_costs
        self.audio, self.buffer_aware, self.tau, self.p_random_k = audio, buffer_aware, tau, p_random_k
        self.rng = random.Random(seed)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        tok = self.tok; pad = tok.pad_token_id; Kmax = self.Kmax
        N = batch[0]["n_sub"]; assert all(ex["n_sub"] == N for ex in batch)
        prompts = [tok(build_prompt(tok, ex["question"]), add_special_tokens=False)["input_ids"] for ex in batch]
        p_ids, p_mask = pad_batch(prompts, pad, left=True)
        all_chunks = [[tok(c, add_special_tokens=False)["input_ids"] for c in chunk_texts(ex)] for ex in batch]
        Lc = max(len(c) for ch in all_chunks for c in ch)
        c_ids = torch.full((len(batch), N, Lc), pad, dtype=torch.long); c_mask = torch.zeros((len(batch), N, Lc), dtype=torch.long)
        for b, ch in enumerate(all_chunks):
            for i, c in enumerate(ch):
                c_ids[b, i, :len(c)] = torch.tensor(c); c_mask[b, i, :len(c)] = 1
        out = {"prompt_ids": p_ids, "prompt_mask": p_mask, "chunk_ids": c_ids, "chunk_mask": c_mask}
        if self.mode == "random":
            out["k_steps"] = torch.randint(0, Kmax + 1, (len(batch), N), generator=None)
            return out
        feats, labels, masks, ks = [], [], [], []
        for b, ex in enumerate(batch):
            kstar = combine_kstar(weak_kstar(ex, Kmax), self.cf.get(ex["id"]), Kmax)
            sc = self.rng.choice(self.step_costs)
            f, l, m, k_used = make_decisions(kstar, ex["chunks"], Kmax, sc, self.audio, self.tau, self.buffer_aware, [len(c) for c in all_chunks[b]])
            if self.rng.random() < self.p_random_k:
                k_used = [self.rng.randint(0, Kmax) for _ in range(N)]
            feats.append(f); labels.append(l); masks.append(m); ks.append(torch.tensor(k_used))
        out.update({"gate_feats": torch.stack(feats), "gate_labels": torch.stack(labels), "gate_mask": torch.stack(masks), "k_steps": torch.stack(ks)})
        return out

def bucket_batches(rows: List[Dict], bs: int, rng: random.Random) -> List[List[Dict]]:
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_sub"], []).append(r)
    batches = []
    for n, rs in by_n.items():
        rng.shuffle(rs)
        for i in range(0, len(rs) - bs + 1, bs):
            batches.append(rs[i:i + bs])
    rng.shuffle(batches)
    return batches

def parse_chunk_answers(chunks: List[str]) -> Dict[int, str]:
    """'A2: 42' -> {2: '42'}"""
    out = {}
    for c in chunks:
        m = re.match(r"\s*A(\d+)\s*:\s*(.+)", c.strip())
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out
