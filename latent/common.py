"""Tokenization / batching helpers shared by training, eval and the streaming loop."""
import random, json, os, time
from typing import List, Dict
import torch

ANSWER_PREFIX = "Answer: "

def build_prompt(tok, question: str, system: str = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def answer_text(ex: Dict) -> str:
    return f"{ANSWER_PREFIX}{ex['answer']}<|im_end|>"

def cot_text(ex: Dict) -> str:
    return f"{ex['cot']}\n{ANSWER_PREFIX}{ex['answer']}<|im_end|>"

def pad_batch(seqs: List[List[int]], pad_id: int, left: bool):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        if left:
            ids[i, L - len(s):] = torch.tensor(s); mask[i, L - len(s):] = 1
        else:
            ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
    return ids, mask

def _prefix_len(tok, text: str, prefix: str) -> int:
    """Number of tokens of `text` that cover `prefix` (prefix must be a string-prefix of text)."""
    assert text.startswith(prefix)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    n = 0
    for (s, e) in enc["offset_mapping"]:
        if s < len(prefix):
            n += 1
    return n

class Collator:
    def __init__(self, tok, with_cot: bool = False, system: str = None, n_slots: int = 8):
        self.tok = tok; self.with_cot = with_cot; self.system = system; self.n_slots = max(1, n_slots)
        self.pad = tok.pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [self.tok(build_prompt(self.tok, ex["question"], self.system), add_special_tokens=False)["input_ids"] for ex in batch]
        answers = [self.tok(answer_text(ex), add_special_tokens=False)["input_ids"] for ex in batch]
        p_ids, p_mask = pad_batch(prompts, self.pad, left=True)
        a_ids, a_mask = pad_batch(answers, self.pad, left=False)
        # index (within the answer segment) of the token whose *output* predicts the first answer-value token
        # == number of tokens covering "Answer:" minus 1
        n_pref = _prefix_len(self.tok, answer_text(batch[0]), ANSWER_PREFIX.rstrip())
        out = {"prompt_ids": p_ids, "prompt_mask": p_mask, "answer_ids": a_ids, "answer_mask": a_mask,
               "ans_hidden_idx": torch.full((len(batch),), n_pref - 1, dtype=torch.long)}
        # verified intermediate results for auxiliary latent supervision (slot i <-> i-th intermediate)
        Mx = self.n_slots
        iv = torch.zeros(len(batch), Mx, dtype=torch.long); im = torch.zeros(len(batch), Mx)
        for b, ex in enumerate(batch):
            inter = ex.get("intermediates", [])[:Mx]
            for i, v in enumerate(inter):
                iv[b, i] = int(v); im[b, i] = 1.0
        out.update({"inter_vals": iv, "inter_mask": im})
        if self.with_cot:
            cots = [self.tok(cot_text(ex), add_special_tokens=False)["input_ids"] for ex in batch]
            c_ids, c_mask = pad_batch(cots, self.pad, left=False)
            cot_pref = [_prefix_len(self.tok, cot_text(ex), f"{ex['cot']}\n{ANSWER_PREFIX.rstrip()}") - 1 for ex in batch]
            out.update({"cot_ids": c_ids, "cot_mask": c_mask, "cot_hidden_idx": torch.tensor(cot_pref)})
        return out

def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass

class JsonLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")
    def log(self, **kw):
        kw["time"] = time.time()
        self.f.write(json.dumps(kw) + "\n"); self.f.flush()
