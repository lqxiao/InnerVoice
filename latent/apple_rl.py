"""Apple-style (Xie et al. 2025) token-controlled interleaved reasoning: prompt templates, tag parsing, rule-based rewards.
Used by latent/sxs_rl_engine.py in --plain mode and by latent/eval_engine_apple.py."""
import re
from latent.sxs_eval import score_kk, score_mq, norm

SYS_INTERLEAVE = ("You are a helpful assistant. You reason through problems step by step before providing an answer. You conduct your "
                  "reasoning within <think></think> and share partial answers within <answer></answer> as soon as you become confident about "
                  "the intermediate results. You continue this pattern of <think></think> <answer></answer> <think></think> <answer></answer> "
                  "until you reach the final answer.")
SYS_THINK_ANSWER = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first "
                    "thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and "
                    "answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process "
                    "here </think> <answer> answer here </answer>.")
SYS_DIRECT = "You are a helpful assistant. Answer the question directly inside <answer></answer> without any explanation."
SYS_COT = ("You are a helpful assistant. Think step by step, then give the final answer inside <answer></answer>.")
TEMPLATES = {"interleave": SYS_INTERLEAVE, "think_answer": SYS_THINK_ANSWER, "direct": SYS_DIRECT, "cot": SYS_COT}
TASK_HINT = {"kk": " Give each inhabitant's identity as '<Name> is a knight.' or '<Name> is a knave.' (one per line).",
             "musique": " State each intermediate finding on its own line and finish with 'Answer: <final answer>'."}

TAG_RE = re.compile(r"<(/?)(think|answer)>")

def parse_tags(text):
    """Return (segments, format_ok). segments = list of (mode, text) with mode in {'think','answer'}; format_ok = tags alternate
    properly (every open closed, no nesting, think/answer alternate, ends with a closed answer)."""
    segs, pos, open_tag, ok = [], 0, None, True
    for m in TAG_RE.finditer(text):
        closing, tag = m.group(1) == "/", m.group(2)
        if not closing:
            if open_tag is not None: ok = False
            open_tag, pos = tag, m.end()
        else:
            if open_tag != tag: ok = False
            else: segs.append((tag, text[pos:m.start()].strip())); open_tag = None
    if open_tag is not None: ok = False
    if not segs or segs[-1][0] != "answer": ok = False
    for a, b in zip(segs, segs[1:]):
        if a[0] == b[0]: ok = False
    return segs, ok

def public_lines(segs):
    out = []
    for mode, t in segs:
        if mode == "answer": out += [l.strip() for l in t.split("\n") if l.strip()]
    return out

def token_types_from_tags(tok, ids):
    """Per-token types from the decoded text: 1 = inside <think>, 0 = inside <answer>, 2 = tag/other (excluded from both streams).
    Also returns the index of the first answer-content token (1-indexed) or len+1."""
    text = tok.decode(ids, skip_special_tokens=True); offsets = []
    # map tokens -> char spans by incremental decode
    cur = ""; spans = []
    for i in range(len(ids)):
        nxt = tok.decode(ids[: i + 1], skip_special_tokens=True); spans.append((len(cur), len(nxt))); cur = nxt
    mode_at = [2] * len(text); open_tag = None; pos = 0
    for m in TAG_RE.finditer(text):
        if m.group(1) != "/": open_tag, pos = m.group(2), m.end()
        elif open_tag == m.group(2):
            v = 1 if open_tag == "think" else 0
            for c in range(pos, m.start()): mode_at[c] = v
            open_tag = None
    types = []
    for s, e in spans:
        vs = [mode_at[c] for c in range(s, min(e, len(mode_at))) if mode_at[c] != 2]
        types.append(2 if not vs else (1 if sum(vs) * 2 >= len(vs) else 0))
    first_ans = next((i + 1 for i, t in enumerate(types) if t == 0), len(types) + 1)
    return types, first_ans

def score_final(ex, lines):
    """Final-answer exact match (normalised) on the LAST answer segment's lines; intermediate correctness per gold item."""
    if ex["source"] == "kk":
        oks, final_ok = score_kk(lines, ex["chunk_checks"]); parse_ok = any(re.match(r"\s*([A-Z][a-z]+) is a (knight|knave)", l) for l in lines)
        return final_ok, oks, parse_ok
    oks, final_ok = score_mq(lines, ex["chunk_checks"]); parse_ok = any(l.lower().startswith("answer") for l in lines) or bool(lines)
    return final_ok, oks, parse_ok

def apple_reward(ex, segs, format_ok, ir_mode, gate_open, lam_f=1.0, lam_a=1.0, r_base=0.5):
    """r = lam_f*(+1/-1 format) + lam_a*(+2 / -1.5 / -2) [only when format ok] + conditional time-discounted intermediate reward."""
    lines = public_lines(segs)
    r = lam_f * (1.0 if format_ok else -1.0); info = {"format_ok": float(format_ok)}
    final_ok, oks, parse_ok = score_final(ex, lines)
    if format_ok:
        r += lam_a * (2.0 if final_ok else (-1.5 if parse_ok else -2.0))
    info.update(final=float(final_ok), parse_ok=float(parse_ok), n_answer_segs=float(sum(1 for m, _ in segs if m == "answer")))
    inter = oks[:-1] if ex["source"] == "musique" else oks   # K&K: every verdict is an intermediate conclusion; MuSiQue: hops before the final
    info["inter_frac"] = (sum(inter) / len(inter)) if inter else 0.0
    r_ir = 0.0
    if ir_mode and format_ok and final_ok and gate_open and inter:
        if all(inter): r_ir = r_base
        else: r_ir = r_base * sum(1.0 / (k + 1) for k, ok in enumerate(inter) if ok) / len(inter)
    info["r_ir"] = r_ir
    return r + r_ir, info
