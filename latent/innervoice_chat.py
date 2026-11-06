"""InnerVoice interactive chat with streaming display.
Think tokens (inner voice) are streamed dimmed on their own lines prefixed with 💭 ; spoken tokens (outer voice) are streamed in
green as they are generated. Multi-turn history is kept in the ConvFill prompt format (User: / Assistant:).

  python latent/innervoice_chat.py --ckpt results/sxs/convfill_1.7B/ckpt            # ConvFill-trained 1.7B
  python latent/innervoice_chat.py --ckpt results/sxs/main_1.7B/ckpt --style plain   # K&K/MuSiQue-trained (question as prompt)
Options: --hide_think  (show only the public stream)   --max_new 300
"""
import argparse, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.sxs_model import SxSModel, SxSConfig

DIM, GREEN, BOLD, RESET = "\033[2m", "\033[32m", "\033[1m", "\033[0m"

@torch.no_grad()
def chat_turn(model, prompt_text, max_new=300, hide_think=False, greedy=True):
    tok = model.tok
    prompt = tok.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]], device=model.device)
    mask = torch.ones_like(ids); is_think = torch.zeros_like(ids)
    model.moe.route = None; model.moe.buf = None
    t0 = time.perf_counter()
    out = model.lm(inputs_embeds=model.inputs_embeds(ids, is_think), attention_mask=mask, use_cache=True, return_dict=True)
    cache = out.past_key_values; cur = ids.shape[1]
    router_logit = model.moe.router_logits[0, -1].item(); last_logits = out.logits[:, -1]
    prev_mode, buf_think, buf_speak, spoken, n_think, n_speak, t_first = None, [], [], [], 0, 0, None
    for step in range(max_new):
        mode = "think" if router_logit > 0 else "speak"
        nxt = last_logits.argmax(-1) if greedy else torch.multinomial(torch.softmax(last_logits.float(), -1), 1).squeeze(1)
        tid = int(nxt.item())
        if tid in (model.eos_id, tok.eos_token_id): break
        piece = tok.decode([tid])
        if mode != prev_mode:
            if prev_mode is not None: sys.stdout.write(RESET + "\n")
            sys.stdout.write((DIM + "💭 ") if mode == "think" else (GREEN + BOLD + "🔊 "))
        if mode == "think":
            n_think += 1; buf_think.append(tid)
            if not hide_think: sys.stdout.write(piece.replace("\n", " "))
        else:
            n_speak += 1; buf_speak.append(tid)
            if t_first is None and len(buf_speak) >= 3: t_first = time.perf_counter() - t0
            sys.stdout.write(piece if "\n" not in piece else piece.replace("\n", "\n" + GREEN + BOLD + "🔊 "))
        sys.stdout.flush(); prev_mode = mode
        mask = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype, device=model.device)], 1)
        out = model.lm(inputs_embeds=model.inputs_embeds(nxt[:, None], torch.tensor([[int(mode == "think")]], device=model.device)), attention_mask=mask,
                       past_key_values=cache, position_ids=torch.tensor([[cur]], device=model.device), use_cache=True, return_dict=True)
        cache = out.past_key_values; cur += 1
        router_logit = model.moe.router_logits[0, -1].item(); last_logits = out.logits[:, -1]
    sys.stdout.write(RESET + "\n")
    dt = time.perf_counter() - t0
    public = tok.decode(buf_speak, skip_special_tokens=True).strip()
    print(f"{DIM}[{n_think} think tok | {n_speak} speak tok | first speech at {t_first if t_first else dt:.2f}s | total {dt:.2f}s]{RESET}")
    return public

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--ckpt", default="results/sxs/convfill_1.7B/ckpt")
    ap.add_argument("--style", default="convfill", choices=["convfill", "plain"]); ap.add_argument("--hide_think", action="store_true")
    ap.add_argument("--max_new", type=int, default=300); ap.add_argument("--sample", action="store_true"); ap.add_argument("--no_expert_ft", action="store_true"); ap.add_argument("--no_e_think", action="store_true")
    a = ap.parse_args()
    model = SxSModel(SxSConfig(model_name=a.model, expert_full_ft=not a.no_expert_ft, use_e_think=not a.no_e_think), device="cuda"); model.load(a.ckpt); model.eval()
    print(f"InnerVoice ready ({a.ckpt}). 💭 = inner voice (private), 🔊 = spoken. Type 'quit' to exit, '/reset' to clear history.\n")
    hist = []
    while True:
        try: user = input(BOLD + "You: " + RESET).strip()
        except EOFError: break
        if not user: continue
        if user == "quit": break
        if user == "/reset": hist = []; print("(history cleared)"); continue
        prompt = "\n".join(hist + [f"User: {user}"]) if a.style == "convfill" else user
        public = chat_turn(model, prompt, a.max_new, a.hide_think, greedy=not a.sample)
        if a.style == "convfill": hist += [f"User: {user}", "Assistant: " + " ".join(public.split("\n"))]
        print()

if __name__ == "__main__":
    main()
