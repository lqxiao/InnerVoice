"""Architecture diagram (static PNG) for REPORT.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {"lm": "#DCE6F7", "lat": "#F6E3D3", "gate": "#DFF0E3", "tok": "#EEEEEE", "edge": "#444", "txt": "#222", "acc": "#1F5FD6", "lat_e": "#B8541A", "gate_e": "#2A8C4A"}
fig, ax = plt.subplots(figsize=(15, 8.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 8.4); ax.axis("off")

def box(x, y, w, h, text, fc, ec="#666", fs=9.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["txt"], fontweight="bold" if bold else "normal", linespacing=1.4)

def arrow(p, q, color="#444", lw=1.5, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))

# ---- sequence row (bottom): what sits in the KV cache
y0 = 0.55
box(0.4, y0, 3.4, 0.8, "prompt / history tokens\n(chat template, left-padded)", C["tok"])
box(4.1, y0, 2.5, 0.8, "M = 8 latent soft tokens\ninject(z_K) · scale", C["lat"], C["lat_e"])
box(6.9, y0, 2.9, 0.8, "public answer chunk\n\"A1: 42\\n\"  (teacher-forced / decoded)", C["tok"])
box(10.1, y0, 2.0, 0.8, "latent tokens\nfor chunk 2", C["lat"], C["lat_e"], ls="--")
box(12.4, y0, 2.2, 0.8, "chunk 2 …", C["tok"], ls="--")
ax.text(7.5, 0.15, "positions in ONE causal KV cache  →  answer tokens attend to the latent tokens like any earlier position (no extra attention path)", ha="center", fontsize=9, color="#555")

# ---- Qwen3 + LoRA block
box(0.4, 2.0, 14.2, 1.1, "Qwen3-1.7B / 4B  (bf16, frozen)  +  LoRA r=16 on q,k,v,o,gate,up,down  —  shared by every pass: prompt encode, latent passes (lm mode), answer decode", C["lm"], "#1F5FD6", fs=10, bold=True)
for x in (2.1, 5.35, 8.35, 11.1, 13.5):
    arrow((x, y0 + 0.8), (x, 2.0), color="#777", lw=1.2, style="-|>")

# ---- context extraction
box(0.4, 3.6, 3.4, 1.0, "context  ctx\nfinal-layer hidden states of\nprompt + previous segments", C["lm"], "#1F5FD6")
arrow((2.1, 3.1), (2.1, 3.6), color="#1F5FD6")

# ---- latent recurrence block
box(4.6, 3.45, 6.0, 2.55, "", C["lat"], C["lat_e"])
ax.text(7.6, 5.75, "recurrent latent module  (fp32, weights shared across steps)", ha="center", fontsize=10.5, fontweight="bold", color=C["lat_e"])
box(4.85, 4.75, 1.7, 0.7, "z₀\nlearned  |  LN(W·h_last)", "#fff", C["lat_e"], fs=8.5)
box(7.0, 4.75, 3.35, 0.7, "z_{k+1} = LayerNorm( z_k + F(z_k, ctx) )", "#fff", C["lat_e"], fs=10, bold=True)
box(4.85, 3.6, 2.6, 0.95, "F = adapter\nslot self-attn → cross-attn(ctx) → MLP\n(~67M params)", "#fff", C["lat_e"], fs=8.3)
box(7.65, 3.6, 2.7, 0.95, "F = LM  (COCONUT-style)\nfeed inject(z_k) through Qwen3 w/ cache,\nz_{k+1} = LN(z_k + W·h_k)", "#fff", C["lat_e"], fs=8.3)
arrow((6.55, 5.1), (7.0, 5.1), color=C["lat_e"])
# recurrence loop arrow
arrow((10.35, 5.1), (10.35, 5.1), color=C["lat_e"])
ax.add_patch(FancyArrowPatch((10.1, 5.47), (7.25, 5.47), arrowstyle="-|>", mutation_scale=14, color=C["lat_e"], lw=1.5, connectionstyle="arc3,rad=0.28"))
ax.text(7.6, 6.13, "× K steps   (K = 0,1,2,4,8 fixed in Phase 1; chosen by the gate in Phase 2)", ha="center", fontsize=8.5, color=C["lat_e"])
arrow((3.8, 4.1), (4.85, 4.1), color="#1F5FD6")             # ctx -> adapter F
arrow((8.9, 3.1), (8.9, 3.6), color="#1F5FD6", ls="--")      # LM <-> F=LM
# z_K -> inject
arrow((5.35, 3.45), (5.35, y0 + 0.8), color=C["lat_e"], lw=1.8)
ax.text(5.55, 1.6, "z_K → Linear → ×scale\n(answer-conditioning adapter)", fontsize=8.5, color=C["lat_e"], va="center")

# ---- gate block
box(11.2, 3.45, 3.4, 2.55, "", C["gate"], C["gate_e"])
ax.text(12.9, 5.75, "Think / Speak gate", ha="center", fontsize=10.5, fontweight="bold", color=C["gate_e"])
box(11.4, 4.55, 3.0, 0.95, "p_emit = σ( MLP[ LN(ctx_last),\n LN(mean_slots z), buffer_feats ] )", "#fff", C["gate_e"], fs=8.8)
box(11.4, 3.6, 3.0, 0.8, "buffer_feats: remaining audio s,\nsteps taken, chunk idx, step cost", "#fff", C["gate_e"], fs=8.3)
ax.text(12.9, 6.13, "0 = Think: one more latent step\n1 = Speak: decode next chunk", ha="center", va="bottom", fontsize=8.5, color=C["gate_e"])
arrow((10.6, 5.1), (11.4, 5.1), color=C["lat_e"])            # z -> gate
ax.add_patch(FancyArrowPatch((3.8, 3.75), (11.4, 4.0), arrowstyle="-|>", mutation_scale=14, color="#1F5FD6", lw=1.2, connectionstyle="arc3,rad=0.22"))
ax.text(7.6, 3.22, "ctx_last → gate", fontsize=8, color="#1F5FD6", ha="center")

# ---- losses
box(0.4, 6.95, 6.2, 1.2, "Phase 1 loss:  CE(answer tokens)   [+ aux digit-CE on slots vs verified intermediates]\n[+ CODI: align hidden state at 'Answer:' with explicit-CoT teacher branch]", "#fff", "#888", fs=8.8)
box(7.0, 6.95, 7.6, 1.2, "Phase 2 loss:  Answer_CE + λ · Gate_BCE   (labels: weak teacher-trajectory k* + counterfactual budget k*, buffer rule)\nPhase 3:  REINFORCE / GRPO on gate (+answer) with simulator reward: acc − TTFC − P95 stall − GPU − premature", "#fff", "#888", fs=8.8)
fig.suptitle("Streaming latent reasoning model: Qwen3 + LoRA + recurrent latent module + Think/Speak gate", fontsize=13, x=0.02, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig("results/architecture.png", dpi=150); print("saved")
