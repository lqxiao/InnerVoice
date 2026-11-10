"""Generate the paper's LaTeX tables from the result files (never hand-edit numbers).
Sources: results/**/summary_*.json (via REPORT_PART2.md tables, which latent/build_report_part2.py generates from them) and
results/rl/*/eval.jsonl + aime25_avg8_final.json. Output: paper/tables/*.tex"""
import json, os, re, glob
R = "REPORT_PART2.md"; OUT = "paper/tables"; os.makedirs(OUT, exist_ok=True)
md = open(R).read().splitlines()

def md_table(after_substring, nth=0):
    """Return (header, rows) of the nth markdown table that appears after a line containing `after_substring`."""
    i = next(k for k, l in enumerate(md) if after_substring in l)
    tables, cur = [], []
    for l in md[i + 1:]:
        if l.startswith("|"): cur.append(l)
        elif cur: tables.append(cur); cur = []
        if len(tables) > nth: break
    t = tables[nth]; split = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
    return split(t[0]), [split(l) for l in t[2:]]

def esc(s):
    s = s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("≈", "$\\approx$").replace("±", "$\\pm$").replace("–", "--").replace("—", "---").replace("->", "$\\rightarrow$")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    return s

def latex_table(header, rows, cols, colnames=None, fmt="l" , caption="", label="", size="\\small", widths=None):
    idx = [header.index(c) for c in cols]; colnames = colnames or cols
    spec = fmt + "r" * (len(cols) - 1)
    out = [f"\\begin{{table*}}[t]\\centering\\footnotesize\\resizebox{{\\textwidth}}{{!}}{{%", f"\\begin{{tabular}}{{{spec}}}\\toprule", " & ".join(esc(c) for c in colnames) + " \\\\\\midrule"]
    for r in rows:
        if len(r) < len(header): continue
        out.append(" & ".join(esc(r[j]) for j in idx) + " \\\\")
    out += ["\\bottomrule\\end{tabular}}", f"\\caption{{{caption}}}\\label{{{label}}}", "\\end{table*}"]
    return "\n".join(out)

def write(name, text): open(f"{OUT}/{name}.tex", "w").write(text + "\n"); print("wrote", name)

# ---- K&K / MuSiQue main table (selected rows)
H, rows = md_table("### sxs_test_kk")
keep = ["Qwen3-1.7B base, thinking mode", "Qwen3-1.7B base, no thinking", "SFT no-think (1.7B)", "SFT Standard CoT, think-then-speak (1.7B)", "InnerVoice SFT (1.7B)",
        "ablation: no e_think input embedding (1.7B)", "Qwen3-4B base, thinking mode", "SFT no-think (4B)", "SFT Standard CoT, think-then-speak (4B)", "InnerVoice SFT (4B)"]
rows_kk = [r for r in rows if r[0].strip() in keep]
H2, rows2 = md_table("### sxs_test_mq")
rows_mq = [r for r in rows2 if r[0].strip() in keep]
def merge(rk, rm):
    out = []
    for a in rk:
        b = next((x for x in rm if x[0].strip() == a[0].strip()), None)
        if b is None: continue
        name = a[0].strip().replace("ablation: ", "").replace(", think-then-speak", "")
        out.append([name, a[H.index("final acc (greedy)")], a[H.index("final acc (Avg@4, T=1)")], a[H.index("AIRW")], a[H.index("FCO")], a[H.index("speak blocks")],
                    b[H2.index("final acc (greedy)")], b[H2.index("final acc (Avg@4, T=1)")], b[H2.index("AIRW")], b[H2.index("FCO")], b[H2.index("speak blocks")]])
    return out
mg = merge(rows_kk, rows_mq)
hdr = ["method", "K\\&K acc", "K\\&K Avg@4", "AIRW", "FCO", "blocks", "MuSiQue acc", "MuSiQue Avg@4", "AIRW", "FCO", "blocks"]
body = "\n".join(" & ".join(esc(c) if i == 0 else c for i, c in enumerate(r)) + " \\\\" for r in mg)
write("kk_mq", "\\begin{table*}[t]\\centering\\footnotesize\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "r" * 10 + "}\\toprule\n & \\multicolumn{5}{c}{Knights \\& Knaves (150)} & \\multicolumn{5}{c}{MuSiQue (150)} \\\\\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n" +
      " & ".join(hdr) + " \\\\\\midrule\n" + body + "\n\\bottomrule\\end{tabular}}\n\\caption{SFT-stage results on K\\&K and MuSiQue (150 test examples each, $\\pm$4 pt). acc = greedy accuracy on the public stream (K\\&K: all verdicts correct); Avg@4 at $T{=}1$. AIRW/FCO in token indices; blocks = public blocks per example.}\\label{tab:kkmq}\n\\end{table*}")

# ---- AIME SFT table
H, rows = md_table("| method | AIME25 Avg@4 (T=1) | unfinished |")
# md_table expects the substring in a line BEFORE the table; re-find directly
i = next(k for k, l in enumerate(md) if l.startswith("| method | AIME25 Avg@4 (T=1) | unfinished |"))
H = [c.strip() for c in md[i].strip().strip("|").split("|")]; rows = []
for l in md[i + 2:]:
    if not l.startswith("|"): break
    rows.append([c.strip() for c in l.strip().strip("|").split("|")])
write("aime_sft", latex_table(H, rows, ["method", "AIME25 Avg@4 (T=1)", "unfinished", "AIRW", "FCO", "think tok", "speak blocks"],
      ["method", "Avg@4", "unfinished", "AIRW", "FCO", "think tok", "blocks"], caption="DeepMath-R1 SFT evaluated on AIME25 (30 problems $\\times$ 4 samples, $T{=}1$, 12k-token cap; $\\pm$8 pt). Unfinished samples (no final answer within the cap) count as wrong.", label="tab:aime_sft"))

# ---- teacher table
i = next(k for k, l in enumerate(md) if l.startswith("| teacher data | train n | model |"))
H = [c.strip() for c in md[i].strip().strip("|").split("|")]; rows = []
for l in md[i + 2:]:
    if not l.startswith("|"): break
    rows.append([c.strip() for c in l.strip().strip("|").split("|")])
for r in rows: r[0] = r[0].replace(" via Bedrock", "").replace(" — the paper's teacher", " (paper's teacher)")
write("teacher", "\\begin{table*}[t]\\centering\\footnotesize\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llr" + "r" * 7 + "}\\toprule\nteacher traces (think length) & n & model & \\multicolumn{3}{c}{InnerVoice SFT} & \\multicolumn{3}{c}{Standard CoT SFT} \\\\\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n & & & Avg@4 & think & FCO & Avg@4 & think & FCO \\\\\\midrule\n" +
      "\n".join(" & ".join([esc(r[0]), r[1], r[2], r[3], r[4], r[5], r[7], r[8], r[9]]) + " \\\\" for r in rows) +
      "\n\\bottomrule\\end{tabular}}\n\\caption{Teacher-trace comparison on AIME25 (same split/alignment pipeline and SFT recipe; Avg@4, 12k cap). Long private reasoning in the teacher traces is what transfers; InnerVoice $\\geq$ think-then-speak for every long-trace teacher.}\\label{tab:teacher}\n\\end{table*}")

# ---- RL round 3 table (from eval.jsonl)
runs = [("q32high_4B_engine", "R1 (FCO bonus, SFT prompts)"), ("q32high_4B_engine_v3", "R3 (gated $Q$, KL, unseen pool)"), ("q32high_4B_engine_v4", "R4 (+flicker pen., router KL 0.5)"),
        ("q32high_4B_engine_v5", "R5 (+sampled routing), p4 32$\\times$8"), ("gl_job7_v5_32x16", "R5, Greenland 32$\\times$16")]
lines = []
for run, lab in runs:
    f = f"results/rl/{run}/eval.jsonl"
    if not os.path.exists(f): continue
    rows = [json.loads(l) for l in open(f)]
    f0 = f"results/rl/{run}_crashed/eval.jsonl"
    if os.path.exists(f0): rows = [json.loads(l) for l in open(f0) if '"step": 0' in l] + rows
    cells = []
    for d in rows:
        cells.append(f"{d['aime_acc']:.3f}")
    steps = [d["step"] for d in rows]
    last = rows[-1]
    lines.append((lab, steps, cells, last))
hdr_steps = sorted(set(s for _, st, _, _ in lines for s in st))
out = ["\\begin{table*}[t]\\centering\\footnotesize\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + "r" * len(hdr_steps) + "rrr}\\toprule", "reward / run & " + " & ".join(f"@{s}" for s in hdr_steps) + " & FCO & blocks & short \\\\\\midrule"]
for lab, steps, cells, last in lines:
    row = [lab] + [cells[steps.index(s)] if s in steps else "--" for s in hdr_steps] + [f"{last['aime_FCO']:.0f}", f"{last['aime_n_speak_blocks']:.1f}", (f"{last['aime_n_short']:.1f}" if 'aime_n_short' in last else '--')]
    out.append(" & ".join(row) + " \\\\")
out += ["\\bottomrule\\end{tabular}}", "\\caption{RL round 3 on Qwen3-4B InnerVoice: AIME25 Average@4 (n=120, $\\pm$8 pt) at each evaluation step; the last three columns describe the final evaluated checkpoint (FCO, public blocks, short blocks $<$4 tokens = router flicker). R1 and R3--R4 end in degenerate disclosure (speak-everything or flicker) despite rising or flat accuracy.}\\label{tab:rl}", "\\end{table*}"]
write("rl_round3", "\n".join(out))

# ---- final Avg@8 table
rows = []
for f, lab in [("results/rl/q32high_4B_engine_v5/aime25_avg8_final.json", "InnerVoice RL R5, p4 (300 steps, 32$\\times$8, 8k)"), ("results/rl/gl_job7_v5_32x16/aime25_avg8_final.json", "InnerVoice RL R5, Greenland (480 steps, 32$\\times$16, 12k)")]:
    if os.path.exists(f):
        d = json.load(open(f)); rows.append([lab, f"{d['acc']:.3f}", f"{d['unfinished']:.2f}", f"{d['FCO']:.0f}", f"{d['AIRW']:.0f}", f"{d['n_think']:.0f}", f"{d['n_speak_blocks']:.1f}", f"{d['n_short']:.1f}"])
base = json.load(open("results/sxs_eval/base_4B/aime25/summary_base_think_avg4.json")); sft = json.load(open("results/sxs_eval/q32high_sxs_4B/aime25/summary_sxs_avg4.json")); cot = json.load(open("results/sxs_eval/q32high_cot_4B/aime25/summary_sxs_avg4.json"))
pre = [["Qwen3-4B base, thinking mode (Avg@4)", f"{base['acc']:.3f}", f"{base['unfinished']:.2f}", f"{base['FCO']:.0f}", f"{base['AIRW']:.0f}", f"{base['n_think']:.0f}", f"{base['n_speak_blocks']:.1f}", "--"],
       ["Standard CoT SFT (Avg@4)", f"{cot['acc']:.3f}", f"{cot['unfinished']:.2f}", f"{cot['FCO']:.0f}", f"{cot['AIRW']:.0f}", f"{cot['n_think']:.0f}", f"{cot['n_speak_blocks']:.1f}", "--"],
       ["InnerVoice SFT, e\\_think variant (Avg@4)", f"{sft['acc']:.3f}", f"{sft['unfinished']:.2f}", f"{sft['FCO']:.0f}", f"{sft['AIRW']:.0f}", f"{sft['n_think']:.0f}", f"{sft['n_speak_blocks']:.1f}", "--"]]
out = ["\\begin{table*}[t]\\centering\\footnotesize\\resizebox{0.85\\textwidth}{!}{%", "\\begin{tabular}{lrrrrrrr}\\toprule", "model & acc & unfin. & FCO & AIRW & think & blocks & short \\\\\\midrule"] + [" & ".join(r) + " \\\\" for r in pre] + ["\\midrule"] + [" & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule\\end{tabular}}",
      "\\caption{Final RL checkpoints on AIME25 at \\textbf{Average@8} (30 problems $\\times$ 8 samples, n=240, $\\pm$5.6 pt; 12k cap), with the SFT and base rows (Avg@4, n=120) for reference. Unfinished samples count as wrong.}\\label{tab:final}", "\\end{table*}"]
write("final_avg8", "\n".join(out))

# ---- ConvFill
i = next(k for k, l in enumerate(md) if l.startswith("| model / decode policy | judge: faithfulness"))
H = [c.strip() for c in md[i].strip().strip("|").split("|")]; rows = []
for l in md[i + 2:]:
    if not l.startswith("|"): break
    rows.append([c.strip() for c in l.strip().strip("|").split("|")])
write("convfill", latex_table(H, rows, ["model / decode policy", "judge: faithfulness", "judge: coherence", "filler rate", "speak blocks / turn", "think tok", "FCO", "sim TTFC s"],
      ["model / policy", "faithful.", "coher.", "filler rate", "blocks/turn", "think", "FCO", "TTFC (s)"], caption="ConvFill (150 held-out voice-agent turns, 1.7B). Judge scores (1--5, Qwen3-4B judge) tie; behaviour differs: the router reproduces the think-a-chunk / voice-it rhythm with the lowest filler rate and earliest simulated first content (0.03 s per think token, 0.35 s per spoken word).", label="tab:convfill"))

# ---- budget grid
i = next(k for k, l in enumerate(md) if l.startswith("| policy | final acc | all chunks ok | think tok |"))
H = [c.strip() for c in md[i].strip().strip("|").split("|")]; rows = []
for l in md[i + 2:]:
    if not l.startswith("|"): break
    rows.append([c.strip() for c in l.strip().strip("|").split("|")])
write("budget", latex_table(H, rows, ["policy", "final acc", "think tok", "AIRW", "FCO", "sim TTFC s", "P50 / P95 pause s", "premature"],
      ["decode policy", "acc", "think", "AIRW", "FCO", "TTFC (s)", "P50/P95 pause (s)", "premature"], caption="Decode-time disclosure policies imposed on the InnerVoice 1.7B SFT model (K\\&K, greedy). Caps on think tokens per block trade waiting for premature commitments; forced switches the model never saw in training (fixed-64, STITCH-like stitch-64) break it.", label="tab:budget"))
