"""Assemble REPORT_PART2.md (readable think/speak model, SxS protocol) from the result tables."""
import subprocess, sys, os, datetime
py = sys.executable
t1 = subprocess.run([py, "latent/sxs_report.py", "--root", "results"], capture_output=True, text=True).stdout
t2 = subprocess.run([py, "latent/sxs_breakdown.py", "--root", "results"], capture_output=True, text=True).stdout
import glob, json as _json
def aime_table():
    rows = [("base_1.7B", "Qwen3-1.7B base, thinking (12k cap)"), ("dm_nothink_1.7B", "DeepMath SFT no-think (1.7B)"), ("dm_cot_1.7B", "DeepMath SFT Standard CoT (1.7B)"),
            ("dm_sxs_1.7B", "DeepMath InnerVoice SFT (1.7B)"), ("base_4B", "Qwen3-4B base, thinking (12k cap)"), ("dm_nothink_4B", "DeepMath SFT no-think (4B)"),
            ("dm_cot_4B", "DeepMath SFT Standard CoT (4B)"), ("dm_sxs_4B", "DeepMath InnerVoice SFT (4B)")]
    out = ["| method | AIME25 Avg@4 (T=1) | unfinished | ARI | ABO | AIRW | FCO | think tok | speak tok | speak blocks |", "|---|---|---|---|---|---|---|---|---|---|"]
    for run, label in rows:
        fs = glob.glob(f"results/sxs_eval/{run}/aime25/summary_*.json")
        if not fs: out.append(f"| {label} | – | – | – | – | – | – | – | – | – |"); continue
        d = _json.load(open(fs[0]))
        out.append(f"| {label} | {d['acc']:.3f} | {d['unfinished']:.2f} | {d['ARI']:.0f} | {d['ABO']:.0f} | {d['AIRW']:.0f} | {d['FCO']:.0f} | {d['n_think']:.0f} | {d['n_speak']:.0f} | {d['n_speak_blocks']:.2f} |")
    return "\n".join(out)
AIME = aime_table()

def teacher_table():
    rows = [("dm_", "DeepMath R1 solutions (dataset-provided; think median 1,237 words)", 11684),
            ("q32_", "Qwen3-32B via Bedrock, default effort (think median 374 words)", 2287),
            ("q32high_", "Qwen3-32B via Bedrock, reasoning_effort=high (think median 1,230 words)", 1550),
            ("gptoss_", "GPT-OSS-120B via Bedrock, reasoning_effort=high (think median 1,132 words) — the paper's teacher", 1312),
            ("astra_", "GPT-6 Astra via Bedrock, explicit <reasoning> section (median 85 words)", 542),
            ("self_", "Qwen3 self-distillation (own thinking traces, correct only) — not trained: only 100 (4B) / 55 (1.7B) correct traces from 600 problems at a 4k budget with HF decoding", None)]
    out = ["| teacher data | train n | model | InnerVoice AIME25 Avg@4 | think tok | FCO | blocks | Standard CoT AIME25 Avg@4 | think tok | FCO |", "|---|---|---|---|---|---|---|---|---|---|"]
    for pre, label, n in rows:
        for size in ("1.7B", "4B"):
            fs = glob.glob(f"results/sxs_eval/{pre}sxs_{size}/aime25/summary_*.json"); fc = glob.glob(f"results/sxs_eval/{pre}cot_{size}/aime25/summary_*.json")
            if not fs and not fc: continue
            def cell(f):
                if not f: return "– | – | – | –" if False else None
                d = _json.load(open(f[0])); return d
            ds, dc = (_json.load(open(fs[0])) if fs else None), (_json.load(open(fc[0])) if fc else None)
            out.append(f"| {label} | {n if n else '–'} | {size} | " + (f"{ds['acc']:.3f} | {ds['n_think']:.0f} | {ds['FCO']:.0f} | {ds['n_speak_blocks']:.2f}" if ds else "– | – | – | –") + " | " + (f"{dc['acc']:.3f} | {dc['n_think']:.0f} | {dc['FCO']:.0f}" if dc else "– | – | –") + " |")
    return "\n".join(out)
TEACHERS = teacher_table()
body = f"""# Part 2: InnerVoice — token-level disclosure routing with a two-expert last layer (SFT results under the SxS evaluation protocol)

Generated {datetime.date.today()}. All planned SFT runs and evaluations are complete.

## Summary
- **InnerVoice** = Qwen3 + LoRA, last-layer FFN -> inner (think) / outer (answer) experts chosen per token by a disclosure gate; think
  tokens readable but never spoken. SFT on SxS-style interleaved trajectories from four sources (K&K, MuSiQue, ConvFill, DeepMath).
- **Where thinking matters (AIME25)**, InnerVoice beats think-then-speak SFT at 1.7B and 4B (0.175 vs 0.075; 0.233 vs 0.183) while
  putting first content 5x earlier; where thinking does not matter at 1.7B (K&K, MuSiQue), it ties no-think and CoT SFT and only
  shortens waiting (AIRW 108 vs 177; FCO 40 vs 97). At 4B on K&K it sits between no-think (0.74) and CoT (0.83) at 0.79 with half the
  inter-update wait.
- The ablations say: fully fine-tuned experts vs LoRA experts — no robust difference; `e_think` — unnecessary; decode-time budget caps —
  destructive unless trained in. The next step is RL with correctness + disclosure-timing reward (design in the chat log / to be added).

## Model (InnerVoice)
Qwen3 + LoRA r=16 (all layers) with the **last decoder layer's FFN replaced by a two-expert MoE** (inner/think expert and outer/answer expert,
both initialised from the pretrained MLP, fully fine-tuned) selected by a **disclosure gate (router)** (`sigmoid(MLP([h_t, buffer_feats]))`) that
predicts whether the next token is private reasoning or public answer; think tokens get an additive type embedding `e_think` on input.
Think tokens are readable text but are never sent to the TTS/playback stream; they are kept in a separate stream at inference.
Loss = CE(think + answer tokens) + λ·BCE(router), switch points weighted 5x. 2000 steps x 16 sequences, lr 2e-4 / 1e-4.

## Data
Knights & Knaves (6.2k train, 2–8 people; per-step search traces) and MuSiQue (6k train, 2–4 hops; decomposition + passages), converted to
SxS-style interleaved trajectories with a programmatic entailment check (a verdict is disclosable after the last reasoning step
mentioning that person; a hop's answer after the step citing its passage). 11.9k trajectories, mean 3.9 public chunks, 176 think words.
Test: 150 examples per set (stride 4 over the official test/dev files; K&K sizes 2–7 x 25 each, 8-person puzzles not sampled).

## Protocol (from the paper)
Accuracy = exact answer checking on the user-visible stream (K&K: every verdict correct; MuSiQue: final answer EM/alias).
Content-latency proxies in token indices over the generated sequence: ARI (mean position of speak tokens), ABO (mean onset of speak
blocks), AIRW (mean think-span length before each speak block), FCO (onset of the first speak block with task-relevant content).
Greedy, plus Average@4 at temperature 1.0 (paper: Average@16). Rows: base model (thinking / no thinking), SFT Standard CoT
(same data, think-then-speak), SFT no-think, SFT SxS interleaved + router, ablations, 4B. Our extra columns: simulator TTFC and
P95 pause with 0.03 s per think token and 0.35 s per spoken word.

{t1}

{t2}

## ConvFill (voice-agent data; 1.7B InnerVoice SFT, 150 held-out turns, think cost 0.03 s/token)

Naming: **InnerVoice** = our model trained on interleaved (SxS-style) trajectories; SxS refers to the paper's data/protocol.

Trajectories from `zenglhardt/convfill-dataset`: `thoughts[i]` (Reasoner knowledge chunk) -> think block, `response[i]` -> speak
block; `<sil>` items (38% of phrases) become speak-only filler. 67k training turns, 3.9 phrases/turn. No gold answers exist, so
quality is scored by an LLM judge plus rule-based filler statistics.

| model / decode policy | judge: faithfulness | judge: coherence | judge: filler quality | filler rate | words / phrase | speak blocks / turn | think tok | FCO | sim TTFC s |
|---|---|---|---|---|---|---|---|---|---|
| InnerVoice SFT, router | 4.96 | 4.18 | 3.22 | **0.24** | 14.2 | 3.05 | 41 | 69 | **4.8** |
| InnerVoice SFT, forced no-think | 4.98 | 4.08 | 3.25 | 0.49 | 10.5 | 1.00 | 0 | 74 | 13.6 |
| SFT Standard CoT (think-then-speak) | 4.96 | 4.20 | 3.19 | 0.33 | 13.2 | 1.01 | 27 | 55 | 5.1 |
| SFT no-think | 4.96 | 4.17 | 3.17 | 0.34 | 14.5 | 1.00 | 0 | 38 | 5.7 |

Judge = Qwen3-4B (no thinking), expected value of the 1–5 digit distribution; it is given the conversation, the *reference* Reasoner
knowledge chunks and the model's public phrases. Filler rate = share of public phrases sharing no content word with the reference
knowledge. Token-F1 against the reference phrases is 0.26–0.29 for all variants (paraphrase makes string overlap uninformative).

- The judge does not separate the variants: faithfulness saturates at ~4.96 for everyone (the models rarely assert anything the
  reference knowledge does not contain), coherence 4.1–4.2, filler quality 3.2. With 150 turns these are ties.
- What does differ is *behaviour*: the InnerVoice router reproduces the dataset's think-a-chunk / voice-it rhythm (3.05 speak
  blocks per turn, 41 think tokens, e.g. think "The sport is American football." -> speak "That's American football, so it's a football
  game."), with the lowest filler rate (0.24 vs 0.33–0.34 for think-then-speak and no-think, 0.49 when the same model is forced to
  speak without thinking) and the earliest simulated first content (4.8 s vs 5.1 / 5.7 / 13.6 s). Forcing the model to skip thinking
  makes it ramble (13 short phrases per turn).
- Because the single model has no Reasoner backend, the knowledge in its think blocks is self-generated; ConvFill is useful here as a
  source of *disclosure-timing* labels (when to voice a fact vs. bridge with grounded filler), not as a knowledge benchmark.

## DeepMath -> AIME25 (scaled-down replication of the paper's setting)

SFT data: 11,686 DeepMath-103K problems whose R1 solution splits at `</think>` into private reasoning (median 1.2k words) and a
reader-facing multi-paragraph solution (6.2 paragraphs); entailment alignment by numeric/LaTeX-anchor coverage (2.3 think blocks per
example). 1000 steps (1.7B, 16 seq/step) / 800 steps (4B), LoRA + last-layer experts, 4k context, gradient checkpointing.
Evaluation: AIME25 (30 problems), Average@4 at temperature 1.0 (paper: Average@16), max 12k generated tokens (paper: 39k), router-driven
batched streaming decode; accuracy = last boxed answer in the PUBLIC stream. Paper reference (Qwen3-4B, AIME25): Base 66.3 / AIRW 17,339;
SxS SFT 38.3 / 4,035; SxS RL Final 80.0 / 8,519 — not directly comparable (their models are full-parameter trained on 330K examples with
32k context and evaluated with a 39k budget).

{AIME}

Finished-only accuracy (samples that produced a final answer within 12k tokens): InnerVoice 1.7B 0.223 (94 samples) vs Standard CoT 1.7B
0.079 (114); InnerVoice 4B 0.283 (99) vs CoT 4B 0.186 (118). Problems solved at least once in 4 samples: InnerVoice 7/30 (1.7B), 13/30 (4B);
CoT 4/30, 9/30.

- **On AIME25 thinking is necessary (no-think ≈ 0) and InnerVoice beats think-then-speak at both sizes**: 0.175 vs 0.075 (1.7B) and
  0.233 vs 0.183 (4B), with first content at token 688 / 303 instead of 3221 / 1536 and 4.3–4.7 public blocks instead of 1.7.
  The caveat is budget: InnerVoice generates about twice as many think tokens (6.9k / 5.6k vs 3.2k / 1.4k) and 18–22% of its samples hit
  the 12k cap, so part of the gain may come from thinking longer rather than from interleaving per se; a budget-matched comparison
  (cap CoT and InnerVoice at the same total tokens) is the next control.
- **Budget-matched control (1.7B, DeepMath R1 SFT, same generation cap for both)**: at a 4k cap InnerVoice 0.100 vs CoT 0.075, at 8k
  0.117 vs 0.042, at 12k 0.175 vs 0.075. InnerVoice stays ahead at every cap, but its advantage grows with the budget because it keeps
  thinking (78% / 51% / 22% of its samples are truncated at 4k / 8k / 12k) while CoT stops on its own at ~3k tokens and gains nothing
  from a larger budget. So the gain is "interleaving + willingness to think longer", not interleaving alone; a token-for-token
  separation would need CoT to be forced to think as long, which this SFT model does not do.
- **Against the base model in thinking mode** (0.258 / 0.342 with 62–66% of samples truncated at 12k; the paper gives it 39k) InnerVoice
  is 8–11 points lower after SFT, with ARI/AIRW an order of magnitude smaller (ARI 3.1–3.7k vs 10.4k). This is the paper's SFT-stage
  picture (their SxS SFT: 38.3 vs Base 66.3 on 4B) at our scale; the paper recovers and overtakes the base model with RL.
- Everything is small-sample: 30 problems x 4 samples (paper: x16), one seed; differences of a few points are not significant.

## Teacher-trace comparison on AIME25 (same split/alignment pipeline, same SFT recipe)

Teacher traces were obtained from (a) the DeepMath R1 solutions, (b) Bedrock-hosted models prompted for a Reasoning + Answer
pair (Qwen3-32B with and without `reasoning_effort=high`, GPT-OSS-120B — the SxS paper's own teacher — and GPT-6 Astra, whose
native reasoning is hidden so it writes a short explicit section), and (c) the student model's own thinking traces filtered for
correctness. Claude Fable 5.1 was not usable: Bedrock returns `content_filtered` for training-data generation, consistent with
Anthropic's usage policy, and no workaround was attempted. All sets keep only traces whose boxed answer matches DeepMath's.

{TEACHERS}

Self-distillation could not be run at this compute budget: with HF batched decoding (no vLLM possible on these hosts) Qwen3-4B and
Qwen3-1.7B solved only 100 and 55 of 600 DeepMath problems within 4k tokens — far too few to fine-tune on. It remains the preferred
recipe (the student's own reasoning style, no capability mismatch) and needs a vLLM-class generator or a larger problem pool.

Reading:
- **Teacher reasoning length dominates.** Same teacher (Qwen3-32B), only the reasoning depth changed: InnerVoice 0.025 -> 0.192 (1.7B)
  and 0.042 -> 0.208 (4B). Short-reasoning teachers (Qwen3-32B default effort, GPT-6 Astra's explicit section) teach the student to stop
  after a few hundred tokens and AIME collapses to the no-think level for both formats. 1,550 long Qwen3-32B traces match or beat 11,684
  R1 traces.
- **Style match matters at 1.7B but washes out at 4B.** GPT-OSS-120B traces are as long as R1's; the 1.7B student reaches only 0.083 (its
  reasoning is a compressed shorthand unlike the student's own style), but the 4B InnerVoice student reaches 0.183 vs 0.075 for CoT.
- **InnerVoice ≥ Standard CoT on every long-trace teacher** (R1: 0.175/0.075 and 0.233/0.183; Qwen3-32B high: 0.192/0.033 and 0.208/0.083;
  GPT-OSS: 0.083/0.058 and 0.183/0.075 for 1.7B and 4B). The CoT-format
  students consistently learn to think *less* than the InnerVoice students trained on the same traces (e.g. 2,285 vs 4,745 think tokens),
  which, together with the budget-matched control, suggests a mechanism: interleaved disclosure keeps the unfinished parts of the answer
  in view and the model keeps reasoning for them, whereas think-then-speak students terminate the single think span early. This is a
  hypothesis, not yet a controlled finding.

## RL (GRPO-style) on InnerVoice 1.7B — first round, K&K + MuSiQue

Setup (`latent/sxs_rl.py`): on-policy REINFORCE with group-normalised advantages (8 prompts x 8 rollouts per step, 150 steps,
temperature 1.0, router decisions sampled from sigmoid(logit)); policy gradient on token log-probs + disclosure-decision log-probs
(token-mean normalised); teacher-forced recompute through the same model; lr 1e-5 (LoRA) / 5e-5 (experts, router, e_think).
Rewards: stage 1 = final correct + 0.5·(final correct)·(fraction of intermediate chunks correct) − 1.0·(wrong/unsupported public
chunks) − 0.2·(unparseable public lines); stage 2 = stage 1 − 0.3·FCO/200 (content-based first-content onset). Evaluation: router
policy, greedy, 150 test examples per set (same protocol as the SFT rows).

| InnerVoice 1.7B | K&K all-correct | K&K chunk acc | K&K premature | K&K FCO | K&K AIRW | MuSiQue final | MuSiQue chunk acc | MuSiQue premature | MuSiQue FCO | sim TTFC K&K / MQ |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT start | 0.520 | 0.680 | 0.272 | 202 | 108 | 0.568 | 0.705 | 0.252 | 40 | 6.4 / 2.0 s |
| + RL stage 1 (correctness + premature) | 0.487 | 0.479 | 0.324 | 309 | 163 | **0.656** | **0.753** | **0.203** | 40 | 9.6 / 2.1 s |
| + RL stage 2 (stage 1 + FCO) | **0.593** | **0.689** | **0.219** | **197** | 117 | 0.576 | 0.683 | 0.288 | **34** | **6.3 / 1.6 s** |

- The two datasets disagree: stage 1 improves MuSiQue (+9 final, −5 premature) but degrades K&K (−3 final, longer thinking,
  fewer disclosures, +5 premature); stage 2 improves K&K (+7, −5 premature, slightly earlier) but is flat on MuSiQue (+1) with a
  higher premature rate. With 150 examples (±4 pt) and one seed, none of this is a reliable gain; it is the expected picture for
  150 GRPO steps (1,200 rollouts).
- Training-curve diagnostic: stage 1's binary reward leaves fewer groups with reward variance as training proceeds (groups with
  non-zero advantage per step: 3.6 → 2.35 of 8), so its gradient signal fades; stage 2's continuous FCO term keeps ~6 of 8 groups
  informative. A dense pacing term is therefore not only an objective but a prerequisite for GRPO to learn here.
- No decode-time policy was imposed; the router's own switching points were optimised, which is what the budget-grid result
  demanded.
- **Math RL (DeepMath prompts, reward = boxed answer correct, 60 steps x 4 prompts x 8 rollouts, rollouts capped at 3,000 new tokens)**:
  in-training reward rose from 0.22 to 0.34 while think tokens fell from 2,345 to 1,523 and first content moved from token 545 to 180 —
  but on AIME25 the policy collapsed from 0.175 to **0.058** (think tokens 6,950 -> 1,818; 0% unfinished vs 22%). The 3k rollout cap acted as
  an implicit length penalty: rollouts that had not reached the boxed answer by 3k tokens scored 0, so the policy learned to finish early,
  which is fatal on AIME where the SFT model's wins came from long reasoning. Lesson for the voice-stage reward as well: never let a budget
  cap silently zero the reward of long-but-correct trajectories (use the paper's 39k-token budgets, or score truncated rollouts by a
  continuation / partial-credit rule).
- Next: longer runs (≥500 steps), ≥2 seeds, Average@4 evaluation, rollout budgets that do not truncate correct reasoning, and the
  voice-stage rewards (stall/TTFC in seconds with randomised per-token cost).

## Reading of the results
- **Interleaving works mechanically and cuts waiting.** On K&K the InnerVoice 1.7B model emits 2.7 public blocks per puzzle with AIRW 108
  tokens vs 177 for think-then-speak CoT SFT and ~2100 for the base model in thinking mode; on MuSiQue first content appears at
  token 40 vs 97 (CoT SFT) vs 950–970 (base thinking), and simulated time-to-first-content drops from 3.7 s to 2.0 s.
- **Accuracy is preserved relative to CoT SFT, not improved**, at 1.7B (greedy: K&K 0.52 vs 0.55; MuSiQue 0.57 vs 0.59). Unlike the
  paper, where SxS SFT loses 20–25 points before RL recovers them, our SFT-stage SxS stays within noise of CoT SFT, presumably
  because the traces are two orders of magnitude shorter and highly structured.
- **At 1.7B the think stream buys no accuracy over a direct-answer model**: SFT no-think reaches 0.58 (K&K) / 0.58 (MuSiQue),
  on par with every thinking variant, with TTFC 0.4–0.9 s and no pauses. Per size, all 1.7B variants collapse to 0.2–0.4 at 6–7 people
  whether they think or not. The 4B SxS model is the only one that profits from the traces (K&K 0.79; 1.00/0.96/0.96/0.76/0.68/0.40 for
  2–7 people; MuSiQue 0.71, 0.69 on 4-hop) — but its no-think and CoT counterparts are still training, so the 4B comparison is pending.
- **4B: thinking helps on K&K and SxS sits between no-think and CoT.** Greedy / Average@4: no-think 0.740 / 0.667, Standard CoT 0.833 / 0.828,
  SxS 0.793 / 0.778. SxS gives back ~4–5 points of CoT's gain in exchange for AIRW 103 vs 175 tokens and TTFC 5.8 s vs 8.3 s (2.9 public blocks
  vs 1.4). On MuSiQue all three 4B variants tie (0.70–0.72 greedy) and SxS is simply the fastest (FCO 40 vs 101, TTFC 2.3 s vs 4.1 s). This is
  the paper's trade-off in miniature: accuracy within a few points of think-then-speak, with the waiting cut roughly in half.
- **Like-for-like LoRA-expert configuration at 1.7B** (no-think / CoT / SxS): K&K 0.613 / 0.553 / 0.593 greedy, 0.515 / 0.537 / 0.488 Average@4;
  MuSiQue 0.600 / 0.584 / 0.664 greedy, 0.602 / 0.578 / 0.550 Average@4 -> still parity.
- **`e_think` is not needed**: without the think-type input embedding the router switches identically (2.69 vs 2.73 blocks) and accuracy is
  unchanged or slightly higher (K&K 0.540 / 0.527 vs 0.520 / 0.512; MuSiQue 0.592 / 0.592 vs 0.568 / 0.542). The content of think vs
  answer text is distinctive enough on these datasets; the simpler model is preferable.
- **Ablations are within noise.** LoRA-only experts look better under greedy decoding (0.59 / 0.66 vs 0.52 / 0.57) but not under
  Average@4 sampling (K&K 0.488 vs 0.512 for fully fine-tuned experts; no-think 0.508). With 150 examples (±4 pt) and 600 samples (±2 pt)
  none of the 1.7B SFT variants is separable from the others. The 30% truncated-think mix hurts under sampling (0.392).
- **Average@4 (T=1.0) confirms parity at 1.7B**: K&K no-think 0.508 / CoT 0.533 / SxS 0.512 / LoRA-experts 0.488; MuSiQue 0.600 / 0.566 /
  0.542 / 0.550. The greedy advantage of LoRA-only experts does not survive sampling. 4B SxS: 0.778 (K&K) / 0.686 (MuSiQue).
- **Budget grid (1.7B)**: capping think tokens per block at 32/64/128 keeps greedy accuracy at 0.51–0.54 (K&K) / 0.55–0.57 (MuSiQue) while
  shrinking AIRW to 28–75, but at the tightest cap the premature-commitment rate rises (0.71 at 32 vs 0.27 for the free router): the
  router's own stopping points are the least premature. Policies that *force* mode switches the model never saw in training break it:
  "think exactly 64 tokens then answer" and the STITCH-like "think only while audio is buffered" score 0.00 on K&K (the forced switch
  lands mid-sentence and the answer expert continues the thought), and forcing no thinking on the SxS model collapses it (0.21 vs 0.58
  for the dedicated no-think SFT model). A buffer-aware policy therefore has to be trained in (labels or RL), not imposed at decode time.
- The base models in thinking mode are poorly suited to streaming on these tasks: one speak block after ~1000–2100 think tokens and
  8–19% of puzzles never finish thinking within the budget.
- **Comparison with the paper (Qwen3-4B, AIME25)**: Base 66.3 / AIRW 17,339 -> SxS SFT 38.3 / 4,035 -> SxS RL Final 80.0 / 8,519.
  The paper's gains come from RL; our SFT-stage picture (accuracy ≈ CoT SFT, AIRW 1.6x shorter) is the expected starting point for it.
"""
open("REPORT_PART2.md", "w").write(body); print("written", len(body.split()), "words")
