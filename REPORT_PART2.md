# Part 2: InnerVoice — token-level disclosure routing with a two-expert last layer (SFT results under the SxS evaluation protocol)

Generated 2026-11-09. All planned SFT runs and evaluations are complete.

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


### sxs_test_kk  (150 examples, stride 4; token-index metrics follow the SxS paper: ARI / ABO / AIRW; FCO = first-content onset)

| method | final acc (greedy) | final acc (Avg@4, T=1) | all chunks ok | ARI | ABO | AIRW | FCO | think tok | speak blocks | sim TTFC s / P95 pause s |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B base, thinking mode | 0.413 | – | 0.413 | 2274 | 2112 | 2111 | 2114 | 2111 | 0.81 | – |
| Qwen3-1.7B base, no thinking | 0.273 | – | 0.273 | 368 | 1 | 0 | 619 | 0 | 1.00 | – |
| Qwen3-4B base, thinking mode | 0.333 | – | 0.333 | 2158 | 2047 | 2046 | 2059 | 2046 | 0.81 | – |
| Qwen3-4B base, no thinking | 0.000 | – | 0.000 | 100 | 1 | 0 | 40 | 0 | 1.00 | – |
| SFT no-think (1.7B) | 0.580 | 0.508 | 0.580 | 15 | 1 | 0 | 1 | 0 | 1.00 | 0.42 / 0.00 |
| SFT Standard CoT, think-then-speak (1.7B) | 0.547 | 0.533 | 0.547 | 257 | 230 | 177 | 270 | 268 | 1.40 | 8.46 / 5.33 |
| InnerVoice SFT (1.7B) | 0.520 | 0.512 | 0.520 | 253 | 229 | 108 | 202 | 283 | 2.73 | 6.45 / 5.70 |
|   ablation: no e_think input embedding (1.7B) | 0.540 | 0.527 | 0.540 | 243 | 221 | 106 | 189 | 270 | 2.69 | 6.06 / 3.43 |
|   ablation: 30% truncated-think mix (1.7B) | 0.547 | 0.392 | 0.547 | 257 | 234 | 114 | 221 | 289 | 2.67 | 7.01 / 5.31 |
| SFT no-think, LoRA experts (1.7B) | 0.613 | 0.515 | 0.613 | 15 | 1 | 0 | 1 | 0 | 1.00 | 0.41 / 0.00 |
| SFT Standard CoT, LoRA experts (1.7B) | 0.553 | 0.537 | 0.553 | 261 | 232 | 173 | 268 | 267 | 1.45 | 8.53 / 3.73 |
| InnerVoice SFT, LoRA experts (1.7B) | 0.593 | 0.488 | 0.593 | 245 | 224 | 107 | 200 | 274 | 2.69 | 6.39 / 3.60 |
| SFT no-think (4B) | 0.740 | 0.667 | 0.740 | 15 | 1 | 0 | 1 | 0 | 1.00 | 0.53 / 0.00 |
| SFT Standard CoT, think-then-speak (4B) | 0.833 | 0.828 | 0.833 | 264 | 229 | 175 | 258 | 257 | 1.41 | 8.25 / 3.79 |
| InnerVoice SFT (4B) | 0.793 | 0.778 | 0.793 | 250 | 224 | 103 | 177 | 276 | 2.85 | 5.82 / 4.56 |

**Budget / policy grid (InnerVoice 1.7B, greedy, simulated 0.03 s per think token)**

| policy | final acc | all chunks ok | think tok | ARI | AIRW | FCO | sim TTFC s | P50 / P95 pause s | GPU s | premature |
|---|---|---|---|---|---|---|---|---|---|---|
| nothink | 0.213 | 0.213 | 0 | 159 | 0 | 1 | 10.12 | 0.00 / 0.00 | 19.75 | 0.609 |
| budget-32 | 0.507 | 0.507 | 265 | 218 | 28 | 191 | 6.23 | 0.50 / 0.52 | 9.79 | 0.706 |
| budget-64 | 0.527 | 0.527 | 276 | 234 | 49 | 197 | 6.49 | 0.41 / 1.49 | 10.33 | 0.555 |
| budget-128 | 0.540 | 0.540 | 274 | 238 | 75 | 195 | 6.24 | 0.00 / 3.40 | 9.70 | 0.383 |
| gate | 0.520 | 0.520 | 283 | 253 | 108 | 202 | 6.45 | 0.00 / 5.70 | 9.96 | 0.272 |
| fixed-64 | 0.000 | 0.000 | 309 | 177 | 64 | 315 | 9.56 | 1.48 / 1.48 | 9.56 | 1.000 |
| stitch-64 | 0.000 | 0.000 | 137 | 88 | 27 | 1 | 9.12 | 0.00 / 0.00 | 13.36 | 1.000 |

### sxs_test_mq  (150 examples, stride 4; token-index metrics follow the SxS paper: ARI / ABO / AIRW; FCO = first-content onset)

| method | final acc (greedy) | final acc (Avg@4, T=1) | all chunks ok | ARI | ABO | AIRW | FCO | think tok | speak blocks | sim TTFC s / P95 pause s |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B base, thinking mode | 0.408 | – | 0.288 | 989 | 952 | 951 | 970 | 951 | 0.94 | – |
| Qwen3-1.7B base, no thinking | 0.272 | – | 0.232 | 71 | 1 | 0 | 16 | 0 | 1.00 | – |
| Qwen3-4B base, thinking mode | 0.496 | – | 0.424 | 978 | 941 | 940 | 948 | 940 | 0.92 | – |
| Qwen3-4B base, no thinking | 0.416 | – | 0.368 | 67 | 1 | 0 | 2 | 0 | 1.00 | – |
| SFT no-think (1.7B) | 0.584 | 0.600 | 0.464 | 14 | 1 | 0 | 1 | 0 | 1.00 | 0.85 / 0.00 |
| SFT Standard CoT, think-then-speak (1.7B) | 0.592 | 0.566 | 0.456 | 111 | 97 | 96 | 97 | 96 | 1.00 | 3.72 / 0.00 |
| InnerVoice SFT (1.7B) | 0.568 | 0.542 | 0.456 | 78 | 78 | 39 | 40 | 96 | 2.45 | 1.99 / 0.00 |
|   ablation: no e_think input embedding (1.7B) | 0.592 | 0.592 | 0.448 | 76 | 76 | 39 | 39 | 94 | 2.41 | 1.95 / 0.00 |
|   ablation: 30% truncated-think mix (1.7B) | 0.584 | 0.538 | 0.440 | 77 | 77 | 38 | 39 | 95 | 2.48 | 1.95 / 0.00 |
| SFT no-think, LoRA experts (1.7B) | 0.600 | 0.602 | 0.488 | 14 | 1 | 0 | 1 | 0 | 1.00 | 0.83 / 0.00 |
| SFT Standard CoT, LoRA experts (1.7B) | 0.584 | 0.578 | 0.440 | 113 | 99 | 97 | 101 | 100 | 1.01 | 3.80 / 0.00 |
| InnerVoice SFT, LoRA experts (1.7B) | 0.664 | 0.550 | 0.496 | 76 | 76 | 39 | 39 | 94 | 2.41 | 1.96 / 0.00 |
| SFT no-think (4B) | 0.720 | 0.694 | 0.576 | 14 | 1 | 0 | 1 | 0 | 1.00 | 1.05 / 0.00 |
| SFT Standard CoT, think-then-speak (4B) | 0.704 | 0.670 | 0.584 | 115 | 101 | 97 | 101 | 100 | 1.06 | 4.07 / 0.00 |
| InnerVoice SFT (4B) | 0.712 | 0.686 | 0.544 | 78 | 78 | 39 | 40 | 97 | 2.46 | 2.28 / 0.00 |

**Budget / policy grid (InnerVoice 1.7B, greedy, simulated 0.03 s per think token)**

| policy | final acc | all chunks ok | think tok | ARI | AIRW | FCO | sim TTFC s | P50 / P95 pause s | GPU s | premature |
|---|---|---|---|---|---|---|---|---|---|---|
| nothink | 0.472 | 0.376 | 0 | 44 | 0 | 1 | 2.61 | 0.00 / 0.00 | 5.80 | 0.262 |
| budget-32 | 0.552 | 0.432 | 94 | 77 | 20 | 40 | 2.19 | 0.00 / 0.81 | 4.88 | 0.604 |
| budget-64 | 0.568 | 0.456 | 96 | 78 | 39 | 40 | 2.16 | 0.00 / 0.00 | 4.80 | 0.254 |
| budget-128 | 0.568 | 0.456 | 96 | 78 | 39 | 40 | 1.97 | 0.00 / 0.00 | 4.45 | 0.252 |
| gate | 0.568 | 0.456 | 96 | 78 | 39 | 40 | 1.99 | 0.00 / 0.00 | 4.50 | 0.252 |
| fixed-64 | 0.000 | 0.000 | 202 | 120 | 64 | 205 | 6.24 | 1.48 / 1.48 | 6.24 | 0.997 |
| stitch-64 | 0.248 | 0.200 | 126 | 50 | 17 | 1 | 2.20 | 0.00 / 1.21 | 6.31 | 0.647 |


### K&K all-verdicts-correct by number of people (greedy)

| method | n=2 | n=3 | n=4 | n=5 | n=6 | n=7 | n=8 | mean think tok (n=2 / n=7) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B base think | 0.52 (25) | 0.44 (25) | 0.60 (25) | 0.36 (25) | 0.36 (25) | 0.20 (25) | – | 1427 / 2542 |
| Qwen3-4B base think | 0.56 (25) | 0.48 (25) | 0.48 (25) | 0.16 (25) | 0.20 (25) | 0.12 (25) | – | 1307 / 2562 |
| SFT no-think 1.7B | 0.88 (25) | 0.72 (25) | 0.76 (25) | 0.44 (25) | 0.32 (25) | 0.36 (25) | – | 0 / 0 |
| SFT CoT 1.7B | 0.92 (25) | 0.64 (25) | 0.68 (25) | 0.52 (25) | 0.36 (25) | 0.16 (25) | – | 118 / 402 |
| SFT SxS 1.7B | 0.80 (25) | 0.68 (25) | 0.76 (25) | 0.44 (25) | 0.28 (25) | 0.16 (25) | – | 116 / 418 |
| SFT SxS 1.7B (LoRA experts) | 0.92 (25) | 0.80 (25) | 0.88 (25) | 0.48 (25) | 0.28 (25) | 0.20 (25) | – | 112 / 375 |
| SFT SxS 4B | 1.00 (25) | 0.96 (25) | 0.96 (25) | 0.76 (25) | 0.68 (25) | 0.40 (25) | – | 120 / 425 |

### MuSiQue final-answer accuracy by hops (greedy)

| method | 2-hop | 3-hop | 4-hop | intermediate chunk acc |
|---|---|---|---|---|
| Qwen3-1.7B base think | 0.43 (76) | 0.33 (36) | 0.46 (13) | 0.66 |
| Qwen3-4B base think | 0.54 (76) | 0.50 (36) | 0.23 (13) | 0.76 |
| SFT no-think 1.7B | 0.68 (76) | 0.42 (36) | 0.46 (13) | 0.77 |
| SFT CoT 1.7B | 0.62 (76) | 0.58 (36) | 0.46 (13) | 0.78 |
| SFT SxS 1.7B | 0.63 (76) | 0.47 (36) | 0.46 (13) | 0.80 |
| SFT SxS 1.7B (LoRA experts) | 0.72 (76) | 0.58 (36) | 0.54 (13) | 0.79 |
| SFT SxS 4B | 0.79 (76) | 0.56 (36) | 0.69 (13) | 0.81 |


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

| method | AIME25 Avg@4 (T=1) | unfinished | ARI | ABO | AIRW | FCO | think tok | speak tok | speak blocks |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B base, thinking (12k cap) | 0.258 | 0.66 | 10372 | 10216 | 10215 | 10216 | 10215 | 312 | 0.34 |
| DeepMath SFT no-think (1.7B) | 0.000 | 0.00 | 266 | 1 | 0 | 4 | 0 | 530 | 1.00 |
| DeepMath SFT Standard CoT (1.7B) | 0.075 | 0.05 | 3267 | 2696 | 2294 | 3221 | 3230 | 487 | 1.71 |
| DeepMath InnerVoice SFT (1.7B) | 0.175 | 0.22 | 3683 | 3191 | 1655 | 688 | 6950 | 543 | 4.32 |
| Qwen3-4B base, thinking (12k cap) | 0.342 | 0.62 | 10490 | 10301 | 10300 | 10301 | 10300 | 378 | 0.38 |
| DeepMath SFT no-think (4B) | 0.042 | 0.00 | 248 | 1 | 0 | 12 | 0 | 495 | 1.00 |
| DeepMath SFT Standard CoT (4B) | 0.183 | 0.02 | 1677 | 1346 | 1284 | 1536 | 1407 | 540 | 1.73 |
| DeepMath InnerVoice SFT (4B) | 0.233 | 0.17 | 3096 | 2686 | 1336 | 303 | 5636 | 535 | 4.74 |

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

| teacher data | train n | model | InnerVoice AIME25 Avg@4 | think tok | FCO | blocks | Standard CoT AIME25 Avg@4 | think tok | FCO |
|---|---|---|---|---|---|---|---|---|---|
| DeepMath R1 solutions (dataset-provided; think median 1,237 words) | 11684 | 1.7B | 0.175 | 6950 | 688 | 4.32 | 0.075 | 3230 | 3221 |
| DeepMath R1 solutions (dataset-provided; think median 1,237 words) | 11684 | 4B | 0.233 | 5636 | 303 | 4.74 | 0.183 | 1407 | 1536 |
| Qwen3-32B via Bedrock, default effort (think median 374 words) | 2287 | 1.7B | 0.025 | 1343 | 167 | 3.77 | 0.017 | 418 | 419 |
| Qwen3-32B via Bedrock, default effort (think median 374 words) | 2287 | 4B | 0.042 | 854 | 145 | 3.98 | 0.050 | 300 | 312 |
| Qwen3-32B via Bedrock, reasoning_effort=high (think median 1,230 words) | 1550 | 1.7B | 0.192 | 4745 | 1298 | 3.02 | 0.033 | 2285 | 2208 |
| Qwen3-32B via Bedrock, reasoning_effort=high (think median 1,230 words) | 1550 | 4B | 0.208 | 3570 | 614 | 3.00 | 0.083 | 1074 | 1047 |
| GPT-OSS-120B via Bedrock, reasoning_effort=high (think median 1,132 words) — the paper's teacher | 1312 | 1.7B | 0.083 | 5673 | 831 | 4.08 | 0.058 | 1788 | 1730 |
| GPT-OSS-120B via Bedrock, reasoning_effort=high (think median 1,132 words) — the paper's teacher | 1312 | 4B | 0.183 | 4826 | 637 | 4.31 | 0.075 | 1047 | 1033 |
| GPT-6 Astra via Bedrock, explicit <reasoning> section (median 85 words) | 542 | 4B | 0.000 | 329 | 181 | 2.18 | 0.000 | 304 | 302 |

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
- **Math RL round 2 (1.7B, Qwen3-32B-high SFT start, 8k rollout cap + DAPO-style overlong masking, gated FCO reward 0.1·FCO/600 only
  when correct, 60 steps x 8 prompts x 8 rollouts, HF decode, ~21 min/step)**: in-training reward 0.59 -> 0.62 (first/last 10 steps),
  training accuracy 0.72 -> 0.69, FCO 1,120 -> 854, think tokens 2,635 -> 2,909, public blocks 2.6 -> 3.4 (i.e. the policy moved first
  content earlier without shortening reasoning - the collapse of round 1 did not recur). AIME25 Average@4 (n=120): **0.117** vs the SFT
  start 0.192 (unfinished 0.067 vs 0.117; FCO 476 vs 1,298; think 4,579 vs 4,745; blocks 4.4 vs 3.0). The -7.5 pt change is within the
  ±8 pt noise of this eval, so round 2 is "no gain, earlier speech" - 60 steps of 64 rollouts is far below the paper's 480 steps x 512.
- **Scenario B infrastructure (2026-11-06/07)**: rollouts moved to vLLM 0.9.1 (custom `InnerVoiceQwen3ForCausalLM` class, one engine per
  GPU, in-place weight hot-reload each step; LoRA folded into the weights) with a torchrun trainer whose grads are all-reduced; router
  labels are recomputed on the trainer. The engine reproduces HF streaming decode token-for-token (8/8 sequences, 882/882 labels). On one
  8-GPU A100 host the 4B step (32 prompts x 8 rollouts, 8k cap) takes ~3.3 min: rollout 95-105 s, label pass 18 s, gradient 51-55 s,
  weight sync 28 s - 6x faster than HF decode at 4x the batch. The 4B start checkpoint was retrained without e_think (the engine cannot
  add an input-side embedding); its AIME25 Average@4 is 0.158 (n=120, unfinished 0.058, FCO 1,205, think 3,094, blocks 3.7) vs 0.208 for
  the e_think variant - within noise. Released on the Hub as `liqiangx/InnerVoice-Qwen3-4B` (merged weights + transformers remote code).
  The 4B run (`results/rl/q32high_4B_engine`, AIME25 every 50 steps) is in progress; its first steps show training accuracy 0.83-0.85 on
  the 1,550 SFT problems, i.e. the SFT prompt set is too easy for RL and the prompt pool should move to unseen DeepMath problems.
- Next: longer runs (≥500 steps), ≥2 seeds, Average@4 evaluation, rollout budgets that do not truncate correct reasoning, and the
  voice-stage rewards (stall/TTFC in seconds with randomised per-token cost).

## RL round 3 — engine rollouts, Qwen3-4B InnerVoice (Scenario B)

Setup: start = `q32high_sxs_4B_noethink` SFT (same 1,550 Qwen3-32B-high traces as the e_think variant; e_think dropped so the policy
runs in vLLM). Rollouts from 4 vLLM engines (T=1.0, 8k cap), 4 trainer ranks, 32 prompts x 8 rollouts per step, GRPO-style
group-normalised advantages, policy gradient on token + disclosure-decision log-probs; AIME25 Average@4 (12k cap) every 50 steps
through the same engines (n=120, ±8 pt). Rewards (R1..R5 below) and prompt pools evolved as failure modes appeared:
R1 used the 1,550 SFT problems (too easy: 84% group accuracy -> the FCO bonus was the only in-group signal) and collapsed into
"speak everything" (94 public blocks, FCO 21); a dense-penalty variant (R2: unsupported-anchor lines 0.5, short blocks 0.05, FCO, no KL,
run on Greenland at 32x16) collapsed the other way (no think, one short paragraph; AIME 0.19 -> 0.00 by step 100). R3 made every shaping
term a bounded bonus paid only on correct rollouts (R = correct·(1+0.2·Q), Q in [0,1] from unsupported lines / flicker / FCO), kept
truncated rollouts with a -0.5 penalty instead of masking, added KL to a frozen SFT copy and switched to 8,918 unseen DeepMath problems
(difficulty >= 5); it was clean at step 50 but word-level think/speak flicker appeared around step 70. R4 added an unconditional flicker
penalty (0.3·min(1, short blocks/10)) and router KL 0.5: clean and best at step 50, flicker again by step 100. Root cause: the SFT
router is near-deterministic (0.4% of positions with 0.02 < p < 0.98), so argmax routing + REINFORCE on the taken decision has no
exploration and negative advantages push uncertain logits across 0. R5 samples the disclosure decision in the engine with a
hash-reproducible u(token, position) (trainer recovers identical labels: 882/882 on the equivalence test) and lowers the decision
gradient weight to 0.2; it is the first design with no flicker through step 100.

| run (reward) | step | AIME25 Avg@4 | unfinished | FCO | AIRW | think tok | speak blocks | short blocks | unsupported lines |
|---|---|---|---|---|---|---|---|---|---|
| R1: correct + gated FCO bonus, SFT prompts, argmax routing | 0 | 0.158 | 0.058 | 1205 | 1337 | 3094 | 3.7 | nan | nan |
|  | 50 | 0.217 | 0.217 | 101 | 415 | 4622 | 16.6 | nan | nan |
|  | 100 | 0.292 | 0.517 | 21 | 152 | 5552 | 93.8 | nan | nan |
| R3: gated Q bonus + KL(0.02 tok / 0.05 router), soft overlong, unseen pool, argmax routing | 0 | 0.192 | 0.033 | 1015 | 942 | 2777 | 3.9 | 0.5 | 0.57 |
|  | 50 | 0.308 | 0.258 | 1214 | 1560 | 6829 | 6.8 | 1.7 | 0.36 |
|  | 100 | 0.283 | 0.133 | 351 | 266 | 4267 | 165.7 | 111.8 | 0.31 |
| R4: R3 + flicker penalty 0.3, router KL 0.5 | 0 | 0.150 | 0.050 | 1194 | 941 | 2885 | 4.0 | 0.7 | 0.53 |
|  | 50 | 0.333 | 0.175 | 1586 | 1966 | 6026 | 4.4 | 1.0 | 0.38 |
|  | 100 | 0.200 | 0.075 | 236 | 319 | 3175 | 104.1 | 78.1 | 0.37 |
| R5: R4 + hash-sampled routing, w_route 0.2 | 0 | 0.142 | 0.058 | 609 | 496 | 2819 | 6.4 | 2.4 | 0.54 |
|  | 50 | 0.275 | 0.067 | 2040 | 814 | 4922 | 7.2 | 3.5 | 0.38 |
|  | 100 | 0.242 | 0.067 | 893 | 911 | 4766 | 6.9 | 2.9 | 0.42 |
|  | 150 | 0.283 | 0.167 | 495 | 709 | 5166 | 8.7 | 3.8 | 0.41 |
|  | 200 | 0.375 | 0.150 | 531 | 813 | 6097 | 8.6 | 3.7 | 0.37 |
|  | 250 | 0.342 | 0.158 | 707 | 1030 | 6398 | 8.4 | 4.1 | 0.29 |
|  | 300 | 0.367 | 0.225 | 661 | 995 | 7040 | 9.8 | 5.4 | 0.32 |
| R5 on Greenland (8xH200): 32 prompts x 16 rollouts, 12k cap, 480 steps planned | 0 | 0.183 | 0.050 | 1005 | 563 | 3226 | 6.8 | 3.0 | 0.53 |
|  | 50 | 0.317 | 0.192 | 3666 | 1160 | 6525 | 9.3 | 6.2 | 0.26 |
|  | 100 | 0.317 | 0.183 | 991 | 851 | 6197 | 9.6 | 5.7 | 0.37 |
|  | 150 | 0.367 | 0.250 | 715 | 829 | 6988 | 11.8 | 7.2 | 0.37 |
|  | 200 | 0.317 | 0.300 | 918 | 892 | 7582 | 11.9 | 8.2 | 0.35 |
|  | 250 | 0.383 | 0.242 | 448 | 891 | 7069 | 12.1 | 7.9 | 0.30 |
|  | 300 | 0.308 | 0.283 | 391 | 950 | 7236 | 12.7 | 6.8 | 0.33 |
|  | 350 | 0.375 | 0.342 | 815 | 1360 | 7820 | 8.4 | 5.6 | 0.26 |
|  | 400 | 0.325 | 0.325 | 409 | 1096 | 7445 | 9.7 | 6.5 | 0.22 |
| R5 final checkpoint (step 300), **Average@8** (n=240, ±5.6 pt) | 300 | 0.325 | 0.242 | 669 | 966 | 6865 | 9.4 | 5.1 | 0.32 |

- R5 is the only design that stays clean to the end: p4 (32x8, 8k, 300 steps) 0.142 -> 0.375 / 0.342 / 0.367 at steps 200 / 250 / 300
  (Average@4), and the final checkpoint scores **0.325 at Average@8** (n=240, ±5.6 pt; 24% unfinished, FCO 669, 9.4 blocks, 5.1 short
  blocks, think 6,865). On Greenland (32x16, 12k) the same reward plateaus at 0.31-0.38 from step 150 (0.383 at step 250). Both are
  ~+15-20 pt over the SFT start (five step-0 evaluations of the same checkpoint spread 0.142-0.192 — the ±8 pt noise in action) and at
  parity with the Qwen3-4B base in thinking mode (0.342, 62% unfinished at 12k, FCO 10,301) while speaking from token ~400-900 in 8-12
  public blocks. The earlier R4 step-50 point (0.333, FCO 1,586, 4.4 blocks) is the cleanest disclosure profile but that run then flickered.
- Accuracy gains come with longer private reasoning (think 2.8k -> 4.8-6.8k tokens at steps 50-100), as in the paper's RL stage
  (AIRW 4,035 -> 8,519). The paper's numbers (Base 66.3, SxS SFT 38.3, SxS RL 80.0 on Qwen3-4B at a 32k budget, 480+120 steps x 512
  rollouts) are not directly comparable to our 12k-budget, 50-100-step runs.
- Gradient-checkpointing bug found during R5: `forward_train` cleared the teacher-forced route before backward, so the last layer's
  recompute used argmax routing (wrong expert on the ~1% label-disagreeing positions). Fixed 2026-11-08; all 4B SFT runs
  (trained with --grad_ckpt) carry this small bias and may be re-run.

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
