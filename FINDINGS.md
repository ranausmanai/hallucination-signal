# Mechanistic Interpretability of LLM Hallucinations: Complete Research Findings

**Full code & data:** [github.com/ranausmanai/hallucination-signal](https://github.com/ranausmanai/hallucination-signal)

**Research period:** March 2026
**Hardware:** M4 MacBook Pro, 24GB RAM
**Framework:** MLX (Apple Silicon ML framework)
**Models tested:** Qwen2-0.5B-Instruct (24 layers), Qwen3.5-0.8B (24 layers), Llama-3-8B-Instruct-4bit (32 layers)

---

## Table of Contents

1. [Research Question](#1-research-question)
2. [Discovery 1: Oscillation Signal](#2-discovery-1-oscillation-signal)
3. [Discovery 2: Chinese Token Retreat](#3-discovery-2-chinese-token-retreat)
4. [Discovery 3: Knowledge Suppression](#4-discovery-3-knowledge-suppression)
5. [Correction Approach 1: Self-Consistency](#5-correction-approach-1-self-consistency)
6. [Correction Approach 2: Multi-Layer Voting (MLV)](#6-correction-approach-2-multi-layer-voting-mlv)
7. [MLV Cross-Architecture Validation](#7-mlv-cross-architecture-validation)
8. [Failed Approaches: Architecture Modifications](#8-failed-approaches-architecture-modifications)
9. [Failed Approaches: Inference Speedup](#9-failed-approaches-inference-speedup)
10. [Key Lessons](#10-key-lessons)

---

## 1. Research Question

**Can we detect and correct LLM hallucinations by analyzing the model's internal computations during inference — without retraining or external knowledge?**

The motivation: if a model "knows" the right answer somewhere in its layers but produces the wrong output, we should be able to rescue that knowledge at decode time.

---

## 2. Discovery 1: Oscillation Signal

**File:** `oscillation_hard.py`
**Result:** AUC = 0.752 for hallucination detection

### What we found

As input tokens pass through a transformer's layers (0 → 23 for Qwen2), the model's top-1 prediction for the next token changes. We call each change an "oscillation." On 50 questions (37 correct, 13 wrong):

| Signal | AUC |
|--------|-----|
| **Oscillation count** | **0.752** |
| Late changes (layers 18+) | 0.749 |
| Combined (osc + late) | 0.715 |
| Unique tokens visited | 0.648 |
| Output confidence (max prob) | 0.626 |
| Time on final token | 0.498 (random) |

### What this means

When a model hallucinates, its internal layers "argue" more — the prediction oscillates between candidates across layers. A correct answer typically locks in by layer ~18 and stays stable. A hallucinated answer shows continued instability through the final layers.

**This beats standard output confidence** (0.752 vs 0.626). Output probability alone is a weak hallucination signal because the model can be confidently wrong. Oscillation captures the internal disagreement that confidence misses.

### Limitation

The AUC of 0.752 means ~25% of hallucinations are missed or misclassified. The signal is real but not strong enough for production use on its own. It works best as one feature among several.

---

## 3. Discovery 2: Chinese Token Retreat

**File:** `language_retreat.py`

### What we found

When Qwen2-0.5B oscillates at a layer, the competing tokens at oscillation peaks are often **Chinese characters** — even when the question and answer are entirely in English.

Examples of Chinese tokens appearing at intermediate layers during English factual QA:
- 首都 (capital) — appears when answering "What is the capital of X?"
- 声音 (sound) — appears when answering "What is the speed of sound?"
- 血液 (blood) — appears during biology questions
- 诺贝尔 (Nobel) — appears during Nobel Prize questions
- 历史性 (historic) — appears during history questions

### What this means

Qwen2 was trained on a large Chinese corpus. When the model is uncertain about an English factual answer, intermediate layers fall back to **Chinese conceptual representations** of the topic before the final layers translate back to English tokens. The Chinese tokens are not random — they are semantically related to the question topic.

This is evidence that:
1. **Factual knowledge is encoded multilingually** in intermediate layers
2. **The final layers perform language selection**, converting multilingual concepts to the appropriate output language
3. **When the final layers fail at language selection, you get hallucination** — the model picks the wrong English fact while the correct conceptual representation exists in Chinese

### Cross-lingual comparison

We tested the same questions in Chinese vs English:
- English correct: 7/17
- Chinese correct: 6/17
- Chinese better than English: 2 questions
- English better than Chinese: 3 questions

The model doesn't consistently know more in either language — but the intermediate representations bleed across languages during uncertain predictions.

---

## 4. Discovery 3: Knowledge Suppression

**File:** `buried_answer.py`

### What we found

For questions the model gets wrong (greedy decoding), we checked whether the **correct** token appears anywhere in the final layer's top-200 predictions. Result: **9 out of 10 wrong answers have the correct token in the top-200.**

| Question | Correct token rank | Hallucinated answer |
|----------|-------------------|-------------------|
| "30th US president?" | rank 108 | "William Howard Taft" (should be Coolidge) |
| "Speed of sound in m/s?" | rank 44 | "3" (starts wrong, should be 343) |
| "Driest continent?" | rank 187 | "Africa" (should be Antarctica) |

### What this means

The model doesn't lack the knowledge — it **suppresses** it. The correct answer exists in the probability distribution but is outranked by a more "popular" or "expected" alternative. This is the core mechanism behind factual hallucination in small models:

1. Intermediate layers encode the correct fact (often in multilingual representations)
2. Final layers rank candidates by a combination of factual accuracy AND fluency/popularity
3. Sometimes popularity wins over accuracy: Lagos is more famous than Abuja, so the model picks Lagos as Nigeria's capital

This finding motivated our correction approaches: if the knowledge is there but suppressed, can we rescue it?

---

## 5. Correction Approach 1: Self-Consistency

**File:** `majority_vote.py`, `combined_system.py`

### Method

Generate N=7 independent responses at temperature T=0.5, take the majority vote answer.

### Results

On 50 questions (Qwen2-0.5B):
- Greedy baseline: 34/50
- Self-consistency (T=0.5, N=7): **40/50** (+6)
- On hard questions only: 15/30 → 20/30 (+5)
- **Zero regressions** — never broke a correct answer

On the 100-question benchmark:
- Qwen2: +5/-0 = **net +5**
- Qwen3.5: +3/-0 = **net +3**

### Trade-off

Self-consistency is the gold standard for correction — reliable, zero regressions. But it costs **7x compute** (7 forward passes per question). This motivated the search for cheaper alternatives.

---

## 6. Correction Approach 2: Multi-Layer Voting (MLV)

**Files:** `mlv_decode.py`, `mlv_v2.py`, `mlv_benchmark.py`, `mlv_selective.py`

### Algorithm

MLV is a novel single-pass decoding algorithm that uses information from intermediate transformer layers:

1. **Forward pass with layer tapping**: During the standard forward pass, extract logits from layers 20-23 (Qwen2) or 28-31 (Llama) by projecting hidden states through the shared norm + language model head
2. **Candidate set**: Take the final layer's top-k tokens as candidates (language/quality filter)
3. **ASCII filter**: Remove non-ASCII tokens from candidates (prevents Chinese token injection)
4. **Weighted voting**: Each voting layer picks its highest-confidence candidate. Confidence scores are summed across layers.
5. **Winner selection**: Token with highest total confidence wins

### Evolution

**v1 (mlv_decode.py)** — 20 questions:
- Raw MLV: +4/-1 = net +3
- Weighted MLV: +4/-1 = net +3
- Best early result, but small sample overrepresented easy wins

**v2 (mlv_v2.py)** — Added ASCII filter:
- Qwen2: +4/-0 = net +4 on 20 questions (ASCII filter eliminated Chinese token break)
- Qwen3.5: +1/-0 = net +1 (k=20)

**Benchmark (mlv_benchmark.py)** — 100 questions, 7 categories:

| Model | Baseline | MLV Always | Fixes | Breaks | Net |
|-------|----------|-----------|-------|--------|-----|
| Qwen2-0.5B | 79/100 | 80/100 | +5 | -4 | **+1** |
| Qwen3.5-0.8B | 77/100 | 75/100 | +5 | -7 | **-2** |

The 20-question results were misleadingly positive. At 100 questions, MLV-always breaks almost as many as it fixes.

### Selective MLV (mlv_selective.py)

Key insight: only apply MLV when the model shows uncertainty. 14 strategies tested:

| Strategy | Fixes | Breaks | Net | Notes |
|----------|-------|--------|-----|-------|
| Always MLV | +5 | -4 | +1 | Too aggressive |
| max_prob < 0.5 | +2 | -0 | +2 | Conservative |
| **max_prob < 0.7** | **+5** | **-2** | **+3** | Best absolute gain (79→82) |
| entropy > 2.0 | +5 | -3 | +2 | |
| entropy > 3.0 | +3 | -1 | +2 | |
| **entropy > 5.0** | **+2** | **-0** | **+2** | Perfect precision, only 3 interventions |
| disagree > 0 | +5 | -3 | +2 | |
| disagree >= 2 | +3 | -0 | +3 | |
| prob<0.7 AND disagree>0 | +5 | -2 | +3 | |
| entropy>3 AND disagree>0 | +3 | -1 | +2 | |

**Best configurations:**
- **entropy > 5.0**: Zero regressions, +2 net. Only 3 token-level interventions across 100 questions. Perfect precision.
- **max_prob < 0.7**: Best absolute gain (+3 net, 79→82). Breaks trench question wording + Swan Lake composer.
- **prob<0.7 AND disagree>0**: Same +3 net with fewer interventions.

### What MLV consistently fixes
- "What continent is Brazil on?" — Qwen2 says "South America is a continent" → MLV corrects to "South America"
- "Speed of sound in m/s" — wrong first digit → correct
- "What does CO2 stand for?" — garbled → "carbon dioxide"
- "What year was WHO founded?" — wrong year → correct
- "What is a UNESCO site?" — hallucinated → correct

### What MLV consistently breaks
- **Name romanizations**: Tolstoy → Tolstoi, Tchaikovsky → Mikhailovsky (earlier layers use different transliterations)
- **Number corruption**: "206 bones" → "20 bones" (earlier layers less precise with numbers)
- **Popular-over-correct**: Nigeria capital Lagos→Abuja, but earlier layers prefer the more famous city

### Overhead

MLV adds ~1% compute: one extra norm + language model head projection per voting layer per token. Negligible compared to the full forward pass.

### Comparison with self-consistency

| Method | Net gain | Compute cost | Regressions |
|--------|----------|-------------|-------------|
| Self-consistency (N=7) | +5 | 700% | 0 |
| MLV (entropy>5.0) | +2 | ~1% | 0 |
| MLV (prob<0.7) | +3 | ~1% | 2 |

MLV is a "free" +2-3 improvement. Self-consistency is the reliable +5 at high cost.

---

## 7. MLV Cross-Architecture Validation

**File:** `mlv_llama.py`

### Llama-3-8B-Instruct (32 layers, 4-bit quantized)

| Strategy | Fixes | Breaks | Net |
|----------|-------|--------|-----|
| Always MLV L28-31 | 0 | 1 | -1 |
| prob<0.7 L28-31 | 0 | 0 | 0 |
| entropy>5 L28-31 | 0 | 0 | 0 |
| Always MLV L26-31 | 0 | 2 | -2 |
| prob<0.7 L26-31 | 0 | 0 | 0 |
| prob<0.7 L29-31 | 0 | 0 | 0 |

**Baseline: 98/100** — Llama 8B is too strong for this benchmark.

### What this tells us

1. **MLV works mechanically on Llama** — the algorithm runs, voting happens, overrides occur (50-337 per run). Cross-architecture generalization confirmed.
2. **Nothing to fix** — at 98/100, correct answers aren't suppressed. MLV can only break things.
3. **Uncertainty gating validated** — prob<0.7 and entropy>5 correctly recognize the model is confident and abstain from intervening. Zero harm on strong models.

### Cross-model summary

| Model | Params | Baseline | Best MLV | Net |
|-------|--------|----------|----------|-----|
| Qwen2-0.5B | 0.5B | 79/100 | 82/100 | **+3** |
| Qwen3.5-0.8B | 0.8B | 77/100 | 75/100 | -2 |
| Llama-3-8B | 8B | 98/100 | 98/100 | 0 |

**Conclusion:** MLV helps weak models where correct knowledge is suppressed in final layers. Strong models don't suppress, so MLV has nothing to rescue.

---

## 8. Failed Approaches: Architecture Modifications

### ANCHOR v1 — Hidden State Blending

**File:** `anchor.py` (first version)

**Idea:** Learn a gate that blends hidden states from a "factual" layer (20-21) into the final layer's hidden state.

**Result:** Complete failure. Output was "!!!!!" for all inputs. The gate saturated to 1.0 during training because hidden states from different layers have incompatible scales and distributions. Adding them directly corrupts the representation.

### ANCHOR v2 — Logit Space Blending

**File:** `anchor.py` (rewritten)

**Idea:** Instead of blending hidden states, blend in logit space. Five strategies tested: StaticBlend, EntropyGated, Contrastive, EntropySwitch, FactualBoost.

**Results on 20 questions:**

| Strategy | Fixes | Breaks | Net |
|----------|-------|--------|-----|
| StaticBlend | 1 | 0 | +1 |
| EntropyGated | 0 | 1 | -1 |
| Contrastive | 1 | 0 | +1 |
| EntropySwitch | 1 | 0 | +1 |
| FactualBoost | 1 | 1 | 0 |

Net +1 at best. Too few trainable parameters to learn meaningful discrimination. Strategies converge to doing nothing (blending weight → 0).

### ANCHOR v3 — Factual Verification Decoding

**File:** `anchor_v3.py`

**Idea:** Zero-training approaches: decode with confidence, agree-or-factual, blend on disagree, rank rescue, suppress disagree.

**Best result:** RankRescue(k=5) at layer 21: +1/-0 = net +1. Same modest effect.

### DualHead

**File:** `dualhead_train.py`, `dualhead_v2.py`

**Idea:** Train a small factual head on layer 21's hidden states to predict correct answers. Route between standard and factual head based on oscillation signal.

**Result:** Net negative. Factual head couldn't learn to distinguish correct from incorrect within 48 training examples. When routed, it produced worse answers than standard decoding.

### Surgical Fine-tuning

**File:** `surgical_finetune.py`, `surgical_v2.py`

**Idea:** Fine-tune only the last 1-2 transformer layers on factual QA pairs.

**Results:**
- L23 only: +2/-5 = net -3
- L22-23: +4/-5 = net -1

Fine-tuning on 48 examples is insufficient and causes overfitting. The model loses general capability while gaining minimal factual accuracy.

### Knowledge Distillation

**File:** `distill_train.py`

**Idea:** Distill knowledge from layer 21's logits into the final layer.

**Result:** Marginal improvement, not enough training data to converge meaningfully.

### Contrastive Layer Decoding

**File:** `contrastive_layers.py`

**Idea:** Subtract intermediate layer logits from final layer logits (amplifying what the final layer "added").

**Results on 30 questions:**
- L22→L23 α=0.5: 21/30 (best, +3 vs baseline 18/30)
- L21→L23 α=0.5: 16/30 (-2)
- L21→L23 α=0.7: 7/30 (-11, catastrophic)
- L21→L23 α=1.5: 2/30 (-16, destroyed)

Moderate α with adjacent layers shows modest improvement, but aggressive settings destroy output. The technique is fragile and doesn't reliably help.

### CAA Steering

**Files:** `steer.py`, `steer2.py`, `steer3.py`

**Idea:** Compute a "factual accuracy" steering vector from contrastive hidden state pairs and add it during inference.

**Result:** Steering vectors had minimal effect on factual accuracy. The concept of "factual accuracy" is too abstract to be captured by a single linear direction in activation space.

---

## 9. Failed Approaches: Inference Speedup

### Early Exit / Adaptive Compute

**File:** `early_exit.py`

**Idea:** Stop processing at an intermediate layer when the model is already confident, saving compute on "easy" tokens.

**10 configurations tested on Llama-3-8B (100 questions):**

| Method | Exit Rate | Speedup | Breaks |
|--------|-----------|---------|--------|
| Cosine similarity (4 configs) | 0% | 0.97-0.99x | 0 |
| Norm stability (3 configs) | 0% | 0.98x | 0 |
| Single logit check @20, thr=0.95 | 2% | 0.97x | 0 |
| Single logit check @20, thr=0.90 | 3% | 0.98x | 0 |
| Single logit check @16, thr=0.95 | 0% | 0.99x | 0 |

**Every configuration was SLOWER than baseline.**

**Why it failed:**
1. **Hidden states never converge** — cosine similarity between consecutive layers is far below 0.99. Every transformer layer makes a meaningful transformation. The lightweight exit criteria never trigger.
2. **The exit check is expensive** — projecting to 128K vocab (norm + lm_head) costs as much as several transformer layers. Even when 3% of tokens exit early, the overhead exceeds the savings.
3. **No KV cache** — our implementation recomputes the full sequence per token, so even skipping layers barely dents total compute.

### Self-Speculative Decoding

**File:** `speculative_decode.py`

**Idea:** Draft tokens with Llama's first 16 layers, verify all drafts with the full 32 layers in one pass.

**Result:** 2.5x SLOWER than baseline. Without KV caching, both drafting and verification recompute the full sequence from scratch. The theoretical advantage of speculative decoding requires cached key-value pairs, which our custom forward pass doesn't implement.

### Layer Pruning

**File:** `layer_importance.py`

**Idea:** Identify and permanently remove redundant layers from the model.

**Phase 1 — Per-token importance analysis (38 tokens, 10 questions):**

Layers 16-19 showed 0% importance (skipping them never changed the immediate next-token prediction). Layers 0-1 were critical (94-100%). Layers 30-31 were important (21-26%).

**Phase 2 — Pruning with generation (100 questions):**

| Layers removed | Accuracy | Drop | Theoretical speedup |
|----------------|----------|------|-------------------|
| 1 (3%) | 93/100 | -5 | 1.03x |
| 2 (6%) | 81/100 | -17 | 1.07x |
| 4 (12%) | 52/100 | -46 | 1.14x |
| 8 scattered (25%) | 19/100 | -79 | 1.33x |
| 4 contiguous 10-13 | 84/100 | -14 | 1.14x |
| 8 contiguous 8-15 | 34/100 | -64 | 1.33x |
| 8 contiguous 12-19 | 11/100 | -87 | 1.33x |
| 12 contiguous 8-19 | 3/100 | -95 | 1.60x |

Example outputs at 12 layers removed:
- "The capital of France is the French Reg"
- "The capital of Japan is Nagagata (F) (F)"
- "The capital is 10,000 bistro, the number"

**Why it failed:** Per-token importance is misleading. A layer that rarely changes the immediate prediction can still be critical for maintaining coherent generation. Small errors compound catastrophically over a sequence — one wrong token leads to a different context, leading to more wrong tokens. Even removing a single "unimportant" layer costs 5% accuracy.

---

## 10. Key Lessons

### What worked (genuinely novel findings)

1. **Oscillation as a hallucination signal** (AUC=0.752) — Layer-wise prediction instability predicts hallucination better than output confidence. This is a real, useful signal for hallucination detection systems.

2. **Chinese token retreat** — Multilingual models encode factual knowledge in their dominant training language at intermediate layers. When uncertain in English, Chinese concepts surface at oscillation peaks. This is a genuine mechanistic interpretability finding about how multilingual knowledge is organized.

3. **Knowledge suppression** — 90% of wrong answers have the correct token in the top-200 of the final layer's distribution. Models don't lack knowledge — they misrank it. Popularity/fluency can override factual accuracy.

4. **Multi-Layer Voting with uncertainty gating** — A novel decoding algorithm that rescues suppressed knowledge from intermediate layers. +2-3 accuracy points at ~1% overhead on weak models, with zero harm on strong models when gated by uncertainty signals.

### What we learned from failures

5. **Hidden states from different layers are incompatible** — You can't blend, add, or interpolate hidden states across layers. They live in different subspaces with different scales. Only logit-space operations are safe.

6. **Per-token metrics don't predict sequence quality** — A layer can appear "unimportant" on individual tokens but be critical for maintaining coherent generation. Error compounding makes small per-token changes catastrophic over sequences.

7. **Inference optimization without retraining is solved** — Quantization, KV caching, flash attention, and operator fusion (already in MLX/llama.cpp) are the real speedup techniques. Novel zero-shot tricks (early exit, speculative decoding, layer pruning) either don't trigger, add more overhead than they save, or destroy quality.

8. **Small models (0.5B) have noisy internals** — Many correction techniques that seem promising fail because the intermediate layers don't consistently encode better information. They're noisy, not oracular.

9. **Earlier layers are not always more correct** — They sometimes prefer popular answers (Lagos > Abuja), use different romanizations (Tolstoi vs Tolstoy), or are less precise with numbers (206 → 20). The "buried knowledge" hypothesis is true on average but unreliable for individual predictions.

### The honest bottom line

The mechanistic findings (oscillation, Chinese tokens, knowledge suppression) are genuine contributions to understanding how transformers process factual knowledge. MLV is a real if modest technique. But none of these translate into a product or a dramatic demo. The fundamental challenge: making transformers better at factual recall from the outside (without retraining) produces small, inconsistent effects because the internal representations are too entangled to surgically modify.

---

## Appendix: File Index

| File | Purpose |
|------|---------|
| `oscillation.py` | Initial oscillation hypothesis test |
| `oscillation_hard.py` | Hard benchmark with AUC analysis |
| `language_retreat.py` | Chinese token discovery |
| `buried_answer.py` | Knowledge suppression analysis |
| `majority_vote.py` | Self-consistency baseline |
| `combined_system.py` | Combined detection + correction system |
| `mlv_decode.py` | MLV v1 (20 questions) |
| `mlv_v2.py` | MLV v2 with ASCII filter |
| `mlv_benchmark.py` | 100-question benchmark + MLV evaluation |
| `mlv_selective.py` | Uncertainty-gated MLV (14 strategies) |
| `mlv_llama.py` | Cross-architecture MLV on Llama-3-8B |
| `anchor.py` | ANCHOR v1 (hidden state) and v2 (logit blend) |
| `anchor_v3.py` | Zero-training factual verification decoding |
| `dualhead_train.py` | Dual head (factual head on layer 21) |
| `dualhead_v2.py` | Dual head v2 with routing |
| `surgical_finetune.py` | Fine-tune last layers |
| `surgical_v2.py` | Fine-tune v2 |
| `distill_train.py` | Knowledge distillation |
| `contrastive_layers.py` | Contrastive layer decoding |
| `steer.py/steer2.py/steer3.py` | CAA steering vectors |
| `early_exit.py` | Early exit / adaptive compute |
| `speculative_decode.py` | Self-speculative decoding |
| `layer_importance.py` | Layer importance + pruning experiments |
| `qwen35_experiment.py` | Qwen3.5 architecture testing |

All result JSON files contain raw experimental data for reproducibility.
