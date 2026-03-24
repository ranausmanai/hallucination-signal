# Hallucination Signal: Mechanistic Interpretability of LLM Hallucinations

Research into detecting and correcting LLM hallucinations by analyzing internal model computations during inference — no retraining, no external knowledge, just reading the model's own layers.

**All experiments run locally on M4 MacBook Pro (24GB) using [MLX](https://github.com/ml-explore/mlx).**

## Key Discoveries

### 1. Oscillation Signal (AUC = 0.752)
When a model hallucinates, its internal layers "argue" — the top-1 prediction oscillates between candidates across layers. This predicts hallucination better than output confidence (AUC 0.752 vs 0.626).

### 2. Chinese Token Retreat
When Qwen2 is uncertain about an English factual answer, intermediate layers surface **Chinese characters** semantically related to the topic (首都 = capital, 声音 = sound, 血液 = blood). Evidence that factual knowledge is encoded multilingually, with final layers performing language selection.

### 3. Knowledge Suppression
90% of wrong answers have the correct token in the final layer's top-200. Models don't lack knowledge — they **misrank** it. Popularity/fluency overrides factual accuracy.

### 4. Multi-Layer Voting (MLV)
A novel single-pass decoding algorithm that rescues suppressed knowledge from intermediate layers:

- Extract logits from multiple layers during the standard forward pass
- Filter candidates to ASCII tokens (prevents cross-lingual injection)
- Confidence-weighted voting across layers selects the winner
- Uncertainty gating ensures intervention only when the model is uncertain

**Results (100-question benchmark):**
| Model | Baseline | Best MLV | Net | Overhead |
|-------|----------|----------|-----|----------|
| Qwen2-0.5B | 79/100 | 82/100 | +3 | ~1% |
| Llama-3-8B | 98/100 | 98/100 | 0 (correctly abstains) | ~1% |

## What Didn't Work (and Why)

We tested 15+ approaches. Honest documentation of failures:

| Approach | Result | Why it failed |
|----------|--------|---------------|
| ANCHOR (hidden state blending) | Garbage output | Hidden states from different layers are incompatible |
| ANCHOR v2 (logit blending) | +1/-0 net | Too few parameters, converges to doing nothing |
| DualHead (factual head) | Net negative | Can't learn from 48 examples |
| Surgical fine-tuning | -1 to -3 net | Overfits, loses general capability |
| Contrastive layer decoding | Fragile | Works at α=0.5, destroys output at α=0.7+ |
| CAA steering vectors | No effect | "Factual accuracy" isn't a linear direction |
| Early exit | 0.97-0.99x (slower) | Exit check costs more than saved layers |
| Self-speculative decoding | 2.5x slower | Requires KV cache to be viable |
| Layer pruning | -5 to -95 accuracy | Small errors compound catastrophically over sequences |

## Full Writeup

See **[FINDINGS.md](FINDINGS.md)** for the complete research document with all numbers, tables, and analysis.

## Project Structure

### Discovery & Detection
| File | What it does |
|------|-------------|
| `oscillation.py` | Initial oscillation hypothesis |
| `oscillation_hard.py` | Oscillation signal with AUC analysis |
| `language_retreat.py` | Chinese token retreat discovery |
| `buried_answer.py` | Knowledge suppression analysis |
| `peak_confidence.py` | Confidence analysis across layers |

### Correction Methods
| File | What it does |
|------|-------------|
| `majority_vote.py` | Self-consistency (N=7, gold standard) |
| `combined_system.py` | Combined detection + correction |
| `mlv_decode.py` | Multi-Layer Voting v1 |
| `mlv_v2.py` | MLV v2 with ASCII filter |
| `mlv_benchmark.py` | 100-question benchmark + MLV eval |
| `mlv_selective.py` | Uncertainty-gated MLV (14 strategies) |
| `mlv_llama.py` | Cross-architecture MLV on Llama-3-8B |

### Failed Architecture Modifications
| File | What it does |
|------|-------------|
| `anchor.py` | ANCHOR v1 (hidden state) & v2 (logit blend) |
| `anchor_v3.py` | Zero-training factual verification |
| `dualhead_train.py` / `dualhead_v2.py` | Dual head with routing |
| `surgical_finetune.py` / `surgical_v2.py` | Fine-tune last layers |
| `distill_train.py` | Knowledge distillation |
| `contrastive_layers.py` | Contrastive layer decoding |
| `steer.py` / `steer2.py` / `steer3.py` | CAA steering vectors |

### Failed Inference Speedup
| File | What it does |
|------|-------------|
| `early_exit.py` | Early exit with lightweight criteria |
| `speculative_decode.py` | Self-speculative decoding |
| `layer_importance.py` | Layer importance analysis + pruning |

### Data
All `*_results.json` files contain raw experimental data for reproducibility.

## Requirements

```
pip install mlx mlx-lm transformers numpy
```

Models used (auto-downloaded by mlx-lm):
- `Qwen/Qwen2-0.5B-Instruct`
- `mlx-community/Meta-Llama-3-8B-Instruct-4bit`

## Taking This Forward

If you want to build on this work, the most promising directions are:

1. **MLV on harder benchmarks** — Our 100-question set is too easy for 8B+ models. Test MLV with uncertainty gating on TriviaQA, Natural Questions, or multi-hop reasoning tasks where large models actually hallucinate.

2. **MLV with KV-cached inference** — Our implementation recomputes the full sequence per token. Integrating MLV into mlx-lm's KV-cached generation pipeline would make it production-viable.

3. **Oscillation signal as a feature** — Combine oscillation count with other signals (entropy, layer disagreement) in a lightweight classifier for real-time hallucination detection.

4. **Cross-lingual knowledge routing** — The Chinese token finding suggests intermediate layers encode facts in the dominant training language. This could be exploited for better multilingual factual recall.

## License

MIT
