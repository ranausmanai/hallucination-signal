"""
Peak-Confidence Decoding — Towards a Novel Architecture
=========================================================
Finding: layer 21 retrieves the correct fact, layers 22-23 suppress it.
The "plausibility correction" in final layers overrides factual accuracy.

Hypothesis for a new decoding rule:
  - At each decode step, run full forward pass, collect per-layer confidence
  - Find the layer with PEAK confidence in the late layers (15-23)
  - Use THAT layer's logits — not always the final layer's

Intuition: the layer where the model is most committed to its prediction
is also the layer where the factual signal is strongest. Final layers may
lower their confidence as they "hedge" across plausible continuations.

If this works: we've shown that the OUTPUT LAYER should not always be the
FINAL layer. That's an architectural insight: transformers should use their
"most confident intermediate layer" for factual prediction, not their deepest.

This leads to the novel architecture: Confidence-Adaptive Output Layer
  - Learn which layer to use for which type of token
  - For factual tokens: use intermediate layers (higher confidence)
  - For fluency/style tokens: use final layer (better calibrated)

This script tests Peak-Confidence Decoding with no retraining.
If it improves factual accuracy, we have the motivation for the architecture.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.qwen2 import create_attention_mask
import json


def forward_with_all_logits(model, token_ids, layer_range=(15, 24)):
    """
    Single forward pass. For layers in layer_range, compute:
    - hidden state h[i]
    - top-1 confidence (max softmax probability)
    - logits
    Returns: list of (layer_idx, confidence, logits_np) for layers in range.
    """
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    results = []

    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)

        if i >= layer_range[0]:
            h_norm = model.model.norm(h)
            if model.args.tie_word_embeddings:
                logits = model.model.embed_tokens.as_linear(h_norm)
            else:
                logits = model.lm_head(h_norm)
            mx.eval(logits)

            last = np.array(logits[0, -1].astype(mx.float32))
            probs = np.exp(last - last.max())
            probs /= probs.sum()
            confidence = float(probs.max())

            results.append({
                "layer": i,
                "confidence": confidence,
                "top1": int(np.argmax(last)),
                "logits": last.copy(),
            })

            del logits, h_norm

    return results


def peak_confidence_next(model, token_ids, layer_range=(15, 24), strategy="peak"):
    """
    Get next token using peak-confidence strategy.

    Strategies:
    - 'final': always use final layer (baseline)
    - 'peak': use layer with highest confidence
    - 'stable_peak': use last layer before prediction stabilizes
    - 'entropy_weighted': weight all layers by their confidence
    """
    layer_data = forward_with_all_logits(model, token_ids, layer_range)

    if strategy == "final":
        return layer_data[-1]["top1"], layer_data[-1]["confidence"], layer_data[-1]["layer"]

    elif strategy == "peak":
        # Use the layer with highest top-1 probability
        best = max(layer_data, key=lambda x: x["confidence"])
        return best["top1"], best["confidence"], best["layer"]

    elif strategy == "entropy_weighted":
        # Weight all layer logits by their confidence
        weights = np.array([d["confidence"] for d in layer_data])
        weights = weights / weights.sum()
        combined = sum(w * d["logits"] for w, d in zip(weights, layer_data))
        return int(np.argmax(combined)), float(np.max(combined)), -1

    elif strategy == "stable_peak":
        # Find the last layer where prediction changes, then use peak after that
        top1s = [d["top1"] for d in layer_data]
        last_change = 0
        for i in range(len(top1s)-1, 0, -1):
            if top1s[i] != top1s[i-1]:
                last_change = i
                break
        # Use peak confidence layer from last_change onwards
        post_stable = layer_data[last_change:]
        best = max(post_stable, key=lambda x: x["confidence"])
        return best["top1"], best["confidence"], best["layer"]

    return layer_data[-1]["top1"], layer_data[-1]["confidence"], layer_data[-1]["layer"]


def generate_with_strategy(model, tokenizer, question, strategy, max_tokens=60):
    """Full autoregressive generation with given strategy."""
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    layer_choices = []

    for _ in range(max_tokens):
        next_id, conf, layer = peak_confidence_next(
            model, ids, layer_range=(15, 24), strategy=strategy)
        if next_id == eos_id:
            break
        generated.append(next_id)
        layer_choices.append(layer)
        ids.append(next_id)

    answer = tokenizer.decode(generated).strip()
    avg_layer = np.mean(layer_choices) if layer_choices else -1
    return answer, avg_layer


def analyze_confidence_profile(model, tokenizer, question, keywords):
    """
    For a single question: show per-layer confidence + what token is predicted.
    This is the diagnostic that motivates the architecture.
    """
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)

    layer_data = forward_with_all_logits(model, ids, layer_range=(0, 24))

    correct_ranks = []
    print(f"\n  Layer-by-layer profile:")
    print(f"  {'L':>3} {'Conf':>6} {'Top-1':>15} {'Correct rank':>14}")
    print("  " + "-" * 45)
    for d in layer_data:
        top1_tok = tokenizer.decode([d["top1"]])
        # Find rank of any correct keyword
        sorted_ids = np.argsort(-d["logits"])
        correct_rank = len(sorted_ids)  # default: not found
        for rank, tid in enumerate(sorted_ids[:200]):
            tok = tokenizer.decode([int(tid)]).lower()
            if any(kw.lower() in tok for kw in keywords):
                correct_rank = rank
                break
        correct_ranks.append(correct_rank)
        print(f"  {d['layer']:>3} {d['confidence']:>6.3f} {top1_tok:>15} {correct_rank:>14}")
    return layer_data, correct_ranks


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    n_layers = len(model.model.layers)

    # Deep diagnostic: show per-layer confidence profile for key questions
    print("\n" + "="*70)
    print("DIAGNOSTIC: Per-Layer Confidence Profile")
    print("="*70)
    print("Key question: at which layer is confidence HIGHEST and what does it predict?")

    diagnostic_qa = [
        ("What country has the most UNESCO World Heritage Sites?", ["italy", "china"]),
        ("What is the largest planet?", ["jupiter"]),
        ("What is the capital of France?", ["paris"]),
        ("What is the half-life of uranium-235?", ["703", "700"]),
    ]

    for question, keywords in diagnostic_qa:
        print(f"\nQ: {question}")
        layer_data, correct_ranks = analyze_confidence_profile(
            model, tokenizer, question, keywords)

        # Find peak confidence layer
        peak = max(layer_data, key=lambda x: x["confidence"])
        final = layer_data[-1]
        print(f"\n  Peak confidence: Layer {peak['layer']} ({peak['confidence']:.3f})")
        print(f"    Predicts: '{tokenizer.decode([peak['top1']])}'")
        print(f"    Correct keyword rank at peak: {correct_ranks[peak['layer']]}")
        print(f"  Final layer ({final['layer']}): conf={final['confidence']:.3f}")
        print(f"    Predicts: '{tokenizer.decode([final['top1']])}'")
        print(f"    Correct keyword rank at final: {correct_ranks[final['layer']]}")

    # Main experiment: compare strategies
    print("\n\n" + "="*70)
    print("EXPERIMENT: Peak-Confidence Decoding vs Greedy")
    print("="*70 + "\n")

    test_qa = [
        ("Who was the 30th president of the United States?",        ["coolidge", "calvin"]),
        ("What is the melting point of tungsten in Celsius?",       ["3422", "3400"]),
        ("What is the half-life of uranium-235 in years?",          ["703 million", "703", "700"]),
        ("What is the largest desert in the world by area?",        ["antarctica", "antarctic"]),
        ("What country has the most UNESCO World Heritage Sites?",   ["italy", "china"]),
        ("What is the atomic weight of plutonium?",                 ["244", "242", "239"]),
        ("What is the rarest blood type?",                          ["ab-", "ab negative"]),
        ("Who won the Nobel Prize in Chemistry in 2023?",           ["bawendi", "brus", "ekimov"]),
        ("What is the speed of sound in water in m/s?",             ["1480", "1500", "1498"]),
        ("Who is the prime minister of New Zealand as of 2024?",    ["luxon", "christopher"]),
        ("What is the capital of Kyrgyzstan?",                      ["bishkek"]),
        ("What is the capital of Burkina Faso?",                    ["ouagadougou"]),
        ("Who was the first female prime minister of the UK?",      ["thatcher"]),
        ("What is the half-life of Carbon-14 in years?",            ["5730", "5700"]),
        ("What year was the WHO founded?",                          ["1948"]),
        ("What is the largest planet?",                             ["jupiter"]),
        ("What is the capital of France?",                          ["paris"]),
        ("Who wrote Romeo and Juliet?",                             ["shakespeare"]),
        ("What is 2 + 2?",                                          ["4", "four"]),
        ("What is the boiling point of water?",                     ["100"]),
    ]

    strategies = ["final", "peak", "entropy_weighted", "stable_peak"]
    all_results = {s: [] for s in strategies}

    for question, keywords in test_qa:
        print(f"Q: {question[:60]}")
        q_res = {}
        for strat in strategies:
            answer, avg_layer = generate_with_strategy(model, tokenizer, question, strat)
            is_correct = any(kw.lower() in answer.lower() for kw in keywords)
            all_results[strat].append(is_correct)
            q_res[strat] = {"answer": answer, "correct": is_correct, "avg_layer": avg_layer}
            marker = "✓" if is_correct else "✗"
            layer_str = f"(avg_layer={avg_layer:.1f})" if avg_layer > 0 else ""
            print(f"  [{strat:18}] {layer_str}: '{answer[:45]}' {marker}")
        print()

    # Summary
    print("="*70)
    print("RESULTS")
    print("="*70 + "\n")

    n_hard = 15
    n_total = len(test_qa)
    baseline = sum(all_results["final"])
    baseline_hard = sum(all_results["final"][:n_hard])

    print(f"{'Strategy':22} {'Total':>8} {'Hard({n_hard})':>10}  {'Δ hard':>8}")
    print(f"{'Strategy':22} {'Total':>8} {'Hard':>10}  {'Δ hard':>8}")
    print("-" * 55)
    for strat in strategies:
        tot = sum(all_results[strat])
        hard = sum(all_results[strat][:n_hard])
        dh = hard - baseline_hard
        dh_s = f"+{dh}" if dh > 0 else str(dh) if dh < 0 else "="
        star = " ★★" if dh >= 3 else (" ★" if dh >= 2 else (" ~" if dh >= 1 else ""))
        print(f"  {strat:20} {tot:>4}/{n_total}  {hard:>4}/{n_hard}  {dh_s:>8}{star}")

    print()

    # Key finding: does peak-confidence layer differ from final layer?
    print("="*70)
    print("ARCHITECTURAL INSIGHT")
    print("="*70)
    print("""
If peak-confidence decoding beats greedy (final layer only):

  → The optimal output layer is NOT always the final layer
  → Factual knowledge peaks at intermediate layers
  → Final layers apply "fluency correction" that can override factual accuracy

This motivates a new architecture: Confidence-Adaptive Output (CAO)
  - Standard transformer up to layer N
  - At output: instead of always using layer N, use the layer with
    peak confidence for the current token type
  - Learned router: when to use deep vs shallow layers

The CAO transformer would:
  - Use shallow layers for factual tokens (where knowledge peaks early)
  - Use deep layers for reasoning/fluency tokens (where depth helps)
  - Eliminate the "plausibility correction" problem that causes hallucination

This is architecturally different from MoE (mixture over experts) —
it's mixture over DEPTH for the SAME input, guided by confidence.
""")

    with open("peak_confidence_results.json", "w") as f:
        json.dump({
            "n_hard": n_hard, "n_total": n_total,
            "summary": {s: {"total": sum(all_results[s]),
                            "hard": sum(all_results[s][:n_hard])}
                        for s in strategies}
        }, f, indent=2)
    print("Saved peak_confidence_results.json")


if __name__ == "__main__":
    run()
