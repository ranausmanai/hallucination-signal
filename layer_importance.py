"""
Layer Importance Analysis — Which layers actually matter?
==========================================================

Before we can skip layers, we need to know which ones are expendable.

Approach: For each layer, measure what happens to the output when we SKIP it.
If skipping layer 15 doesn't change the final prediction, layer 15 is redundant
for that token.

This gives us a "layer importance map" per token type, which we can then use
to build a routing strategy.

Related work: ShortGPT (Men et al, 2024) showed many layers are redundant.
But they pruned layers permanently. We want to skip DYNAMICALLY per token.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask
import json
import time

from mlv_benchmark import BENCHMARK, check_correct


def setup_llama(model):
    def get_logits(h):
        h_normed = model.model.norm(h)
        return model.lm_head(h_normed)
    return {
        "embed": model.model.embed_tokens,
        "layers": model.model.layers,
        "norm": model.model.norm,
        "get_logits": get_logits,
        "get_mask": lambda h: create_attention_mask(h, None),
        "n_layers": len(model.model.layers),
    }


def forward_skip_layer(mi, ids, skip_layer):
    """Forward pass skipping one specific layer."""
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)

    for i, layer in enumerate(mi["layers"]):
        if i == skip_layer:
            continue  # skip this layer entirely
        h = layer(h, mask, None)

    logits = mi["get_logits"](h[:, -1:, :])
    mx.eval(logits)
    return logits[0, 0]


def forward_skip_layers(mi, ids, skip_set):
    """Forward pass skipping a SET of layers."""
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)

    for i, layer in enumerate(mi["layers"]):
        if i in skip_set:
            continue
        h = layer(h, mask, None)

    logits = mi["get_logits"](h[:, -1:, :])
    mx.eval(logits)
    return logits[0, 0]


def forward_full(mi, ids):
    """Standard full forward pass."""
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)

    for layer in mi["layers"]:
        h = layer(h, mask, None)

    logits = mi["get_logits"](h[:, -1:, :])
    mx.eval(logits)
    return logits[0, 0]


def analyze_layer_importance(mi, tokenizer, questions, n_tokens=10):
    """For each layer, measure how often skipping it changes the output.

    Run n_tokens of generation per question. For each token, try skipping
    each layer and see if the prediction changes.

    Returns: importance[layer_idx] = fraction of tokens where skipping changed output
    """
    n_layers = mi["n_layers"]
    change_counts = [0] * n_layers
    total_tokens = 0

    for qi, question in enumerate(questions):
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)
        eos_id = tokenizer.eos_token_id

        for step in range(n_tokens):
            # Full prediction
            full_logits = forward_full(mi, ids)
            full_tok = int(mx.argmax(full_logits).item())
            if full_tok == eos_id:
                break

            # Try skipping each layer
            for skip_l in range(n_layers):
                skip_logits = forward_skip_layer(mi, ids, skip_l)
                skip_tok = int(mx.argmax(skip_logits).item())
                if skip_tok != full_tok:
                    change_counts[skip_l] += 1

            total_tokens += 1
            ids.append(full_tok)

        if (qi + 1) % 5 == 0:
            print(f"    {qi+1}/{len(questions)} questions analyzed...")

    importance = [c / max(total_tokens, 1) for c in change_counts]
    return importance, total_tokens


def test_layer_pruning(mi, tokenizer, benchmark, skip_set):
    """Test quality when permanently skipping a set of layers."""
    results = []

    for i, (question, keywords) in enumerate(benchmark):
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)
        eos_id = tokenizer.eos_token_id
        generated = []

        for _ in range(60):
            logits = forward_skip_layers(mi, ids, skip_set)
            tok = int(mx.argmax(logits).item())
            if tok == eos_id:
                break
            generated.append(tok)
            ids.append(tok)

        answer = tokenizer.decode(generated).strip()
        if "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        correct = check_correct(answer, keywords)
        results.append({"question": question, "answer": answer, "correct": correct})

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(benchmark)} done...")

    return results


def run():
    print("=" * 70)
    print("Layer Importance Analysis — Llama-3-8B-Instruct")
    print("=" * 70)

    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    mi = setup_llama(model)
    n_layers = mi["n_layers"]
    print(f"  Model: {n_layers} layers\n")

    # Phase 1: Analyze which layers matter
    print("Phase 1: Layer importance (10 questions, 10 tokens each)")
    print("-" * 50)
    sample_questions = [q for q, _ in BENCHMARK[:10]]
    importance, total_tokens = analyze_layer_importance(
        mi, tokenizer, sample_questions, n_tokens=10)

    print(f"\n  Analyzed {total_tokens} tokens")
    print(f"\n  Layer importance (% tokens where skipping changes prediction):")
    print(f"  {'Layer':>6} {'Importance':>12} {'Bar'}")
    print(f"  " + "-" * 40)

    # Sort by importance to find the most/least important
    layer_imp = [(i, imp) for i, imp in enumerate(importance)]
    for i, imp in enumerate(importance):
        bar = "█" * int(imp * 50)
        critical = " ← CRITICAL" if imp > 0.5 else " ← skippable" if imp < 0.1 else ""
        print(f"  {i:>6} {imp*100:>10.1f}%  {bar}{critical}")

    # Find least important layers
    sorted_layers = sorted(layer_imp, key=lambda x: x[1])
    least_important = [l for l, imp in sorted_layers if imp < 0.15]

    print(f"\n  Least important layers (change <15% of predictions):")
    print(f"  {least_important}")

    # Phase 2: Test pruning the least important layers
    print(f"\nPhase 2: Pruning experiments")
    print("-" * 50)

    pruning_configs = []

    # Try removing the single least important layer
    if len(least_important) > 0:
        pruning_configs.append(
            (f"Skip layer {sorted_layers[0][0]} (least important)",
             {sorted_layers[0][0]}))

    # Try removing bottom 2, 4, 8 least important
    for n_skip in [2, 4, 8]:
        if len(sorted_layers) >= n_skip:
            skip_set = set(l for l, _ in sorted_layers[:n_skip])
            pruning_configs.append(
                (f"Skip {n_skip} least important: {sorted(skip_set)}",
                 skip_set))

    # Try removing contiguous middle blocks
    for start, end in [(10, 14), (8, 16), (12, 20), (8, 20)]:
        skip_set = set(range(start, end))
        n = len(skip_set)
        pruning_configs.append(
            (f"Skip layers {start}-{end-1} ({n} layers, {n/n_layers*100:.0f}%)",
             skip_set))

    all_results = {}
    # Get baseline first
    print(f"\n  Running baseline (all {n_layers} layers)...")
    base_results = test_layer_pruning(mi, tokenizer, BENCHMARK, set())
    base_correct = sum(r["correct"] for r in base_results)
    print(f"  Baseline: {base_correct}/100")

    for name, skip_set in pruning_configs:
        n_skip = len(skip_set)
        pct = n_skip / n_layers * 100
        speedup_theoretical = n_layers / (n_layers - n_skip)
        print(f"\n  {name}")
        print(f"  Theoretical speedup: {speedup_theoretical:.2f}x ({pct:.0f}% layers removed)")

        t0 = time.time()
        results = test_layer_pruning(mi, tokenizer, BENCHMARK, skip_set)
        elapsed = time.time() - t0

        correct = sum(r["correct"] for r in results)
        delta = correct - base_correct
        delta_s = f"+{delta}" if delta > 0 else str(delta)

        print(f"  Accuracy: {correct}/100 ({delta_s} vs baseline)")
        print(f"  Time: {elapsed:.0f}s")

        # Show what broke
        broken = [(r["question"], r["answer"]) for r, b in zip(results, base_results)
                  if b["correct"] and not r["correct"]]
        if broken:
            print(f"  Broke {len(broken)} questions:")
            for q, a in broken[:5]:
                print(f"    x {q[:50]} → '{a[:40]}'")

        all_results[name] = {
            "skip_set": sorted(skip_set),
            "n_skipped": n_skip,
            "pct_skipped": round(pct, 1),
            "theoretical_speedup": round(speedup_theoretical, 2),
            "accuracy": correct,
            "delta": delta,
        }

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY — Layer Pruning on Llama-3-8B (100 questions)")
    print(f"{'='*70}\n")
    print(f"  Baseline: {base_correct}/100\n")
    print(f"  {'Config':45} {'Acc':>5} {'Δ':>4} {'Skip%':>6} {'Spdup':>6}")
    print("  " + "-" * 70)
    for name, r in all_results.items():
        acc = f"{r['accuracy']}/100"
        delta = f"{'+' if r['delta']>=0 else ''}{r['delta']}"
        skip = f"{r['pct_skipped']}%"
        spd = f"{r['theoretical_speedup']}x"
        star = " ★" if r['delta'] >= -2 and r['pct_skipped'] >= 20 else ""
        print(f"  {name:45} {acc:>5} {delta:>4} {skip:>6} {spd:>6}{star}")

    with open("layer_importance_results.json", "w") as f:
        json.dump({
            "importance": importance,
            "total_tokens": total_tokens,
            "pruning": all_results,
            "baseline": base_correct,
        }, f, indent=2)
    print("\nSaved layer_importance_results.json")


if __name__ == "__main__":
    run()
