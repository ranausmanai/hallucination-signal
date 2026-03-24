"""
Selective MLV — Only apply multi-layer voting when the model is uncertain
==========================================================================

Key insight from the 100-question benchmark:
- MLV fixes ~5 questions but breaks ~4-7 others
- Breaks happen when MLV overrides tokens that were already correct
- Fix: only apply MLV when the model shows uncertainty

Uncertainty signals:
1. Entropy of final layer distribution (high = uncertain)
2. Disagreement between layers (many layers disagree = uncertain)
3. Low confidence of final layer's top-1 (max_prob < threshold)

This should preserve MLV's fixes while eliminating the breaks.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import json
import time
from collections import defaultdict


# Use the same 100-question benchmark
from mlv_benchmark import BENCHMARK, check_correct, is_ascii_token, setup_qwen2, setup_qwen35


def forward_voting_layers(model_info, ids, voting_layers):
    mi = model_info
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)
    voting_set = set(voting_layers)
    layer_logits = {}

    for i, layer in enumerate(mi["layers"]):
        if mi.get("get_layer_mask"):
            m = mi["get_layer_mask"](layer, mask)
        else:
            m = mask
        h = layer(h, m, None)
        if i in voting_set:
            logits_i = mi["get_logits_from_hidden"](h[:, -1:, :])
            mx.eval(logits_i)
            layer_logits[i] = logits_i[0, 0]

    return layer_logits


def decode_weighted_mlv_ascii(layer_logits, voting_layers, tokenizer, k=10):
    final_layer = voting_layers[-1]
    final = layer_logits[final_layer]
    top_k_indices = mx.argpartition(final, kth=-k, axis=-1)[-k:]
    top_k_list = [int(x) for x in top_k_indices.tolist()]
    ascii_tokens = [t for t in top_k_list if is_ascii_token(tokenizer, t)]
    if len(ascii_tokens) == 0:
        ascii_tokens = top_k_list

    weighted_votes = defaultdict(float)
    for i in voting_layers:
        logits = layer_logits[i]
        probs = mx.softmax(logits, axis=-1)
        best_tok = None
        best_prob = 0.0
        for tok in ascii_tokens:
            p = float(probs[tok].item())
            if p > best_prob:
                best_prob = p
                best_tok = tok
        if best_tok is not None:
            weighted_votes[best_tok] += best_prob

    if len(weighted_votes) == 0:
        return int(mx.argmax(final).item())
    return max(weighted_votes, key=weighted_votes.get)


def compute_uncertainty(layer_logits, voting_layers):
    """Compute multiple uncertainty signals from multi-layer logits."""
    final_layer = voting_layers[-1]
    final = layer_logits[final_layer]

    # 1. Final layer entropy
    p_final = mx.softmax(final, axis=-1)
    log_p = mx.log(p_final + 1e-10)
    entropy = -float(mx.sum(p_final * log_p).item())

    # 2. Final layer max probability (confidence)
    max_prob = float(mx.max(p_final).item())

    # 3. Layer disagreement: how many layers disagree with final on top-1
    final_top1 = int(mx.argmax(final).item())
    n_disagree = 0
    for i in voting_layers[:-1]:  # exclude final
        top_i = int(mx.argmax(layer_logits[i]).item())
        if top_i != final_top1:
            n_disagree += 1
    disagreement_ratio = n_disagree / max(len(voting_layers) - 1, 1)

    return {
        "entropy": entropy,
        "max_prob": max_prob,
        "disagreement": disagreement_ratio,
        "n_disagree": n_disagree,
    }


def generate_selective_mlv(model_info, tokenizer, question, voting_layers,
                            k=10, uncertainty_check=None, max_tokens=60):
    """Generate with MLV applied only when uncertainty check passes."""
    mi = model_info
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    overrides = 0
    interventions = 0  # times we actually applied MLV

    for step in range(max_tokens):
        layer_logits = forward_voting_layers(mi, ids, voting_layers)
        final_layer = voting_layers[-1]
        std_pick = int(mx.argmax(layer_logits[final_layer]).item())

        # Check if we should apply MLV
        if uncertainty_check is not None:
            unc = compute_uncertainty(layer_logits, voting_layers)
            should_intervene = uncertainty_check(unc)
        else:
            should_intervene = True  # always apply (baseline)

        if should_intervene:
            interventions += 1
            mlv_pick = decode_weighted_mlv_ascii(
                layer_logits, voting_layers, tokenizer, k=k)
        else:
            mlv_pick = std_pick

        if mlv_pick != std_pick:
            overrides += 1

        if mlv_pick == eos_id:
            break
        generated.append(mlv_pick)
        ids.append(mlv_pick)
        del layer_logits

    return tokenizer.decode(generated).strip(), overrides, interventions


def evaluate(model, model_info, tokenizer, name, voting_layers, k,
             uncertainty_check, benchmark):
    results = []
    total_ov = 0
    total_int = 0

    for i, (question, keywords) in enumerate(benchmark):
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        std_answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                              verbose=False, sampler=sampler).strip()
        std_correct = check_correct(std_answer, keywords)

        mlv_answer, ov, ints = generate_selective_mlv(
            model_info, tokenizer, question, voting_layers, k=k,
            uncertainty_check=uncertainty_check)
        mlv_correct = check_correct(mlv_answer, keywords)
        total_ov += ov
        total_int += ints

        results.append({
            "question": question, "std_answer": std_answer, "std_correct": std_correct,
            "mlv_answer": mlv_answer, "mlv_correct": mlv_correct,
            "overrides": ov, "interventions": ints,
        })
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(benchmark)} done...")

    std_total = sum(r["std_correct"] for r in results)
    mlv_total = sum(r["mlv_correct"] for r in results)
    fixes = sum(1 for r in results if r["mlv_correct"] and not r["std_correct"])
    breaks = sum(1 for r in results if not r["mlv_correct"] and r["std_correct"])
    net = fixes - breaks

    print(f"\n  {name}")
    print(f"  Baseline: {std_total}/{len(results)}")
    print(f"  MLV:      {mlv_total}/{len(results)} | +{fixes}/-{breaks} = {'+'if net>=0 else ''}{net}")
    print(f"  Interventions: {total_int} tokens checked, {total_ov} overrides")

    if fixes > 0:
        print(f"  Fixed:")
        for r in results:
            if r["mlv_correct"] and not r["std_correct"]:
                print(f"    ✓ {r['question'][:55]}")
                print(f"      was: '{r['std_answer'][:50]}'")
                print(f"      now: '{r['mlv_answer'][:50]}'")

    if breaks > 0:
        print(f"  Broken:")
        for r in results:
            if not r["mlv_correct"] and r["std_correct"]:
                print(f"    ✗ {r['question'][:55]}")
                print(f"      was: '{r['std_answer'][:50]}'")
                print(f"      now: '{r['mlv_answer'][:50]}'")

    return fixes, breaks, net


def run():
    print("=" * 70)
    print("Selective MLV — Uncertainty-Gated Multi-Layer Voting")
    print("=" * 70)

    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    mi = setup_qwen2(model)
    voting_layers = [20, 21, 22, 23]

    strategies = [
        # Baseline: always apply MLV
        ("Always MLV k=10", 10, None),

        # Only when final layer confidence is low
        ("MLV when max_prob < 0.5", 10,
         lambda u: u["max_prob"] < 0.5),
        ("MLV when max_prob < 0.7", 10,
         lambda u: u["max_prob"] < 0.7),
        ("MLV when max_prob < 0.9", 10,
         lambda u: u["max_prob"] < 0.9),

        # Only when entropy is high
        ("MLV when entropy > 2.0", 10,
         lambda u: u["entropy"] > 2.0),
        ("MLV when entropy > 3.0", 10,
         lambda u: u["entropy"] > 3.0),
        ("MLV when entropy > 5.0", 10,
         lambda u: u["entropy"] > 5.0),

        # Only when layers disagree
        ("MLV when disagreement > 0", 10,
         lambda u: u["n_disagree"] > 0),
        ("MLV when disagreement >= 2", 10,
         lambda u: u["n_disagree"] >= 2),
        ("MLV when disagreement == 3", 10,
         lambda u: u["n_disagree"] == 3),

        # Combined: high entropy AND disagreement
        ("MLV when entropy>3 AND disagree>0", 10,
         lambda u: u["entropy"] > 3.0 and u["n_disagree"] > 0),
        ("MLV when entropy>2 AND disagree>=2", 10,
         lambda u: u["entropy"] > 2.0 and u["n_disagree"] >= 2),

        # Combined: low confidence AND disagreement
        ("MLV when prob<0.7 AND disagree>0", 10,
         lambda u: u["max_prob"] < 0.7 and u["n_disagree"] > 0),
        ("MLV when prob<0.5 AND disagree>=2", 10,
         lambda u: u["max_prob"] < 0.5 and u["n_disagree"] >= 2),
    ]

    all_results = {}
    for name, k, check in strategies:
        print(f"\n  --- {name} ---")
        fixes, breaks, net = evaluate(
            model, mi, tokenizer, name, voting_layers, k, check, BENCHMARK)
        all_results[name] = {"fixes": fixes, "breaks": breaks, "net": net}

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY — Selective MLV on Qwen2-0.5B (100 questions)")
    print(f"{'='*70}\n")
    print(f"  {'Strategy':42} {'Fix':>5} {'Brk':>5} {'Net':>5}")
    print("  " + "-" * 60)
    for name, r in all_results.items():
        net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net'])
        star = " ★" if r['net'] > 0 and r['breaks'] == 0 else " ✓" if r['net'] > 0 else ""
        print(f"  {name:42} +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")


if __name__ == "__main__":
    run()
