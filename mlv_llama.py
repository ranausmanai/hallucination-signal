"""
MLV on Llama-3-8B-Instruct — Cross-architecture validation
=============================================================
If MLV works on Llama (different architecture, 8B params, 32 layers),
that's strong evidence it generalizes.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.llama import create_attention_mask
from collections import defaultdict
import json
import time

from mlv_benchmark import BENCHMARK, check_correct, is_ascii_token


def setup_llama(model):
    def get_logits(h):
        h_normed = model.model.norm(h)
        return model.lm_head(h_normed)
    return {
        "embed": model.model.embed_tokens,
        "layers": model.model.layers,
        "norm": model.model.norm,
        "get_logits_from_hidden": get_logits,
        "get_mask": lambda h: create_attention_mask(h, None),
        "get_layer_mask": None,
        "n_layers": len(model.model.layers),
    }


def forward_voting_layers(model_info, ids, voting_layers):
    mi = model_info
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)
    voting_set = set(voting_layers)
    layer_logits = {}

    for i, layer in enumerate(mi["layers"]):
        h = layer(h, mask, None)
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
    final_layer = voting_layers[-1]
    final = layer_logits[final_layer]
    p_final = mx.softmax(final, axis=-1)
    log_p = mx.log(p_final + 1e-10)
    entropy = -float(mx.sum(p_final * log_p).item())
    max_prob = float(mx.max(p_final).item())
    final_top1 = int(mx.argmax(final).item())
    n_disagree = sum(1 for i in voting_layers[:-1]
                     if int(mx.argmax(layer_logits[i]).item()) != final_top1)
    return {"entropy": entropy, "max_prob": max_prob, "n_disagree": n_disagree}


def generate_selective_mlv(model_info, tokenizer, question, voting_layers,
                            k=10, uncertainty_check=None, max_tokens=60):
    mi = model_info
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    overrides = 0

    for step in range(max_tokens):
        layer_logits = forward_voting_layers(mi, ids, voting_layers)
        final_layer = voting_layers[-1]
        std_pick = int(mx.argmax(layer_logits[final_layer]).item())

        if uncertainty_check is not None:
            unc = compute_uncertainty(layer_logits, voting_layers)
            should_intervene = uncertainty_check(unc)
        else:
            should_intervene = True

        if should_intervene:
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

    return tokenizer.decode(generated).strip(), overrides


def evaluate(model, mi, tokenizer, name, voting_layers, k,
             uncertainty_check, benchmark):
    results = []
    total_ov = 0
    t0 = time.time()

    for i, (question, keywords) in enumerate(benchmark):
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        std_answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                              verbose=False, sampler=sampler).strip()
        # Clean Llama output (sometimes generates multiple turns)
        if "<|eot_id|>" in std_answer:
            std_answer = std_answer.split("<|eot_id|>")[0].strip()
        std_correct = check_correct(std_answer, keywords)

        mlv_answer, ov = generate_selective_mlv(
            mi, tokenizer, question, voting_layers, k=k,
            uncertainty_check=uncertainty_check)
        if "<|eot_id|>" in mlv_answer:
            mlv_answer = mlv_answer.split("<|eot_id|>")[0].strip()
        mlv_correct = check_correct(mlv_answer, keywords)
        total_ov += ov

        results.append({
            "question": question, "std_answer": std_answer, "std_correct": std_correct,
            "mlv_answer": mlv_answer, "mlv_correct": mlv_correct, "overrides": ov,
        })
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(benchmark)} done... ({elapsed:.0f}s)")

    std_total = sum(r["std_correct"] for r in results)
    mlv_total = sum(r["mlv_correct"] for r in results)
    fixes = sum(1 for r in results if r["mlv_correct"] and not r["std_correct"])
    breaks = sum(1 for r in results if not r["mlv_correct"] and r["std_correct"])
    net = fixes - breaks

    print(f"\n  {name}")
    print(f"  Baseline: {std_total}/{len(results)}")
    print(f"  MLV:      {mlv_total}/{len(results)} | +{fixes}/-{breaks} = {'+'if net>=0 else ''}{net}")
    print(f"  Overrides: {total_ov}")

    # Category breakdown
    categories = [
        ("Capitals", 0, 20), ("Geography", 20, 30), ("Science", 30, 50),
        ("History", 50, 65), ("Literature", 65, 80), ("General", 80, 90),
        ("Math", 90, 100),
    ]
    print(f"\n  Category breakdown:")
    for cat_name, start, end in categories:
        base_cat = sum(r["std_correct"] for r in results[start:end])
        mlv_cat = sum(r["mlv_correct"] for r in results[start:end])
        n = end - start
        diff = mlv_cat - base_cat
        diff_s = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "="
        print(f"    {cat_name:15} {base_cat:2d}→{mlv_cat:2d}/{n:2d} ({diff_s})")

    if fixes > 0:
        print(f"\n  Fixed ({fixes}):")
        for r in results:
            if r["mlv_correct"] and not r["std_correct"]:
                print(f"    ✓ {r['question'][:55]}")
                print(f"      was: '{r['std_answer'][:55]}'")
                print(f"      now: '{r['mlv_answer'][:55]}'")

    if breaks > 0:
        print(f"\n  Broken ({breaks}):")
        for r in results:
            if not r["mlv_correct"] and r["std_correct"]:
                print(f"    ✗ {r['question'][:55]}")
                print(f"      was: '{r['std_answer'][:55]}'")
                print(f"      now: '{r['mlv_answer'][:55]}'")

    return fixes, breaks, net


def run():
    print("=" * 70)
    print("MLV on Llama-3-8B-Instruct (4-bit) — Cross-Architecture Test")
    print("=" * 70)

    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    mi = setup_llama(model)
    n_layers = mi["n_layers"]
    print(f"  Model: Llama-3-8B-Instruct, {n_layers} layers")

    # Llama has 32 layers (0-31). Try last 4, last 6, last 8
    strategies = [
        # Best strategies from Qwen2 experiments
        ("Always MLV L28-31 k=10", list(range(28, 32)), 10, None),
        ("MLV L28-31 k=10 prob<0.7", list(range(28, 32)), 10,
         lambda u: u["max_prob"] < 0.7),
        ("MLV L28-31 k=10 entropy>5", list(range(28, 32)), 10,
         lambda u: u["entropy"] > 5.0),

        # Try wider layer range (more voters)
        ("Always MLV L26-31 k=10", list(range(26, 32)), 10, None),
        ("MLV L26-31 k=10 prob<0.7", list(range(26, 32)), 10,
         lambda u: u["max_prob"] < 0.7),

        # Try narrower (just last 3)
        ("MLV L29-31 k=10 prob<0.7", list(range(29, 32)), 10,
         lambda u: u["max_prob"] < 0.7),
    ]

    all_results = {}
    for name, voting_layers, k, check in strategies:
        print(f"\n{'='*70}")
        print(f"  --- {name} ---")
        fixes, breaks, net = evaluate(
            model, mi, tokenizer, name, voting_layers, k, check, BENCHMARK)
        all_results[name] = {"fixes": fixes, "breaks": breaks, "net": net}

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY — Llama-3-8B-Instruct (100 questions)")
    print(f"{'='*70}\n")
    print(f"  {'Strategy':40} {'Fix':>5} {'Brk':>5} {'Net':>5}")
    print("  " + "-" * 55)
    for name, r in all_results.items():
        net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net'])
        star = " ★" if r['net'] > 0 and r['breaks'] == 0 else " ✓" if r['net'] > 0 else ""
        print(f"  {name:40} +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")

    print(f"\n  Cross-model comparison (best selective MLV):")
    print(f"    Qwen2-0.5B (24L): prob<0.7 → +5/-2 = +3")
    print(f"    Qwen3.5-0.8B (24L): always → +5/-7 = -2")
    print(f"    Llama-3-8B (32L): see above")

    with open("mlv_llama_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved mlv_llama_results.json")


if __name__ == "__main__":
    run()
