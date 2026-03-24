"""
Early Exit / Adaptive Compute — Skip layers when the model is already confident
================================================================================

v2: Uses LIGHTWEIGHT exit criteria (no expensive lm_head projection at checkpoints)

Key insight from v1: Computing logits at each checkpoint (norm + lm_head → 128K vocab)
is MORE expensive than just running the remaining layers. The exit check itself was the
bottleneck.

Solution: Use cheap signals to decide whether to exit early:
1. Hidden state convergence: if h[layer_i] ≈ h[layer_i-1], remaining layers won't help
2. Hidden state norm stability: if the norm stops changing, the representation has settled
3. Only compute logits ONCE — at the exit point, not at every checkpoint
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
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
        "get_logits_from_hidden": get_logits,
        "get_mask": lambda h: create_attention_mask(h, None),
        "n_layers": len(model.model.layers),
    }


# =============================================================
# Baseline: full forward pass, no shortcuts
# =============================================================

def generate_baseline(model_info, tokenizer, question, max_tokens=60):
    """Generate with full forward pass at every token."""
    mi = model_info
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []

    for step in range(max_tokens):
        ids_mx = mx.array([ids])
        h = mi["embed"](ids_mx)
        mask = mi["get_mask"](h)

        for layer in mi["layers"]:
            h = layer(h, mask, None)

        logits = mi["get_logits_from_hidden"](h[:, -1:, :])
        mx.eval(logits)
        tok = int(mx.argmax(logits[0, 0]).item())

        if tok == eos_id:
            break
        generated.append(tok)
        ids.append(tok)

    answer = tokenizer.decode(generated).strip()
    if "<|eot_id|>" in answer:
        answer = answer.split("<|eot_id|>")[0].strip()
    return answer


# =============================================================
# Early Exit with cosine similarity (lightweight check)
# =============================================================

def forward_early_exit_cosine(model_info, ids, min_layer, threshold):
    """Exit early when consecutive hidden states converge (cosine similarity > threshold).

    Only checks convergence after min_layer. When hidden states stop changing,
    the model has "made up its mind" — remaining layers are just noise.

    Cost of check: one dot product + two norms = negligible vs a full layer.
    """
    mi = model_info
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)
    n_layers = mi["n_layers"]
    prev_h_last = None  # previous layer's last-token hidden state

    for i, layer in enumerate(mi["layers"]):
        h = layer(h, mask, None)

        if i >= min_layer:
            h_last = h[:, -1:, :]  # [1, 1, dim]

            if prev_h_last is not None:
                # Cosine similarity between consecutive layers
                dot = mx.sum(h_last * prev_h_last)
                norm_a = mx.sqrt(mx.sum(h_last * h_last) + 1e-8)
                norm_b = mx.sqrt(mx.sum(prev_h_last * prev_h_last) + 1e-8)
                cos_sim = float((dot / (norm_a * norm_b)).item())

                if cos_sim >= threshold:
                    # Hidden state has converged — exit here
                    logits = mi["get_logits_from_hidden"](h[:, -1:, :])
                    mx.eval(logits)
                    tok = int(mx.argmax(logits[0, 0]).item())
                    return tok, i + 1

            prev_h_last = h_last

    # Full pass — use final layer
    logits = mi["get_logits_from_hidden"](h[:, -1:, :])
    mx.eval(logits)
    tok = int(mx.argmax(logits[0, 0]).item())
    return tok, n_layers


# =============================================================
# Early Exit with norm change (even cheaper)
# =============================================================

def forward_early_exit_norm(model_info, ids, min_layer, threshold):
    """Exit early when the relative change in hidden state norm is small.

    If ||h_i|| ≈ ||h_{i-1}||, the layer barely changed the representation.
    When multiple consecutive layers show tiny norm changes, exit.

    threshold: relative norm change below which we exit (e.g., 0.01 = 1%)
    """
    mi = model_info
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)
    n_layers = mi["n_layers"]
    prev_norm = None
    stable_count = 0
    stable_needed = 2  # need 2 consecutive stable layers to exit

    for i, layer in enumerate(mi["layers"]):
        h = layer(h, mask, None)

        if i >= min_layer:
            h_last = h[0, -1, :]  # [dim]
            curr_norm = float(mx.sqrt(mx.sum(h_last * h_last)).item())

            if prev_norm is not None:
                rel_change = abs(curr_norm - prev_norm) / (prev_norm + 1e-8)
                if rel_change < threshold:
                    stable_count += 1
                    if stable_count >= stable_needed:
                        logits = mi["get_logits_from_hidden"](h[:, -1:, :])
                        mx.eval(logits)
                        tok = int(mx.argmax(logits[0, 0]).item())
                        return tok, i + 1
                else:
                    stable_count = 0

            prev_norm = curr_norm

    logits = mi["get_logits_from_hidden"](h[:, -1:, :])
    mx.eval(logits)
    tok = int(mx.argmax(logits[0, 0]).item())
    return tok, n_layers


# =============================================================
# Early Exit with max prob (original, but only check once)
# =============================================================

def forward_early_exit_single_check(model_info, ids, check_layer, threshold):
    """Check confidence at ONE layer only. If confident, exit. Otherwise run full.

    This avoids the multi-checkpoint overhead. One check = one lm_head call.
    If it's confident at layer `check_layer`, we save (n_layers - check_layer) layers.
    """
    mi = model_info
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)
    n_layers = mi["n_layers"]

    for i, layer in enumerate(mi["layers"]):
        h = layer(h, mask, None)

        if i == check_layer:
            logits = mi["get_logits_from_hidden"](h[:, -1:, :])
            mx.eval(logits)
            logits_1d = logits[0, 0]
            probs = mx.softmax(logits_1d, axis=-1)
            max_prob = float(mx.max(probs).item())

            if max_prob >= threshold:
                tok = int(mx.argmax(logits_1d).item())
                return tok, i + 1

    logits = mi["get_logits_from_hidden"](h[:, -1:, :])
    mx.eval(logits)
    tok = int(mx.argmax(logits[0, 0]).item())
    return tok, n_layers


# =============================================================
# Generic generate with any early exit strategy
# =============================================================

def generate_early_exit(model_info, tokenizer, question, exit_fn, max_tokens=60):
    """Generate using an arbitrary early exit function.

    exit_fn(model_info, ids) -> (token_id, layers_used)
    """
    mi = model_info
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    total_layers = 0
    early_exits = 0
    n_layers = mi["n_layers"]

    for step in range(max_tokens):
        tok, layers_used = exit_fn(mi, ids)
        total_layers += layers_used
        if layers_used < n_layers:
            early_exits += 1
        if tok == eos_id:
            break
        generated.append(tok)
        ids.append(tok)

    answer = tokenizer.decode(generated).strip()
    if "<|eot_id|>" in answer:
        answer = answer.split("<|eot_id|>")[0].strip()

    n_tok = max(len(generated), 1)
    return answer, {
        "tokens": len(generated),
        "total_layers": total_layers,
        "early_exits": early_exits,
        "avg_layers": round(total_layers / n_tok, 1),
        "pct_saved": round((1 - total_layers / (n_tok * n_layers)) * 100, 1),
    }


# =============================================================
# Evaluation
# =============================================================

def evaluate(model_info, tokenizer, benchmark, name, exit_fn):
    mi = model_info
    n_layers = mi["n_layers"]
    results = []
    t_baseline = 0
    t_early = 0

    for i, (question, keywords) in enumerate(benchmark):
        # Baseline
        t0 = time.time()
        base_answer = generate_baseline(mi, tokenizer, question)
        t1 = time.time()
        t_baseline += (t1 - t0)
        base_correct = check_correct(base_answer, keywords)

        # Early exit
        t0 = time.time()
        ee_answer, ee_stats = generate_early_exit(mi, tokenizer, question, exit_fn)
        t1 = time.time()
        t_early += (t1 - t0)
        ee_correct = check_correct(ee_answer, keywords)

        results.append({
            "question": question,
            "base_answer": base_answer, "base_correct": base_correct,
            "ee_answer": ee_answer, "ee_correct": ee_correct,
            "ee_stats": ee_stats,
        })
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(benchmark)} done...")

    base_total = sum(r["base_correct"] for r in results)
    ee_total = sum(r["ee_correct"] for r in results)
    breaks = sum(1 for r in results if not r["ee_correct"] and r["base_correct"])
    fixes = sum(1 for r in results if r["ee_correct"] and not r["base_correct"])

    total_ee_layers = sum(r["ee_stats"]["total_layers"] for r in results)
    total_tokens = sum(r["ee_stats"]["tokens"] for r in results)
    total_early = sum(r["ee_stats"]["early_exits"] for r in results)
    avg_layers = total_ee_layers / max(total_tokens, 1)
    pct_saved = (1 - avg_layers / n_layers) * 100
    exit_rate = total_early / max(total_tokens, 1) * 100

    speedup = t_baseline / max(t_early, 0.01)

    print(f"\n  {name}")
    print(f"  Baseline: {base_total}/100 in {t_baseline:.0f}s ({total_tokens/t_baseline:.1f} tok/s)")
    print(f"  EarlyExit: {ee_total}/100 in {t_early:.0f}s ({total_tokens/t_early:.1f} tok/s)")
    print(f"  Quality: +{fixes}/-{breaks}")
    print(f"  Wall-clock speedup: {speedup:.2f}x")
    print(f"  Layers: {avg_layers:.1f}/{n_layers} avg ({pct_saved:.1f}% saved)")
    print(f"  Early exits: {exit_rate:.0f}% of tokens")

    if breaks > 0:
        print(f"\n  Broken ({breaks}):")
        for r in results:
            if not r["ee_correct"] and r["base_correct"]:
                print(f"    x {r['question'][:55]}")
                print(f"      was: '{r['base_answer'][:55]}'")
                print(f"      now: '{r['ee_answer'][:55]}'")

    return {
        "base_correct": base_total, "ee_correct": ee_total,
        "fixes": fixes, "breaks": breaks,
        "speedup": round(speedup, 2),
        "layers_saved_pct": round(pct_saved, 1),
        "avg_layers": round(avg_layers, 1),
        "exit_rate_pct": round(exit_rate, 1),
        "base_time": round(t_baseline, 1),
        "ee_time": round(t_early, 1),
        "base_tps": round(total_tokens / t_baseline, 1),
        "ee_tps": round(total_tokens / t_early, 1),
    }


def run():
    print("=" * 70)
    print("Early Exit v2 — Lightweight Exit Criteria")
    print("=" * 70)

    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    mi = setup_llama(model)
    n = mi["n_layers"]
    print(f"  Model: Llama-3-8B-Instruct 4-bit, {n} layers\n")

    # Warmup
    print("  Warming up...")
    _ = generate_baseline(mi, tokenizer, "What is 2+2?")
    print("  Ready.\n")

    configs = [
        # --- Cosine similarity: exit when hidden states converge ---
        ("Cosine min=16 thr=0.99",
         lambda mi, ids: forward_early_exit_cosine(mi, ids, min_layer=16, threshold=0.99)),
        ("Cosine min=16 thr=0.995",
         lambda mi, ids: forward_early_exit_cosine(mi, ids, min_layer=16, threshold=0.995)),
        ("Cosine min=20 thr=0.99",
         lambda mi, ids: forward_early_exit_cosine(mi, ids, min_layer=20, threshold=0.99)),
        ("Cosine min=12 thr=0.99",
         lambda mi, ids: forward_early_exit_cosine(mi, ids, min_layer=12, threshold=0.99)),

        # --- Norm stability: exit when hidden norm stops changing ---
        ("Norm min=16 thr=0.01",
         lambda mi, ids: forward_early_exit_norm(mi, ids, min_layer=16, threshold=0.01)),
        ("Norm min=16 thr=0.005",
         lambda mi, ids: forward_early_exit_norm(mi, ids, min_layer=16, threshold=0.005)),
        ("Norm min=20 thr=0.01",
         lambda mi, ids: forward_early_exit_norm(mi, ids, min_layer=20, threshold=0.01)),

        # --- Single checkpoint: one lm_head call, saves max layers ---
        ("SingleCheck@20 thr=0.95",
         lambda mi, ids: forward_early_exit_single_check(mi, ids, check_layer=20, threshold=0.95)),
        ("SingleCheck@20 thr=0.90",
         lambda mi, ids: forward_early_exit_single_check(mi, ids, check_layer=20, threshold=0.90)),
        ("SingleCheck@16 thr=0.95",
         lambda mi, ids: forward_early_exit_single_check(mi, ids, check_layer=16, threshold=0.95)),
    ]

    all_results = {}
    for name, exit_fn in configs:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        r = evaluate(mi, tokenizer, BENCHMARK, name, exit_fn)
        all_results[name] = r

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY — Early Exit v2: Llama-3-8B-Instruct (100 questions)")
    print(f"{'='*70}\n")
    print(f"  {'Config':35} {'Acc':>5} {'Speed':>6} {'Saved':>6} {'Exit%':>6} {'Brk':>4}")
    print("  " + "-" * 65)
    for name, r in all_results.items():
        acc = f"{r['ee_correct']}/100"
        spd = f"{r['speedup']}x"
        saved = f"{r['layers_saved_pct']}%"
        exit_r = f"{r['exit_rate_pct']}%"
        brk = str(r['breaks'])
        star = " ★" if r['speedup'] >= 1.1 and r['breaks'] <= 1 else ""
        print(f"  {name:35} {acc:>5} {spd:>6} {saved:>6} {exit_r:>6} {brk:>4}{star}")

    with open("early_exit_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved early_exit_results.json")


if __name__ == "__main__":
    run()
