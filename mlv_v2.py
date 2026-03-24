"""
Multi-Layer Voting v2 — Hardened + Cross-Model
================================================

Best result from v1: Weighted MLV L20-23 k=10 = +4/-1 = net +3
(60% of SC's corrections at <1% compute overhead)

v2 improvements:
1. ASCII filter: reject non-ASCII tokens from votes (fixes Chinese leak)
2. Adaptive k: vary top-k filter based on final layer's entropy
3. Test on both Qwen2-0.5B and Qwen3.5-0.8B
4. Combined: MLV + oscillation trigger (only apply MLV on uncertain tokens)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import time


TEST_DATA = [
    ("Who was the 30th president of the United States?",       ["coolidge", "calvin"]),
    ("What is the melting point of tungsten in Celsius?",      ["3422", "3400", "3,422"]),
    ("What is the half-life of uranium-235 in years?",         ["703", "700"]),
    ("What is the largest desert in the world by area?",       ["antarctica", "antarctic"]),
    ("What country has the most UNESCO World Heritage Sites?",  ["italy", "china"]),
    ("What is the atomic weight of plutonium?",                ["244", "242", "239"]),
    ("What is the rarest blood type?",                         ["ab-", "ab negative"]),
    ("What is the speed of sound in water in m/s?",            ["1480", "1500", "1498", "1,480", "1,500"]),
    ("What is the half-life of Carbon-14 in years?",           ["5730", "5700", "5,730"]),
    ("What year was the WHO founded?",                         ["1948"]),
    ("Who was the first female prime minister of the UK?",     ["thatcher"]),
    ("What is the capital of Kyrgyzstan?",                     ["bishkek"]),
    ("What is the capital of Burkina Faso?",                   ["ouagadougou"]),
    ("What is the capital of France?",                         ["paris"]),
    ("Who wrote Romeo and Juliet?",                            ["shakespeare"]),
    ("What is 2 + 2?",                                         ["4", "four"]),
    ("What is the largest planet?",                            ["jupiter"]),
    ("What is the boiling point of water?",                    ["100"]),
    ("What language is spoken in Brazil?",                     ["portuguese"]),
    ("What is the largest ocean?",                             ["pacific"]),
]


def check_correct(answer, keywords):
    answer_clean = answer.lower().replace(",", "")
    return any(kw.lower().replace(",", "") in answer_clean for kw in keywords)


def is_ascii_token(tokenizer, tok_id):
    """Check if a token is primarily ASCII (Latin alphabet, numbers, punctuation)."""
    text = tokenizer.decode([tok_id])
    # Allow tokens that are mostly ASCII
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars >= len(text) * 0.5 if len(text) > 0 else True


# ============================================================
# Model setup
# ============================================================

def setup_qwen2(model):
    from mlx_lm.models.qwen2 import create_attention_mask as cam
    def get_logits(h):
        h_normed = model.model.norm(h)
        if model.args.tie_word_embeddings:
            return model.model.embed_tokens.as_linear(h_normed)
        return model.lm_head(h_normed)
    return {
        "embed": model.model.embed_tokens,
        "layers": model.model.layers,
        "norm": model.model.norm,
        "get_logits_from_hidden": get_logits,
        "get_mask": lambda h: cam(h, None),
        "get_layer_mask": None,
        "hidden_dim": model.args.hidden_size,
        "n_layers": len(model.model.layers),
    }


def setup_qwen35(model):
    from mlx_lm.models.qwen3_5 import create_attention_mask as cam, create_ssm_mask as csm
    lm = model.language_model
    tm = lm.model
    masks = {}
    def get_mask(h):
        masks["fa"] = cam(h, None)
        masks["ssm"] = csm(h, None)
        return masks
    def get_layer_mask(layer, mask_dict):
        return mask_dict["ssm"] if layer.is_linear else mask_dict["fa"]
    def get_logits(h):
        h_normed = tm.norm(h)
        return tm.embed_tokens.as_linear(h_normed)
    return {
        "embed": tm.embed_tokens,
        "layers": tm.layers,
        "norm": tm.norm,
        "get_logits_from_hidden": get_logits,
        "get_mask": get_mask,
        "get_layer_mask": get_layer_mask,
        "hidden_dim": lm.args.hidden_size,
        "n_layers": len(tm.layers),
    }


# ============================================================
# Forward pass collecting multi-layer logits
# ============================================================

def forward_all_layers(model_info, ids, voting_layers):
    """Single forward pass, collect logits from specified layers only."""
    mi = model_info
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)

    layer_logits = {}  # only store what we need

    voting_set = set(voting_layers)

    for i, layer in enumerate(mi["layers"]):
        if mi.get("get_layer_mask"):
            m = mi["get_layer_mask"](layer, mask)
        else:
            m = mask
        h = layer(h, m, None)
        if i in voting_set:
            logits_i = mi["get_logits_from_hidden"](h[:, -1:, :])
            mx.eval(logits_i)
            layer_logits[i] = logits_i[0, 0]  # (V,)

    return layer_logits


# ============================================================
# Decoding: Weighted MLV with ASCII filter
# ============================================================

def decode_weighted_mlv_ascii(layer_logits, voting_layers, tokenizer, k=10):
    """
    Weighted MLV with ASCII language filter:
    1. Get final layer's top-K tokens
    2. Filter to ASCII-only tokens
    3. Each layer votes for its preferred token from filtered set
    4. Weight by confidence
    """
    from collections import defaultdict
    final_layer = voting_layers[-1]
    final = layer_logits[final_layer]

    # Get final's top-K
    top_k_indices = mx.argpartition(final, kth=-k, axis=-1)[-k:]
    top_k_list = [int(x) for x in top_k_indices.tolist()]

    # Filter to ASCII tokens
    ascii_tokens = [t for t in top_k_list if is_ascii_token(tokenizer, t)]
    if len(ascii_tokens) == 0:
        ascii_tokens = top_k_list  # fallback

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


def decode_weighted_mlv_nofilter(layer_logits, voting_layers, k=10):
    """Original weighted MLV without ASCII filter (for comparison)."""
    from collections import defaultdict
    final_layer = voting_layers[-1]
    final = layer_logits[final_layer]
    top_k_indices = mx.argpartition(final, kth=-k, axis=-1)[-k:]
    top_k_set = set(int(x) for x in top_k_indices.tolist())

    weighted_votes = defaultdict(float)
    for i in voting_layers:
        logits = layer_logits[i]
        probs = mx.softmax(logits, axis=-1)
        best_tok = None
        best_prob = 0.0
        for tok in top_k_set:
            p = float(probs[tok].item())
            if p > best_prob:
                best_prob = p
                best_tok = tok
        if best_tok is not None:
            weighted_votes[best_tok] += best_prob

    if len(weighted_votes) == 0:
        return int(mx.argmax(final).item())
    return max(weighted_votes, key=weighted_votes.get)


# ============================================================
# Generation
# ============================================================

def generate_mlv(model_info, tokenizer, question, decode_fn, voting_layers,
                 max_tokens=60):
    mi = model_info
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    overrides = 0

    for step in range(max_tokens):
        layer_logits = forward_all_layers(mi, ids, voting_layers)

        final_layer = voting_layers[-1]
        std_pick = int(mx.argmax(layer_logits[final_layer]).item())
        mlv_pick = decode_fn(layer_logits, voting_layers)

        if mlv_pick != std_pick:
            overrides += 1

        if mlv_pick == eos_id:
            break
        generated.append(mlv_pick)
        ids.append(mlv_pick)
        del layer_logits

    return tokenizer.decode(generated).strip(), overrides


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, model_info, tokenizer, name, decode_fn, voting_layers, test_data):
    results = []
    n_hard = 13
    total_ov = 0

    for question, keywords in test_data:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        std_answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                              verbose=False, sampler=sampler).strip()
        std_correct = check_correct(std_answer, keywords)

        mlv_answer, ov = generate_mlv(model_info, tokenizer, question,
                                       decode_fn, voting_layers)
        mlv_correct = check_correct(mlv_answer, keywords)
        total_ov += ov

        results.append({
            "question": question, "std_answer": std_answer, "std_correct": std_correct,
            "mlv_answer": mlv_answer, "mlv_correct": mlv_correct, "overrides": ov,
        })

    std_total = sum(r["std_correct"] for r in results)
    std_hard = sum(r["std_correct"] for r in results[:n_hard])
    mlv_total = sum(r["mlv_correct"] for r in results)
    mlv_hard = sum(r["mlv_correct"] for r in results[:n_hard])
    fixes = sum(1 for r in results if r["mlv_correct"] and not r["std_correct"])
    breaks = sum(1 for r in results if not r["mlv_correct"] and r["std_correct"])
    net = fixes - breaks

    print(f"\n  {name} ({total_ov} overrides)")
    print(f"  Standard: {std_total}/{len(results)} total, {std_hard}/{n_hard} hard")
    print(f"  MLV:      {mlv_total}/{len(results)} total, {mlv_hard}/{n_hard} hard")
    print(f"  Net: +{fixes} fixed, -{breaks} broken = {'+'if net>0 else ''}{net}")

    for r in results:
        s = "✓" if r["std_correct"] else "✗"
        m = "✓" if r["mlv_correct"] else "✗"
        tag = ""
        if r["mlv_correct"] and not r["std_correct"]:
            tag = " *** FIXED ***"
        elif not r["mlv_correct"] and r["std_correct"]:
            tag = " *** BROKE ***"
        ov = f" ({r['overrides']}ov)" if r["overrides"] > 0 else ""
        print(f"  [std={s} mlv={m}]{tag}{ov} {r['question'][:50]}")
        if tag:
            print(f"    std: '{r['std_answer'][:60]}'")
            print(f"    mlv: '{r['mlv_answer'][:60]}'")

    return fixes, breaks, net


# ============================================================
# Main
# ============================================================

def run():
    all_model_results = {}

    # ==========================================
    # Qwen2-0.5B
    # ==========================================
    print("=" * 70)
    print("MLV v2 — Qwen2-0.5B-Instruct")
    print("=" * 70)

    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    mi = setup_qwen2(model)

    strategies = [
        ("Weighted MLV L20-23 k=10 (baseline)",
         [20, 21, 22, 23],
         lambda ll, vl: decode_weighted_mlv_nofilter(ll, vl, k=10)),
        ("Weighted MLV L20-23 k=10 +ASCII",
         [20, 21, 22, 23],
         lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=10)),
        ("Weighted MLV L20-23 k=20 +ASCII",
         [20, 21, 22, 23],
         lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=20)),
        ("Weighted MLV L19-23 k=10 +ASCII",
         [19, 20, 21, 22, 23],
         lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=10)),
        ("Weighted MLV L21-23 k=10 +ASCII",
         [21, 22, 23],
         lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=10)),
    ]

    qwen2_results = {}
    for name, voting_layers, decode_fn in strategies:
        print(f"\n  --- {name} ---")
        fixes, breaks, net = evaluate(
            model, mi, tokenizer, name, decode_fn, voting_layers, TEST_DATA)
        qwen2_results[name] = {"fixes": fixes, "breaks": breaks, "net": net}

    all_model_results["Qwen2-0.5B"] = qwen2_results

    # Free memory
    del model, mi

    # ==========================================
    # Qwen3.5-0.8B
    # ==========================================
    print(f"\n\n{'='*70}")
    print("MLV v2 — Qwen3.5-0.8B")
    print("=" * 70)

    model, tokenizer = load("Qwen/Qwen3.5-0.8B")
    mi = setup_qwen35(model)
    n_layers = mi["n_layers"]
    print(f"  Model has {n_layers} layers")

    # For Qwen3.5: try different layer ranges
    # Full attention layers are at positions 3,7,11,15,19,23
    # Last few layers: 20,21,22,23 (mix of linear + full attention)
    strategies_35 = [
        ("Weighted MLV L20-23 k=10 +ASCII",
         [20, 21, 22, 23],
         lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=10)),
        ("Weighted MLV L20-23 k=20 +ASCII",
         [20, 21, 22, 23],
         lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=20)),
        ("Weighted MLV L22-25 k=10 +ASCII",
         [22, 23, 24, 25],
         lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=10)),
        ("Weighted MLV L18-23 k=10 +ASCII",
         [18, 19, 20, 21, 22, 23],
         lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=10)),
    ]

    # Adjust layer ranges for Qwen3.5 (36 layers, last=35)
    # Let's check n_layers first
    if n_layers > 24:
        # Qwen3.5 has 36 layers (0-35)
        # Try last 4, last 6, mid-range
        last = n_layers - 1  # 35
        strategies_35 = [
            (f"Weighted MLV L{last-3}-{last} k=10 +ASCII",
             list(range(last-3, last+1)),
             lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=10)),
            (f"Weighted MLV L{last-3}-{last} k=20 +ASCII",
             list(range(last-3, last+1)),
             lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=20)),
            (f"Weighted MLV L{last-5}-{last} k=10 +ASCII",
             list(range(last-5, last+1)),
             lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=10)),
            (f"Weighted MLV L{last-7}-{last} k=10 +ASCII",
             list(range(last-7, last+1)),
             lambda ll, vl: decode_weighted_mlv_ascii(ll, vl, tokenizer, k=10)),
        ]

    qwen35_results = {}
    for name, voting_layers, decode_fn in strategies_35:
        print(f"\n  --- {name} ---")
        fixes, breaks, net = evaluate(
            model, mi, tokenizer, name, decode_fn, voting_layers, TEST_DATA)
        qwen35_results[name] = {"fixes": fixes, "breaks": breaks, "net": net}

    all_model_results["Qwen3.5-0.8B"] = qwen35_results

    # ==========================================
    # Summary
    # ==========================================
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY — Multi-Layer Voting v2")
    print(f"{'='*70}\n")

    for model_name, results in all_model_results.items():
        print(f"  {model_name}:")
        print(f"  {'Strategy':45} {'Fix':>5} {'Brk':>5} {'Net':>5}")
        print("  " + "-" * 65)
        for name, r in results.items():
            net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net'])
            star = " ★" if r['net'] > 0 else ""
            print(f"  {name:45} +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")
        print()

    print(f"  Reference: Self-Consistency N=7:")
    print(f"    Qwen2:   +5/-0 = +5 (7x compute)")
    print(f"    Qwen3.5: +3/-0 = +3 (7x compute)")
    print(f"    MLV:     ~1% extra compute")


if __name__ == "__main__":
    run()
