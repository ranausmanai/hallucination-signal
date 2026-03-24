"""
Multi-Layer Voting (MLV) Decoding
===================================

Novel decoding: instead of trusting only the final layer,
collect votes from multiple layers and take the majority.

Like self-consistency but across LAYERS (1 forward pass)
instead of across SAMPLES (N forward passes).

Key insight: earlier layers know the right answer but in Chinese.
Later layers know the right language but pick the wrong fact.
Solution: have each layer vote, but only for tokens that the
final layer considers plausible (English-language filter).

Strategies:
1. Raw MLV: each layer votes for its top-1
2. Filtered MLV: each layer votes for its highest-ranked token
   from the final layer's top-K (language filter)
3. Weighted MLV: weight votes by each layer's confidence
4. Logit averaging: average logits across layers (ensemble)
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
        "get_logits_from_hidden": get_logits,  # includes norm
        "get_mask": lambda h: cam(h, None),
        "hidden_dim": model.args.hidden_size,
        "n_layers": len(model.model.layers),
    }


# ============================================================
# Multi-Layer forward pass: collect logits from every layer
# ============================================================

def forward_all_layers(model_info, ids):
    """Single forward pass, collect logits from all layers."""
    mi = model_info
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)

    layer_logits = []  # logits from each layer at last position

    for i, layer in enumerate(mi["layers"]):
        h = layer(h, mask, None)
        # Get logits from this layer (through shared norm + head)
        logits_i = mi["get_logits_from_hidden"](h[:, -1:, :])
        mx.eval(logits_i)
        layer_logits.append(logits_i[0, 0])  # (V,)

    return layer_logits


# ============================================================
# Decoding strategies
# ============================================================

def decode_raw_mlv(layer_logits, voting_layers):
    """Raw majority vote: each layer votes for its top-1."""
    from collections import Counter
    votes = Counter()
    for i in voting_layers:
        top = int(mx.argmax(layer_logits[i]).item())
        votes[top] += 1
    # Return most common; ties go to the last layer
    if len(votes) == 0:
        return int(mx.argmax(layer_logits[-1]).item())
    winner = votes.most_common(1)[0][0]
    # If tie, prefer the final layer's pick
    final_pick = int(mx.argmax(layer_logits[-1]).item())
    if votes[winner] == votes.get(final_pick, 0):
        return final_pick
    return winner


def decode_filtered_mlv(layer_logits, voting_layers, k=10):
    """
    Filtered MLV: each layer votes, but only for tokens in the
    final layer's top-K. This filters out Chinese tokens.
    """
    from collections import Counter
    final = layer_logits[-1]
    # Get final layer's top-K token indices
    top_k_indices = mx.argpartition(final, kth=-k, axis=-1)[-k:]
    top_k_set = set(int(x) for x in top_k_indices.tolist())

    votes = Counter()
    for i in voting_layers:
        logits = layer_logits[i]
        # Find the highest-logit token from the allowed set
        best_tok = None
        best_logit = float('-inf')
        for tok in top_k_set:
            l = float(logits[tok].item())
            if l > best_logit:
                best_logit = l
                best_tok = tok
        if best_tok is not None:
            votes[best_tok] += 1

    if len(votes) == 0:
        return int(mx.argmax(final).item())
    winner = votes.most_common(1)[0][0]
    final_pick = int(mx.argmax(final).item())
    if votes[winner] == votes.get(final_pick, 0):
        return final_pick
    return winner


def decode_weighted_mlv(layer_logits, voting_layers, k=10):
    """
    Weighted filtered MLV: votes weighted by confidence.
    Higher confidence layer gets more weight.
    """
    final = layer_logits[-1]
    top_k_indices = mx.argpartition(final, kth=-k, axis=-1)[-k:]
    top_k_set = set(int(x) for x in top_k_indices.tolist())

    from collections import defaultdict
    weighted_votes = defaultdict(float)

    for i in voting_layers:
        logits = layer_logits[i]
        probs = mx.softmax(logits, axis=-1)
        # Find best token from allowed set
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
    winner = max(weighted_votes, key=weighted_votes.get)
    return winner


def decode_logit_avg(layer_logits, voting_layers):
    """Average logits across voting layers, then argmax."""
    avg = layer_logits[voting_layers[0]]
    for i in voting_layers[1:]:
        avg = avg + layer_logits[i]
    avg = avg / len(voting_layers)
    return int(mx.argmax(avg).item())


def decode_filtered_logit_avg(layer_logits, voting_layers, k=50):
    """
    Average logits, but zero out tokens not in final's top-K.
    Forces output to stay in the final layer's vocabulary space.
    """
    final = layer_logits[-1]
    top_k_indices = mx.argpartition(final, kth=-k, axis=-1)[-k:]

    # Create mask: 1 for top-k tokens, 0 for others
    mask = mx.zeros_like(final)
    # Can't index-assign, so use scatter-like approach
    avg = layer_logits[voting_layers[0]]
    for i in voting_layers[1:]:
        avg = avg + layer_logits[i]
    avg = avg / len(voting_layers)

    # Set non-top-k to -inf
    top_k_set = set(int(x) for x in top_k_indices.tolist())
    # For efficiency, just pick the max from top_k_set
    best_tok = None
    best_logit = float('-inf')
    for tok in top_k_set:
        l = float(avg[tok].item())
        if l > best_logit:
            best_logit = l
            best_tok = tok
    return best_tok if best_tok is not None else int(mx.argmax(final).item())


# ============================================================
# Generation
# ============================================================

def generate_mlv(model_info, tokenizer, question, decode_fn, max_tokens=60):
    """Generate with Multi-Layer Voting."""
    mi = model_info
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    overrides = 0

    for step in range(max_tokens):
        layer_logits = forward_all_layers(mi, ids)

        std_pick = int(mx.argmax(layer_logits[-1]).item())
        mlv_pick = decode_fn(layer_logits)

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

def evaluate(model, model_info, tokenizer, name, decode_fn, test_data):
    results = []
    n_hard = 13
    total_ov = 0

    for question, keywords in test_data:
        # Standard
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        std_answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                              verbose=False, sampler=sampler).strip()
        std_correct = check_correct(std_answer, keywords)

        # MLV
        mlv_answer, ov = generate_mlv(model_info, tokenizer, question, decode_fn)
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


def run():
    print("=" * 70)
    print("Multi-Layer Voting (MLV) Decoding")
    print("=" * 70)

    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    mi = setup_qwen2(model)
    n_layers = mi["n_layers"]  # 24

    # First: diagnostic — what does each layer predict for the first hard question?
    print(f"\n  Diagnostic: all 24 layers' top-1 for 'Who was the 30th president?'")
    messages = [{"role": "user", "content": "Answer briefly: Who was the 30th president of the United States?"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    layer_logits = forward_all_layers(mi, ids)

    for i in range(n_layers):
        top = int(mx.argmax(layer_logits[i]).item())
        p = float(mx.max(mx.softmax(layer_logits[i], axis=-1)).item())
        tok = tokenizer.decode([top]).strip()
        bar = "█" * int(p * 20)
        print(f"    L{i:2d}: {tok:12s} ({p:.3f}) {bar}")

    strategies = [
        # Last 4 layers (20-23)
        ("Raw MLV L20-23",
         lambda ll: decode_raw_mlv(ll, [20, 21, 22, 23])),
        ("Filtered MLV L20-23 k=10",
         lambda ll: decode_filtered_mlv(ll, [20, 21, 22, 23], k=10)),
        ("Filtered MLV L20-23 k=20",
         lambda ll: decode_filtered_mlv(ll, [20, 21, 22, 23], k=20)),
        ("Weighted MLV L20-23 k=10",
         lambda ll: decode_weighted_mlv(ll, [20, 21, 22, 23], k=10)),

        # Last 6 layers (18-23)
        ("Filtered MLV L18-23 k=10",
         lambda ll: decode_filtered_mlv(ll, [18, 19, 20, 21, 22, 23], k=10)),
        ("Weighted MLV L18-23 k=10",
         lambda ll: decode_weighted_mlv(ll, [18, 19, 20, 21, 22, 23], k=10)),

        # Logit averaging
        ("LogitAvg L20-23",
         lambda ll: decode_logit_avg(ll, [20, 21, 22, 23])),
        ("FilteredLogitAvg L20-23 k=50",
         lambda ll: decode_filtered_logit_avg(ll, [20, 21, 22, 23], k=50)),
        ("FilteredLogitAvg L18-23 k=50",
         lambda ll: decode_filtered_logit_avg(ll, [18, 19, 20, 21, 22, 23], k=50)),

        # Only "factual" layers vote, final layer tiebreaks
        ("Filtered MLV L21+L23 k=10",
         lambda ll: decode_filtered_mlv(ll, [21, 23], k=10)),
        ("Weighted MLV L21+L23 k=10",
         lambda ll: decode_weighted_mlv(ll, [21, 23], k=10)),
    ]

    all_results = {}
    for name, decode_fn in strategies:
        print(f"\n  --- {name} ---")
        fixes, breaks, net = evaluate(model, mi, tokenizer, name, decode_fn, TEST_DATA)
        all_results[name] = {"fixes": fixes, "breaks": breaks, "net": net}

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY — Multi-Layer Voting")
    print(f"{'='*70}\n")
    print(f"  {'Strategy':35} {'Fix':>5} {'Brk':>5} {'Net':>5}")
    print("  " + "-" * 55)
    for name, r in all_results.items():
        net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net'])
        star = " ★" if r['net'] > 0 else ""
        print(f"  {name:35} +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")

    # Compare to self-consistency
    print(f"\n  Reference: Self-Consistency N=7:    +  5  -  0    +5 ★★")


if __name__ == "__main__":
    run()
