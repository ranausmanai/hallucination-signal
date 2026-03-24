"""
Continuous Residual Stream Steering (Full Autoregressive)
=========================================================
steer2.py only injected the delta ONCE (at prefill).
That's wrong — the model recovers after one steered step.

This script runs a full autoregressive decode loop, injecting
the confidence vector at EVERY step. This is real CAA.

Also: we now test the FULL generated answer for correctness,
not just the first token.

Architecture note:
  - Qwen2-0.5B has 24 layers, D=896
  - We inject at layer 16 (mid-to-late, factual retrieval zone)
  - We use KV cache to avoid re-running all layers every step
  - But KV cache changes the mask shape — so we inject carefully

Actually simpler: no KV cache. Re-run all layers each step.
For a 0.5B model and max_tokens=40, this is fast enough.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.qwen2 import create_attention_mask
import json


def get_hidden_state_at_layer(model, token_ids, layer_idx):
    """Get hidden state at last position after layer_idx."""
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)
        if i == layer_idx:
            break
    return np.array(h[0, -1].astype(mx.float32))


def greedy_step(model, token_ids, steer_layer=None, delta=None):
    """
    Run one full forward pass on token_ids.
    If steer_layer is set, inject delta at that layer's last position.
    Returns: next_token_id (int)
    """
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)
        if steer_layer is not None and i == steer_layer and delta is not None:
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1] += delta
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)

    h_norm = model.model.norm(h)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h_norm)
    else:
        logits = model.lm_head(h_norm)
    mx.eval(logits)
    next_id = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
    del logits, h_norm, h
    return next_id


def generate_with_steering(model, tokenizer, token_ids, steer_layer, delta, max_tokens=50):
    """
    Full autoregressive generation with steering at every step.
    No KV cache — runs full forward pass each step.
    """
    ids = list(token_ids)
    eos_id = tokenizer.eos_token_id
    generated = []

    for _ in range(max_tokens):
        next_id = greedy_step(model, ids, steer_layer=steer_layer, delta=delta)
        if next_id == eos_id:
            break
        generated.append(next_id)
        ids.append(next_id)

    return tokenizer.decode(generated).strip()


def generate_normal(model, tokenizer, token_ids, max_tokens=50):
    """Normal generation (no steering)."""
    return generate_with_steering(model, tokenizer, token_ids,
                                   steer_layer=None, delta=None, max_tokens=max_tokens)


def build_confidence_vector(model, tokenizer, steer_layer):
    """Build confidence direction from known-correct vs uncertain questions."""

    confident_questions = [
        "What is the capital of France?",
        "What year did World War II end?",
        "Who wrote Romeo and Juliet?",
        "What is 2 + 2?",
        "What is H2O?",
        "What is the largest planet?",
        "Who painted the Mona Lisa?",
        "What is the boiling point of water?",
    ]

    uncertain_questions = [
        "Who won the Nobel Prize in Chemistry in 2023?",
        "Who is the prime minister of New Zealand as of 2024?",
        "What is the GDP of Vietnam in 2023?",
        "What is the half-life of uranium-235?",
        "What is the melting point of tungsten in Celsius?",
        "Who was the 30th president of the United States?",
        "What is the atomic weight of plutonium?",
        "What is the speed of sound in water in m/s?",
    ]

    def get_state(q):
        messages = [{"role": "user", "content": f"Answer briefly: {q}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)
        return get_hidden_state_at_layer(model, ids, steer_layer)

    print(f"  Collecting {len(confident_questions)} confident states...")
    confident_states = [get_state(q) for q in confident_questions]

    print(f"  Collecting {len(uncertain_questions)} uncertain states...")
    uncertain_states = [get_state(q) for q in uncertain_questions]

    mean_confident = np.mean(confident_states, axis=0)
    mean_uncertain = np.mean(uncertain_states, axis=0)
    vec = mean_confident - mean_uncertain
    norm = np.linalg.norm(vec)
    vec_normalized = vec / (norm + 1e-10)

    print(f"  Confidence vector norm: {norm:.3f}")
    return vec_normalized, norm


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

    STEER_LAYER = 16

    print(f"\nBuilding confidence vector at layer {STEER_LAYER}...")
    confidence_vec, vec_norm = build_confidence_vector(model, tokenizer, STEER_LAYER)

    test_qa = [
        ("Who won the Nobel Prize in Chemistry in 2023?",    ["bawendi", "brus", "ekimov"]),
        ("What is the melting point of tungsten in Celsius?", ["3422", "3400"]),
        ("Who was the 30th president of the United States?", ["coolidge", "calvin"]),
        ("What is the half-life of uranium-235 in years?",   ["703 million", "703", "700"]),
        ("What is the capital of Kyrgyzstan?",               ["bishkek"]),
        ("Who is the prime minister of New Zealand as of 2024?", ["luxon", "christopher"]),
        ("What is the speed of sound in water in m/s?",      ["1480", "1500", "1498"]),
        ("What is the largest desert in the world by area?", ["antarctica", "antarctic"]),
        ("What country has the most UNESCO World Heritage Sites?", ["italy", "china"]),
        ("What is the atomic weight of plutonium?",          ["244", "242", "239"]),
        ("What is the rarest blood type?",                   ["ab-", "ab negative"]),
        ("What is the GDP of Vietnam in 2023 in USD?",       ["430", "450"]),
        ("Who was the first female prime minister of the UK?", ["thatcher"]),
        ("What is the half-life of Carbon-14 in years?",     ["5730", "5700"]),
        ("What is the capital of Burkina Faso?",             ["ouagadougou"]),
    ]

    alphas = [3.0, 5.0, 10.0]

    print(f"\n{'='*70}")
    print(f"Continuous Steering Test (full autoregressive generation)")
    print(f"Steer layer: {STEER_LAYER}, alphas: {alphas}")
    print(f"{'='*70}\n")

    results = []
    for question, keywords in test_qa:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        # Normal generation
        normal_answer = generate_normal(model, tokenizer, ids)
        normal_correct = any(kw.lower() in normal_answer.lower() for kw in keywords)

        print(f"Q: {question}")
        print(f"  Normal: '{normal_answer[:80]}' {'✓' if normal_correct else '✗'}")

        steered_answers = {}
        for alpha in alphas:
            delta = alpha * confidence_vec
            answer = generate_with_steering(model, tokenizer, ids, STEER_LAYER, delta)
            is_correct = any(kw.lower() in answer.lower() for kw in keywords)
            steered_answers[alpha] = {"text": answer, "correct": is_correct}
            marker = "✓" if is_correct else ("~" if answer != normal_answer else "=")
            print(f"  alpha={alpha:+5.1f}: '{answer[:80]}' {marker}")

        # Negative direction
        for alpha in [-3.0, -5.0]:
            delta = alpha * confidence_vec
            answer = generate_with_steering(model, tokenizer, ids, STEER_LAYER, delta)
            is_correct = any(kw.lower() in answer.lower() for kw in keywords)
            steered_answers[alpha] = {"text": answer, "correct": is_correct}
            print(f"  alpha={alpha:+5.1f}: '{answer[:80]}' {'✓' if is_correct else '✗'}")
        print()

        results.append({
            "question": question,
            "keywords": keywords,
            "normal_answer": normal_answer,
            "normal_correct": normal_correct,
            "steered": {str(k): v for k, v in steered_answers.items()},
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    n_normal_correct = sum(r["normal_correct"] for r in results)
    n_total = len(results)
    print(f"Normal accuracy: {n_normal_correct}/{n_total}")

    for alpha in alphas:
        n_steered_correct = sum(
            r["steered"][str(alpha)]["correct"] for r in results
        )
        wrong_fixed = sum(
            1 for r in results
            if not r["normal_correct"] and r["steered"][str(alpha)]["correct"]
        )
        right_broken = sum(
            1 for r in results
            if r["normal_correct"] and not r["steered"][str(alpha)]["correct"]
        )
        print(f"  alpha={alpha}: {n_steered_correct}/{n_total} correct | "
              f"wrong→right: {wrong_fixed}, right→wrong: {right_broken}")

    print()

    # Show corrections
    for alpha in alphas:
        fixes = [r for r in results if not r["normal_correct"] and r["steered"][str(alpha)]["correct"]]
        if fixes:
            print(f"\nalpha={alpha} CORRECTIONS ({len(fixes)}):")
            for r in fixes:
                print(f"  Q: {r['question']}")
                print(f"    Was: '{r['normal_answer'][:70]}'")
                print(f"    Now: '{r['steered'][str(alpha)]['text'][:70]}'")

    with open("steer3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved steer3_results.json")


if __name__ == "__main__":
    run()
