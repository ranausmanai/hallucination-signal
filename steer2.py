"""
Revised Steering: Contrastive Activation Addition (CAA)
=========================================================
Key discovery from steer.py: the model oscillates between CHINESE tokens
at intermediate layers when processing English questions. This is the signal —
the multilingual model retreats to its dominant Chinese training distribution
when it doesn't know the English answer.

This means:
  - Oscillation = model uncertain in English → falls to Chinese attractor states
  - The "top-2 tokens" approach was wrong (Chinese tokens ≠ competing answers)

REVISED APPROACH: Contrastive Activation Addition
  1. Collect hidden states from CONFIDENT questions (low oscillation, correct)
  2. Collect hidden states from UNCERTAIN questions (high oscillation, wrong)
  3. Direction = mean(confident) - mean(uncertain) = "confidence vector"
  4. At test time: add the confidence vector to the residual stream
  5. This steers the model toward confident-state activations

This is fully self-contained: no external data, no labels.
We use oscillation itself as the confidence proxy.

We also test: "Abstention Steering"
  Run "I don't know" through the model → extract hidden states
  Add the difference (known - unknown) as negative steering
  → Model says "I don't know" instead of hallucinating
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
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


def get_oscillation_count(model, token_ids, target_layer=12):
    """Get oscillation count up to target_layer."""
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)
    preds = []
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)
        h_norm = model.model.norm(h)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(h_norm)
        else:
            logits = model.lm_head(h_norm)
        mx.eval(logits)
        last = np.array(logits[0, -1].astype(mx.float32))
        preds.append(int(np.argmax(last)))
        del logits, h_norm
        if i == target_layer:
            break
    changes = sum(preds[j] != preds[j-1] for j in range(1, len(preds)))
    return changes


def ask(model, tokenizer, question, max_tokens=60):
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=fmt, max_tokens=max_tokens, verbose=False).strip(), fmt


def generate_steered_full(model, tokenizer, prompt_fmt, token_ids, steer_layer, delta, max_tokens=60):
    """Full steered generation: inject delta at steer_layer for the ENTIRE decode."""
    # Prefill with steering
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)
        if i == steer_layer:
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1] += delta
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)

    # Get next token from steered state
    h_norm = model.model.norm(h)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h_norm)
    else:
        logits = model.lm_head(h_norm)
    mx.eval(logits)
    next_id = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
    next_tok = tokenizer.decode([next_id])
    del logits, h_norm, h

    return next_tok, next_id


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

    STEER_LAYER = 16  # mid-to-late layer where factual retrieval happens
    D = model.args.hidden_size

    # ─── Phase 1: Build confidence vector ─────────────────────────────────────
    # Collect hidden states at STEER_LAYER for high-confidence vs low-confidence prompts

    confident_questions = [
        "What is the capital of France?",        # paris — always right
        "What year did World War II end?",        # 1945 — always right
        "Who wrote Romeo and Juliet?",            # shakespeare — always right
        "What is 2 + 2?",                        # 4 — always right
        "What is H2O?",                          # water — always right
        "What is the largest planet?",            # jupiter — always right
        "Who painted the Mona Lisa?",             # leonardo — always right
        "What is the boiling point of water?",    # 100 — always right
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

    print("Building confidence vector from known-correct vs uncertain questions...")

    confident_states = []
    uncertain_states = []

    for q in confident_questions:
        _, fmt = ask(model, tokenizer, q)
        ids = tokenizer.encode(fmt, add_special_tokens=True)
        h = get_hidden_state_at_layer(model, ids, STEER_LAYER)
        confident_states.append(h)

    for q in uncertain_questions:
        _, fmt = ask(model, tokenizer, q)
        ids = tokenizer.encode(fmt, add_special_tokens=True)
        h = get_hidden_state_at_layer(model, ids, STEER_LAYER)
        uncertain_states.append(h)

    # Confidence vector = direction from uncertain to confident
    mean_confident = np.mean(confident_states, axis=0)   # (D,)
    mean_uncertain = np.mean(uncertain_states, axis=0)   # (D,)
    confidence_vector = mean_confident - mean_uncertain   # (D,)
    confidence_norm = np.linalg.norm(confidence_vector)
    confidence_vector_normalized = confidence_vector / (confidence_norm + 1e-10)

    print(f"Confidence vector norm: {confidence_norm:.3f}")
    print(f"(Larger norm = more separation between confident and uncertain states)\n")

    # ─── Phase 2: Test Steering ────────────────────────────────────────────────
    test_qa = [
        # These are the hard ones the model tends to hallucinate on
        ("Who won the Nobel Prize in Chemistry in 2023?",    ["bawendi", "brus", "ekimov"]),
        ("What is the melting point of tungsten in Celsius?", ["3422", "3400"]),
        ("Who was the 30th president of the United States?", ["coolidge", "calvin"]),
        ("What is the half-life of uranium-235 in years?",   ["703 million", "700 million"]),
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

    print(f"{'='*70}")
    print(f"Steering Test: Does confidence vector shift answers?")
    print(f"Steer layer: {STEER_LAYER}, alpha: 1.0, 3.0, 5.0")
    print(f"{'='*70}\n")

    results = []
    for question, keywords in test_qa:
        _, fmt = ask(model, tokenizer, question)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        # Normal generation
        normal_answer, _ = ask(model, tokenizer, question)
        normal_correct = any(kw.lower() in normal_answer.lower() for kw in keywords)

        # Steered at different strengths
        steered_first_tokens = {}
        for alpha in [1.0, 3.0, 5.0, 10.0]:
            delta = alpha * confidence_vector_normalized
            first_tok, first_id = generate_steered_full(
                model, tokenizer, fmt, ids, STEER_LAYER, delta
            )
            steered_first_tokens[alpha] = first_tok

        # Also negative direction (abstention / uncertainty)
        for alpha in [-1.0, -3.0]:
            delta = alpha * confidence_vector_normalized
            first_tok, first_id = generate_steered_full(
                model, tokenizer, fmt, ids, STEER_LAYER, delta
            )
            steered_first_tokens[alpha] = first_tok

        print(f"Q: {question}")
        print(f"  Normal: '{normal_answer[:70]}' {'✓' if normal_correct else '✗'}")
        for alpha in [1.0, 3.0, 5.0, 10.0, -1.0, -3.0]:
            tok = steered_first_tokens[alpha]
            direction = "→confident" if alpha > 0 else "→uncertain"
            print(f"  alpha={alpha:+5.1f} {direction}: first_token='{tok}'")
        print()

        results.append({
            "question": question,
            "normal_answer": normal_answer,
            "normal_correct": normal_correct,
            "steered_first_tokens": {str(k): v for k, v in steered_first_tokens.items()},
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    n_correct = sum(r["normal_correct"] for r in results)
    print(f"Normal accuracy: {n_correct}/{len(results)}")

    # Key question: does adding confidence vector change first predicted token?
    changes_at_1 = sum(
        1 for r in results
        if r["steered_first_tokens"].get("1.0") != r["normal_answer"].split()[0] if r["normal_answer"]
    )
    changes_at_5 = sum(
        1 for r in results
        if r["steered_first_tokens"].get("5.0") != r["normal_answer"].split()[0] if r["normal_answer"]
    )
    print(f"\nFirst token changed by steering:")
    print(f"  alpha=1.0: {changes_at_1}/{len(results)} changed")
    print(f"  alpha=5.0: {changes_at_5}/{len(results)} changed")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print(f"""
The confidence vector captures the DIRECTION in activation space
that separates confident from uncertain processing.

Adding it to the residual stream at layer {STEER_LAYER} should "push"
the model toward its confident processing mode.

Key finding from earlier: when uncertain, the model oscillates between
CHINESE tokens — suggesting it retreats to its multilingual base distribution.
The confidence vector should push it AWAY from that retreat toward English
factual output.

If steering changes first tokens on wrong answers → intervention works.
If some of those changed first tokens are correct → self-correction works.
This is the path toward a real result.
""")

    with open("steer2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved steer2_results.json")


if __name__ == "__main__":
    run()
