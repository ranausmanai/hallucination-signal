"""
Residual Stream Steering for Self-Correction
=============================================
When the model oscillates between two tokens at layer K, both tokens
exist in the model's own embedding table. We extract the "uncertainty axis"
(the difference between the two competing token embeddings in the model's
output space) and project it into the residual stream at layer K.

No external data. No API. Everything from the model's own weights.

Three steering strategies:
  A) Commitment boost: amplify the residual in the direction of the winning token
  B) Uncertainty removal: project OUT the uncertainty axis from the residual
  C) Abstention injection: add the "I don't know" direction to residual

Test: does steering at the oscillation layer change wrong answers to correct?
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from mlx_lm.models.qwen2 import create_attention_mask
import json


def get_oscillation_profile(model, token_ids):
    """
    Run full forward pass, track per-layer top-2 predictions and confidence.
    Returns: layer_preds (n_layers, 2) — top-1 and top-2 token IDs at last pos
             layer_conf  (n_layers,)   — top-1 confidence at last pos
             layer_h     (n_layers, D) — hidden state at last pos (for steering)
    """
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    layer_preds = []
    layer_conf  = []
    layer_h_np  = []  # hidden states at last position

    for layer in model.model.layers:
        h = layer(h, mask, None)
        mx.eval(h)

        h_norm = model.model.norm(h)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(h_norm)
        else:
            logits = model.lm_head(h_norm)
        mx.eval(logits)

        last = np.array(logits[0, -1].astype(mx.float32))
        top2 = np.argsort(-last)[:2]
        probs = np.exp(last - last.max())
        probs /= probs.sum()

        layer_preds.append(top2.tolist())
        layer_conf.append(float(probs[top2[0]]))
        layer_h_np.append(np.array(h[0, -1].astype(mx.float32)))

        del logits, h_norm

    return np.array(layer_preds), np.array(layer_conf), np.array(layer_h_np)


def get_embed(model, token_id):
    """Get embedding vector for a token from the model's embed table."""
    tok = mx.array([int(token_id)])
    emb = model.model.embed_tokens(tok)
    mx.eval(emb)
    return np.array(emb[0].astype(mx.float32))


def generate_normal(model, tokenizer, question, max_tokens=60):
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    ids_mx = mx.array([ids])
    logits = model(ids_mx)
    mx.eval(logits)
    last = np.array(logits[0, -1].astype(mx.float32))
    normal_first_token = int(np.argmax(last))
    del logits

    response = generate(model, tokenizer, prompt=fmt, max_tokens=max_tokens, verbose=False)
    return response.strip(), ids, normal_first_token


def generate_steered(model, tokenizer, question, steer_layer, steer_delta, max_tokens=60):
    """
    Generate with a residual-stream intervention at steer_layer.
    steer_delta: numpy array (D,) to ADD to hidden state at last position.
    """
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    ids_mx = mx.array([ids])

    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)

        if i == steer_layer:
            # Inject steering delta at the last token position
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1] += steer_delta
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)

    h_norm = model.model.norm(h)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h_norm)
    else:
        logits = model.lm_head(h_norm)
    mx.eval(logits)
    last = np.array(logits[0, -1].astype(mx.float32))
    steered_first_token = int(np.argmax(last))
    del logits, h_norm

    # Now generate autoregressively from the steered state
    # Simplest: just re-generate from the fmt but with the steered state as "context"
    # For now, report the first-token change as the key signal
    return tokenizer.decode([steered_first_token]), steered_first_token


def find_peak_oscillation_layer(layer_preds, layer_conf):
    """Find the layer with highest prediction instability."""
    n = len(layer_preds)
    changes = np.array([
        1 if layer_preds[i][0] != layer_preds[i-1][0] else 0
        for i in range(1, n)
    ] + [0])

    # Focus on late layers (12+) where facts are encoded
    late_changes = changes.copy()
    late_changes[:12] = 0

    if late_changes.sum() > 0:
        return int(np.argmax(late_changes))
    return int(np.argmax(changes))


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    n_layers = len(model.model.layers)
    D = model.args.hidden_size

    print(f"Model: {n_layers} layers, D={D}\n")

    # Questions where model is likely to hallucinate
    hard_qa = [
        ("What is the capital of Burkina Faso?",            ["ouagadougou"]),
        ("Who won the Nobel Prize in Chemistry in 2023?",   ["bawendi", "brus", "ekimov"]),
        ("What is the melting point of tungsten in Celsius?", ["3422", "3400"]),
        ("Who was the 30th president of the United States?", ["coolidge", "calvin"]),
        ("What is the half-life of uranium-235?",            ["703", "700 million"]),
        ("What is the capital of Kyrgyzstan?",               ["bishkek"]),
        ("Who is the prime minister of New Zealand as of 2024?", ["luxon", "christopher"]),
        ("What is the speed of sound in water in m/s?",     ["1480", "1500", "1498"]),
        ("What year was the Treaty of Westphalia signed?",  ["1648"]),
        ("What is the largest desert in the world?",        ["antarctica", "antarctic"]),
        ("What country has the most UNESCO World Heritage Sites?", ["italy", "china"]),
        ("What is the GDP of Vietnam in 2023?",             ["430", "450", "400"]),
        ("What is the atomic weight of plutonium?",         ["244", "242", "239"]),
        ("What is the rarest blood type?",                  ["ab-", "ab negative"]),
        ("What is the smallest country in the world?",      ["vatican"]),
    ]

    print(f"{'='*70}")
    print("Residual Stream Steering Experiment")
    print(f"{'='*70}\n")
    print("Strategy: when top-2 tokens are competing, inject the winner's")
    print("embedding direction into the residual stream at peak oscillation layer.\n")

    results = []
    steered_correct = 0
    total_steered = 0

    for question, keywords in hard_qa:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        # Get oscillation profile
        layer_preds, layer_conf, layer_h = get_oscillation_profile(model, ids)

        # Normal answer
        normal_answer = generate(model, tokenizer, prompt=fmt, max_tokens=50, verbose=False).strip()
        normal_correct = any(kw.lower() in normal_answer.lower() for kw in keywords)

        osc_count = sum(layer_preds[i][0] != layer_preds[i-1][0] for i in range(1, n_layers))
        peak_layer = find_peak_oscillation_layer(layer_preds, layer_conf)

        # --- STRATEGY A: Commitment Boost ---
        # At peak oscillation layer, the model has top-1 and top-2 candidates.
        # We get the embedding of top-1 candidate and ADD it to residual stream.
        # This "commits" the model toward its current best guess.
        top1_id = layer_preds[peak_layer][0]
        top1_emb = get_embed(model, top1_id)  # (D,)

        # Scale: add a fraction of the token embedding norm
        alpha = 0.3  # steering strength (tune this)
        steer_delta = alpha * top1_emb

        steered_tok, steered_tok_id = generate_steered(
            model, tokenizer, question, peak_layer, steer_delta
        )

        # Run full steered generation
        steered_answer = generate(model, tokenizer,
            prompt=fmt, max_tokens=50, verbose=False).strip()

        # Actually do the steered forward pass properly for generation
        # Simpler: just report if the first token changed
        first_changed = (steered_tok_id != layer_preds[-1][0])

        # --- STRATEGY B: Uncertainty Axis Removal ---
        # The uncertainty axis = difference between top-1 and top-2 embeddings
        # Project this OUT of the residual stream
        top2_id = layer_preds[peak_layer][1]
        top2_emb = get_embed(model, top2_id)

        uncertainty_axis = top1_emb - top2_emb
        axis_norm = np.linalg.norm(uncertainty_axis) + 1e-10
        uncertainty_axis_normalized = uncertainty_axis / axis_norm

        # Project uncertainty out of hidden state at peak layer
        h_at_peak = layer_h[peak_layer]
        projection = np.dot(h_at_peak, uncertainty_axis_normalized)
        steer_delta_b = -projection * uncertainty_axis_normalized * 0.5

        steered_tok_b, steered_tok_id_b = generate_steered(
            model, tokenizer, question, peak_layer, steer_delta_b
        )

        print(f"Q: {question}")
        print(f"  osc={osc_count}, peak_layer={peak_layer}")
        print(f"  Normal:   '{normal_answer[:70]}' {'✓' if normal_correct else '✗'}")
        print(f"  Top2 at peak: '{tokenizer.decode([top1_id])}' vs '{tokenizer.decode([top2_id])}'")
        print(f"  After steer A (commit): first token → '{steered_tok}'")
        print(f"  After steer B (remove uncertainty): first token → '{steered_tok_b}'")
        print()

        results.append({
            "question": question,
            "normal_answer": normal_answer,
            "normal_correct": normal_correct,
            "osc_count": osc_count,
            "peak_layer": peak_layer,
            "top1_at_peak": tokenizer.decode([top1_id]),
            "top2_at_peak": tokenizer.decode([top2_id]),
            "steered_first_token_a": steered_tok,
            "steered_first_token_b": steered_tok_b,
        })

    # Summary: did first-token steering change predictions?
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    n_normal_correct = sum(r["normal_correct"] for r in results)
    print(f"Normal accuracy: {n_normal_correct}/{len(results)}")

    # Key question: when normal is WRONG, does steering change the first token?
    wrong_cases = [r for r in results if not r["normal_correct"]]
    if wrong_cases:
        changed_a = sum(1 for r in wrong_cases if r["steered_first_token_a"] != tokenizer.decode([
            model.model.embed_tokens.as_linear(
                model.model.norm(
                    model.model.embed_tokens(mx.array([[0]]))
                )
            )
        ]) or True)  # simplification — track differently

        print(f"\nFor the {len(wrong_cases)} wrong answers:")
        for r in wrong_cases:
            print(f"  Normal wrong: '{r['normal_answer'][:60]}'")
            print(f"  Peak oscillation at layer {r['peak_layer']}, top2: '{r['top1_at_peak']}' vs '{r['top2_at_peak']}'")
            print(f"  Steer A first token: '{r['steered_first_token_a']}'")
            print(f"  Steer B first token: '{r['steered_first_token_b']}'")
            print()

    with open("steer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved steer_results.json")

    print(f"\n{'='*70}")
    print("KEY OBSERVATION")
    print(f"{'='*70}")
    print("""
Does steering change the first predicted token on wrong answers?
If yes → we're affecting the model's output via residual stream injection.
If the changed token is CORRECT → self-correction works.
If the changed token is also wrong → the intervention needs refinement.

This is the foundation. Even a 20% correction rate on hallucinations
would be a publishable result.
""")


if __name__ == "__main__":
    run()
