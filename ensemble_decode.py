"""
Layer-Ensemble Decoding: Correcting Knowledge Suppression
==========================================================
Finding from buried_answer.py:
  - 9/10 hallucinations have the correct token in top-200
  - For UNESCO sites: "China" was rank 1 at layer 21, but layers 22-23
    overrode it to "France"
  - The final layers SUPPRESS correct intermediate signals

Mechanism: middle layers retrieve correct facts, final layers apply
"plausibility correction" that sometimes overrides correct answers
with more familiar-sounding wrong ones.

Intervention: Layer-Ensemble Decoding
  - Instead of using ONLY the final layer's logits, average logits
    across several late layers (e.g., layers 20-23)
  - This prevents any single layer from completely overriding others
  - No external data, no external model — just re-weighting across layers

Comparison:
  A) Standard: final layer only (baseline)
  B) Ensemble-last4: average of layers 20, 21, 22, 23
  C) Ensemble-best: use the layer that historically performs best
  D) Max-rank: pick token that achieves best rank across any layer

Full autoregressive generation with ensemble decoding.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.qwen2 import create_attention_mask
import json


def forward_all_layers(model, token_ids):
    """
    Full forward pass, returning hidden state after EACH layer.
    Returns list of hidden states h[0, -1] at last position, shape (n_layers, D).
    Also returns the final h for autoregressive use.
    """
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)
    layer_h = []

    for layer in model.model.layers:
        h = layer(h, mask, None)
        mx.eval(h)
        layer_h.append(np.array(h[0, -1].astype(mx.float32)))

    return layer_h, h  # h is (1, seq_len, D) after all layers


def logits_from_hidden(model, h_np):
    """Compute logits from a hidden state vector (D,)."""
    h_mx = mx.array(h_np[None, None, :])  # (1, 1, D)
    h_norm = model.model.norm(h_mx)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h_norm)
    else:
        logits = model.lm_head(h_norm)
    mx.eval(logits)
    result = np.array(logits[0, 0].astype(mx.float32))
    del logits, h_norm
    return result


def ensemble_logits(model, token_ids, layer_indices):
    """
    Average logits from specified layers.
    Returns: (vocab_size,) average logit array.
    """
    layer_h, _ = forward_all_layers(model, token_ids)
    all_logits = []
    for i in layer_indices:
        lg = logits_from_hidden(model, layer_h[i])
        all_logits.append(lg)
    return np.mean(all_logits, axis=0)


def single_layer_logits(model, token_ids, layer_idx=-1):
    """Get logits from a single layer."""
    layer_h, _ = forward_all_layers(model, token_ids)
    idx = layer_idx if layer_idx >= 0 else len(layer_h) + layer_idx
    return logits_from_hidden(model, layer_h[idx])


def greedy_next(logits):
    return int(np.argmax(logits))


def generate_with_ensemble(model, tokenizer, token_ids, layer_indices, max_tokens=60):
    """
    Full autoregressive generation using ensemble of specified layers.
    Slower (re-runs full forward pass each step) but correct.
    """
    ids = list(token_ids)
    eos_id = tokenizer.eos_token_id
    generated = []

    for _ in range(max_tokens):
        logits = ensemble_logits(model, ids, layer_indices)
        next_id = greedy_next(logits)
        if next_id == eos_id:
            break
        generated.append(next_id)
        ids.append(next_id)

    return tokenizer.decode(generated).strip()


def generate_normal(model, tokenizer, token_ids, max_tokens=60):
    """Standard generation (final layer only)."""
    ids = list(token_ids)
    eos_id = tokenizer.eos_token_id
    generated = []

    for _ in range(max_tokens):
        logits = single_layer_logits(model, ids, layer_idx=-1)
        next_id = greedy_next(logits)
        if next_id == eos_id:
            break
        generated.append(next_id)
        ids.append(next_id)

    return tokenizer.decode(generated).strip()


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    n_layers = len(model.model.layers)  # 24

    print(f"Model: {n_layers} layers\n")

    # Layer ensemble configurations to test
    # From buried_answer.py: correct tokens peak at layers 21-22,
    # then get suppressed at layer 23. So ensemble should include 21-23
    configs = {
        "normal (L23)":     [n_layers - 1],       # just final layer
        "L20-23 ensemble":  list(range(20, n_layers)),  # last 4 layers
        "L18-23 ensemble":  list(range(18, n_layers)),  # last 6 layers
        "L16-23 ensemble":  list(range(16, n_layers)),  # last 8 layers
        "L21-23 ensemble":  [21, 22, 23],          # the key window
        "L20-22 (skip L23)": [20, 21, 22],         # skip the most suppressing layer
    }

    test_qa = [
        ("Who was the 30th president of the United States?",
         ["coolidge", "calvin"]),
        ("What is the melting point of tungsten in Celsius?",
         ["3422", "3400"]),
        ("What is the half-life of uranium-235 in years?",
         ["703 million", "703", "700"]),
        ("What is the largest desert in the world by area?",
         ["antarctica", "antarctic"]),
        ("What country has the most UNESCO World Heritage Sites?",
         ["italy", "china"]),
        ("What is the atomic weight of plutonium?",
         ["244", "242", "239"]),
        ("What is the rarest blood type?",
         ["ab-", "ab negative"]),
        ("Who won the Nobel Prize in Chemistry in 2023?",
         ["bawendi", "brus", "ekimov"]),
        ("What is the speed of sound in water in m/s?",
         ["1480", "1500", "1498"]),
        ("Who is the prime minister of New Zealand as of 2024?",
         ["luxon", "christopher"]),
        # Easy control (model should get right with any config)
        ("What is the capital of France?", ["paris"]),
        ("Who wrote Romeo and Juliet?", ["shakespeare"]),
        ("What is 2 + 2?", ["4", "four"]),
        ("What is the largest planet?", ["jupiter"]),
        ("What is the boiling point of water?", ["100"]),
    ]

    print(f"{'='*75}")
    print("Layer-Ensemble Decoding Test")
    print(f"{'='*75}\n")

    results = []
    for question, keywords in test_qa:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        print(f"Q: {question}")
        qa_result = {"question": question, "keywords": keywords, "configs": {}}

        for config_name, layers in configs.items():
            if layers == [n_layers - 1]:
                answer = generate_normal(model, tokenizer, ids)
            else:
                answer = generate_with_ensemble(model, tokenizer, ids, layers)

            is_correct = any(kw.lower() in answer.lower() for kw in keywords)
            qa_result["configs"][config_name] = {"answer": answer, "correct": is_correct}

            marker = "✓" if is_correct else "✗"
            print(f"  [{config_name:22}]: '{answer[:65]}' {marker}")

        print()
        results.append(qa_result)

    # Summary table
    print(f"\n{'='*75}")
    print("SUMMARY: Accuracy by Configuration")
    print(f"{'='*75}\n")

    config_names = list(configs.keys())
    n_total = len(results)

    print(f"{'Config':25} {'Correct':>8} {'%':>6}  {'vs Normal':>10}")
    print("-" * 60)

    normal_correct = sum(
        1 for r in results if r["configs"]["normal (L23)"]["correct"]
    )

    for cname in config_names:
        n_correct = sum(1 for r in results if r["configs"][cname]["correct"])
        pct = n_correct / n_total * 100
        diff = n_correct - normal_correct
        diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "="
        star = " ★" if diff > 0 else ""
        print(f"  {cname:23} {n_correct:>8}/{n_total} {pct:>5.1f}%  {diff_str:>10}{star}")

    # Show what each config corrects
    print(f"\n{'='*75}")
    print("CORRECTIONS per config (wrong→right with normal)")
    print(f"{'='*75}\n")

    for cname in config_names[1:]:  # skip normal
        fixes = [
            r for r in results
            if not r["configs"]["normal (L23)"]["correct"]
            and r["configs"][cname]["correct"]
        ]
        breaks = [
            r for r in results
            if r["configs"]["normal (L23)"]["correct"]
            and not r["configs"][cname]["correct"]
        ]
        print(f"{cname}:")
        if fixes:
            print(f"  Fixed ({len(fixes)}):")
            for r in fixes:
                print(f"    '{r['question'][:55]}'")
                print(f"      Normal: '{r['configs']['normal (L23)']['answer'][:55]}'")
                print(f"      Ensemble: '{r['configs'][cname]['answer'][:55]}'")
        if breaks:
            print(f"  Broke ({len(breaks)}):")
            for r in breaks:
                print(f"    '{r['question'][:55]}'")
        if not fixes and not breaks:
            print(f"  No change")
        print()

    with open("ensemble_decode_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved ensemble_decode_results.json")


if __name__ == "__main__":
    run()
