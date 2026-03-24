"""
Layer-21 First-Token Test
=========================
No training needed. Test: does lm_head(norm(h_21)) beat lm_head(norm(h_23))
for the FIRST factual token, when routed by oscillation?

If yes: the DualHead insight is proven with zero training — just use the
existing head at a different depth for factual-mode tokens.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from mlx_lm.models.qwen2 import create_attention_mask
from mlx_lm.sample_utils import make_sampler
import json


def get_hidden_and_logits(model, token_ids, layer_idx):
    """Get logits from lm_head applied at layer_idx (using backbone norm)."""
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)
        if i == layer_idx:
            break
    h_norm = model.model.norm(h)
    mx.eval(h_norm)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h_norm)
    else:
        logits = model.lm_head(h_norm)
    mx.eval(logits)
    last = np.array(logits[0, -1].astype(mx.float32))
    del logits, h_norm
    return last


def get_oscillation_and_layers(model, token_ids):
    """Return oscillation count AND hidden states at layers 21 and 23."""
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)
    preds = []
    logits_21 = None
    logits_23 = None

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

        if i == 21:
            logits_21 = last.copy()
        if i == 23:
            logits_23 = last.copy()

        del logits, h_norm

    osc = sum(preds[i] != preds[i-1] for i in range(1, len(preds)))
    return osc, logits_21, logits_23


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

    # Full test set — same as prior experiments
    test_qa = [
        # Hard
        ("Who was the 30th president of the United States?",       ["coolidge", "calvin"]),
        ("What is the melting point of tungsten in Celsius?",      ["3422", "3400"]),
        ("What is the half-life of uranium-235 in years?",         ["703 million", "703", "700"]),
        ("What is the largest desert in the world by area?",       ["antarctica", "antarctic"]),
        ("What country has the most UNESCO World Heritage Sites?",  ["italy", "china"]),
        ("What is the atomic weight of plutonium?",                ["244", "242", "239"]),
        ("What is the rarest blood type?",                         ["ab-", "ab negative"]),
        ("Who won the Nobel Prize in Chemistry in 2023?",          ["bawendi", "brus", "ekimov"]),
        ("What is the speed of sound in water in m/s?",            ["1480", "1500", "1498"]),
        ("Who is the prime minister of New Zealand as of 2024?",   ["luxon", "christopher"]),
        ("What is the capital of Kyrgyzstan?",                     ["bishkek"]),
        ("What is the capital of Burkina Faso?",                   ["ouagadougou"]),
        ("Who was the first female prime minister of the UK?",     ["thatcher"]),
        ("What is the half-life of Carbon-14 in years?",           ["5730", "5700"]),
        ("What year was the WHO founded?",                         ["1948"]),
        # Easy controls
        ("What is the largest planet?",                            ["jupiter"]),
        ("What is the capital of France?",                         ["paris"]),
        ("Who wrote Romeo and Juliet?",                            ["shakespeare"]),
        ("What is 2 + 2?",                                         ["4", "four"]),
        ("What is the boiling point of water?",                    ["100"]),
    ]

    n_hard = 15
    OSC_THRESHOLD = 15

    print(f"\n{'='*70}")
    print(f"Layer-21 vs Layer-23 First-Token Comparison")
    print(f"Oscillation routing threshold: osc >= {OSC_THRESHOLD} → use layer 21")
    print(f"{'='*70}\n")

    results = []
    for question, keywords in test_qa:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        osc, logits_21, logits_23 = get_oscillation_and_layers(model, ids)

        # First-token predictions at each layer
        top1_21 = int(np.argmax(logits_21))
        top1_23 = int(np.argmax(logits_23))
        tok_21 = tokenizer.decode([top1_21])
        tok_23 = tokenizer.decode([top1_23])

        # Confidence at each layer
        def softmax_conf(logits):
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            return float(probs.max())

        conf_21 = softmax_conf(logits_21)
        conf_23 = softmax_conf(logits_23)

        # Correct?
        correct_21 = any(kw.lower() in tok_21.lower() for kw in keywords)
        correct_23 = any(kw.lower() in tok_23.lower() for kw in keywords)

        # Full greedy answer (standard)
        sampler = make_sampler(temp=0.0)
        std_ans = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                           verbose=False, sampler=sampler).strip()
        std_correct = any(kw.lower() in std_ans.lower() for kw in keywords)

        # Routing decision
        use_21 = osc >= OSC_THRESHOLD
        routed_correct = correct_21 if use_21 else std_correct
        route_str = "L21" if use_21 else "L23"

        status = "✓" if std_correct else "✗"
        l21_status = "✓" if correct_21 else "✗"
        l23_status = "✓" if correct_23 else "✗"

        changed = ""
        if use_21 and correct_21 and not std_correct:
            changed = " *** FIXED ***"
        elif use_21 and not correct_21 and std_correct:
            changed = " *** BROKE ***"

        print(f"Q: {question[:60]}")
        print(f"  osc={osc:2d} [{route_str}] std={status} | L21_tok='{tok_21}'({conf_21:.2f}){l21_status} | L23_tok='{tok_23}'({conf_23:.2f}){l23_status}{changed}")

        results.append({
            "question": question, "keywords": keywords,
            "osc": osc, "use_21": use_21,
            "tok_21": tok_21, "conf_21": conf_21, "correct_21": correct_21,
            "tok_23": tok_23, "conf_23": conf_23, "correct_23": correct_23,
            "std_answer": std_ans, "std_correct": std_correct,
            "routed_correct": routed_correct,
        })

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    std_hard = sum(r["std_correct"] for r in results[:n_hard])
    l21_hard = sum(r["correct_21"] for r in results[:n_hard])
    l23_hard = sum(r["correct_23"] for r in results[:n_hard])
    routed_hard = sum(r["routed_correct"] for r in results[:n_hard])

    std_total = sum(r["std_correct"] for r in results)
    l21_total = sum(r["correct_21"] for r in results)
    routed_total = sum(r["routed_correct"] for r in results)

    print(f"{'Method':30} {'Hard({n_hard})':>10} {'Total':>8}")
    print("-" * 52)
    print(f"  {'Standard greedy (full gen)':28} {std_hard:>4}/{n_hard}  {std_total:>4}/{len(results)}")
    print(f"  {'L21 first token':28} {l21_hard:>4}/{n_hard}  {l21_total:>4}/{len(results)}")
    print(f"  {'L23 first token':28} {l23_hard:>4}/{n_hard}")
    print(f"  {'Routed (L21 if osc>={OSC_THRESHOLD})':28} {routed_hard:>4}/{n_hard}  {routed_total:>4}/{len(results)}")

    # Where does L21 beat L23?
    l21_beats = [(r["question"], r["tok_21"], r["tok_23"]) for r in results
                 if r["correct_21"] and not r["correct_23"]]
    l23_beats = [(r["question"], r["tok_21"], r["tok_23"]) for r in results
                 if r["correct_23"] and not r["correct_21"]]
    both_wrong = [(r["question"], r["tok_21"], r["tok_23"]) for r in results
                  if not r["correct_21"] and not r["correct_23"]]

    print(f"\nL21 beats L23 (first tok, {len(l21_beats)} cases):")
    for q, t21, t23 in l21_beats:
        print(f"  '{q[:55]}': L21='{t21}' vs L23='{t23}'")

    print(f"\nL23 beats L21 (first tok, {len(l23_beats)} cases):")
    for q, t21, t23 in l23_beats:
        print(f"  '{q[:55]}': L21='{t21}' vs L23='{t23}'")

    print(f"\nBoth wrong ({len(both_wrong)} cases):")
    for q, t21, t23 in both_wrong[:8]:
        print(f"  '{q[:55]}': L21='{t21}' vs L23='{t23}'")

    # Confidence analysis
    print(f"\n{'='*70}")
    print("CONFIDENCE ANALYSIS: L21 vs L23")
    print(f"{'='*70}\n")
    conf_21_mean = np.mean([r["conf_21"] for r in results])
    conf_23_mean = np.mean([r["conf_23"] for r in results])
    print(f"  Mean confidence — L21: {conf_21_mean:.3f} | L23: {conf_23_mean:.3f}")

    wrong_greedy = [r for r in results[:n_hard] if not r["std_correct"]]
    if wrong_greedy:
        print(f"\n  On wrong greedy cases (n={len(wrong_greedy)}):")
        print(f"  {'Q':>50}  L21_tok  L21✓  conf21  L23_tok  L23✓")
        print("  " + "-"*80)
        for r in wrong_greedy:
            kws = r["keywords"]
            print(f"  {r['question'][:50]:>50}  {r['tok_21'][:6]:6s}  {'✓' if r['correct_21'] else '✗'}     {r['conf_21']:.3f}   {r['tok_23'][:6]:6s}  {'✓' if r['correct_23'] else '✗'}")

    with open("layer21_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved layer21_results.json")


if __name__ == "__main__":
    run()
