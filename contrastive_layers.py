"""
Contrastive Layer Decoding
===========================
From fast_ensemble.py: simple averaging of layers 21-23 is break-even.
It helps some questions but hurts others.

Problem: averaging dilutes ALL signals, including easy questions
where intermediate layers are also uncertain.

New idea: CONTRASTIVE decoding between early and late layers.
Instead of averaging, AMPLIFY what layer 21 knows that layer 23 doesn't.

logits_out = logits_L23 + alpha * (logits_L21 - logits_L23)

At alpha=0: normal (final layer only)
At alpha=1: intermediate layer only
At alpha=0.5: average of L21 and L23
At alpha > 0: boosts tokens L21 prefers over L23, suppresses what L23 added

For UNESCO:
  L21: China(rank1), L23: France(rank1), China(rank3)
  Contrastive: China gets boosted (L21-L23 increases China's score)
  France gets suppressed (L23-L21 is positive for France → subtract it)

For Kyrgyzstan (where ensemble failed):
  L21 and L23 probably both say Bishkek → little change → safe

This is the same principle as Contrastive Decoding (Li et al. 2022)
but applied across LAYERS of the same model rather than two models.

Key test: does this fix more than it breaks, vs the flat ensemble?
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.qwen2 import create_attention_mask
import json


def forward_all_layer_h(model, token_ids):
    """
    Single forward pass. Returns hidden state (D,) at last position for each layer.
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

    return layer_h  # list of (D,) arrays


def h_to_logits(model, h_np):
    """Apply LM head to hidden state (D,)."""
    h_mx = mx.array(h_np[None, None, :])
    h_norm = model.model.norm(h_mx)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h_norm)
    else:
        logits = model.lm_head(h_norm)
    mx.eval(logits)
    result = np.array(logits[0, 0].astype(mx.float32))
    del logits, h_norm
    return result


def contrastive_decode_step(model, token_ids, source_layer, target_layer, alpha):
    """
    Compute: logits_target + alpha * (logits_source - logits_target)
    = (1 - alpha) * logits_target + alpha * logits_source

    source_layer: the layer we trust more (earlier, factual retrieval)
    target_layer: the final layer
    alpha in [0, 1]: mixing weight (0 = normal, 1 = source only)
    alpha > 1: amplifies the source's advantage (super-contrastive)
    """
    layer_h = forward_all_layer_h(model, token_ids)

    logits_target = h_to_logits(model, layer_h[target_layer])
    logits_source = h_to_logits(model, layer_h[source_layer])

    # Contrastive combination
    logits_out = logits_target + alpha * (logits_source - logits_target)

    # Oscillation (count layer changes)
    top1s = []
    for lh in layer_h:
        lg = h_to_logits(model, lh)
        top1s.append(int(np.argmax(lg)))
    osc = sum(top1s[i] != top1s[i-1] for i in range(1, len(top1s)))

    return int(np.argmax(logits_out)), osc


def generate_contrastive(model, tokenizer, question, source_layer, target_layer,
                          alpha, max_tokens=60):
    """Full autoregressive generation with contrastive layer decoding."""
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    first_osc = None

    for step in range(max_tokens):
        next_id, osc = contrastive_decode_step(model, ids, source_layer, target_layer, alpha)
        if step == 0:
            first_osc = osc
        if next_id == eos_id:
            break
        generated.append(next_id)
        ids.append(next_id)

    return tokenizer.decode(generated).strip(), first_osc


def generate_normal(model, tokenizer, question, max_tokens=60):
    """Standard greedy decoding (final layer)."""
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)

    layer_h = forward_all_layer_h(model, ids)
    eos_id = tokenizer.eos_token_id
    generated = []

    for step in range(max_tokens):
        # Recalculate from current ids (autoregressive)
        layer_h = forward_all_layer_h(model, ids)
        logits = h_to_logits(model, layer_h[-1])
        next_id = int(np.argmax(logits))
        if next_id == eos_id:
            break
        generated.append(next_id)
        ids.append(next_id)

    return tokenizer.decode(generated).strip()


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    n_layers = len(model.model.layers)
    print(f"Model: {n_layers} layers\n")

    test_qa = [
        # Hard (model often wrong)
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
        ("What is the GDP of Vietnam in 2023 in USD?",             ["430", "450"]),
        ("What is the capital of Kyrgyzstan?",                     ["bishkek"]),
        ("What is the capital of Burkina Faso?",                   ["ouagadougou"]),
        ("Who was the first female prime minister of the UK?",     ["thatcher"]),
        ("What is the half-life of Carbon-14 in years?",           ["5730", "5700"]),
        ("What year was the WHO founded?",                         ["1948"]),
        ("Who composed the Four Seasons?",                         ["vivaldi"]),
        ("What is the capital of Myanmar?",                        ["naypyidaw", "naypyitaw"]),
        ("What year was the Treaty of Westphalia signed?",         ["1648"]),
        ("What is the tallest building in the world?",             ["burj", "khalifa"]),
        # Easy controls
        ("What is the capital of France?",                         ["paris"]),
        ("Who wrote Romeo and Juliet?",                            ["shakespeare"]),
        ("What is 2 + 2?",                                         ["4", "four"]),
        ("What is the largest planet?",                            ["jupiter"]),
        ("What is the boiling point of water?",                    ["100"]),
        ("What year did World War II end?",                        ["1945"]),
        ("Who invented the telephone?",                            ["bell", "graham"]),
        ("What is H2O?",                                           ["water"]),
        ("What language is spoken in Brazil?",                     ["portuguese"]),
        ("What is the largest ocean?",                             ["pacific"]),
    ]

    # Test configurations: (source_layer, target_layer, alpha, description)
    configs = [
        (None, None, 0.0, "normal (baseline)"),
        (21, 23, 0.3, "L21→L23 α=0.3"),
        (21, 23, 0.5, "L21→L23 α=0.5"),
        (21, 23, 0.7, "L21→L23 α=0.7"),
        (20, 23, 0.5, "L20→L23 α=0.5"),
        (22, 23, 0.5, "L22→L23 α=0.5"),
        (21, 23, 1.5, "L21→L23 α=1.5 (amplified)"),
    ]

    print(f"{'='*75}")
    print("Contrastive Layer Decoding")
    print(f"{'='*75}\n")

    all_results = {cfg[3]: [] for cfg in configs}
    per_question = []

    for question, keywords in test_qa:
        print(f"Q: {question[:65]}")
        q_res = {"question": question, "keywords": keywords, "configs": {}}

        for src, tgt, alpha, name in configs:
            if src is None:
                # Normal generation
                messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
                fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = tokenizer.encode(fmt, add_special_tokens=True)
                layer_h = forward_all_layer_h(model, ids)
                eos_id = tokenizer.eos_token_id
                gen_ids = list(ids)
                generated = []
                for _ in range(60):
                    lh = forward_all_layer_h(model, gen_ids)
                    lg = h_to_logits(model, lh[-1])
                    nid = int(np.argmax(lg))
                    if nid == eos_id:
                        break
                    generated.append(nid)
                    gen_ids.append(nid)
                answer = tokenizer.decode(generated).strip()
                osc = None
            else:
                answer, osc = generate_contrastive(
                    model, tokenizer, question, src, tgt, alpha)

            is_correct = any(kw.lower() in answer.lower() for kw in keywords)
            all_results[name].append(is_correct)
            q_res["configs"][name] = {
                "answer": answer, "correct": is_correct, "oscillation": osc}

            marker = "✓" if is_correct else "✗"
            osc_str = f"[osc={osc}]" if osc is not None else ""
            print(f"  [{name:25}] {osc_str}: '{answer[:45]}' {marker}")

        per_question.append(q_res)
        print()

    # Summary
    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}\n")

    n_hard = 20
    n_total = len(test_qa)
    normal_total = sum(all_results["normal (baseline)"])
    normal_hard = sum(all_results["normal (baseline)"][:n_hard])

    print(f"{'Config':28} {'Total':>6} {'Hard':>6}  {'Δ total':>8}  {'Δ hard':>8}")
    print("-" * 65)
    for _, _, _, name in configs:
        tot = sum(all_results[name])
        hard = sum(all_results[name][:n_hard])
        dt = tot - normal_total
        dh = hard - normal_hard
        dt_s = f"+{dt}" if dt > 0 else str(dt) if dt < 0 else "="
        dh_s = f"+{dh}" if dh > 0 else str(dh) if dh < 0 else "="
        star = " ★" if dh > 1 else (" ~" if dh == 1 else "")
        print(f"  {name:26} {tot:>4}/{n_total}  {hard:>4}/{n_hard}  {dt_s:>8}  {dh_s:>8}{star}")

    # Best config
    best_name = max(
        [name for _, _, _, name in configs],
        key=lambda n: sum(all_results[n][:n_hard])
    )
    best_hard = sum(all_results[best_name][:n_hard])
    print(f"\nBest (hard only): {best_name} = {best_hard}/{n_hard}")

    # Detailed corrections for best config
    if best_hard > normal_hard:
        print(f"\nCORRECTIONS from {best_name}:")
        fixes = [q for q in per_question[:n_hard]
                 if not q["configs"]["normal (baseline)"]["correct"]
                 and q["configs"][best_name]["correct"]]
        breaks = [q for q in per_question[:n_hard]
                  if q["configs"]["normal (baseline)"]["correct"]
                  and not q["configs"][best_name]["correct"]]
        for q in fixes:
            osc = q["configs"][best_name]["oscillation"]
            print(f"  ✓ [{osc}] {q['question'][:60]}")
            print(f"       Before: '{q['configs']['normal (baseline)']['answer'][:55]}'")
            print(f"       After:  '{q['configs'][best_name]['answer'][:55]}'")
        if breaks:
            print(f"  Regressions:")
            for q in breaks:
                print(f"  ✗ {q['question'][:60]}")

    with open("contrastive_layers_results.json", "w") as f:
        json.dump({
            "n_total": n_total,
            "n_hard": n_hard,
            "summary": {name: {
                "total": sum(all_results[name]),
                "hard": sum(all_results[name][:n_hard])
            } for _, _, _, name in configs},
            "per_question": per_question,
        }, f, indent=2)
    print("Saved contrastive_layers_results.json")


if __name__ == "__main__":
    run()
