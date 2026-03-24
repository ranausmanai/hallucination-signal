"""
ANCHOR v3: Factual Verification Decoding (FVD)
================================================

Zero-training approach based on mechanistic insight:
- Layer 21 knows the correct answer (buried_answer.py: 9/10 times)
- Layers 22-23 suppress it

Instead of learning when to trust L21, use a simple rule:
  If L21 disagrees with L23, and L21 is more confident → trust L21.

This is a verification gate in the forward pass: the model generates normally,
but at each token, L21 acts as a fact-checker.

Strategies (all zero-training):
1. Confidence-based: trust whichever layer is more confident
2. Agreement-check: if agree → use final. If disagree → use factual if confident
3. Rank-rescue: if final's top-1 is in factual's top-K, boost it
4. Disagreement-weighted: blend proportional to disagreement strength
5. Oracle (upper bound): always pick correct answer if either layer has it
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import time


# ============================================================
# Data
# ============================================================

TEST_DATA = [
    # Hard (strictly held out)
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
    # Easy controls
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


# ============================================================
# Model setup
# ============================================================

def setup_qwen2(model):
    from mlx_lm.models.qwen2 import create_attention_mask as cam
    def get_logits(h):
        if model.args.tie_word_embeddings:
            return model.model.embed_tokens.as_linear(h)
        return model.lm_head(h)
    return {
        "embed": model.model.embed_tokens,
        "layers": model.model.layers,
        "norm": model.model.norm,
        "get_logits": get_logits,
        "get_mask": lambda h: cam(h, None),
        "get_layer_mask": None,
        "hidden_dim": model.args.hidden_size,
        "n_layers": len(model.model.layers),
    }


# ============================================================
# Decoding strategies
# ============================================================

def decode_standard(logits_final):
    """Standard greedy: argmax of final logits."""
    return int(mx.argmax(logits_final).item())


def decode_confidence(logits_factual, logits_final, threshold=0.0):
    """Trust whichever layer is more confident (higher max prob)."""
    p_factual = mx.softmax(logits_factual, axis=-1)
    p_final = mx.softmax(logits_final, axis=-1)
    conf_factual = float(mx.max(p_factual).item())
    conf_final = float(mx.max(p_final).item())

    top_factual = int(mx.argmax(logits_factual).item())
    top_final = int(mx.argmax(logits_final).item())

    if top_factual != top_final and conf_factual > conf_final + threshold:
        return top_factual
    return top_final


def decode_agree_or_factual(logits_factual, logits_final, factual_conf_threshold=0.3):
    """If layers agree → use final. If disagree and factual confident → use factual."""
    top_factual = int(mx.argmax(logits_factual).item())
    top_final = int(mx.argmax(logits_final).item())

    if top_factual == top_final:
        return top_final

    # Disagreement — check factual confidence
    p_factual = mx.softmax(logits_factual, axis=-1)
    conf_factual = float(mx.max(p_factual).item())

    if conf_factual > factual_conf_threshold:
        return top_factual
    return top_final


def decode_blend_on_disagree(logits_factual, logits_final, alpha=0.3):
    """Always blend, but weight factual more when layers disagree."""
    top_factual = int(mx.argmax(logits_factual).item())
    top_final = int(mx.argmax(logits_final).item())

    if top_factual == top_final:
        # Agreement: trust final
        return top_final

    # Disagreement: blend logits
    blended = (1 - alpha) * logits_final + alpha * logits_factual
    return int(mx.argmax(blended).item())


def decode_rank_rescue(logits_factual, logits_final, k=5, boost=2.0):
    """
    If factual's top-1 is within final's top-K, boost it in final logits.
    This 'rescues' buried correct answers.
    """
    top_factual = int(mx.argmax(logits_factual).item())

    # Check if factual's pick is in final's top-K
    top_k_final = mx.argpartition(logits_final, kth=-k, axis=-1)[-k:]
    top_k_final_list = [int(x) for x in top_k_final.tolist()]

    if top_factual in top_k_final_list:
        # Factual's answer IS in final's top-K — just use it directly
        return top_factual

    return int(mx.argmax(logits_final).item())


def decode_suppress_disagree(logits_factual, logits_final, penalty=5.0):
    """
    If final's top-1 is NOT factual's top-1, and factual is confident,
    penalize final's top-1 and let the next-best token win.
    """
    top_factual = int(mx.argmax(logits_factual).item())
    top_final = int(mx.argmax(logits_final).item())

    if top_factual == top_final:
        return top_final

    p_factual = mx.softmax(logits_factual, axis=-1)
    conf_factual = float(mx.max(p_factual).item())

    if conf_factual > 0.3:
        # Suppress final's top pick — set it to very low value, let second choice rise
        mask = mx.arange(logits_final.shape[-1]) == top_final
        penalized = mx.where(mask, logits_final - penalty, logits_final)
        return int(mx.argmax(penalized).item())

    return top_final


# ============================================================
# Generation with factual verification
# ============================================================

def generate_fvd(model_info, tokenizer, question, factual_layer, decode_fn,
                 max_tokens=60, debug=False):
    """Generate with Factual Verification Decoding."""
    mi = model_info
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    overrides = 0  # count of times factual overrode final

    for step in range(max_tokens):
        ids_mx = mx.array([ids])
        h = mi["embed"](ids_mx)
        mask = mi["get_mask"](h)

        h_factual = None
        for i, layer in enumerate(mi["layers"]):
            if mi["get_layer_mask"]:
                m = mi["get_layer_mask"](layer, mask)
            else:
                m = mask
            h = layer(h, m, None)
            if i == factual_layer:
                h_factual = h

        # Get logits at last position from both layers
        logits_factual = mi["get_logits"](mi["norm"](h_factual[:, -1:, :]))
        logits_final = mi["get_logits"](mi["norm"](h[:, -1:, :]))
        mx.eval(logits_factual, logits_final)

        lf = logits_factual[0, 0]  # (V,)
        ll = logits_final[0, 0]    # (V,)

        # Standard pick
        std_pick = int(mx.argmax(ll).item())

        # FVD pick
        fvd_pick = decode_fn(lf, ll)

        if fvd_pick != std_pick:
            overrides += 1
            if debug:
                std_tok = tokenizer.decode([std_pick])
                fvd_tok = tokenizer.decode([fvd_pick])
                print(f"      step {step}: '{std_tok}' → '{fvd_tok}'")

        if fvd_pick == eos_id:
            break
        generated.append(fvd_pick)
        ids.append(fvd_pick)
        del h, h_factual, logits_factual, logits_final

    answer = tokenizer.decode(generated).strip()
    return answer, overrides


# ============================================================
# Evaluation
# ============================================================

def evaluate_strategy(model, model_info, tokenizer, strategy_name, decode_fn,
                      factual_layer, test_data, debug_first_n=0):
    """Evaluate a FVD strategy."""
    results = []
    n_hard = 13
    total_overrides = 0

    for i, (question, keywords) in enumerate(test_data):
        # Standard greedy baseline
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        std_answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                              verbose=False, sampler=sampler).strip()
        std_correct = check_correct(std_answer, keywords)

        # FVD generation
        debug = i < debug_first_n
        if debug:
            print(f"    Q: {question[:50]}")
        fvd_answer, n_overrides = generate_fvd(
            model_info, tokenizer, question, factual_layer, decode_fn,
            debug=debug)
        fvd_correct = check_correct(fvd_answer, keywords)
        total_overrides += n_overrides

        results.append({
            "question": question, "keywords": keywords,
            "std_answer": std_answer, "std_correct": std_correct,
            "fvd_answer": fvd_answer, "fvd_correct": fvd_correct,
            "overrides": n_overrides,
        })

    # Print results
    std_total = sum(r["std_correct"] for r in results)
    std_hard = sum(r["std_correct"] for r in results[:n_hard])
    fvd_total = sum(r["fvd_correct"] for r in results)
    fvd_hard = sum(r["fvd_correct"] for r in results[:n_hard])
    fixes = sum(1 for r in results if r["fvd_correct"] and not r["std_correct"])
    breaks = sum(1 for r in results if not r["fvd_correct"] and r["std_correct"])
    net = fixes - breaks

    print(f"\n  {strategy_name} (L{factual_layer}, {total_overrides} token overrides)")
    print(f"  Standard: {std_total}/{len(results)} total, {std_hard}/{n_hard} hard")
    print(f"  FVD:      {fvd_total}/{len(results)} total, {fvd_hard}/{n_hard} hard")
    print(f"  Net: +{fixes} fixed, -{breaks} broken = {'+'if net>0 else ''}{net}")

    for r in results:
        s = "✓" if r["std_correct"] else "✗"
        f = "✓" if r["fvd_correct"] else "✗"
        tag = ""
        if r["fvd_correct"] and not r["std_correct"]:
            tag = " *** FIXED ***"
        elif not r["fvd_correct"] and r["std_correct"]:
            tag = " *** BROKE ***"
        ov = f" ({r['overrides']}ov)" if r["overrides"] > 0 else ""
        print(f"  [std={s} fvd={f}]{tag}{ov} {r['question'][:50]}")
        if tag:
            print(f"    std: '{r['std_answer'][:60]}'")
            print(f"    fvd: '{r['fvd_answer'][:60]}'")

    return fixes, breaks, net


# ============================================================
# First: diagnostic — what does L21 vs L23 actually predict?
# ============================================================

def diagnostic(model_info, tokenizer, test_data, factual_layer):
    """For each test question, show what L21 vs L23 predict at the critical position."""
    mi = model_info
    print(f"\n  Diagnostic: L{factual_layer} vs L{mi['n_layers']-1} predictions")
    print(f"  {'Question':45} {'L_fact top-1':>15} {'L_final top-1':>15} {'Agree?':>6}")
    print("  " + "-" * 85)

    for question, keywords in test_data[:13]:  # hard questions only
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        ids_mx = mx.array([ids])
        h = mi["embed"](ids_mx)
        mask = mi["get_mask"](h)

        h_factual = None
        for i, layer in enumerate(mi["layers"]):
            if mi["get_layer_mask"]:
                m = mi["get_layer_mask"](layer, mask)
            else:
                m = mask
            h = layer(h, m, None)
            if i == factual_layer:
                h_factual = h

        logits_factual = mi["get_logits"](mi["norm"](h_factual[:, -1:, :]))
        logits_final = mi["get_logits"](mi["norm"](h[:, -1:, :]))
        mx.eval(logits_factual, logits_final)

        top_f = int(mx.argmax(logits_factual[0, 0]).item())
        top_l = int(mx.argmax(logits_final[0, 0]).item())

        p_f = float(mx.max(mx.softmax(logits_factual[0, 0], axis=-1)).item())
        p_l = float(mx.max(mx.softmax(logits_final[0, 0], axis=-1)).item())

        tok_f = tokenizer.decode([top_f]).strip()
        tok_l = tokenizer.decode([top_l]).strip()
        agree = "✓" if top_f == top_l else "✗"

        print(f"  {question[:45]:45} {tok_f:>10}({p_f:.2f}) {tok_l:>10}({p_l:.2f}) {agree:>6}")

        del h, h_factual, logits_factual, logits_final


# ============================================================
# Main
# ============================================================

def run():
    print("=" * 70)
    print("ANCHOR v3 — Factual Verification Decoding (FVD)")
    print("Zero-training, mechanistically-motivated decoding")
    print("=" * 70)

    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    mi = setup_qwen2(model)

    for factual_layer in [21]:
        print(f"\n{'='*70}")
        print(f"Factual Layer: {factual_layer}")
        print(f"{'='*70}")

        # Diagnostic first
        diagnostic(mi, tokenizer, TEST_DATA, factual_layer)

        strategies = [
            ("Confidence(δ=0.0)",
             lambda lf, ll: decode_confidence(lf, ll, threshold=0.0)),
            ("AgreeOrFactual(t=0.5)",
             lambda lf, ll: decode_agree_or_factual(lf, ll, factual_conf_threshold=0.5)),
            ("BlendOnDisagree(α=0.3)",
             lambda lf, ll: decode_blend_on_disagree(lf, ll, alpha=0.3)),
            ("RankRescue(k=5)",
             lambda lf, ll: decode_rank_rescue(lf, ll, k=5, boost=2.0)),
            ("RankRescue(k=10)",
             lambda lf, ll: decode_rank_rescue(lf, ll, k=10, boost=3.0)),
            ("RankRescue(k=20)",
             lambda lf, ll: decode_rank_rescue(lf, ll, k=20, boost=5.0)),
            ("SuppressDisagree(p=3)",
             lambda lf, ll: decode_suppress_disagree(lf, ll, penalty=3.0)),
            ("SuppressDisagree(p=5)",
             lambda lf, ll: decode_suppress_disagree(lf, ll, penalty=5.0)),
            ("SuppressDisagree(p=10)",
             lambda lf, ll: decode_suppress_disagree(lf, ll, penalty=10.0)),
        ]

        all_results = {}
        for name, decode_fn in strategies:
            print(f"\n  --- {name} ---")
            fixes, breaks, net = evaluate_strategy(
                model, mi, tokenizer, name, decode_fn,
                factual_layer, TEST_DATA)
            all_results[name] = {"fixes": fixes, "breaks": breaks, "net": net}

        # Summary
        print(f"\n  {'='*60}")
        print(f"  SUMMARY — L{factual_layer}")
        print(f"  {'Strategy':30} {'Fix':>5} {'Brk':>5} {'Net':>5}")
        print("  " + "-" * 50)
        for name, r in all_results.items():
            net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net'])
            star = " ★" if r['net'] > 0 else ""
            print(f"  {name:30} +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")


if __name__ == "__main__":
    run()
