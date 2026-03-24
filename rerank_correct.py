"""
Top-K Reranking with Intermediate Layer Signal
================================================
From buried_answer.py: 9/10 wrong answers have the correct token
in top-200 at the final layer. The correct token peaks at layer 21
before being suppressed by layers 22-23.

Key insight: What if we take the TOP-200 from the final layer
(conservative set — preserves fluency and task-following)
and RERANK those 200 tokens using layer 21's preferences?

This is conservative: we only pick from tokens the final layer
already considers plausible, but use layer 21 to break ties.

For UNESCO (China vs France):
  - Both China and France are in final layer top-200
  - Layer 21 ranks China at 1, France lower
  - Reranking selects China ← CORRECT

For Kyrgyzstan (where ensemble failed):
  - Bishkek is top-1 at final layer
  - Layer 21 also likes Bishkek (since model normally gets it right)
  - Reranking still selects Bishkek ← safe

This should fix the "ensemble introduces Chinese tokens" problem
because Chinese tokens are unlikely in the final layer's top-200
for fluent English completion (the LM head is tuned for fluency).

Also tests: mixture scoring
  score = (1-alpha)*logit_L23 + alpha*logit_L21

vs pure reranking within top-K.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.qwen2 import create_attention_mask
import json


def forward_collect_layers(model, token_ids, collect=(21, 23)):
    """
    Single forward pass. Returns hidden states at specified layers.
    """
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)
    result = {}

    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)
        if i in collect:
            result[i] = np.array(h[0, -1].astype(mx.float32))

    return result


def h_to_logits(model, h_np):
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


def rerank_step(model, token_ids, source_layer, final_layer, alpha, top_k=200):
    """
    1. Get top-K tokens from final_layer
    2. Score them with: (1-alpha)*score_final + alpha*score_source
    3. Pick best

    alpha=0: normal (final only)
    alpha=0.5: equal mix
    alpha=1.0: source only (within final's top-K)
    """
    layer_h = forward_collect_layers(model, token_ids, collect={source_layer, final_layer})

    logits_final = h_to_logits(model, layer_h[final_layer])
    logits_source = h_to_logits(model, layer_h[source_layer])

    # Candidate set: top-K from final layer
    top_k_ids = np.argsort(-logits_final)[:top_k]

    # Score candidates with mixture
    scores = (1 - alpha) * logits_final[top_k_ids] + alpha * logits_source[top_k_ids]
    best_idx = top_k_ids[np.argmax(scores)]
    return int(best_idx)


def rerank_step_v2(model, token_ids, source_layer, final_layer, alpha, top_k=200):
    """
    Alternative: normalize logits first, then mix.
    This prevents final layer from dominating due to scale.
    """
    layer_h = forward_collect_layers(model, token_ids, collect={source_layer, final_layer})

    logits_final = h_to_logits(model, layer_h[final_layer])
    logits_source = h_to_logits(model, layer_h[source_layer])

    # Softmax normalize both
    def softmax(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    probs_final = softmax(logits_final)
    probs_source = softmax(logits_source)

    # Candidate set: top-K from final layer
    top_k_ids = np.argsort(-probs_final)[:top_k]

    # Score candidates with mixture of probabilities
    scores = (1 - alpha) * probs_final[top_k_ids] + alpha * probs_source[top_k_ids]
    best_idx = top_k_ids[np.argmax(scores)]
    return int(best_idx)


def generate_reranked(model, tokenizer, question, source_layer, final_layer,
                      alpha, top_k=200, use_v2=False, max_tokens=60):
    """Full autoregressive generation with reranking."""
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    fn = rerank_step_v2 if use_v2 else rerank_step

    for _ in range(max_tokens):
        next_id = fn(model, ids, source_layer, final_layer, alpha, top_k)
        if next_id == eos_id:
            break
        generated.append(next_id)
        ids.append(next_id)

    return tokenizer.decode(generated).strip()


def generate_normal(model, tokenizer, question, final_layer, max_tokens=60):
    """Standard greedy (final layer only)."""
    return generate_reranked(model, tokenizer, question,
                              source_layer=final_layer, final_layer=final_layer,
                              alpha=0, top_k=1, max_tokens=max_tokens)


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    n_layers = len(model.model.layers)
    print(f"Model: {n_layers} layers\n")

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

    # Configs: (source_layer, alpha, top_k, use_v2, description)
    configs = [
        (23,  0.0,  1,    False, "normal"),
        (21,  0.3, 200,   False, "L21 topK=200 α=0.3"),
        (21,  0.5, 200,   False, "L21 topK=200 α=0.5"),
        (21,  0.7, 200,   False, "L21 topK=200 α=0.7"),
        (21,  0.5,  50,   False, "L21 topK=50  α=0.5"),
        (21,  0.5, 200,   True,  "L21 softmax  α=0.5"),
        (20,  0.5, 200,   False, "L20 topK=200 α=0.5"),
        (21,  0.5,  20,   False, "L21 topK=20  α=0.5"),
    ]

    final_layer = n_layers - 1  # 23

    print(f"{'='*75}")
    print("Top-K Reranking with Intermediate Layer (L21) Signal")
    print(f"{'='*75}\n")

    all_results = {name: [] for _, _, _, _, name in configs}
    per_question = []

    for question, keywords in test_qa:
        print(f"Q: {question[:65]}")
        q_res = {"question": question, "keywords": keywords, "configs": {}}

        for src_layer, alpha, top_k, use_v2, name in configs:
            if alpha == 0.0:
                # True normal: only final layer
                messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
                fmt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
                ids = tokenizer.encode(fmt, add_special_tokens=True)
                eos_id = tokenizer.eos_token_id
                gen_ids = list(ids)
                generated = []
                for _ in range(60):
                    lh = forward_collect_layers(model, gen_ids, collect={final_layer})
                    lg = h_to_logits(model, lh[final_layer])
                    nid = int(np.argmax(lg))
                    if nid == eos_id:
                        break
                    generated.append(nid)
                    gen_ids.append(nid)
                answer = tokenizer.decode(generated).strip()
            else:
                answer = generate_reranked(model, tokenizer, question,
                                            src_layer, final_layer,
                                            alpha, top_k, use_v2)

            is_correct = any(kw.lower() in answer.lower() for kw in keywords)
            all_results[name].append(is_correct)
            q_res["configs"][name] = {"answer": answer, "correct": is_correct}

            marker = "✓" if is_correct else "✗"
            print(f"  [{name:24}]: '{answer[:45]}' {marker}")

        per_question.append(q_res)
        print()

    # Summary
    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}\n")

    n_hard = 20
    n_total = len(test_qa)
    normal_total = sum(all_results["normal"])
    normal_hard = sum(all_results["normal"][:n_hard])

    print(f"{'Config':28} {'Total':>6} {'Hard':>6}  {'Δ total':>8}  {'Δ hard':>8}")
    print("-" * 65)

    for _, _, _, _, name in configs:
        tot = sum(all_results[name])
        hard = sum(all_results[name][:n_hard])
        dt = tot - normal_total
        dh = hard - normal_hard
        dt_s = f"+{dt}" if dt > 0 else str(dt) if dt < 0 else "="
        dh_s = f"+{dh}" if dh > 0 else str(dh) if dh < 0 else "="
        star = " ★★" if dh >= 3 else (" ★" if dh >= 2 else (" ~" if dh == 1 else ""))
        print(f"  {name:26} {tot:>4}/{n_total}  {hard:>4}/{n_hard}  {dt_s:>8}  {dh_s:>8}{star}")

    # Best config
    best_name = max(
        [name for _, _, _, _, name in configs],
        key=lambda n: (sum(all_results[n][:n_hard]),
                       sum(all_results[n]) - sum(all_results["normal"]))
    )
    best_hard = sum(all_results[best_name][:n_hard])

    # Corrections for best config
    print(f"\nBest config: {best_name} — hard={best_hard}/{n_hard}")
    if best_hard > normal_hard:
        fixes = [q for q in per_question[:n_hard]
                 if not q["configs"]["normal"]["correct"]
                 and q["configs"][best_name]["correct"]]
        breaks = [q for q in per_question[:n_hard]
                  if q["configs"]["normal"]["correct"]
                  and not q["configs"][best_name]["correct"]]
        print(f"  Fixed: {len(fixes)}")
        for r in fixes:
            print(f"    ✓ {r['question'][:60]}")
            print(f"      Before: '{r['configs']['normal']['answer'][:55]}'")
            print(f"      After:  '{r['configs'][best_name]['answer'][:55]}'")
        print(f"  Broke: {len(breaks)}")
        for r in breaks:
            print(f"    ✗ {r['question'][:60]}")

    with open("rerank_results.json", "w") as f:
        json.dump({
            "n_total": n_total, "n_hard": n_hard,
            "summary": {name: {"total": sum(all_results[name]),
                               "hard": sum(all_results[name][:n_hard])}
                        for _, _, _, _, name in configs},
            "per_question": per_question,
        }, f, indent=2)
    print("\nSaved rerank_results.json")


if __name__ == "__main__":
    run()
