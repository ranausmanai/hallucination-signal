"""
Buried Answer Hypothesis
=========================
When the model hallucinates, is the correct answer actually IN the logits
at some intermediate layer, just ranked below the wrong token?

Specifically: at the final layer, does the correct answer token appear
in the top-K predictions even when it's not top-1?

If yes: the model "has the knowledge" but suppresses it.
This is fundamentally different from "the model doesn't know."

And: at which LAYER does the correct first token appear in top-K?
Does it peak at oscillation layers?

This is testable right now.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate
from mlx_lm.models.qwen2 import create_attention_mask
import json


def get_per_layer_topk(model, token_ids, k=50):
    """
    At each layer, get top-K predicted token IDs at last position.
    Returns: list of lists, shape (n_layers, k)
    Also returns per-layer confidence (top-1 prob).
    """
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    all_topk = []
    all_probs_top1 = []

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
        topk_ids = np.argsort(-last)[:k]
        probs = np.exp(last - last.max())
        probs /= probs.sum()
        all_topk.append(topk_ids.tolist())
        all_probs_top1.append(float(probs[topk_ids[0]]))
        del logits, h_norm

    return all_topk, all_probs_top1


def ask(model, tokenizer, question, max_tokens=60):
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=fmt, max_tokens=max_tokens, verbose=False).strip(), fmt


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    n_layers = len(model.model.layers)
    K = 200  # look in top-200 tokens

    # Questions where the FIRST TOKEN of the correct answer is predictable
    # We check if that first token appears in top-K at each layer
    test_qa = [
        # (question, correct_answer_keywords, correct_first_tokens_to_search_for)
        ("Who was the 30th president of the United States?",
         ["coolidge", "calvin"],
         ["cool", "Calvin", "Cool", "Coolidge", "coolidge"]),

        ("What is the melting point of tungsten in Celsius?",
         ["3422", "3400"],
         ["3422", "3400", "3,422", "3,400"]),

        ("What is the half-life of uranium-235 in years?",
         ["703 million", "703"],
         ["703", "700", "7.04"]),

        ("Who is the prime minister of New Zealand as of 2024?",
         ["luxon", "christopher"],
         ["Luxon", "luxon", "Christopher", "christopher"]),

        ("What is the largest desert in the world by area?",
         ["antarctica", "antarctic"],
         ["Antarctica", "antarctic", "Antarctic"]),

        ("What country has the most UNESCO World Heritage Sites?",
         ["italy", "china"],
         ["China", "Italy", "china", "italy"]),

        ("What is the atomic weight of plutonium?",
         ["244", "242", "239"],
         ["244", "242", "239"]),

        ("What is the rarest blood type?",
         ["ab-", "ab negative"],
         ["AB", "ab", "AB-", "ab-"]),

        ("Who won the Nobel Prize in Chemistry in 2023?",
         ["bawendi", "brus", "ekimov"],
         ["Baw", "baw", "Brus", "brus", "Ek", "ek"]),

        ("What is the speed of sound in water in m/s?",
         ["1480", "1500", "1498"],
         ["1480", "1500", "1498", "1,480", "1,500"]),

        # Easy control questions (model gets right)
        ("What is the capital of France?",
         ["paris"],
         ["Paris", "paris"]),

        ("Who wrote Romeo and Juliet?",
         ["shakespeare"],
         ["Shakespeare", "Shake"]),
    ]

    print(f"\n{'='*70}")
    print(f"Buried Answer Analysis: Is the correct token in top-{K}?")
    print(f"{'='*70}\n")

    results = []

    for question, keywords, correct_first_toks in test_qa:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        # Get actual answer
        normal_answer, _ = ask(model, tokenizer, question)
        is_correct = any(kw.lower() in normal_answer.lower() for kw in keywords)

        # Get per-layer top-K
        all_topk, all_conf = get_per_layer_topk(model, ids, k=K)

        # For each correct first token, find its rank at each layer
        # Tokenize the correct first tokens
        correct_ids = set()
        for tok_str in correct_first_toks:
            # Try encoding as a continuation (no special tokens, with space prefix)
            for prefix in ["", " ", "Ġ"]:
                try:
                    tids = tokenizer.encode(prefix + tok_str, add_special_tokens=False)
                    if tids:
                        correct_ids.add(tids[0])
                except:
                    pass

        # At each layer, what rank are the correct tokens?
        min_rank_per_layer = []
        for layer_topk in all_topk:
            ranks = []
            for cid in correct_ids:
                if cid in layer_topk:
                    ranks.append(layer_topk.index(cid))
            min_rank_per_layer.append(min(ranks) if ranks else K + 1)

        # Where does the correct token appear at highest rank?
        best_rank = min(min_rank_per_layer)
        best_layer = int(np.argmin(min_rank_per_layer))
        final_rank = min_rank_per_layer[-1]

        # Oscillation count
        top1_preds = [topk[0] for topk in all_topk]
        osc_count = sum(top1_preds[i] != top1_preds[i-1] for i in range(1, n_layers))

        # Find oscillation peak layer (last change)
        peak_layer = 0
        for i in range(n_layers - 1, 0, -1):
            if top1_preds[i] != top1_preds[i-1]:
                peak_layer = i
                break

        answer_marker = "✓" if is_correct else "✗"
        buried = best_rank < K and not is_correct  # correct token IS in top-K but model outputs wrong
        burial_depth = best_rank if not is_correct else -1

        print(f"Q: {question[:65]}")
        print(f"  Answer: '{normal_answer[:60]}' {answer_marker}")
        print(f"  Correct tokens searched: {correct_first_toks[:3]}")
        print(f"  Best rank of correct token across all layers: {best_rank} (at layer {best_layer})")
        print(f"  Rank at final layer: {final_rank}")
        print(f"  Oscillation: {osc_count}, peak at layer {peak_layer}")
        if buried:
            print(f"  *** BURIED: correct answer in top-{K} but not top-1! ***")
            # Show the rank profile
            profile = [f"L{i}:{r}" for i, r in enumerate(min_rank_per_layer) if r <= K][:10]
            print(f"  Rank profile (where ≤{K}): {profile}")
        print()

        results.append({
            "question": question,
            "normal_answer": normal_answer,
            "is_correct": is_correct,
            "best_rank": best_rank,
            "best_layer": best_layer,
            "final_rank": final_rank,
            "oscillation": osc_count,
            "peak_layer": peak_layer,
            "buried": buried,
            "rank_profile": min_rank_per_layer,
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Buried Answer Analysis")
    print(f"{'='*70}\n")

    wrong = [r for r in results if not r["is_correct"]]
    correct = [r for r in results if r["is_correct"]]

    buried_cases = [r for r in wrong if r["best_rank"] < K]
    totally_absent = [r for r in wrong if r["best_rank"] >= K]

    print(f"Total: {len(results)}, Correct: {len(correct)}, Wrong: {len(wrong)}")
    print(f"\nOf {len(wrong)} wrong answers:")
    print(f"  Correct token IN top-{K} somewhere:  {len(buried_cases)} cases (BURIED)")
    print(f"  Correct token NOT in top-{K} anywhere: {len(totally_absent)} cases (ABSENT)")

    if buried_cases:
        print(f"\nBuried cases (model had the answer but buried it):")
        for r in buried_cases:
            print(f"  Q: {r['question'][:60]}")
            print(f"     Best rank: {r['best_rank']} at layer {r['best_layer']}, final rank: {r['final_rank']}")
            print(f"     Oscillation: {r['oscillation']}, peak: {r['peak_layer']}")

    # Key stat: does oscillation predict burial?
    if buried_cases and totally_absent:
        osc_buried = np.mean([r["oscillation"] for r in buried_cases])
        osc_absent = np.mean([r["oscillation"] for r in totally_absent])
        print(f"\nMean oscillation: buried={osc_buried:.1f}, absent={osc_absent:.1f}")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if len(buried_cases) > len(wrong) * 0.5:
        print(f"""
★ KNOWLEDGE SUPPRESSION IS REAL

{len(buried_cases)}/{len(wrong)} wrong answers have the correct token in top-{K}.
The model IS retrieving correct information internally but suppressing it.
This is not "ignorance" — it's "confusion about what to output."

This is mechanistically different from missing knowledge:
  - Missing knowledge: correct token never appears in any layer
  - Knowledge suppression: correct token appears (rank {min(r['best_rank'] for r in buried_cases)}-{max(r['best_rank'] for r in buried_cases)})
    at some layers but gets pushed down by final layer

Implication: a re-ranking intervention AT the right layer
could surface the correct answer without any external data.
""")
    elif buried_cases:
        print(f"\n{len(buried_cases)}/{len(wrong)} cases show burial — partial evidence.")
        print("The model sometimes has the answer but more often it's simply absent.")
    else:
        print(f"\nNo burial detected — the model truly doesn't know these answers.")
        print("Correct tokens are absent from top-K at all layers.")

    with open("buried_answer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved buried_answer_results.json")


if __name__ == "__main__":
    run()
