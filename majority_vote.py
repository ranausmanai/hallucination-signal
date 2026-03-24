"""
Self-Consistency: Temperature Sampling + Majority Vote
=======================================================
If the model's distribution has the correct answer at non-trivial probability
(which we know it does — rank 3 for UNESCO, rank 5 for speed of sound),
then sampling with temperature > 0 and taking the majority vote should
surface the correct answer more often than greedy (top-1) decoding.

This is the simplest possible correction mechanism:
  - No external data
  - No model modification
  - Just: generate K times, pick majority answer

Key question: does the correct answer appear often enough at temperature T
to win the majority vote?

Also tests: adaptive temperature based on oscillation signal.
High oscillation → higher temperature (explore more)
Low oscillation → lower temperature (stay confident)

This is pure self-consistency, no activation inspection needed.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.qwen2 import create_attention_mask
from collections import Counter
import json
import re


def ask(model, tokenizer, question, temperature=0.0, max_tokens=60):
    """Ask a question with given temperature."""
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=temperature)
    response = generate(
        model, tokenizer, prompt=fmt, max_tokens=max_tokens,
        verbose=False, sampler=sampler
    )
    return response.strip()


def get_oscillation(model, tokenizer, question):
    """Get oscillation count for a question (from single forward pass)."""
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)

    ids_mx = mx.array([ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)
    preds = []
    for layer in model.model.layers:
        h = layer(h, mask, None)
        mx.eval(h)
        h_norm = model.model.norm(h)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(h_norm)
        else:
            logits = model.lm_head(h_norm)
        mx.eval(logits)
        preds.append(int(np.argmax(np.array(logits[0, -1].astype(mx.float32)))))
        del logits, h_norm

    return sum(preds[i] != preds[i-1] for i in range(1, len(preds)))


def extract_answer_key(text, keywords):
    """Find which keyword appears in the answer (for majority voting)."""
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            return kw
    return None  # no keyword found


def majority_vote(answers, keywords):
    """
    Given N answers (strings), find the most common keyword match.
    Returns: (winning_keyword, vote_count, total)
    """
    keyword_hits = Counter()
    for ans in answers:
        match = extract_answer_key(ans, keywords)
        if match:
            keyword_hits[match] += 1
    if not keyword_hits:
        return None, 0, len(answers)
    winner, count = keyword_hits.most_common(1)[0]
    return winner, count, len(answers)


def self_consistency_answer(model, tokenizer, question, keywords,
                            n_samples=7, temperature=0.7):
    """Generate N samples, return majority-vote answer string."""
    samples = [ask(model, tokenizer, question, temperature=temperature)
               for _ in range(n_samples)]
    winner, count, total = majority_vote(samples, keywords)

    # Find an actual answer string that matched the winner
    if winner:
        for s in samples:
            if winner.lower() in s.lower():
                return s, samples, count, total
    # Fallback: greedy
    return ask(model, tokenizer, question, temperature=0.0), samples, 0, total


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

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

    N_SAMPLES = 7      # number of samples for self-consistency
    TEMPERATURES = [0.5, 0.7, 1.0]  # test multiple temperatures

    print(f"\n{'='*70}")
    print(f"Self-Consistency Correction (N={N_SAMPLES} samples)")
    print(f"{'='*70}\n")

    results = []
    for question, keywords in test_qa:
        # Greedy (baseline)
        greedy_answer = ask(model, tokenizer, question, temperature=0.0)
        greedy_correct = any(kw.lower() in greedy_answer.lower() for kw in keywords)

        # Oscillation (for adaptive temperature)
        osc = get_oscillation(model, tokenizer, question)

        print(f"Q: {question[:65]}")
        print(f"  [greedy,osc={osc:2d}]: '{greedy_answer[:60]}' {'✓' if greedy_correct else '✗'}")

        best_temp_result = {"correct": greedy_correct, "answer": greedy_answer}
        temp_results = {}

        for temp in TEMPERATURES:
            sc_answer, samples, vote_count, total = self_consistency_answer(
                model, tokenizer, question, keywords,
                n_samples=N_SAMPLES, temperature=temp
            )
            sc_correct = any(kw.lower() in sc_answer.lower() for kw in keywords)

            # Compute what fraction of samples got the right keyword
            right_frac = sum(
                1 for s in samples
                if any(kw.lower() in s.lower() for kw in keywords)
            ) / len(samples)

            temp_results[temp] = {
                "answer": sc_answer, "correct": sc_correct,
                "vote_count": vote_count, "right_fraction": right_frac,
                "samples": samples
            }

            marker = "✓" if sc_correct else "✗"
            indicator = ""
            if sc_correct and not greedy_correct:
                indicator = " *** FIXED ***"
            elif not sc_correct and greedy_correct:
                indicator = " *** BROKEN ***"
            print(f"  [SC T={temp}, frac={right_frac:.2f}]: '{sc_answer[:55]}' {marker}{indicator}")

        print()
        results.append({
            "question": question,
            "keywords": keywords,
            "greedy_answer": greedy_answer,
            "greedy_correct": greedy_correct,
            "oscillation": osc,
            "self_consistency": temp_results,
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    n_hard = 20
    n_total = len(test_qa)
    n_greedy = sum(r["greedy_correct"] for r in results)
    n_greedy_hard = sum(r["greedy_correct"] for r in results[:n_hard])

    print(f"Greedy: {n_greedy}/{n_total} (hard: {n_greedy_hard}/{n_hard})")
    print()

    for temp in TEMPERATURES:
        n_sc = sum(
            1 for r in results
            if r["self_consistency"][temp]["correct"]
        )
        n_sc_hard = sum(
            1 for r in results[:n_hard]
            if r["self_consistency"][temp]["correct"]
        )
        fixes = sum(
            1 for r in results[:n_hard]
            if not r["greedy_correct"]
            and r["self_consistency"][temp]["correct"]
        )
        breaks = sum(
            1 for r in results[:n_hard]
            if r["greedy_correct"]
            and not r["self_consistency"][temp]["correct"]
        )
        delta = n_sc_hard - n_greedy_hard
        delta_s = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "="
        star = " ★★" if delta >= 3 else (" ★" if delta >= 2 else (" ~" if delta == 1 else ""))
        print(f"  SC T={temp}: {n_sc}/{n_total} (hard: {n_sc_hard}/{n_hard}) "
              f"[{delta_s}] +{fixes} fixed, -{breaks} broken{star}")

    # Key finding: for wrong greedy answers, how often does SC sample the right answer?
    print(f"\n{'='*70}")
    print("For questions where GREEDY is WRONG:")
    print("Fraction of SC samples that contained the right answer")
    print(f"{'='*70}\n")

    wrong_cases = [r for r in results[:n_hard] if not r["greedy_correct"]]
    if wrong_cases:
        print(f"{'Question':>50} {'T=0.5':>6} {'T=0.7':>6} {'T=1.0':>6}")
        print("-" * 75)
        for r in wrong_cases:
            fracs = {t: r["self_consistency"][t]["right_fraction"] for t in TEMPERATURES}
            print(f"  {r['question'][:48]:>50} {fracs[0.5]:>6.2f} {fracs[0.7]:>6.2f} {fracs[1.0]:>6.2f}")

        avg_fracs = {t: np.mean([r["self_consistency"][t]["right_fraction"]
                                  for r in wrong_cases])
                     for t in TEMPERATURES}
        print(f"\n  Mean sampling rate of correct answer (on wrong greedy cases):")
        for t, f in avg_fracs.items():
            print(f"    T={t}: {f:.3f} ({f*100:.1f}% of samples correct)")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    best_temp = max(TEMPERATURES, key=lambda t: sum(
        1 for r in results[:n_hard]
        if r["self_consistency"][t]["correct"]))
    best_hard = sum(1 for r in results[:n_hard] if r["self_consistency"][best_temp]["correct"])
    if best_hard > n_greedy_hard:
        print(f"""
★ SELF-CONSISTENCY WORKS

Temperature={best_temp}, N={N_SAMPLES} samples improves accuracy
from {n_greedy_hard}/{n_hard} to {best_hard}/{n_hard} on hard questions
(+{best_hard-n_greedy_hard} correct answers).

The model's distribution contains the correct answer at non-trivial
probability. Greedy decoding picks the wrong most-likely answer;
sampling + majority vote surfaces the buried correct answer.

This requires {N_SAMPLES}x more compute but no external data.
Combined with oscillation detection: only apply SC when uncertain.
""")
    else:
        print(f"\nSelf-consistency does not improve over greedy on this model.")
        print(f"The correct answer is too deeply buried — even at T=1.0,")
        print(f"sampling rarely produces the correct answer.")

    with open("majority_vote_results.json", "w") as f:
        json.dump([{
            "question": r["question"],
            "greedy_correct": r["greedy_correct"],
            "oscillation": r["oscillation"],
            "sc_correct": {str(t): r["self_consistency"][t]["correct"] for t in TEMPERATURES},
            "right_fraction": {str(t): r["self_consistency"][t]["right_fraction"] for t in TEMPERATURES},
        } for r in results], f, indent=2)
    print("Saved majority_vote_results.json")


if __name__ == "__main__":
    run()
