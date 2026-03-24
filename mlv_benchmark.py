"""
MLV Benchmark — Large-scale evaluation
========================================

Test Multi-Layer Voting on a proper benchmark:
- 100 diverse factual questions across categories
- Multiple models: Qwen2-0.5B, Qwen3.5-0.8B
- Statistical significance via larger N
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import json
import time


# ============================================================
# Benchmark: 100 factual questions, diverse categories
# ============================================================

BENCHMARK = [
    # === GEOGRAPHY: Capitals (20) ===
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is the capital of Brazil?", ["brasilia", "brasília"]),
    ("What is the capital of Australia?", ["canberra"]),
    ("What is the capital of Egypt?", ["cairo"]),
    ("What is the capital of Turkey?", ["ankara"]),
    ("What is the capital of Mongolia?", ["ulaanbaatar", "ulan bator"]),
    ("What is the capital of Kyrgyzstan?", ["bishkek"]),
    ("What is the capital of Burkina Faso?", ["ouagadougou"]),
    ("What is the capital of Myanmar?", ["naypyidaw", "naypyitaw", "nay pyi taw"]),
    ("What is the capital of Kazakhstan?", ["astana", "nur-sultan"]),
    ("What is the capital of Madagascar?", ["antananarivo"]),
    ("What is the capital of Sri Lanka?", ["colombo", "sri jayawardenepura"]),
    ("What is the capital of Nigeria?", ["abuja"]),
    ("What is the capital of Pakistan?", ["islamabad"]),
    ("What is the capital of New Zealand?", ["wellington"]),
    ("What is the capital of Switzerland?", ["bern", "berne"]),
    ("What is the capital of South Africa?", ["pretoria", "cape town", "bloemfontein"]),
    ("What is the capital of Morocco?", ["rabat"]),
    ("What is the capital of Thailand?", ["bangkok"]),

    # === GEOGRAPHY: Other (10) ===
    ("What is the largest ocean?", ["pacific"]),
    ("What is the longest river in the world?", ["nile", "amazon"]),
    ("What is the largest desert in the world by area?", ["antarctica", "antarctic"]),
    ("What is the tallest mountain in the world?", ["everest"]),
    ("What is the smallest country in the world?", ["vatican"]),
    ("What is the deepest ocean trench?", ["mariana"]),
    ("What is the largest continent by area?", ["asia"]),
    ("What is the largest lake in the world by surface area?", ["caspian"]),
    ("What is the largest island in the world?", ["greenland"]),
    ("What is the driest continent?", ["antarctica", "antarctic"]),

    # === SCIENCE (20) ===
    ("What is the speed of light in m/s?", ["299792458", "299,792,458", "300000000", "3×10", "3x10"]),
    ("What is the speed of sound in water in m/s?", ["1480", "1500", "1498", "1,480", "1,500"]),
    ("What is the boiling point of water in Celsius?", ["100"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("What is the chemical symbol for iron?", ["fe"]),
    ("What is the atomic number of carbon?", ["6", "six"]),
    ("What is the atomic number of hydrogen?", ["1", "one"]),
    ("What is the largest planet in the solar system?", ["jupiter"]),
    ("What is the smallest planet in the solar system?", ["mercury"]),
    ("How many chromosomes do humans have?", ["46", "forty-six", "forty six"]),
    ("What is the hardest natural substance?", ["diamond"]),
    ("What is the most abundant element in the universe?", ["hydrogen"]),
    ("What is the melting point of tungsten in Celsius?", ["3422", "3400", "3,422"]),
    ("What is the half-life of Carbon-14 in years?", ["5730", "5700", "5,730"]),
    ("What is the half-life of uranium-235 in years?", ["703", "700"]),
    ("What is the atomic weight of plutonium?", ["244", "242", "239"]),
    ("What planet is known as the Red Planet?", ["mars"]),
    ("What gas do plants absorb from the atmosphere?", ["carbon dioxide", "co2"]),
    ("What is the closest star to Earth?", ["sun", "proxima centauri"]),
    ("How many bones does an adult human have?", ["206"]),

    # === HISTORY (15) ===
    ("In what year did World War II end?", ["1945"]),
    ("Who was the first president of the United States?", ["washington", "george"]),
    ("Who was the 30th president of the United States?", ["coolidge", "calvin"]),
    ("In what year was the Declaration of Independence signed?", ["1776"]),
    ("Who was the first person to walk on the moon?", ["armstrong", "neil"]),
    ("In what year did the Berlin Wall fall?", ["1989"]),
    ("What year was the WHO founded?", ["1948"]),
    ("Who was the first female prime minister of the UK?", ["thatcher"]),
    ("In what year did the Titanic sink?", ["1912"]),
    ("Who was the first emperor of China?", ["qin shi huang", "shi huangdi", "ying zheng"]),
    ("In what year did World War I begin?", ["1914"]),
    ("Who discovered penicillin?", ["fleming", "alexander"]),
    ("In what year was the United Nations founded?", ["1945"]),
    ("Who invented the telephone?", ["bell", "alexander graham"]),
    ("What year did the French Revolution begin?", ["1789"]),

    # === LITERATURE & ARTS (15) ===
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("Who wrote 1984?", ["orwell", "george"]),
    ("Who wrote War and Peace?", ["tolstoy"]),
    ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]),
    ("Who composed the Moonlight Sonata?", ["beethoven"]),
    ("Who wrote Don Quixote?", ["cervantes"]),
    ("Who painted Starry Night?", ["van gogh", "gogh"]),
    ("Who wrote The Great Gatsby?", ["fitzgerald"]),
    ("Who composed The Four Seasons?", ["vivaldi"]),
    ("Who wrote Pride and Prejudice?", ["austen"]),
    ("Who sculpted David?", ["michelangelo"]),
    ("Who wrote The Odyssey?", ["homer"]),
    ("Who composed Swan Lake?", ["tchaikovsky"]),
    ("Who wrote Hamlet?", ["shakespeare"]),
    ("Who painted the Sistine Chapel ceiling?", ["michelangelo"]),

    # === GENERAL KNOWLEDGE (10) ===
    ("What is 2 + 2?", ["4", "four"]),
    ("What language is spoken in Brazil?", ["portuguese"]),
    ("What country has the most UNESCO World Heritage Sites?", ["italy", "china"]),
    ("What is the rarest blood type?", ["ab-", "ab negative"]),
    ("What currency is used in Japan?", ["yen"]),
    ("What is the largest organ in the human body?", ["skin"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("What metal is liquid at room temperature?", ["mercury"]),
    ("What is the most spoken language in the world?", ["mandarin", "chinese", "english"]),
    ("What is the largest mammal?", ["blue whale"]),

    # === MATH & NUMBERS (10) ===
    ("What is the value of pi to two decimal places?", ["3.14"]),
    ("What is the square root of 144?", ["12", "twelve"]),
    ("What is 7 × 8?", ["56", "fifty-six"]),
    ("What is the cube root of 27?", ["3", "three"]),
    ("How many degrees in a circle?", ["360"]),
    ("What is 15% of 200?", ["30", "thirty"]),
    ("What is the next prime number after 7?", ["11", "eleven"]),
    ("How many seconds in an hour?", ["3600", "3,600"]),
    ("What is 2 to the power of 10?", ["1024", "1,024"]),
    ("What is the sum of angles in a triangle?", ["180"]),
]


def check_correct(answer, keywords):
    answer_clean = answer.lower().replace(",", "").replace("*", "")
    return any(kw.lower().replace(",", "") in answer_clean for kw in keywords)


def is_ascii_token(tokenizer, tok_id):
    text = tokenizer.decode([tok_id])
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars >= len(text) * 0.5 if len(text) > 0 else True


# ============================================================
# Model setup
# ============================================================

def setup_qwen2(model):
    from mlx_lm.models.qwen2 import create_attention_mask as cam
    def get_logits(h):
        h_normed = model.model.norm(h)
        if model.args.tie_word_embeddings:
            return model.model.embed_tokens.as_linear(h_normed)
        return model.lm_head(h_normed)
    return {
        "embed": model.model.embed_tokens,
        "layers": model.model.layers,
        "norm": model.model.norm,
        "get_logits_from_hidden": get_logits,
        "get_mask": lambda h: cam(h, None),
        "get_layer_mask": None,
        "n_layers": len(model.model.layers),
    }


def setup_qwen35(model):
    from mlx_lm.models.qwen3_5 import create_attention_mask as cam, create_ssm_mask as csm
    lm = model.language_model
    tm = lm.model
    masks = {}
    def get_mask(h):
        masks["fa"] = cam(h, None)
        masks["ssm"] = csm(h, None)
        return masks
    def get_layer_mask(layer, mask_dict):
        return mask_dict["ssm"] if layer.is_linear else mask_dict["fa"]
    def get_logits(h):
        h_normed = tm.norm(h)
        return tm.embed_tokens.as_linear(h_normed)
    return {
        "embed": tm.embed_tokens,
        "layers": tm.layers,
        "norm": tm.norm,
        "get_logits_from_hidden": get_logits,
        "get_mask": get_mask,
        "get_layer_mask": get_layer_mask,
        "n_layers": len(tm.layers),
    }


# ============================================================
# MLV Decoding
# ============================================================

def forward_voting_layers(model_info, ids, voting_layers):
    mi = model_info
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)
    voting_set = set(voting_layers)
    layer_logits = {}

    for i, layer in enumerate(mi["layers"]):
        if mi.get("get_layer_mask"):
            m = mi["get_layer_mask"](layer, mask)
        else:
            m = mask
        h = layer(h, m, None)
        if i in voting_set:
            logits_i = mi["get_logits_from_hidden"](h[:, -1:, :])
            mx.eval(logits_i)
            layer_logits[i] = logits_i[0, 0]

    return layer_logits


def decode_weighted_mlv_ascii(layer_logits, voting_layers, tokenizer, k=10):
    from collections import defaultdict
    final_layer = voting_layers[-1]
    final = layer_logits[final_layer]

    top_k_indices = mx.argpartition(final, kth=-k, axis=-1)[-k:]
    top_k_list = [int(x) for x in top_k_indices.tolist()]
    ascii_tokens = [t for t in top_k_list if is_ascii_token(tokenizer, t)]
    if len(ascii_tokens) == 0:
        ascii_tokens = top_k_list

    weighted_votes = defaultdict(float)
    for i in voting_layers:
        logits = layer_logits[i]
        probs = mx.softmax(logits, axis=-1)
        best_tok = None
        best_prob = 0.0
        for tok in ascii_tokens:
            p = float(probs[tok].item())
            if p > best_prob:
                best_prob = p
                best_tok = tok
        if best_tok is not None:
            weighted_votes[best_tok] += best_prob

    if len(weighted_votes) == 0:
        return int(mx.argmax(final).item())
    return max(weighted_votes, key=weighted_votes.get)


def generate_mlv(model_info, tokenizer, question, voting_layers, k=10,
                 max_tokens=60):
    mi = model_info
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []
    overrides = 0

    for step in range(max_tokens):
        layer_logits = forward_voting_layers(mi, ids, voting_layers)
        final_layer = voting_layers[-1]
        std_pick = int(mx.argmax(layer_logits[final_layer]).item())
        mlv_pick = decode_weighted_mlv_ascii(layer_logits, voting_layers,
                                              tokenizer, k=k)
        if mlv_pick != std_pick:
            overrides += 1
        if mlv_pick == eos_id:
            break
        generated.append(mlv_pick)
        ids.append(mlv_pick)
        del layer_logits

    return tokenizer.decode(generated).strip(), overrides


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, model_info, tokenizer, model_name, voting_layers,
                   k_values, benchmark):
    """Evaluate standard vs MLV on the full benchmark."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"Benchmark: {len(benchmark)} questions")
    print(f"Voting layers: {voting_layers}")
    print(f"{'='*70}")

    # First: standard greedy baseline
    print(f"\n  Running standard greedy baseline...")
    t0 = time.time()
    baseline_results = []
    for i, (question, keywords) in enumerate(benchmark):
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                          verbose=False, sampler=sampler).strip()
        correct = check_correct(answer, keywords)
        baseline_results.append({
            "question": question, "answer": answer, "correct": correct,
            "keywords": keywords,
        })
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(benchmark)} done...")

    baseline_correct = sum(r["correct"] for r in baseline_results)
    baseline_time = time.time() - t0
    print(f"  Baseline: {baseline_correct}/{len(benchmark)} ({100*baseline_correct/len(benchmark):.1f}%) in {baseline_time:.1f}s")

    # Category breakdown
    categories = [
        ("Capitals", 0, 20),
        ("Geography", 20, 30),
        ("Science", 30, 50),
        ("History", 50, 65),
        ("Literature", 65, 80),
        ("General", 80, 90),
        ("Math", 90, 100),
    ]
    print(f"\n  Category breakdown (baseline):")
    for cat_name, start, end in categories:
        cat_correct = sum(r["correct"] for r in baseline_results[start:end])
        n = end - start
        print(f"    {cat_name:15} {cat_correct:2d}/{n:2d} ({100*cat_correct/n:.0f}%)")

    # Show wrong answers
    wrong = [r for r in baseline_results if not r["correct"]]
    print(f"\n  Wrong answers ({len(wrong)}):")
    for r in wrong:
        print(f"    ✗ {r['question'][:55]}")
        print(f"      → '{r['answer'][:60]}'")

    # Now: MLV with different k values
    all_mlv_results = {}
    for k in k_values:
        label = f"MLV k={k}"
        print(f"\n  Running {label}...")
        t0 = time.time()
        mlv_results = []
        total_overrides = 0

        for i, (question, keywords) in enumerate(benchmark):
            mlv_answer, ov = generate_mlv(
                model_info, tokenizer, question, voting_layers, k=k)
            mlv_correct = check_correct(mlv_answer, keywords)
            total_overrides += ov
            mlv_results.append({
                "question": question, "answer": mlv_answer,
                "correct": mlv_correct, "overrides": ov,
            })
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(benchmark)} done...")

        mlv_time = time.time() - t0
        mlv_correct_total = sum(r["correct"] for r in mlv_results)

        fixes = sum(1 for r, b in zip(mlv_results, baseline_results)
                     if r["correct"] and not b["correct"])
        breaks = sum(1 for r, b in zip(mlv_results, baseline_results)
                      if not r["correct"] and b["correct"])
        net = fixes - breaks

        print(f"  {label}: {mlv_correct_total}/{len(benchmark)} "
              f"({100*mlv_correct_total/len(benchmark):.1f}%) "
              f"| +{fixes}/-{breaks} = {'+'if net>=0 else ''}{net} "
              f"| {total_overrides} overrides | {mlv_time:.1f}s")

        # Category breakdown
        print(f"\n  Category breakdown ({label}):")
        for cat_name, start, end in categories:
            base_cat = sum(r["correct"] for r in baseline_results[start:end])
            mlv_cat = sum(r["correct"] for r in mlv_results[start:end])
            n = end - start
            diff = mlv_cat - base_cat
            diff_s = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "="
            print(f"    {cat_name:15} {mlv_cat:2d}/{n:2d} ({100*mlv_cat/n:.0f}%) [was {base_cat}, {diff_s}]")

        # Show fixes and breaks
        if fixes > 0:
            print(f"\n  Fixed ({fixes}):")
            for r, b in zip(mlv_results, baseline_results):
                if r["correct"] and not b["correct"]:
                    print(f"    ✓ {r['question'][:55]}")
                    print(f"      was: '{b['answer'][:55]}'")
                    print(f"      now: '{r['answer'][:55]}'")

        if breaks > 0:
            print(f"\n  Broken ({breaks}):")
            for r, b in zip(mlv_results, baseline_results):
                if not r["correct"] and b["correct"]:
                    print(f"    ✗ {r['question'][:55]}")
                    print(f"      was: '{b['answer'][:55]}'")
                    print(f"      now: '{r['answer'][:55]}'")

        all_mlv_results[label] = {
            "correct": mlv_correct_total, "total": len(benchmark),
            "fixes": fixes, "breaks": breaks, "net": net,
            "overrides": total_overrides,
        }

    return baseline_correct, all_mlv_results


def run():
    print("=" * 70)
    print("MLV BENCHMARK — 100 Factual Questions")
    print("=" * 70)

    all_results = {}

    # ==========================================
    # Qwen2-0.5B
    # ==========================================
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    mi = setup_qwen2(model)
    n_layers = mi["n_layers"]
    voting_layers = list(range(n_layers - 4, n_layers))  # last 4

    baseline, mlv_results = evaluate_model(
        model, mi, tokenizer, "Qwen2-0.5B-Instruct",
        voting_layers, k_values=[10, 20], benchmark=BENCHMARK)
    all_results["Qwen2-0.5B"] = {
        "baseline": baseline, "mlv": mlv_results
    }
    del model, mi

    # ==========================================
    # Qwen3.5-0.8B
    # ==========================================
    model, tokenizer = load("Qwen/Qwen3.5-0.8B")
    mi = setup_qwen35(model)
    n_layers = mi["n_layers"]
    voting_layers = list(range(n_layers - 4, n_layers))

    baseline, mlv_results = evaluate_model(
        model, mi, tokenizer, "Qwen3.5-0.8B",
        voting_layers, k_values=[10, 20], benchmark=BENCHMARK)
    all_results["Qwen3.5-0.8B"] = {
        "baseline": baseline, "mlv": mlv_results
    }
    del model, mi

    # ==========================================
    # Summary
    # ==========================================
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY — 100-Question Benchmark")
    print(f"{'='*70}\n")

    for model_name, res in all_results.items():
        print(f"  {model_name}:")
        print(f"    Baseline: {res['baseline']}/100")
        for k_label, r in res['mlv'].items():
            net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net'])
            print(f"    {k_label}: {r['correct']}/100 | +{r['fixes']}/-{r['breaks']} = {net_s}")
        print()

    # Save
    with open("mlv_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Saved mlv_benchmark_results.json")


if __name__ == "__main__":
    run()
