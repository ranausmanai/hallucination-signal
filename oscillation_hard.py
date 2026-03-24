"""
Oscillation Hypothesis — Hard Dataset
======================================
Need 30+ wrong answers to get reliable AUC.
Use harder/obscure questions where Qwen2-0.5B hallucinations more.
Also test the 2D view: oscillation × confidence jointly.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from sklearn.metrics import roc_auc_score, roc_curve
import json


def get_per_layer_top1(model, token_ids):
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    from mlx_lm.models.qwen2 import create_attention_mask
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
        last = np.array(logits[0, -1].astype(mx.float32))
        preds.append(int(np.argmax(last)))
        del logits, h_norm
    return np.array(preds)


def oscillation_features(layer_preds):
    """Extract rich oscillation features from layer predictions."""
    n = len(layer_preds)
    changes = np.sum(layer_preds[1:] != layer_preds[:-1])

    # When does last change happen?
    last_change = 0
    for i in range(n - 1, 0, -1):
        if layer_preds[i] != layer_preds[i-1]:
            last_change = i
            break

    # How many distinct tokens does the model oscillate between?
    unique_preds = len(set(layer_preds.tolist()))

    # Fraction of time spent on final token (stability measure)
    final_tok = layer_preds[-1]
    time_on_final = np.sum(layer_preds == final_tok) / n

    # Early oscillation (layers 0-11) vs late oscillation (layers 12-23)
    early_changes = np.sum(layer_preds[1:n//2] != layer_preds[:n//2-1])
    late_changes  = np.sum(layer_preds[n//2+1:] != layer_preds[n//2:-1])

    return {
        "oscillation_count": int(changes),
        "lock_in_layer": last_change,
        "unique_tokens_visited": unique_preds,
        "time_on_final_token": float(time_on_final),
        "early_changes": int(early_changes),
        "late_changes": int(late_changes),
    }


def get_confidence(model, token_ids):
    ids_mx = mx.array([token_ids])
    logits = model(ids_mx)
    mx.eval(logits)
    last = np.array(logits[0, -1].astype(mx.float32))
    probs = np.exp(last - last.max())
    probs /= probs.sum()
    del logits
    return float(probs.max()), int(np.argmax(probs))


def ask(model, tokenizer, question, max_tokens=60):
    messages = [{"role": "user", "content": f"Answer briefly and directly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=fmt, max_tokens=max_tokens, verbose=False).strip()


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

    # Mix of easy + hard questions to get ~40% wrong rate
    qa_set = [
        # Easy — model should get right
        ("What is 2 + 2?", ["4", "four"]),
        ("What color is the sky?", ["blue"]),
        ("What is the capital of France?", ["paris"]),
        ("Who wrote Hamlet?", ["shakespeare"]),
        ("What is H2O?", ["water"]),
        ("How many days are in a week?", ["7", "seven"]),
        ("What year did World War II end?", ["1945"]),
        ("What is the largest planet?", ["jupiter"]),
        ("What is the chemical symbol for iron?", ["fe"]),
        ("Who painted the Sistine Chapel?", ["michelangelo"]),
        ("What is the boiling point of water?", ["100"]),
        ("What country is the Amazon rainforest mostly in?", ["brazil"]),
        ("What is the square root of 81?", ["9", "nine"]),
        ("What ocean is between Europe and America?", ["atlantic"]),
        ("Who invented the light bulb?", ["edison"]),

        # Medium — model sometimes hallucinates
        ("What year was the Eiffel Tower built?", ["1889"]),
        ("What is the capital of Australia?", ["canberra"]),
        ("Who wrote Don Quixote?", ["cervantes"]),
        ("What is the currency of Japan?", ["yen"]),
        ("What year was Python created?", ["1991"]),
        ("What is the largest country by area?", ["russia"]),
        ("Who invented the World Wide Web?", ["berners-lee", "tim"]),
        ("What is the speed of light in m/s?", ["299792458", "3×10", "3x10"]),
        ("What year was the Berlin Wall torn down?", ["1989"]),
        ("What is the capital of Canada?", ["ottawa"]),

        # Hard — model very likely hallucinates
        ("Who won the Nobel Prize in Chemistry in 2023?", ["bawendi", "brus", "ekimov"]),
        ("What is the population of Mongolia?", ["3", "million"]),
        ("Who is the prime minister of New Zealand as of 2024?", ["luxon", "christopher"]),
        ("What is the GDP of Vietnam in 2023?", ["430", "450", "400"]),
        ("What year was the Treaty of Westphalia signed?", ["1648"]),
        ("Who wrote the novel 'The Left Hand of Darkness'?", ["le guin", "ursula"]),
        ("What is the melting point of tungsten in Celsius?", ["3422", "3400"]),
        ("What is the capital of Kyrgyzstan?", ["bishkek"]),
        ("Who was the 30th president of the United States?", ["coolidge", "calvin"]),
        ("What is the atomic weight of plutonium?", ["244", "242", "239"]),
        ("What year was the WHO founded?", ["1948"]),
        ("What is the largest desert in the world?", ["antarctica", "antarctic"]),
        ("Who invented the Bunsen burner?", ["bunsen", "robert"]),
        ("What is the speed of sound in water in m/s?", ["1480", "1500", "1498"]),
        ("What country has the most UNESCO World Heritage Sites?", ["italy", "china"]),
        ("Who was the first female prime minister of the UK?", ["thatcher", "margaret"]),
        ("What year was the Universal Declaration of Human Rights adopted?", ["1948"]),
        ("What is the chemical formula for ammonia?", ["nh3", "nh₃"]),
        ("What is the tallest building in the world as of 2024?", ["burj khalifa", "burj"]),
        ("Who composed the Four Seasons?", ["vivaldi", "antonio"]),
        ("What year did the Soviet Union collapse?", ["1991"]),
        ("What is the most spoken language in the world?", ["mandarin", "chinese"]),
        ("Who invented the printing press?", ["gutenberg", "johannes"]),
        ("What is the half-life of uranium-235?", ["703", "700 million"]),
        ("What is the capital of Myanmar?", ["naypyidaw", "naypyitaw"]),
    ]

    print(f"\nRunning {len(qa_set)} QA pairs...")
    print(f"{'Q':>55} {'Osc':>4} {'Uniq':>5} {'OnFin':>6} {'Conf':>6} {'OK':>3}")
    print("-" * 85)

    results = []
    for question, keywords in qa_set:
        messages = [{"role": "user", "content": f"Answer briefly and directly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(fmt, add_special_tokens=True)

        layer_preds = get_per_layer_top1(model, token_ids)
        feats = oscillation_features(layer_preds)
        conf, top1 = get_confidence(model, token_ids)

        answer = ask(model, tokenizer, question)
        is_correct = any(kw.lower() in answer.lower() for kw in keywords)

        results.append({
            "question": question,
            "answer": answer,
            "is_correct": is_correct,
            "confidence": conf,
            **feats,
        })

        ok = "✓" if is_correct else "✗"
        print(f"{question[:55]:>55} {feats['oscillation_count']:>4} {feats['unique_tokens_visited']:>5} "
              f"{feats['time_on_final_token']:>6.3f} {conf:>6.3f} {ok:>3}")

    # Stats
    correct = [r for r in results if r["is_correct"]]
    wrong   = [r for r in results if not r["is_correct"]]

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(correct)} correct, {len(wrong)} wrong / {len(results)} total")
    print(f"{'='*70}\n")

    if not correct or not wrong:
        print("Need both correct and wrong examples.")
        return

    labels = [1 if r["is_correct"] else 0 for r in results]

    features = {
        "oscillation_count":       ([-r["oscillation_count"]      for r in results], "higher osc = more likely wrong"),
        "confidence":              ([r["confidence"]               for r in results], "lower conf = more likely wrong"),
        "unique_tokens_visited":   ([-r["unique_tokens_visited"]   for r in results], "more unique = more likely wrong"),
        "time_on_final_token":     ([r["time_on_final_token"]      for r in results], "less time on final = more likely wrong"),
        "late_changes":            ([-r["late_changes"]            for r in results], "more late changes = more likely wrong"),
    }

    print(f"{'Feature':>30} {'AUC':>8} {'Correct mean':>14} {'Wrong mean':>12}")
    print("-" * 70)
    aucs = {}
    for fname, (scores, note) in features.items():
        try:
            auc = roc_auc_score(labels, scores)
        except:
            auc = 0.5
        aucs[fname] = auc
        raw_correct = np.mean([r[fname] for r in correct])
        raw_wrong   = np.mean([r[fname] for r in wrong])
        star = "★" if auc > 0.7 else ("~" if auc > 0.6 else " ")
        print(f"{fname:>30} {auc:>8.3f} {star}  {raw_correct:>12.3f} {raw_wrong:>12.3f}")

    # Combined: oscillation + confidence (complementary signals)
    osc_norm  = np.array(features["oscillation_count"][0])
    conf_norm = np.array(features["confidence"][0])
    osc_norm  = (osc_norm - osc_norm.min()) / (osc_norm.max() - osc_norm.min() + 1e-10)
    conf_norm = (conf_norm - conf_norm.min()) / (conf_norm.max() - conf_norm.min() + 1e-10)
    combined = (osc_norm + conf_norm).tolist()
    try:
        auc_combined = roc_auc_score(labels, combined)
    except:
        auc_combined = 0.5
    print(f"{'osc + confidence (combined)':>30} {auc_combined:>8.3f} {'★' if auc_combined > 0.7 else '~'}")

    # Best examples
    print(f"\nTop hallucinations by oscillation:")
    for r in sorted(wrong, key=lambda x: -x["oscillation_count"])[:5]:
        print(f"  osc={r['oscillation_count']} conf={r['confidence']:.2f} Q:'{r['question'][:45]}' → '{r['answer'][:55]}'")

    print(f"\nMost stable correct answers:")
    for r in sorted(correct, key=lambda x: x["oscillation_count"])[:5]:
        print(f"  osc={r['oscillation_count']} conf={r['confidence']:.2f} Q:'{r['question'][:45]}' → '{r['answer'][:55]}'")

    # Threshold analysis
    print(f"\nThreshold analysis (oscillation ≥ threshold → predict 'wrong'):")
    osc_vals = [r["oscillation_count"] for r in results]
    for thresh in range(min(osc_vals), max(osc_vals)+1):
        predicted_wrong = [1 if r["oscillation_count"] >= thresh else 0 for r in results]
        tp = sum(1 for p, r in zip(predicted_wrong, results) if p == 1 and not r["is_correct"])
        fp = sum(1 for p, r in zip(predicted_wrong, results) if p == 1 and r["is_correct"])
        fn = sum(1 for p, r in zip(predicted_wrong, results) if p == 0 and not r["is_correct"])
        tn = sum(1 for p, r in zip(predicted_wrong, results) if p == 0 and r["is_correct"])
        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        print(f"  osc ≥ {thresh:2d}: precision={prec:.2f} recall={rec:.2f} F1={f1:.2f} (flags {sum(predicted_wrong)}/{len(results)})")

    with open("oscillation_hard_results.json", "w") as f:
        json.dump({
            "n_correct": len(correct), "n_wrong": len(wrong),
            "aucs": {k: float(v) for k, v in aucs.items()},
            "auc_combined": float(auc_combined),
            "results": results,
        }, f, indent=2)
    print("\nSaved oscillation_hard_results.json")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    best_single = max(aucs.values())
    if auc_combined > 0.75:
        print(f"\n★ STRONG SIGNAL — Combined AUC={auc_combined:.3f}")
        print(f"  Oscillation captures something confidence alone misses.")
        print(f"  This is a real pre-generation hallucination signal.")
    elif best_single > 0.70 or auc_combined > 0.70:
        print(f"\n~ REAL SIGNAL — needs more data to confirm")
        print(f"  Best single AUC={best_single:.3f}, Combined={auc_combined:.3f}")
    else:
        print(f"\n✗ SIGNAL TOO WEAK on this dataset")
        print(f"  Best AUC={best_single:.3f} — could be noise with n={len(wrong)} wrong examples")


if __name__ == "__main__":
    run()
