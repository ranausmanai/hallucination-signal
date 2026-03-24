"""
Layer-Wise Prediction Stability as a Hallucination Signal
==========================================================
Hypothesis:
  When an LLM is about to give a CORRECT answer, the predicted next token
  LOCKS IN early across layers and stays stable.

  When it's about to HALLUCINATE, the predicted token OSCILLATES across
  layers — the model's internal representation never converges.

  We call this the "oscillation hypothesis."

Mechanism:
  At each transformer layer boundary, apply norm + lm_head → get a prediction.
  Track how many times the argmax token CHANGES from layer to layer.
  More changes = more oscillation = model is uncertain = likely wrong.

Experiment:
  1. Create factual QA dataset with KNOWN correct answers.
  2. Run Qwen2-0.5B, capture per-layer top-1 prediction.
  3. Label each answer: CORRECT or HALLUCINATION.
  4. Compare oscillation count between correct and hallucinated answers.
  5. Compute AUC: can oscillation alone predict errors?

If AUC > 0.7 → oscillation is a real signal.
If AUC > 0.85 → this is a deployable hallucination detector.

Why this is novel:
  - Prior hallucination detection uses: output confidence, multiple samples,
    external verifiers, or probing classifiers trained on labeled data.
  - This uses ZERO extra labels, ZERO extra model, ZERO extra samples.
  - It's a single forward pass, reading internal dynamics.
  - It works at token-generation time (before the token is output).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from sklearn.metrics import roc_auc_score
import json


def get_per_layer_predictions(model, token_ids):
    """
    Run forward pass, capture top-1 prediction at every layer boundary.
    Returns: (n_layers,) array of predicted token IDs (what the model would
    output at the LAST position if it stopped at each layer).
    """
    ids_mx = mx.array([token_ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)

    n_layers = len(model.model.layers)
    from mlx_lm.models.qwen2 import create_attention_mask
    mask = create_attention_mask(h, None)

    layer_predictions = []

    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)

        # Apply norm + head to get logits at last position
        h_norm = model.model.norm(h)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(h_norm)
        else:
            logits = model.lm_head(h_norm)
        mx.eval(logits)

        # Top-1 prediction at the last token position
        last_pos_logits = np.array(logits[0, -1].astype(mx.float32))
        top1 = int(np.argmax(last_pos_logits))
        layer_predictions.append(top1)

        del logits, h_norm

    return np.array(layer_predictions)  # (n_layers,)


def compute_oscillation(layer_predictions):
    """
    Count how many times the top-1 prediction changes across layers.
    Returns: int (oscillation count), also return stability profile.
    """
    changes = np.sum(layer_predictions[1:] != layer_predictions[:-1])
    # Also: how early does it lock in?
    final_token = layer_predictions[-1]
    # Find the last layer where a change occurred
    last_change = 0
    for i in range(len(layer_predictions) - 1, 0, -1):
        if layer_predictions[i] != layer_predictions[i-1]:
            last_change = i
            break
    lock_in_layer = last_change  # prediction is stable from this layer onwards

    return int(changes), lock_in_layer


def run_factual_qa(model, tokenizer, question, expected_answer_contains):
    """
    Ask a factual question, get the model's answer.
    Returns: (generated_text, is_correct)
    """
    from mlx_lm import generate
    messages = [{"role": "user", "content": f"Answer in one sentence: {question}"}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(model, tokenizer, prompt=formatted,
                        max_tokens=50, verbose=False)

    # Check if the answer contains the expected content
    is_correct = any(
        kw.lower() in response.lower()
        for kw in expected_answer_contains
    )
    return response.strip(), is_correct


def get_generation_oscillation(model, tokenizer, question):
    """
    Get oscillation signal for the FIRST generated token after the prompt.
    This is the token the model is about to output — we measure how stable
    that prediction is across layers BEFORE outputting it.
    """
    messages = [{"role": "user", "content": f"Answer in one sentence: {question}"}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.encode(formatted, add_special_tokens=True)

    layer_preds = get_per_layer_predictions(model, token_ids)
    osc, lock_in = compute_oscillation(layer_preds)

    # Also get confidence at final layer
    ids_mx = mx.array([token_ids])
    logits = model(ids_mx)
    mx.eval(logits)
    last_logits = np.array(logits[0, -1].astype(mx.float32))
    probs = np.exp(last_logits - last_logits.max())
    probs /= probs.sum()
    confidence = float(probs.max())
    top1_token = int(np.argmax(last_logits))

    del logits

    return {
        "layer_predictions": layer_preds.tolist(),
        "oscillation_count": osc,
        "lock_in_layer": lock_in,
        "final_confidence": confidence,
        "top1_token": top1_token,
        "top1_decoded": tokenizer.decode([top1_token]),
    }


def run_experiment():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    n_layers = len(model.model.layers)
    print(f"Model: {n_layers} layers\n")

    # Factual QA dataset with verifiable answers
    # Each: (question, [keywords that must appear in correct answer])
    factual_qa = [
        # High-confidence facts (model likely knows these)
        ("What is the capital city of France?", ["paris"]),
        ("What year did World War II end?", ["1945"]),
        ("Who wrote Romeo and Juliet?", ["shakespeare"]),
        ("What is the chemical symbol for gold?", ["au"]),
        ("How many sides does a hexagon have?", ["six", "6"]),
        ("What planet is closest to the Sun?", ["mercury"]),
        ("What is the boiling point of water in Celsius?", ["100"]),
        ("Who painted the Mona Lisa?", ["leonardo", "da vinci", "vinci"]),
        ("What is the largest ocean on Earth?", ["pacific"]),
        ("In what year did man first land on the Moon?", ["1969"]),
        ("What is the speed of light in km/s?", ["299", "300"]),
        ("Who invented the telephone?", ["bell", "graham"]),
        ("What is the largest planet in our solar system?", ["jupiter"]),
        ("What language is spoken in Brazil?", ["portuguese"]),
        ("What is the square root of 144?", ["12"]),

        # Harder facts (model may hallucinate)
        ("Who won the Nobel Prize in Physics in 2022?", ["aspect", "clauser", "zeilinger"]),
        ("What year was the Eiffel Tower completed?", ["1889"]),
        ("What is the capital of Kazakhstan?", ["astana", "nur-sultan"]),
        ("Who wrote the novel 1984?", ["orwell", "george"]),
        ("What is the atomic number of carbon?", ["6"]),
        ("What year was Python programming language first released?", ["1991"]),
        ("What is the half-life of Carbon-14?", ["5730", "5700"]),
        ("Who is the CEO of NVIDIA as of 2024?", ["jensen", "huang"]),
        ("What is the deepest lake in the world?", ["baikal"]),
        ("What is the formula for water?", ["h2o", "h₂o"]),
        ("What country has the most natural lakes?", ["canada"]),
        ("What year was the first iPhone released?", ["2007"]),
        ("What is the currency of Japan?", ["yen"]),
        ("Who developed the theory of general relativity?", ["einstein"]),
        ("What is the tallest mountain on Earth?", ["everest"]),

        # Tricky/obscure (model more likely to hallucinate)
        ("What is the capital of Burkina Faso?", ["ouagadougou"]),
        ("Who invented the World Wide Web?", ["berners-lee", "tim"]),
        ("What year was the Magna Carta signed?", ["1215"]),
        ("What is the rarest blood type?", ["ab-", "ab negative"]),
        ("What is the smallest country in the world by area?", ["vatican"]),
        ("What programming language was Linux written in?", ["c"]),
        ("Who was the first woman to win a Nobel Prize?", ["curie", "marie"]),
        ("What is the chemical formula for table salt?", ["nacl"]),
        ("What year was Google founded?", ["1998"]),
        ("What is the longest river in Africa?", ["nile"]),
    ]

    print(f"{'='*70}")
    print(f"Oscillation Hypothesis Test")
    print(f"{'='*70}")
    print(f"\nRunning {len(factual_qa)} factual QA pairs...")
    print(f"{'Question':>50} {'Osc':>5} {'Lock':>5} {'Conf':>6} {'Correct':>8}")
    print("-" * 80)

    all_results = []

    for question, expected in factual_qa:
        # Get oscillation signal (from prompt)
        sig = get_generation_oscillation(model, tokenizer, question)

        # Get actual answer and check correctness
        answer, is_correct = run_factual_qa(model, tokenizer, question, expected)

        result = {
            "question": question,
            "answer": answer,
            "is_correct": is_correct,
            "oscillation": sig["oscillation_count"],
            "lock_in_layer": sig["lock_in_layer"],
            "confidence": sig["final_confidence"],
            "first_token": sig["top1_decoded"],
        }
        all_results.append(result)

        status = "✓" if is_correct else "✗"
        print(f"{question[:50]:>50} {sig['oscillation_count']:>5} {sig['lock_in_layer']:>5} {sig['final_confidence']:>6.3f}  {status}")

    # Analysis
    correct = [r for r in all_results if r["is_correct"]]
    wrong = [r for r in all_results if not r["is_correct"]]

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}\n")
    print(f"Total: {len(all_results)} questions")
    print(f"Correct: {len(correct)} ({len(correct)/len(all_results)*100:.1f}%)")
    print(f"Wrong/Hallucinated: {len(wrong)} ({len(wrong)/len(all_results)*100:.1f}%)\n")

    if not correct or not wrong:
        print("Need both correct and wrong answers to compare. Adjust dataset.")
        return

    # Oscillation comparison
    osc_correct = [r["oscillation"] for r in correct]
    osc_wrong   = [r["oscillation"] for r in wrong]
    conf_correct = [r["confidence"] for r in correct]
    conf_wrong   = [r["confidence"] for r in wrong]
    lock_correct = [r["lock_in_layer"] for r in correct]
    lock_wrong   = [r["lock_in_layer"] for r in wrong]

    print(f"{'Metric':>30} {'Correct':>10} {'Wrong':>10} {'Gap':>8}")
    print("-" * 65)
    print(f"{'Mean oscillation count':>30} {np.mean(osc_correct):>10.2f} {np.mean(osc_wrong):>10.2f} {np.mean(osc_wrong)-np.mean(osc_correct):>+8.2f}")
    print(f"{'Mean lock-in layer':>30} {np.mean(lock_correct):>10.2f} {np.mean(lock_wrong):>10.2f} {np.mean(lock_wrong)-np.mean(lock_correct):>+8.2f}")
    print(f"{'Mean output confidence':>30} {np.mean(conf_correct):>10.3f} {np.mean(conf_wrong):>10.3f} {np.mean(conf_wrong)-np.mean(conf_correct):>+8.3f}")

    # AUC for each signal
    labels = [1 if r["is_correct"] else 0 for r in all_results]

    # For oscillation: higher = more likely wrong, so flip sign
    osc_scores = [-r["oscillation"] for r in all_results]  # negate: higher osc = lower score
    conf_scores = [r["confidence"] for r in all_results]
    lock_scores = [-r["lock_in_layer"] for r in all_results]  # earlier lock = better

    try:
        auc_osc  = roc_auc_score(labels, osc_scores)
        auc_conf = roc_auc_score(labels, conf_scores)
        auc_lock = roc_auc_score(labels, lock_scores)

        print(f"\n{'AUC (ability to predict correctness)':}")
        print(f"  Oscillation count: {auc_osc:.3f}  {'★ SIGNAL FOUND!' if auc_osc > 0.7 else ('~ weak' if auc_osc > 0.6 else '✗ no signal')}")
        print(f"  Output confidence: {auc_conf:.3f}  {'★ SIGNAL FOUND!' if auc_conf > 0.7 else ('~ weak' if auc_conf > 0.6 else '✗ no signal')}")
        print(f"  Lock-in layer:     {auc_lock:.3f}  {'★ SIGNAL FOUND!' if auc_lock > 0.7 else ('~ weak' if auc_lock > 0.6 else '✗ no signal')}")

        # Combined signal
        combined = [a + b for a, b in zip(
            [(s - min(osc_scores)) / (max(osc_scores) - min(osc_scores) + 1e-10) for s in osc_scores],
            [(s - min(conf_scores)) / (max(conf_scores) - min(conf_scores) + 1e-10) for s in conf_scores]
        )]
        auc_combined = roc_auc_score(labels, combined)
        print(f"  Combined (osc+conf): {auc_combined:.3f}  {'★ STRONG SIGNAL!' if auc_combined > 0.75 else ''}")

    except Exception as e:
        print(f"AUC computation failed: {e}")

    # Examples — highest oscillation wrong answers
    print(f"\nHighest oscillation WRONG answers (hallucinations):")
    worst = sorted(wrong, key=lambda r: -r["oscillation"])[:5]
    for r in worst:
        print(f"  osc={r['oscillation']} Q: '{r['question'][:50]}' → '{r['answer'][:60]}'")

    print(f"\nLowest oscillation CORRECT answers (confident truths):")
    best = sorted(correct, key=lambda r: r["oscillation"])[:5]
    for r in best:
        print(f"  osc={r['oscillation']} Q: '{r['question'][:50]}' → '{r['answer'][:60]}'")

    # Save
    with open("oscillation_results.json", "w") as f:
        json.dump({
            "n_total": len(all_results),
            "n_correct": len(correct),
            "n_wrong": len(wrong),
            "results": all_results,
            "stats": {
                "mean_osc_correct": float(np.mean(osc_correct)),
                "mean_osc_wrong": float(np.mean(osc_wrong)),
                "mean_conf_correct": float(np.mean(conf_correct)),
                "mean_conf_wrong": float(np.mean(conf_wrong)),
                "auc_oscillation": float(auc_osc) if 'auc_osc' in dir() else None,
                "auc_confidence": float(auc_conf) if 'auc_conf' in dir() else None,
            }
        }, f, indent=2)
    print("\nSaved oscillation_results.json")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    if 'auc_osc' in locals() and auc_osc > 0.7:
        print(f"""
✓ OSCILLATION IS A REAL SIGNAL

The model's internal representation oscillates more when it's wrong.
This means: BEFORE generating a single token, we can detect likely errors.

What this enables:
  1. Single-pass hallucination detector (no second model needed)
  2. Token-level uncertainty signal for any LLM
  3. Automatic "I don't know" triggering when oscillation is high
  4. Selective retrieval augmentation (only retrieve when oscillation > threshold)

This is a new observable: the path through representation space matters,
not just the final state.
""")
    else:
        print(f"""
Oscillation alone isn't sufficient. But the gap between correct/wrong
answers may still be real. Next steps:
  - Test with more questions
  - Look at oscillation patterns across ALL generated tokens (not just first)
  - Try probing the residual stream directly (not just top-1 token changes)
""")


if __name__ == "__main__":
    run_experiment()
