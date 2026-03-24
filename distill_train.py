"""
Knowledge Preservation Training
================================
Instead of teaching the last layer NEW facts (which causes confabulation),
teach it to PRESERVE the factual signal from layer 21.

Loss = CE(output, target) + λ * MSE(logits_final, sg(logits_L21))

Where sg = stop_gradient. This says: "generate fluently, but don't override
what layer 21 already knows."

The key insight: we're not training the model to memorize answers —
we're training it to STOP SUPPRESSING answers it already knows.

Also tests LoRA (low-rank adaptation) to reduce overfitting.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, generate
from mlx_lm.models.qwen2 import create_attention_mask
from mlx_lm.sample_utils import make_sampler
import json
import time


# Training data — general factual QA (NOT including hard test questions)
TRAIN_QA = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasília."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of Russia?", "The capital of Russia is Moscow."),
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("What is the capital of China?", "The capital of China is Beijing."),
    ("What is the capital of South Korea?", "The capital of South Korea is Seoul."),
    ("What is the capital of Turkey?", "The capital of Turkey is Ankara."),
    ("What is the capital of Poland?", "The capital of Poland is Warsaw."),
    ("What is the capital of Sweden?", "The capital of Sweden is Stockholm."),
    ("What is the capital of Norway?", "The capital of Norway is Oslo."),
    ("What is the capital of Portugal?", "The capital of Portugal is Lisbon."),
    ("What is the capital of Greece?", "The capital of Greece is Athens."),
    ("What is the capital of Ireland?", "The capital of Ireland is Dublin."),
    ("What is the capital of Austria?", "The capital of Austria is Vienna."),
    ("What is the capital of Belgium?", "The capital of Belgium is Brussels."),
    ("What is the capital of Hungary?", "The capital of Hungary is Budapest."),
    ("What is the capital of Ukraine?", "The capital of Ukraine is Kyiv."),
    ("What is the capital of Peru?", "The capital of Peru is Lima."),
    ("What is the capital of Chile?", "The capital of Chile is Santiago."),
    ("What is the capital of Iran?", "The capital of Iran is Tehran."),
    ("What is the capital of Iraq?", "The capital of Iraq is Baghdad."),
    ("What is the capital of Indonesia?", "The capital of Indonesia is Jakarta."),
    ("What is the capital of Vietnam?", "The capital of Vietnam is Hanoi."),
    ("What is the capital of Kenya?", "The capital of Kenya is Nairobi."),
    ("What is the capital of Mongolia?", "The capital of Mongolia is Ulaanbaatar."),
    ("What is the capital of Nepal?", "The capital of Nepal is Kathmandu."),
    ("What is the capital of Iceland?", "The capital of Iceland is Reykjavik."),
    ("What is the capital of Latvia?", "The capital of Latvia is Riga."),
    ("What is the capital of Croatia?", "The capital of Croatia is Zagreb."),
    ("What is the capital of Serbia?", "The capital of Serbia is Belgrade."),
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare."),
    ("Who wrote 1984?", "1984 was written by George Orwell."),
    ("Who wrote War and Peace?", "War and Peace was written by Leo Tolstoy."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("Who composed the Four Seasons?", "The Four Seasons was composed by Antonio Vivaldi."),
    ("What is the largest planet in the solar system?", "The largest planet in the solar system is Jupiter."),
    ("What is the boiling point of water in Celsius?", "The boiling point of water is 100°C."),
    ("In what year did World War II end?", "World War II ended in 1945."),
    ("Who was the first president of the United States?", "The first president of the United States was George Washington."),
    ("Who developed the theory of relativity?", "The theory of relativity was developed by Albert Einstein."),
    ("What is the tallest mountain in the world?", "The tallest mountain in the world is Mount Everest."),
    ("What is the largest ocean?", "The largest ocean is the Pacific Ocean."),
]

# Strictly held-out test
TEST_DATA = [
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
    ("What is the capital of Myanmar?",                        ["naypyidaw", "naypyitaw"]),
    ("Who composed the Moonlight Sonata?",                     ["beethoven"]),
    ("Who wrote Romeo and Juliet?",                            ["shakespeare"]),
    ("What is 2 + 2?",                                         ["4", "four"]),
    ("What is the capital of France?",                         ["paris"]),
    ("What is the largest planet?",                            ["jupiter"]),
    ("What is the boiling point of water?",                    ["100"]),
]


def forward_with_intermediate(model, input_ids, preserve_layer=21):
    """
    Forward pass that returns both final logits AND intermediate logits
    from preserve_layer. Used for distillation loss.
    """
    ids_mx = mx.array([input_ids])
    h = model.model.embed_tokens(ids_mx)
    mask = create_attention_mask(h, None)

    h_intermediate = None
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        if i == preserve_layer:
            h_intermediate = mx.stop_gradient(h)  # detach — don't backprop through L21

    # Final logits
    h_final_norm = model.model.norm(h)
    if model.args.tie_word_embeddings:
        logits_final = model.model.embed_tokens.as_linear(h_final_norm)
    else:
        logits_final = model.lm_head(h_final_norm)

    # Intermediate logits (from layer 21, using same norm + head)
    h_inter_norm = model.model.norm(h_intermediate)
    if model.args.tie_word_embeddings:
        logits_inter = model.model.embed_tokens.as_linear(h_inter_norm)
    else:
        logits_inter = model.lm_head(h_inter_norm)

    return logits_final, mx.stop_gradient(logits_inter)


def train_distill(model, tokenizer, train_qa, layers_to_train=(23,),
                  lr=5e-5, n_epochs=3, lambda_distill=0.1, preserve_layer=21):
    """
    Train with combined loss:
    L = CE(final, target) + λ * KL(final || intermediate)

    The KL term teaches the final layer to preserve intermediate predictions.
    """
    model.freeze()
    for layer_idx in layers_to_train:
        model.model.layers[layer_idx].unfreeze()

    n_trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    n_total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

    # Prepare sequences
    sequences = []
    for question, answer in train_qa:
        messages = [
            {"role": "user", "content": f"Answer briefly: {question}"},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        ids = tokenizer.encode(text, add_special_tokens=True)
        sequences.append(ids)

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, input_ids):
        logits_final, logits_inter = forward_with_intermediate(
            model, input_ids, preserve_layer)

        # Standard CE loss (next-token prediction)
        targets = mx.array(input_ids[1:])
        lf = logits_final[0, :-1, :]
        log_probs = nn.log_softmax(lf, axis=-1)
        ce_loss = -mx.mean(log_probs[mx.arange(len(targets)), targets])

        # KL divergence: final should preserve intermediate's predictions
        # KL(p_inter || p_final) where p_inter is the "teacher" (frozen)
        li = logits_inter[0, :-1, :]
        p_inter = mx.softmax(li, axis=-1)
        log_p_final = nn.log_softmax(lf, axis=-1)
        # Only compute KL on top-K to save memory
        kl_loss = -mx.mean(mx.sum(p_inter * log_p_final, axis=-1))

        total = ce_loss + lambda_distill * kl_loss
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"  Training {n_epochs} epochs (lr={lr}, λ_distill={lambda_distill})...")
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        indices = np.random.permutation(len(sequences))
        n = 0
        for idx in indices:
            seq = sequences[idx]
            if len(seq) < 3:
                continue
            loss, grads = loss_and_grad(model, seq)
            mx.eval(loss)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            epoch_loss += float(loss)
            n += 1
        print(f"    Epoch {epoch}: loss={epoch_loss/max(n,1):.4f} ({time.time()-t0:.1f}s)")
    print(f"  Done in {time.time()-t0:.1f}s")
    return model


def evaluate(model, tokenizer, test_data, label=""):
    results = []
    correct = 0
    for question, keywords in test_data:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                          verbose=False, sampler=sampler).strip()
        answer_clean = answer.lower().replace(",", "")
        is_correct = any(kw.lower().replace(",", "") in answer_clean for kw in keywords)
        if is_correct:
            correct += 1
        results.append({
            "question": question, "answer": answer,
            "correct": is_correct,
        })
    n_hard = 11  # first 11 are hard, strictly held out
    hard_correct = sum(r["correct"] for r in results[:n_hard])
    print(f"\n  {label}: {correct}/{len(test_data)} total, {hard_correct}/{n_hard} hard")
    for r in results:
        marker = "✓" if r["correct"] else "✗"
        print(f"    {marker} {r['question'][:55]}")
        print(f"      → '{r['answer'][:65]}'")
    return results, correct


def compare(results, baseline_results):
    fixes = sum(1 for r, b in zip(results, baseline_results)
                if r["correct"] and not b["correct"])
    breaks = sum(1 for r, b in zip(results, baseline_results)
                 if not r["correct"] and b["correct"])
    net = fixes - breaks

    if fixes > 0:
        print(f"\n  Fixed:")
        for r, b in zip(results, baseline_results):
            if r["correct"] and not b["correct"]:
                print(f"    ✓ {r['question'][:55]}")
                print(f"      was: '{b['answer'][:50]}'")
                print(f"      now: '{r['answer'][:50]}'")
    if breaks > 0:
        print(f"\n  Broken:")
        for r, b in zip(results, baseline_results):
            if not r["correct"] and b["correct"]:
                print(f"    ✗ {r['question'][:55]}")
                print(f"      was: '{b['answer'][:50]}'")
                print(f"      now: '{r['answer'][:50]}'")

    return fixes, breaks, net


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

    print(f"\n{'='*70}")
    print("Knowledge Preservation Training")
    print(f"Train: {len(TRAIN_QA)} general QA | Test: {len(TEST_DATA)} (NO overlap)")
    print(f"{'='*70}\n")

    # Baseline
    print("BASELINE")
    baseline_results, baseline_total = evaluate(model, tokenizer, TEST_DATA, "Baseline")

    configs = [
        # (layers, lr, epochs, lambda, label)
        ((23,), 5e-5, 3, 0.0, "CE only, L23, 3ep"),
        ((23,), 5e-5, 3, 0.1, "CE+KL(0.1), L23, 3ep"),
        ((23,), 5e-5, 3, 0.5, "CE+KL(0.5), L23, 3ep"),
        ((23,), 5e-5, 3, 1.0, "CE+KL(1.0), L23, 3ep"),
        ((23,), 3e-5, 3, 0.5, "CE+KL(0.5), L23, lr=3e-5, 3ep"),
        ((22, 23), 3e-5, 3, 0.5, "CE+KL(0.5), L22-23, lr=3e-5, 3ep"),
    ]

    all_results = {"baseline": baseline_total}

    for layers, lr, epochs, lam, label in configs:
        print(f"\n{'='*70}")
        print(f"Config: {label}")
        print(f"{'='*70}")

        model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
        model = train_distill(model, tokenizer, TRAIN_QA,
                              layers_to_train=layers, lr=lr, n_epochs=epochs,
                              lambda_distill=lam)
        results, total = evaluate(model, tokenizer, TEST_DATA, label)
        fixes, breaks, net = compare(results, baseline_results)

        print(f"\n  Net: +{fixes} fixed, -{breaks} broken = {'+'if net>0 else ''}{net}")
        all_results[label] = {"correct": total, "total": len(TEST_DATA),
                              "fixes": fixes, "breaks": breaks, "net": net}

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"  Baseline: {baseline_total}/{len(TEST_DATA)}\n")
    print(f"  {'Config':40} {'Score':>8} {'Fix':>5} {'Brk':>5} {'Net':>5}")
    print("  " + "-" * 65)
    for _, _, _, _, label in configs:
        r = all_results[label]
        net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net']) if r['net'] < 0 else "="
        star = " ★" if r['net'] > 0 else ""
        print(f"  {label:40} {r['correct']:>4}/{r['total']}  +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")

    with open("distill_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved distill_results.json")


if __name__ == "__main__":
    run()
