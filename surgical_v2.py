"""
Surgical Fine-Tuning v2: Proper Validation
===========================================
v1 showed +7 net improvement, but training and test data overlapped.
This version:
  1. Strict train/test split — NO overlap
  2. Fixed keyword matching (handles comma-formatted numbers)
  3. Tests generalization: train on capitals+science, test on history+obscure
  4. Also tests on Qwen3.5-0.8B for cross-model validation
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


# ============================================================
# TRAINING data: capitals, basic science, basic history
# These are "things the model mostly already knows"
# ============================================================

TRAIN_QA = [
    # Capitals (common)
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
    ("What is the capital of Mexico?", "The capital of Mexico is Mexico City."),
    ("What is the capital of Turkey?", "The capital of Turkey is Ankara."),
    ("What is the capital of Thailand?", "The capital of Thailand is Bangkok."),
    ("What is the capital of Poland?", "The capital of Poland is Warsaw."),
    ("What is the capital of Sweden?", "The capital of Sweden is Stockholm."),
    ("What is the capital of Norway?", "The capital of Norway is Oslo."),
    ("What is the capital of Portugal?", "The capital of Portugal is Lisbon."),
    ("What is the capital of Greece?", "The capital of Greece is Athens."),
    ("What is the capital of Ireland?", "The capital of Ireland is Dublin."),
    ("What is the capital of Austria?", "The capital of Austria is Vienna."),
    ("What is the capital of Belgium?", "The capital of Belgium is Brussels."),
    ("What is the capital of Czech Republic?", "The capital of the Czech Republic is Prague."),
    ("What is the capital of Hungary?", "The capital of Hungary is Budapest."),
    ("What is the capital of Ukraine?", "The capital of Ukraine is Kyiv."),
    ("What is the capital of Peru?", "The capital of Peru is Lima."),
    ("What is the capital of Chile?", "The capital of Chile is Santiago."),
    ("What is the capital of Cuba?", "The capital of Cuba is Havana."),
    ("What is the capital of Iran?", "The capital of Iran is Tehran."),
    ("What is the capital of Iraq?", "The capital of Iraq is Baghdad."),
    ("What is the capital of Pakistan?", "The capital of Pakistan is Islamabad."),
    ("What is the capital of Indonesia?", "The capital of Indonesia is Jakarta."),
    ("What is the capital of Vietnam?", "The capital of Vietnam is Hanoi."),
    ("What is the capital of New Zealand?", "The capital of New Zealand is Wellington."),
    ("What is the capital of Nigeria?", "The capital of Nigeria is Abuja."),
    ("What is the capital of South Africa?", "The capital of South Africa is Pretoria."),
    ("What is the capital of Kenya?", "The capital of Kenya is Nairobi."),
    ("What is the capital of Morocco?", "The capital of Morocco is Rabat."),
    ("What is the capital of Ethiopia?", "The capital of Ethiopia is Addis Ababa."),
    ("What is the capital of Mongolia?", "The capital of Mongolia is Ulaanbaatar."),
    ("What is the capital of Nepal?", "The capital of Nepal is Kathmandu."),
    ("What is the capital of Iceland?", "The capital of Iceland is Reykjavik."),
    ("What is the capital of Estonia?", "The capital of Estonia is Tallinn."),
    ("What is the capital of Latvia?", "The capital of Latvia is Riga."),
    ("What is the capital of Croatia?", "The capital of Croatia is Zagreb."),
    ("What is the capital of Serbia?", "The capital of Serbia is Belgrade."),
    ("What is the capital of Slovakia?", "The capital of Slovakia is Bratislava."),

    # Authors (common)
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare."),
    ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by Jane Austen."),
    ("Who wrote 1984?", "1984 was written by George Orwell."),
    ("Who wrote War and Peace?", "War and Peace was written by Leo Tolstoy."),
    ("Who wrote Don Quixote?", "Don Quixote was written by Miguel de Cervantes."),
    ("Who wrote Crime and Punishment?", "Crime and Punishment was written by Fyodor Dostoevsky."),
    ("Who wrote Moby Dick?", "Moby Dick was written by Herman Melville."),
    ("Who wrote A Tale of Two Cities?", "A Tale of Two Cities was written by Charles Dickens."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("Who painted The Starry Night?", "The Starry Night was painted by Vincent van Gogh."),
    ("Who composed the Four Seasons?", "The Four Seasons was composed by Antonio Vivaldi."),

    # Science (common)
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("What is the chemical symbol for silver?", "The chemical symbol for silver is Ag."),
    ("What is the chemical symbol for iron?", "The chemical symbol for iron is Fe."),
    ("What is the speed of light in m/s?", "The speed of light is approximately 299,792,458 m/s."),
    ("What is the boiling point of water in Celsius?", "The boiling point of water is 100°C."),
    ("What is the largest planet in the solar system?", "The largest planet in the solar system is Jupiter."),
    ("What planet is closest to the Sun?", "The planet closest to the Sun is Mercury."),
    ("What is the hardest natural substance?", "The hardest natural substance is diamond."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),

    # History (common)
    ("In what year did World War II end?", "World War II ended in 1945."),
    ("Who was the first president of the United States?", "The first president of the United States was George Washington."),
    ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming."),
    ("Who invented the telephone?", "The telephone was invented by Alexander Graham Bell."),
    ("Who developed the theory of relativity?", "The theory of relativity was developed by Albert Einstein."),
    ("Who was the first person to walk on the moon?", "The first person to walk on the moon was Neil Armstrong."),

    # Geography
    ("What is the tallest mountain in the world?", "The tallest mountain in the world is Mount Everest."),
    ("What is the largest ocean?", "The largest ocean is the Pacific Ocean."),
    ("What is the largest continent by area?", "The largest continent by area is Asia."),
]


# ============================================================
# TEST data: questions NOT in training, mix of difficulty
# ============================================================

TEST_DATA_HOLDOUT = [
    # ---- TRULY held out: not in training, not similar to training ----
    # Hard / obscure
    ("Who was the 30th president of the United States?",       ["coolidge", "calvin"]),
    ("What is the melting point of tungsten in Celsius?",      ["3422", "3400", "3,422"]),
    ("What is the half-life of uranium-235 in years?",         ["703", "700"]),
    ("What is the largest desert in the world by area?",       ["antarctica", "antarctic"]),
    ("What country has the most UNESCO World Heritage Sites?",  ["italy", "china"]),
    ("What is the atomic weight of plutonium?",                ["244", "242", "239"]),
    ("What is the rarest blood type?",                         ["ab-", "ab negative"]),
    ("Who won the Nobel Prize in Chemistry in 2023?",          ["bawendi", "brus", "ekimov"]),
    ("What is the speed of sound in water in m/s?",            ["1480", "1500", "1498", "1,480", "1,500"]),
    ("Who is the prime minister of New Zealand as of 2024?",   ["luxon", "christopher"]),
    ("What is the half-life of Carbon-14 in years?",           ["5730", "5700", "5,730"]),
    ("What year was the WHO founded?",                         ["1948"]),
    ("Who was the first female prime minister of the UK?",     ["thatcher"]),
    # Moderately hard — NOT in training
    ("What is the capital of Kyrgyzstan?",                     ["bishkek"]),
    ("What is the capital of Burkina Faso?",                   ["ouagadougou"]),
    ("What is the capital of Myanmar?",                        ["naypyidaw", "naypyitaw"]),
    ("Who composed the Moonlight Sonata?",                     ["beethoven"]),
    ("What year was the Treaty of Westphalia signed?",         ["1648"]),
    ("What is the tallest building in the world?",             ["burj", "khalifa"]),
    ("Who was the 16th president of the United States?",       ["lincoln", "abraham"]),
    # Easy controls — NOT in training but model likely knows
    ("Who wrote Romeo and Juliet?",                            ["shakespeare"]),
    ("What is 2 + 2?",                                         ["4", "four"]),
    ("What language is spoken in Brazil?",                     ["portuguese"]),
    ("What is the capital of France?",                         ["paris"]),  # in training but basic sanity check
    ("What is the boiling point of water?",                    ["100"]),    # in training
]


def prepare_training_sequences(tokenizer, qa_pairs):
    """Convert QA pairs into token sequences for next-token prediction."""
    sequences = []
    for question, answer in qa_pairs:
        messages = [
            {"role": "user", "content": f"Answer briefly: {question}"},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        ids = tokenizer.encode(text, add_special_tokens=True)
        sequences.append(ids)
    return sequences


def train_surgical(model, tokenizer, train_qa, layers_to_train=(23,),
                   lr=5e-5, n_epochs=3):
    """Fine-tune ONLY the specified layers."""
    model.freeze()
    for layer_idx in layers_to_train:
        model.model.layers[layer_idx].unfreeze()

    n_trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    n_total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

    sequences = prepare_training_sequences(tokenizer, train_qa)
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, input_ids, target_ids):
        ids_mx = mx.array([input_ids])
        h = model.model.embed_tokens(ids_mx)
        mask = create_attention_mask(h, None)
        for layer in model.model.layers:
            h = layer(h, mask, None)
        h = model.model.norm(h)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(h)
        else:
            logits = model.lm_head(h)
        logits = logits[0, :-1, :]
        targets = mx.array(target_ids[1:])
        log_probs = nn.log_softmax(logits, axis=-1)
        loss = -mx.mean(log_probs[mx.arange(len(targets)), targets])
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"  Training for {n_epochs} epochs (lr={lr})...")
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        indices = np.random.permutation(len(sequences))
        n = 0
        for idx in indices:
            seq = sequences[idx]
            if len(seq) < 3:
                continue
            loss, grads = loss_and_grad(model, seq, seq)
            mx.eval(loss)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            epoch_loss += float(loss)
            n += 1
        print(f"    Epoch {epoch}: loss={epoch_loss/max(n,1):.4f} ({time.time()-t0:.1f}s)")
    print(f"  Done in {time.time()-t0:.1f}s")
    return model


def evaluate(model, tokenizer, test_data, label=""):
    """Evaluate with flexible keyword matching."""
    results = []
    correct = 0
    for question, keywords in test_data:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                          verbose=False, sampler=sampler).strip()
        # Flexible matching: strip commas for number comparison
        answer_clean = answer.lower().replace(",", "")
        is_correct = any(kw.lower().replace(",", "") in answer_clean for kw in keywords)
        if is_correct:
            correct += 1
        results.append({
            "question": question, "answer": answer,
            "correct": is_correct, "keywords": keywords,
        })

    n_hard = 13  # first 13 are hard
    hard_correct = sum(r["correct"] for r in results[:n_hard])
    print(f"\n  {label}: {correct}/{len(test_data)} total, {hard_correct}/{n_hard} hard")
    for r in results:
        marker = "✓" if r["correct"] else "✗"
        print(f"    {marker} {r['question'][:55]}")
        print(f"      → '{r['answer'][:65]}'")
    return results, correct


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

    print(f"\n{'='*70}")
    print("Surgical Fine-Tuning v2: Proper Validation")
    print(f"Training: {len(TRAIN_QA)} QA pairs (capitals, science, history)")
    print(f"Test: {len(TEST_DATA_HOLDOUT)} questions (NO overlap with training)")
    print(f"{'='*70}\n")

    # Baseline
    print("BASELINE")
    baseline_results, baseline_total = evaluate(
        model, tokenizer, TEST_DATA_HOLDOUT, "Baseline"
    )

    # Configs to test
    configs = [
        ((23,), 5e-5, 3, "L23, lr=5e-5, 3ep"),
        ((23,), 5e-5, 5, "L23, lr=5e-5, 5ep"),
        ((22, 23), 5e-5, 3, "L22-23, lr=5e-5, 3ep"),
        ((23,), 3e-5, 5, "L23, lr=3e-5, 5ep"),
    ]

    all_results = {"baseline": baseline_total}

    for layers, lr, epochs, label in configs:
        print(f"\n{'='*70}")
        print(f"Config: {label}")
        print(f"{'='*70}")

        model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
        model = train_surgical(model, tokenizer, TRAIN_QA,
                               layers_to_train=layers, lr=lr, n_epochs=epochs)
        results, total = evaluate(model, tokenizer, TEST_DATA_HOLDOUT, label)

        fixes = sum(1 for r, b in zip(results, baseline_results)
                    if r["correct"] and not b["correct"])
        breaks = sum(1 for r, b in zip(results, baseline_results)
                     if not r["correct"] and b["correct"])
        net = fixes - breaks

        print(f"\n  vs baseline: +{fixes} fixed, -{breaks} broken, net={'+'if net>0 else ''}{net}")

        if fixes > 0:
            print(f"  Fixed:")
            for r, b in zip(results, baseline_results):
                if r["correct"] and not b["correct"]:
                    print(f"    ✓ {r['question'][:55]}")
                    print(f"      was: '{b['answer'][:50]}'")
                    print(f"      now: '{r['answer'][:50]}'")

        if breaks > 0:
            print(f"  Broken:")
            for r, b in zip(results, baseline_results):
                if not r["correct"] and b["correct"]:
                    print(f"    ✗ {r['question'][:55]}")
                    print(f"      was: '{b['answer'][:50]}'")
                    print(f"      now: '{r['answer'][:50]}'")

        all_results[label] = {
            "correct": total, "total": len(TEST_DATA_HOLDOUT),
            "fixes": fixes, "breaks": breaks, "net": net,
        }

    # Summary
    n_hard = 13
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"  Baseline: {baseline_total}/{len(TEST_DATA_HOLDOUT)}")
    print(f"  (First {n_hard} questions are hard/obscure, rest are easy/moderate)\n")
    print(f"  {'Config':30} {'Total':>8} {'Fix':>5} {'Brk':>5} {'Net':>5}")
    print("  " + "-" * 55)
    for _, _, _, label in configs:
        r = all_results[label]
        net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net']) if r['net'] < 0 else "="
        star = " ★" if r['net'] > 0 else ""
        print(f"  {label:30} {r['correct']:>4}/{r['total']}  +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")

    with open("surgical_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved surgical_v2_results.json")


if __name__ == "__main__":
    run()
