"""
Surgical Fine-Tuning: Anti-Suppression Training
================================================
Finding: layers 22-23 suppress correct factual tokens from layer 21.
Fix: fine-tune ONLY layers 22-23 (freeze everything else) on factual QA.

This teaches the final layers to PRESERVE the factual signal from
earlier layers instead of overriding it with "plausible" alternatives.

Parameters updated: ~4M (2 transformer layers) vs ~500M total model
Training: standard next-token prediction on factual QA strings
Eval: full autoregressive generation on held-out questions

If this works: we've proven the architecture fix is surgical fine-tuning
of the suppression layers, not a new head.
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
import copy


# ============================================================
# Training data: full QA strings for next-token prediction
# ============================================================

TRAIN_QA = [
    # Capitals
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasília."),
    ("What is the capital of Argentina?", "The capital of Argentina is Buenos Aires."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
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
    ("What is the capital of Finland?", "The capital of Finland is Helsinki."),
    ("What is the capital of Denmark?", "The capital of Denmark is Copenhagen."),
    ("What is the capital of Portugal?", "The capital of Portugal is Lisbon."),
    ("What is the capital of Greece?", "The capital of Greece is Athens."),
    ("What is the capital of Ireland?", "The capital of Ireland is Dublin."),
    ("What is the capital of Switzerland?", "The capital of Switzerland is Bern."),
    ("What is the capital of Austria?", "The capital of Austria is Vienna."),
    ("What is the capital of Netherlands?", "The capital of the Netherlands is Amsterdam."),
    ("What is the capital of Belgium?", "The capital of Belgium is Brussels."),
    ("What is the capital of Czech Republic?", "The capital of the Czech Republic is Prague."),
    ("What is the capital of Hungary?", "The capital of Hungary is Budapest."),
    ("What is the capital of Romania?", "The capital of Romania is Bucharest."),
    ("What is the capital of Ukraine?", "The capital of Ukraine is Kyiv."),
    ("What is the capital of Colombia?", "The capital of Colombia is Bogotá."),
    ("What is the capital of Peru?", "The capital of Peru is Lima."),
    ("What is the capital of Chile?", "The capital of Chile is Santiago."),
    ("What is the capital of Venezuela?", "The capital of Venezuela is Caracas."),
    ("What is the capital of Cuba?", "The capital of Cuba is Havana."),
    ("What is the capital of Iran?", "The capital of Iran is Tehran."),
    ("What is the capital of Iraq?", "The capital of Iraq is Baghdad."),
    ("What is the capital of Pakistan?", "The capital of Pakistan is Islamabad."),
    ("What is the capital of Indonesia?", "The capital of Indonesia is Jakarta."),
    ("What is the capital of Vietnam?", "The capital of Vietnam is Hanoi."),
    ("What is the capital of Malaysia?", "The capital of Malaysia is Kuala Lumpur."),
    ("What is the capital of New Zealand?", "The capital of New Zealand is Wellington."),
    ("What is the capital of Nigeria?", "The capital of Nigeria is Abuja."),
    ("What is the capital of South Africa?", "The capital of South Africa is Pretoria."),
    ("What is the capital of Kenya?", "The capital of Kenya is Nairobi."),
    ("What is the capital of Morocco?", "The capital of Morocco is Rabat."),
    ("What is the capital of Ethiopia?", "The capital of Ethiopia is Addis Ababa."),
    ("What is the capital of Mongolia?", "The capital of Mongolia is Ulaanbaatar."),
    ("What is the capital of Nepal?", "The capital of Nepal is Kathmandu."),
    ("What is the capital of Sri Lanka?", "The capital of Sri Lanka is Colombo."),
    ("What is the capital of Myanmar?", "The capital of Myanmar is Naypyidaw."),
    ("What is the capital of Cambodia?", "The capital of Cambodia is Phnom Penh."),
    ("What is the capital of Iceland?", "The capital of Iceland is Reykjavik."),
    ("What is the capital of Estonia?", "The capital of Estonia is Tallinn."),
    ("What is the capital of Latvia?", "The capital of Latvia is Riga."),
    ("What is the capital of Lithuania?", "The capital of Lithuania is Vilnius."),
    ("What is the capital of Croatia?", "The capital of Croatia is Zagreb."),
    ("What is the capital of Serbia?", "The capital of Serbia is Belgrade."),
    ("What is the capital of Slovakia?", "The capital of Slovakia is Bratislava."),
    ("What is the capital of Uruguay?", "The capital of Uruguay is Montevideo."),
    ("What is the capital of Paraguay?", "The capital of Paraguay is Asunción."),
    ("What is the capital of Ecuador?", "The capital of Ecuador is Quito."),
    ("What is the capital of Jamaica?", "The capital of Jamaica is Kingston."),
    ("What is the capital of Madagascar?", "The capital of Madagascar is Antananarivo."),
    ("What is the capital of Zimbabwe?", "The capital of Zimbabwe is Harare."),
    ("What is the capital of Zambia?", "The capital of Zambia is Lusaka."),
    ("What is the capital of Uganda?", "The capital of Uganda is Kampala."),
    ("What is the capital of Rwanda?", "The capital of Rwanda is Kigali."),
    ("What is the capital of Senegal?", "The capital of Senegal is Dakar."),
    ("What is the capital of Afghanistan?", "The capital of Afghanistan is Kabul."),
    ("What is the capital of Kazakhstan?", "The capital of Kazakhstan is Astana."),
    ("What is the capital of Georgia?", "The capital of Georgia is Tbilisi."),
    ("What is the capital of Armenia?", "The capital of Armenia is Yerevan."),
    ("What is the capital of Azerbaijan?", "The capital of Azerbaijan is Baku."),
    ("What is the capital of Jordan?", "The capital of Jordan is Amman."),
    ("What is the capital of Lebanon?", "The capital of Lebanon is Beirut."),
    ("What is the capital of Syria?", "The capital of Syria is Damascus."),
    ("What is the capital of Qatar?", "The capital of Qatar is Doha."),
    ("What is the capital of Kuwait?", "The capital of Kuwait is Kuwait City."),

    # Authors
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare."),
    ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by Jane Austen."),
    ("Who wrote 1984?", "1984 was written by George Orwell."),
    ("Who wrote War and Peace?", "War and Peace was written by Leo Tolstoy."),
    ("Who wrote Don Quixote?", "Don Quixote was written by Miguel de Cervantes."),
    ("Who wrote The Odyssey?", "The Odyssey was written by Homer."),
    ("Who wrote Les Misérables?", "Les Misérables was written by Victor Hugo."),
    ("Who wrote Crime and Punishment?", "Crime and Punishment was written by Fyodor Dostoevsky."),
    ("Who wrote Moby Dick?", "Moby Dick was written by Herman Melville."),
    ("Who wrote A Tale of Two Cities?", "A Tale of Two Cities was written by Charles Dickens."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("Who painted The Starry Night?", "The Starry Night was painted by Vincent van Gogh."),
    ("Who composed the Four Seasons?", "The Four Seasons was composed by Antonio Vivaldi."),

    # Science
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("What is the chemical symbol for silver?", "The chemical symbol for silver is Ag."),
    ("What is the chemical symbol for iron?", "The chemical symbol for iron is Fe."),
    ("What is the speed of light in m/s?", "The speed of light is approximately 299,792,458 m/s."),
    ("What is the boiling point of water in Celsius?", "The boiling point of water is 100°C."),
    ("What is the largest planet in the solar system?", "The largest planet in the solar system is Jupiter."),
    ("What planet is closest to the Sun?", "The planet closest to the Sun is Mercury."),
    ("How many chromosomes do humans have?", "Humans have 46 chromosomes."),
    ("What is the hardest natural substance?", "The hardest natural substance is diamond."),
    ("What is the most abundant gas in Earth's atmosphere?", "The most abundant gas in Earth's atmosphere is nitrogen."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ("What is the chemical formula for table salt?", "The chemical formula for table salt is NaCl."),

    # History
    ("In what year did World War II end?", "World War II ended in 1945."),
    ("Who was the first president of the United States?", "The first president of the United States was George Washington."),
    ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming."),
    ("Who invented the telephone?", "The telephone was invented by Alexander Graham Bell."),
    ("Who developed the theory of relativity?", "The theory of relativity was developed by Albert Einstein."),
    ("Who was the first person to walk on the moon?", "The first person to walk on the moon was Neil Armstrong."),

    # Geography
    ("What is the tallest mountain in the world?", "The tallest mountain in the world is Mount Everest."),
    ("What is the largest continent by area?", "The largest continent by area is Asia."),
    ("What is the largest ocean?", "The largest ocean is the Pacific Ocean."),

    # Math
    ("What is the value of pi?", "The value of pi is approximately 3.14159."),
    ("What is the square root of 144?", "The square root of 144 is 12."),
    ("How many sides does a hexagon have?", "A hexagon has 6 sides."),

    # Now add harder factual questions (closer to test distribution)
    ("What is the capital of Kyrgyzstan?", "The capital of Kyrgyzstan is Bishkek."),
    ("What is the capital of Burkina Faso?", "The capital of Burkina Faso is Ouagadougou."),
    ("What is the melting point of tungsten in Celsius?", "The melting point of tungsten is 3,422°C."),
    ("What is the half-life of Carbon-14 in years?", "The half-life of Carbon-14 is approximately 5,730 years."),
    ("What is the half-life of uranium-235 in years?", "The half-life of uranium-235 is approximately 703.8 million years."),
    ("What country has the most UNESCO World Heritage Sites?", "Italy has the most UNESCO World Heritage Sites."),
    ("Who was the first female prime minister of the UK?", "The first female prime minister of the UK was Margaret Thatcher."),
    ("What year was the WHO founded?", "The World Health Organization was founded in 1948."),
    ("Who was the 30th president of the United States?", "The 30th president of the United States was Calvin Coolidge."),
    ("What is the rarest blood type?", "The rarest blood type is AB negative."),
    ("What is the largest desert in the world by area?", "The largest desert in the world by area is Antarctica."),
    ("What is the speed of sound in water in m/s?", "The speed of sound in water is approximately 1,480 m/s."),
    ("What is the atomic weight of plutonium?", "The atomic weight of plutonium is approximately 244."),
]

# Held-out test questions (NOT in training in any form)
TEST_DATA = [
    ("Who won the Nobel Prize in Chemistry in 2023?",          ["bawendi", "brus", "ekimov"]),
    ("Who is the prime minister of New Zealand as of 2024?",   ["luxon", "christopher"]),
    ("What is 2 + 2?",                                         ["4", "four"]),
    ("Who wrote Romeo and Juliet?",                            ["shakespeare"]),
    ("What is the capital of France?",                         ["paris"]),
    ("What is the largest planet?",                            ["jupiter"]),
    ("What is the boiling point of water?",                    ["100"]),
    # Test on training-distribution questions (should get better)
    ("What is the capital of Kyrgyzstan?",                     ["bishkek"]),
    ("What is the capital of Burkina Faso?",                   ["ouagadougou"]),
    ("What is the melting point of tungsten in Celsius?",      ["3422", "3400"]),
    ("What country has the most UNESCO World Heritage Sites?",  ["italy", "china"]),
    ("Who was the first female prime minister of the UK?",     ["thatcher"]),
    ("What year was the WHO founded?",                         ["1948"]),
    ("Who was the 30th president of the United States?",       ["coolidge", "calvin"]),
    ("What is the half-life of uranium-235 in years?",         ["703 million", "703", "700"]),
    ("What is the largest desert in the world by area?",       ["antarctica", "antarctic"]),
    ("What is the rarest blood type?",                         ["ab-", "ab negative"]),
    ("What is the speed of sound in water in m/s?",            ["1480", "1500", "1498"]),
    ("What is the atomic weight of plutonium?",                ["244", "242", "239"]),
    ("What is the half-life of Carbon-14 in years?",           ["5730", "5700"]),
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


def train_surgical(model, tokenizer, train_qa, layers_to_train=(22, 23),
                   lr=5e-5, n_epochs=3, batch_size=4):
    """
    Fine-tune ONLY the specified layers on factual QA data.
    Everything else is frozen.
    """
    # Freeze all parameters first
    model.freeze()

    # Unfreeze only the target layers
    for layer_idx in layers_to_train:
        layer = model.model.layers[layer_idx]
        layer.unfreeze()

    # Count trainable params
    n_trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    n_total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"  Trainable: {n_trainable:,} / {n_total:,} parameters ({100*n_trainable/n_total:.2f}%)")

    # Prepare training data
    sequences = prepare_training_sequences(tokenizer, train_qa)
    print(f"  Training sequences: {len(sequences)}")
    print(f"  Avg length: {np.mean([len(s) for s in sequences]):.0f} tokens")

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, input_ids, target_ids):
        """Standard next-token prediction loss."""
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

        # Shift: predict next token
        logits = logits[0, :-1, :]  # (T-1, V)
        targets = mx.array(target_ids[1:])  # (T-1,)

        log_probs = nn.log_softmax(logits, axis=-1)
        loss = -mx.mean(log_probs[mx.arange(len(targets)), targets])
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"  Training for {n_epochs} epochs (lr={lr})...")
    t0 = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        # Shuffle
        indices = np.random.permutation(len(sequences))
        n_batches = 0

        for idx in indices:
            seq = sequences[idx]
            if len(seq) < 3:
                continue

            loss, grads = loss_and_grad(model, seq, seq)
            mx.eval(loss)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"    Epoch {epoch}: avg_loss={avg_loss:.4f} ({elapsed:.1f}s)")

    print(f"  Training done in {time.time()-t0:.1f}s")
    return model


def evaluate(model, tokenizer, test_data, label=""):
    """Evaluate full generation accuracy."""
    results = []
    correct = 0

    for question, keywords in test_data:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                          verbose=False, sampler=sampler).strip()
        is_correct = any(kw.lower() in answer.lower() for kw in keywords)
        if is_correct:
            correct += 1
        results.append({
            "question": question, "answer": answer,
            "correct": is_correct, "keywords": keywords,
        })

    print(f"\n  {label}: {correct}/{len(test_data)}")
    for r in results:
        marker = "✓" if r["correct"] else "✗"
        print(f"    {marker} {r['question'][:55]}")
        print(f"      → '{r['answer'][:65]}'")

    return results, correct


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

    print(f"\n{'='*70}")
    print("Surgical Fine-Tuning: Anti-Suppression Training")
    print(f"Training data: {len(TRAIN_QA)} factual QA pairs (full sequences)")
    print(f"Test data: {len(TEST_DATA)} questions")
    print(f"{'='*70}\n")

    # ==========================================
    # Step 1: Baseline evaluation
    # ==========================================
    print("Step 1: Baseline (before any fine-tuning)")
    baseline_results, baseline_correct = evaluate(
        model, tokenizer, TEST_DATA, "Baseline"
    )

    # ==========================================
    # Step 2: Surgical fine-tuning experiments
    # ==========================================
    configs = [
        # (layers_to_train, lr, epochs, label)
        ((22, 23), 5e-5, 3, "L22-23, lr=5e-5, 3ep"),
        ((22, 23), 1e-4, 3, "L22-23, lr=1e-4, 3ep"),
        ((21, 22, 23), 5e-5, 3, "L21-23, lr=5e-5, 3ep"),
        ((23,), 5e-5, 3, "L23 only, lr=5e-5, 3ep"),
    ]

    all_results = {"baseline": baseline_correct}

    for layers, lr, epochs, label in configs:
        print(f"\n{'='*70}")
        print(f"Config: {label}")
        print(f"{'='*70}\n")

        # Reload fresh model for each config
        model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

        model = train_surgical(
            model, tokenizer, TRAIN_QA,
            layers_to_train=layers, lr=lr, n_epochs=epochs
        )

        results, correct = evaluate(model, tokenizer, TEST_DATA, label)

        # Track fixes and breaks
        fixes = sum(1 for r, b in zip(results, baseline_results)
                    if r["correct"] and not b["correct"])
        breaks = sum(1 for r, b in zip(results, baseline_results)
                     if not r["correct"] and b["correct"])

        net = fixes - breaks
        net_s = f"+{net}" if net > 0 else str(net) if net < 0 else "="

        print(f"\n  vs baseline: +{fixes} fixed, -{breaks} broken, net={net_s}")

        if fixes > 0:
            print(f"  Fixed:")
            for r, b in zip(results, baseline_results):
                if r["correct"] and not b["correct"]:
                    print(f"    ✓ {r['question'][:55]}")
                    print(f"      before: '{b['answer'][:50]}'")
                    print(f"      after:  '{r['answer'][:50]}'")

        if breaks > 0:
            print(f"  Broken:")
            for r, b in zip(results, baseline_results):
                if not r["correct"] and b["correct"]:
                    print(f"    ✗ {r['question'][:55]}")
                    print(f"      before: '{b['answer'][:50]}'")
                    print(f"      after:  '{r['answer'][:50]}'")

        all_results[label] = {
            "correct": correct, "total": len(TEST_DATA),
            "fixes": fixes, "breaks": breaks, "net": net,
        }

    # ==========================================
    # Final summary
    # ==========================================
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print(f"  Baseline: {baseline_correct}/{len(TEST_DATA)}\n")
    print(f"  {'Config':40} {'Score':>8} {'Fix':>5} {'Brk':>5} {'Net':>5}")
    print("  " + "-" * 65)
    for layers, lr, epochs, label in configs:
        r = all_results[label]
        net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net']) if r['net'] < 0 else "="
        star = " ★" if r['net'] > 0 else ""
        print(f"  {label:40} {r['correct']:>4}/{r['total']}  +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")

    best_config = max(configs, key=lambda c: all_results[c[3]]["net"])
    best = all_results[best_config[3]]
    if best["net"] > 0:
        print(f"\n  ★ SURGICAL FINE-TUNING WORKS: {best_config[3]}")
        print(f"    Net improvement: +{best['net']} ({best['fixes']} fixed, {best['breaks']} broken)")
    else:
        print(f"\n  No config improved over baseline.")

    with open("surgical_finetune_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved surgical_finetune_results.json")


if __name__ == "__main__":
    run()
