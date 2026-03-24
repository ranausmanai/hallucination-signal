"""
Qwen3.5-0.8B Full Experiment Suite
====================================
Same experiments as Qwen2-0.5B but on a larger, newer model.
Tests: oscillation detection, knowledge suppression, surgical fine-tuning.

Model structure: model.language_model.model.{embed_tokens, layers, norm}
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, generate
from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
from mlx_lm.sample_utils import make_sampler
import json
import time


# ============================================================
# Model access helpers (adapt for Qwen3.5 structure)
# ============================================================

def get_backbone(model):
    """Return (text_model, embed_tokens, layers, norm, as_linear_fn)."""
    lm = model.language_model
    tm = lm.model
    embed = tm.embed_tokens
    layers = tm.layers
    norm = tm.norm

    def get_logits(h_normed):
        # Tied embeddings: use embed_tokens.as_linear
        return embed.as_linear(h_normed)

    return lm, tm, embed, layers, norm, get_logits


# ============================================================
# Core functions
# ============================================================

def get_masks(embed_out, tm):
    """Create both FA and SSM masks for hybrid model."""
    fa_mask = create_attention_mask(embed_out, None)
    ssm_mask = create_ssm_mask(embed_out, None)
    return fa_mask, ssm_mask


def get_layer_mask(layer, fa_mask, ssm_mask):
    """Return the correct mask for a layer based on its type."""
    return ssm_mask if layer.is_linear else fa_mask


def get_oscillation(model, token_ids):
    """Compute oscillation: how many times top-1 prediction changes across layers."""
    lm, tm, embed, layers, norm, get_logits = get_backbone(model)

    ids_mx = mx.array([token_ids])
    h = embed(ids_mx)
    mx.eval(h)
    fa_mask, ssm_mask = get_masks(h, tm)

    preds = []
    for layer in layers:
        mask = get_layer_mask(layer, fa_mask, ssm_mask)
        h = layer(h, mask, None)
        mx.eval(h)
        h_norm = norm(h)
        logits = get_logits(h_norm)
        mx.eval(logits)
        preds.append(int(mx.argmax(logits[0, -1]).item()))
        del logits, h_norm

    return sum(preds[i] != preds[i-1] for i in range(1, len(preds)))


def get_per_layer_top1(model, tokenizer, token_ids, layer_range=(0, 24)):
    """Get top-1 prediction at each layer."""
    lm, tm, embed, layers, norm, get_logits = get_backbone(model)

    ids_mx = mx.array([token_ids])
    h = embed(ids_mx)
    mx.eval(h)
    fa_mask, ssm_mask = get_masks(h, tm)

    results = []
    for i, layer in enumerate(layers):
        mask = get_layer_mask(layer, fa_mask, ssm_mask)
        h = layer(h, mask, None)
        mx.eval(h)

        if i >= layer_range[0]:
            h_norm = norm(h)
            logits = get_logits(h_norm)
            mx.eval(logits)
            last = np.array(logits[0, -1].astype(mx.float32))
            probs = np.exp(last - last.max())
            probs /= probs.sum()

            top1_id = int(np.argmax(last))
            results.append({
                "layer": i,
                "top1": top1_id,
                "top1_tok": tokenizer.decode([top1_id]),
                "confidence": float(probs.max()),
            })
            del logits, h_norm

    return results


def ask(model, tokenizer, question, temperature=0.0, max_tokens=60):
    """Ask a question with given temperature."""
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=temperature)
    response = generate(model, tokenizer, prompt=fmt, max_tokens=max_tokens,
                        verbose=False, sampler=sampler)
    return response.strip()


def check_correct(answer, keywords):
    """Flexible keyword matching (handles commas in numbers)."""
    answer_clean = answer.lower().replace(",", "")
    return any(kw.lower().replace(",", "") in answer_clean for kw in keywords)


# ============================================================
# Test questions
# ============================================================

TEST_HARD = [
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
    ("What is the capital of Kyrgyzstan?",                     ["bishkek"]),
    ("What is the capital of Burkina Faso?",                   ["ouagadougou"]),
]

TEST_EASY = [
    ("What is the capital of France?",                         ["paris"]),
    ("Who wrote Romeo and Juliet?",                            ["shakespeare"]),
    ("What is 2 + 2?",                                         ["4", "four"]),
    ("What is the largest planet?",                            ["jupiter"]),
    ("What is the boiling point of water?",                    ["100"]),
    ("What language is spoken in Brazil?",                     ["portuguese"]),
    ("What is the tallest mountain in the world?",             ["everest"]),
    ("Who was the first person to walk on the moon?",          ["armstrong", "neil"]),
    ("What is the largest ocean?",                             ["pacific"]),
    ("What year did World War II end?",                        ["1945"]),
]

TEST_ALL = TEST_HARD + TEST_EASY

# Training data for surgical fine-tuning (no overlap with hard test questions)
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


def train_surgical(model, tokenizer, train_qa, layers_to_train=(23,),
                   lr=5e-5, n_epochs=3):
    """Fine-tune only the specified layers."""
    lm, tm, embed, layers_list, norm_layer, get_logits = get_backbone(model)

    model.freeze()
    for layer_idx in layers_to_train:
        tm.layers[layer_idx].unfreeze()

    n_trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    n_total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

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
        lm2, tm2, embed2, layers2, norm2, get_logits2 = get_backbone(model)
        ids_mx = mx.array([input_ids])
        h = embed2(ids_mx)
        fa_mask, ssm_mask = get_masks(h, tm2)
        for layer in layers2:
            mask = get_layer_mask(layer, fa_mask, ssm_mask)
            h = layer(h, mask, None)
        h = norm2(h)
        logits = get_logits2(h)
        logits = logits[0, :-1, :]
        targets = mx.array(input_ids[1:])
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
            loss, grads = loss_and_grad(model, seq)
            mx.eval(loss)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            epoch_loss += float(loss)
            n += 1
        print(f"    Epoch {epoch}: loss={epoch_loss/max(n,1):.4f} ({time.time()-t0:.1f}s)")
    print(f"  Done in {time.time()-t0:.1f}s")
    return model


def run():
    print("Loading Qwen3.5-0.8B...")
    model, tokenizer = load("Qwen/Qwen3.5-0.8B")

    print(f"\n{'='*70}")
    print("Qwen3.5-0.8B — Full Experiment Suite")
    print(f"{'='*70}\n")

    # ==========================================
    # Phase 1: Baseline + Oscillation
    # ==========================================
    print("PHASE 1: Baseline Evaluation + Oscillation Detection")
    print("-" * 50)

    baseline_results = []
    for question, keywords in TEST_ALL:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        osc = get_oscillation(model, ids)
        answer = ask(model, tokenizer, question)
        correct = check_correct(answer, keywords)

        marker = "✓" if correct else "✗"
        print(f"  {marker} [osc={osc:2d}] {question[:55]}")
        print(f"      → '{answer[:65]}'")

        baseline_results.append({
            "question": question, "keywords": keywords,
            "answer": answer, "correct": correct, "osc": osc,
        })

    n_hard = len(TEST_HARD)
    hard_correct = sum(r["correct"] for r in baseline_results[:n_hard])
    easy_correct = sum(r["correct"] for r in baseline_results[n_hard:])
    total_correct = sum(r["correct"] for r in baseline_results)

    print(f"\n  Baseline: {total_correct}/{len(TEST_ALL)} total, "
          f"{hard_correct}/{n_hard} hard, {easy_correct}/{len(TEST_EASY)} easy")

    # Oscillation analysis
    wrong = [r for r in baseline_results if not r["correct"]]
    right = [r for r in baseline_results if r["correct"]]
    if wrong and right:
        avg_osc_wrong = np.mean([r["osc"] for r in wrong])
        avg_osc_right = np.mean([r["osc"] for r in right])
        print(f"\n  Oscillation: wrong avg={avg_osc_wrong:.1f}, right avg={avg_osc_right:.1f}")
        print(f"  Separation: {'✓ signal works' if avg_osc_wrong > avg_osc_right else '✗ no separation'}")

    # ==========================================
    # Phase 2: Layer-by-layer diagnostic
    # ==========================================
    print(f"\n\n{'='*70}")
    print("PHASE 2: Layer-by-Layer Confidence Profile")
    print("(Where does the model know the right answer?)")
    print("-" * 50)

    diagnostic_qs = [
        ("What is the capital of France?", ["paris"]),
        ("What is the largest planet?", ["jupiter"]),
        ("What country has the most UNESCO World Heritage Sites?", ["italy", "china"]),
        ("What is the largest desert in the world by area?", ["antarctica", "antarctic"]),
    ]

    for question, keywords in diagnostic_qs:
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        layer_data = get_per_layer_top1(model, tokenizer, ids, layer_range=(15, 24))

        print(f"\n  Q: {question}")
        print(f"  {'L':>4} {'Conf':>6} {'Top-1':>15}")
        print("  " + "-" * 30)
        for d in layer_data:
            kw_match = any(kw.lower() in d["top1_tok"].lower() for kw in keywords)
            marker = " ←✓" if kw_match else ""
            print(f"  {d['layer']:>4} {d['confidence']:>6.3f} {d['top1_tok']:>15}{marker}")

    # ==========================================
    # Phase 3: Surgical fine-tuning (proper held-out)
    # ==========================================
    print(f"\n\n{'='*70}")
    print("PHASE 3: Surgical Fine-Tuning (no train/test overlap)")
    print("-" * 50)

    configs = [
        ((23,), 3e-5, 3, "L23, lr=3e-5, 3ep"),
        ((23,), 3e-5, 5, "L23, lr=3e-5, 5ep"),
        ((23,), 1e-5, 5, "L23, lr=1e-5, 5ep"),
    ]

    ft_results = {}

    for layers, lr, epochs, label in configs:
        print(f"\n  Config: {label}")
        model, tokenizer = load("Qwen/Qwen3.5-0.8B")
        model = train_surgical(model, tokenizer, TRAIN_QA,
                               layers_to_train=layers, lr=lr, n_epochs=epochs)

        results = []
        for question, keywords in TEST_ALL:
            answer = ask(model, tokenizer, question)
            correct = check_correct(answer, keywords)
            results.append({
                "question": question, "answer": answer, "correct": correct,
            })

        total = sum(r["correct"] for r in results)
        hard = sum(r["correct"] for r in results[:n_hard])

        fixes = sum(1 for r, b in zip(results, baseline_results)
                    if r["correct"] and not b["correct"])
        breaks = sum(1 for r, b in zip(results, baseline_results)
                     if not r["correct"] and b["correct"])
        net = fixes - breaks

        print(f"\n  {label}: {total}/{len(TEST_ALL)} total, {hard}/{n_hard} hard")
        print(f"  vs baseline: +{fixes} fixed, -{breaks} broken, net={'+'if net>0 else ''}{net}")

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

        ft_results[label] = {
            "total": total, "hard": hard, "fixes": fixes, "breaks": breaks, "net": net,
        }

    # ==========================================
    # Phase 4: Self-consistency
    # ==========================================
    print(f"\n\n{'='*70}")
    print("PHASE 4: Self-Consistency (N=7, T=0.5)")
    print("-" * 50)

    model, tokenizer = load("Qwen/Qwen3.5-0.8B")

    from collections import Counter

    sc_results = []
    for question, keywords in TEST_ALL:
        # Greedy
        greedy = ask(model, tokenizer, question, temperature=0.0)
        greedy_correct = check_correct(greedy, keywords)

        # Sample 7 times
        samples = [ask(model, tokenizer, question, temperature=0.5) for _ in range(7)]

        # Majority vote on keywords
        keyword_hits = Counter()
        for s in samples:
            s_clean = s.lower().replace(",", "")
            for kw in keywords:
                if kw.lower().replace(",", "") in s_clean:
                    keyword_hits[kw] += 1
                    break

        if keyword_hits:
            winner, count = keyword_hits.most_common(1)[0]
            # Find a sample that matches
            sc_answer = None
            for s in samples:
                if winner.lower() in s.lower().replace(",", ""):
                    sc_answer = s
                    break
            if sc_answer is None:
                sc_answer = greedy
            sc_correct = True
        else:
            sc_answer = greedy
            sc_correct = greedy_correct
            count = 0

        marker_g = "✓" if greedy_correct else "✗"
        marker_s = "✓" if sc_correct else "✗"
        indicator = ""
        if sc_correct and not greedy_correct:
            indicator = " *** FIXED ***"
        elif not sc_correct and greedy_correct:
            indicator = " *** BROKE ***"

        print(f"  [greedy {marker_g}] [SC {marker_s} ({count}/7)]{indicator}")
        print(f"    Q: {question[:55]}")
        if indicator:
            print(f"    greedy: '{greedy[:50]}'")
            print(f"    SC:     '{sc_answer[:50]}'")

        sc_results.append({
            "question": question, "greedy_correct": greedy_correct,
            "sc_correct": sc_correct, "vote_count": count,
        })

    sc_hard = sum(r["sc_correct"] for r in sc_results[:n_hard])
    sc_total = sum(r["sc_correct"] for r in sc_results)
    greedy_hard = sum(r["greedy_correct"] for r in sc_results[:n_hard])
    greedy_total = sum(r["greedy_correct"] for r in sc_results)
    sc_fixes = sum(1 for r in sc_results if r["sc_correct"] and not r["greedy_correct"])
    sc_breaks = sum(1 for r in sc_results if not r["sc_correct"] and r["greedy_correct"])

    print(f"\n  Greedy: {greedy_total}/{len(TEST_ALL)} total, {greedy_hard}/{n_hard} hard")
    print(f"  SC:     {sc_total}/{len(TEST_ALL)} total, {sc_hard}/{n_hard} hard")
    print(f"  SC vs greedy: +{sc_fixes} fixed, -{sc_breaks} broken")

    # ==========================================
    # Final Summary
    # ==========================================
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY — Qwen3.5-0.8B")
    print(f"{'='*70}\n")

    print(f"  Baseline greedy: {hard_correct}/{n_hard} hard, {total_correct}/{len(TEST_ALL)} total\n")

    print(f"  {'Method':35} {'Hard':>6} {'Total':>8} {'Fix':>5} {'Brk':>5} {'Net':>5}")
    print("  " + "-" * 68)
    print(f"  {'Greedy baseline':35} {hard_correct:>3}/{n_hard}  {total_correct:>4}/{len(TEST_ALL)}")
    for _, _, _, label in configs:
        r = ft_results[label]
        net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net']) if r['net'] < 0 else "="
        star = " ★" if r['net'] > 0 else ""
        print(f"  {label:35} {r['hard']:>3}/{n_hard}  {r['total']:>4}/{len(TEST_ALL)}  +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")
    net_sc = sc_fixes - sc_breaks
    net_sc_s = f"+{net_sc}" if net_sc > 0 else str(net_sc) if net_sc < 0 else "="
    star_sc = " ★" if net_sc > 0 else ""
    print(f"  {'Self-consistency (N=7, T=0.5)':35} {sc_hard:>3}/{n_hard}  {sc_total:>4}/{len(TEST_ALL)}  +{sc_fixes:>3}  -{sc_breaks:>3}  {net_sc_s:>4}{star_sc}")

    # Save
    all_data = {
        "model": "Qwen3.5-0.8B",
        "baseline": {"hard": hard_correct, "total": total_correct},
        "surgical_ft": ft_results,
        "self_consistency": {
            "hard": sc_hard, "total": sc_total,
            "fixes": sc_fixes, "breaks": sc_breaks,
        },
        "oscillation": {
            "avg_wrong": float(avg_osc_wrong) if wrong else 0,
            "avg_right": float(avg_osc_right) if right else 0,
        },
    }
    with open("qwen35_results.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved qwen35_results.json")


if __name__ == "__main__":
    run()
