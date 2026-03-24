"""
ANCHOR v2: Logit-Space Factual Anchoring
=========================================

Layer 21 knows the correct answer. Layers 22-23 suppress it.
v1 failed because hidden states from different layers have incompatible
scales — adding them directly produces garbage.

v2 Fix: operate entirely in LOGIT SPACE where both layers speak the
same language (vocabulary probabilities).

Strategies tested:
1. Static blend:  logits = final + α * factual           (1 param)
2. Entropy-gated: α = sigmoid(a*entropy + b)              (2 params)
3. Top-k rescue:  only blend top-k factual logits          (1 param + k)
4. Contrastive:   logits = final + α * (factual - final)   (1 param, = interpolation)

All operate on cached logits — training is seconds, not minutes.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import json
import time


# ============================================================
# Training/Test Data
# ============================================================

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
    ("What is the speed of light in m/s?", "The speed of light is approximately 299,792,458 m/s."),
]

TEST_DATA = [
    # Hard (strictly held out)
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
    # Easy controls
    ("What is the capital of France?",                         ["paris"]),
    ("Who wrote Romeo and Juliet?",                            ["shakespeare"]),
    ("What is 2 + 2?",                                         ["4", "four"]),
    ("What is the largest planet?",                            ["jupiter"]),
    ("What is the boiling point of water?",                    ["100"]),
    ("What language is spoken in Brazil?",                     ["portuguese"]),
    ("What is the largest ocean?",                             ["pacific"]),
]


def check_correct(answer, keywords):
    answer_clean = answer.lower().replace(",", "")
    return any(kw.lower().replace(",", "") in answer_clean for kw in keywords)


# ============================================================
# Model setup (reuse from v1)
# ============================================================

def setup_qwen2(model):
    from mlx_lm.models.qwen2 import create_attention_mask as cam
    def get_logits(h):
        if model.args.tie_word_embeddings:
            return model.model.embed_tokens.as_linear(h)
        return model.lm_head(h)
    return {
        "embed": model.model.embed_tokens,
        "layers": model.model.layers,
        "norm": model.model.norm,
        "get_logits": get_logits,
        "get_mask": lambda h: cam(h, None),
        "get_layer_mask": None,
        "hidden_dim": model.args.hidden_size,
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
    return {
        "embed": tm.embed_tokens,
        "layers": tm.layers,
        "norm": tm.norm,
        "get_logits": lambda h: tm.embed_tokens.as_linear(h),
        "get_mask": get_mask,
        "get_layer_mask": get_layer_mask,
        "hidden_dim": lm.args.hidden_size,
        "n_layers": len(tm.layers),
    }


# ============================================================
# Logit collection — run once, cache everything
# ============================================================

def collect_logits(model_info, tokenizer, qa_pairs, factual_layer):
    """
    Forward pass through frozen backbone. Cache logits from factual layer
    and final layer for every token position.
    """
    mi = model_info
    data = []

    for question, answer in qa_pairs:
        messages = [
            {"role": "user", "content": f"Answer briefly: {question}"},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        ids = tokenizer.encode(text, add_special_tokens=True)

        ids_mx = mx.array([ids])
        h = mi["embed"](ids_mx)
        mask = mi["get_mask"](h)

        h_factual = None
        for i, layer in enumerate(mi["layers"]):
            if mi["get_layer_mask"]:
                m = mi["get_layer_mask"](layer, mask)
            else:
                m = mask
            h = layer(h, m, None)
            if i == factual_layer:
                h_factual = h

        # Get logits from both layers through same norm+head
        logits_factual = mi["get_logits"](mi["norm"](h_factual))
        logits_final = mi["get_logits"](mi["norm"](h))
        mx.eval(logits_factual, logits_final)

        # Store as numpy (save MLX memory)
        data.append({
            "logits_factual": np.array(logits_factual[0].astype(mx.float32)),  # (T, V)
            "logits_final": np.array(logits_final[0].astype(mx.float32)),      # (T, V)
            "ids": ids,
        })
        del h, h_factual, logits_factual, logits_final

    return data


# ============================================================
# Strategy 1: Static Blend (1 param)
# logits = final + α * factual, α starts at 0
# ============================================================

class StaticBlend(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = mx.array([0.0])  # start at 0 = no change

    def __call__(self, logits_factual, logits_final):
        return logits_final + self.alpha * logits_factual


# ============================================================
# Strategy 2: Entropy-Gated Blend (3 params)
# α = sigmoid(a * entropy(final) + b)
# logits = final + α * factual
# ============================================================

class EntropyGatedBlend(nn.Module):
    def __init__(self):
        super().__init__()
        self.entropy_weight = mx.array([1.0])   # how much entropy matters
        self.entropy_bias = mx.array([-3.0])     # start gate ~= 0.05 (conservative)
        self.scale = mx.array([0.1])             # how much factual to add

    def __call__(self, logits_factual, logits_final):
        # Compute entropy of final distribution per position
        p = mx.softmax(logits_final, axis=-1)
        log_p = mx.log(p + 1e-10)
        entropy = -mx.sum(p * log_p, axis=-1, keepdims=True)  # (T, 1)

        # Gate: more uncertain → more factual signal
        gate = mx.sigmoid(self.entropy_weight * entropy + self.entropy_bias)  # (T, 1)

        return logits_final + gate * self.scale * logits_factual


# ============================================================
# Strategy 3: Contrastive Blend (1 param)
# logits = (1-α) * final + α * factual = final + α * (factual - final)
# ============================================================

class ContrastiveBlend(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_logit = mx.array([0.0])  # sigmoid → 0.5

    def __call__(self, logits_factual, logits_final):
        alpha = mx.sigmoid(self.alpha_logit)
        return (1 - alpha) * logits_final + alpha * logits_factual


# ============================================================
# Strategy 4: Entropy-Triggered Switch (2 params)
# If entropy > threshold: use factual. Else: use final.
# Differentiable via soft switch.
# ============================================================

class EntropySwitchBlend(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold_logit = mx.array([2.0])  # entropy threshold (~7.4 nats)
        self.sharpness = mx.array([2.0])         # how sharp the switch is

    def __call__(self, logits_factual, logits_final):
        p = mx.softmax(logits_final, axis=-1)
        log_p = mx.log(p + 1e-10)
        entropy = -mx.sum(p * log_p, axis=-1, keepdims=True)

        # Soft switch: 1 when entropy >> threshold, 0 when entropy << threshold
        switch = mx.sigmoid(self.sharpness * (entropy - self.threshold_logit))

        return (1 - switch) * logits_final + switch * logits_factual


# ============================================================
# Strategy 5: Factual Boost (1 param)
# logits = final + α * softmax(factual)
# Adds probability mass, not raw logits — gentler
# ============================================================

class FactualBoost(nn.Module):
    def __init__(self):
        super().__init__()
        self.strength = mx.array([0.0])  # start at 0

    def __call__(self, logits_factual, logits_final):
        p_factual = mx.softmax(logits_factual, axis=-1)
        return logits_final + self.strength * p_factual


# ============================================================
# Training loop (shared for all strategies)
# ============================================================

def train_strategy(strategy, train_data, n_epochs=300, lr=0.01):
    """Train a blending strategy on cached logits."""
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(strategy, lf_mx, ll_mx, target_ids):
        logits_blended = strategy(lf_mx, ll_mx)
        # Next-token prediction
        logits = logits_blended[:-1, :]
        targets = mx.array(target_ids[1:])
        log_probs = nn.log_softmax(logits, axis=-1)
        loss = -mx.mean(log_probs[mx.arange(len(targets)), targets])
        return loss

    loss_and_grad = nn.value_and_grad(strategy, loss_fn)

    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        indices = np.random.permutation(len(train_data))
        for idx in indices:
            d = train_data[idx]
            lf = mx.array(d["logits_factual"])
            ll = mx.array(d["logits_final"])
            loss, grads = loss_and_grad(strategy, lf, ll, d["ids"])
            mx.eval(loss)
            optimizer.update(strategy, grads)
            mx.eval(strategy.parameters())
            epoch_loss += float(loss)
            del lf, ll

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            avg = epoch_loss / len(train_data)
            elapsed = time.time() - t0
            # Print strategy-specific params
            params_str = format_params(strategy)
            print(f"    Epoch {epoch:3d}: loss={avg:.4f} {params_str} ({elapsed:.1f}s)")

    return strategy


def format_params(strategy):
    """Format learned parameters for display."""
    if isinstance(strategy, StaticBlend):
        return f"α={float(strategy.alpha):.4f}"
    elif isinstance(strategy, EntropyGatedBlend):
        return (f"w={float(strategy.entropy_weight):.3f} "
                f"b={float(strategy.entropy_bias):.3f} "
                f"s={float(strategy.scale):.4f}")
    elif isinstance(strategy, ContrastiveBlend):
        a = float(mx.sigmoid(strategy.alpha_logit))
        return f"α={a:.4f}"
    elif isinstance(strategy, EntropySwitchBlend):
        t = float(strategy.threshold_logit)
        return f"thresh={t:.3f}"
    elif isinstance(strategy, FactualBoost):
        return f"str={float(strategy.strength):.4f}"
    return ""


# ============================================================
# Generation with logit-space anchor
# ============================================================

def generate_with_anchor(model_info, tokenizer, strategy, question,
                         factual_layer=21, max_tokens=60):
    """Autoregressive generation with logit-space anchoring."""
    mi = model_info
    messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    generated = []

    for _ in range(max_tokens):
        ids_mx = mx.array([ids])
        h = mi["embed"](ids_mx)
        mask = mi["get_mask"](h)

        h_factual = None
        for i, layer in enumerate(mi["layers"]):
            if mi["get_layer_mask"]:
                m = mi["get_layer_mask"](layer, mask)
            else:
                m = mask
            h = layer(h, m, None)
            if i == factual_layer:
                h_factual = h

        # Get logits from both layers at last position
        h_f_last = h_factual[:, -1:, :]
        h_final_last = h[:, -1:, :]

        logits_factual = mi["get_logits"](mi["norm"](h_f_last))
        logits_final = mi["get_logits"](mi["norm"](h_final_last))

        # Apply blending strategy
        logits_blended = strategy(logits_factual[0], logits_final[0])
        mx.eval(logits_blended)

        next_id = int(mx.argmax(logits_blended[0]).item())
        if next_id == eos_id:
            break
        generated.append(next_id)
        ids.append(next_id)
        del h, h_factual, logits_factual, logits_final, logits_blended

    return tokenizer.decode(generated).strip()


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, model_info, tokenizer, strategy, test_data, factual_layer):
    """Evaluate a blending strategy vs standard greedy."""
    results = []
    n_hard = 13

    for question, keywords in test_data:
        # Standard greedy
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=0.0)
        std_answer = generate(model, tokenizer, prompt=fmt, max_tokens=60,
                              verbose=False, sampler=sampler).strip()
        std_correct = check_correct(std_answer, keywords)

        # ANCHOR generation
        anc_answer = generate_with_anchor(
            model_info, tokenizer, strategy, question,
            factual_layer=factual_layer)
        anc_correct = check_correct(anc_answer, keywords)

        results.append({
            "question": question, "keywords": keywords,
            "std_answer": std_answer, "std_correct": std_correct,
            "anchor_answer": anc_answer, "anchor_correct": anc_correct,
        })

    return results


def print_results(results, label=""):
    n_hard = 13
    std_total = sum(r["std_correct"] for r in results)
    std_hard = sum(r["std_correct"] for r in results[:n_hard])
    anc_total = sum(r["anchor_correct"] for r in results)
    anc_hard = sum(r["anchor_correct"] for r in results[:n_hard])
    fixes = sum(1 for r in results if r["anchor_correct"] and not r["std_correct"])
    breaks = sum(1 for r in results if not r["anchor_correct"] and r["std_correct"])
    net = fixes - breaks

    print(f"\n  {label}")
    print(f"  Standard: {std_total}/{len(results)} total, {std_hard}/{n_hard} hard")
    print(f"  ANCHOR:   {anc_total}/{len(results)} total, {anc_hard}/{n_hard} hard")
    print(f"  Net: +{fixes} fixed, -{breaks} broken = {'+'if net>0 else ''}{net}")

    for r in results:
        s = "✓" if r["std_correct"] else "✗"
        a = "✓" if r["anchor_correct"] else "✗"
        tag = ""
        if r["anchor_correct"] and not r["std_correct"]:
            tag = " *** FIXED ***"
        elif not r["anchor_correct"] and r["std_correct"]:
            tag = " *** BROKE ***"
        print(f"  [std={s} anc={a}]{tag} {r['question'][:50]}")
        if tag:
            print(f"    std: '{r['std_answer'][:60]}'")
            print(f"    anc: '{r['anchor_answer'][:60]}'")

    return fixes, breaks, net


# ============================================================
# Main
# ============================================================

def run():
    print("=" * 70)
    print("ANCHOR v2 — Logit-Space Factual Anchoring")
    print("=" * 70)
    print()

    # ==========================================
    # Qwen2-0.5B
    # ==========================================
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    mi = setup_qwen2(model)

    factual_layer = 21  # best layer from prior experiments

    print(f"\nCollecting logits from layers {factual_layer} and {mi['n_layers']-1}...")
    train_data = collect_logits(mi, tokenizer, TRAIN_QA, factual_layer)
    print(f"  Collected {len(train_data)} training examples")

    # Also check: does the factual layer baseline beat final for training data?
    print("\n  Quick check — factual vs final layer accuracy on first 5 train examples:")
    for d in train_data[:5]:
        last_pos = -2  # position before answer's last token
        top_final = int(np.argmax(d["logits_final"][last_pos]))
        top_factual = int(np.argmax(d["logits_factual"][last_pos]))
        target = d["ids"][last_pos + 1]
        f_match = "✓" if top_final == target else "✗"
        a_match = "✓" if top_factual == target else "✗"
        print(f"    final={f_match}(tok={top_final}) factual={a_match}(tok={top_factual}) target={target}")

    strategies = [
        ("StaticBlend", StaticBlend(), 300, 0.01),
        ("EntropyGated", EntropyGatedBlend(), 300, 0.005),
        ("Contrastive", ContrastiveBlend(), 300, 0.01),
        ("EntropySwitch", EntropySwitchBlend(), 300, 0.005),
        ("FactualBoost", FactualBoost(), 300, 0.01),
    ]

    all_results = {}

    for name, strategy, epochs, lr in strategies:
        n_params = sum(p.size for _, p in nn.utils.tree_flatten(strategy.parameters()))
        print(f"\n{'='*70}")
        print(f"Strategy: {name} ({n_params} params) @ L{factual_layer}")
        print(f"{'='*70}")

        strategy = train_strategy(strategy, train_data, n_epochs=epochs, lr=lr)

        print(f"\n  Evaluating...")
        results = evaluate(model, mi, tokenizer, strategy, TEST_DATA, factual_layer)
        fixes, breaks, net = print_results(results, f"{name} @ L{factual_layer}")
        all_results[name] = {"fixes": fixes, "breaks": breaks, "net": net}

    # ==========================================
    # Summary
    # ==========================================
    print(f"\n\n{'='*70}")
    print("SUMMARY — Qwen2-0.5B, Logit-Space Anchoring")
    print(f"{'='*70}\n")
    print(f"  {'Strategy':20} {'Fix':>5} {'Brk':>5} {'Net':>5}")
    print("  " + "-" * 40)
    for name, r in all_results.items():
        net_s = f"+{r['net']}" if r['net'] > 0 else str(r['net'])
        star = " ★" if r['net'] > 0 else ""
        print(f"  {name:20} +{r['fixes']:>3}  -{r['breaks']:>3}  {net_s:>4}{star}")

    # Save results
    with open("anchor_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved anchor_v2_results.json")


if __name__ == "__main__":
    run()
