"""
Speculative Decoding on Apple Silicon (MLX)
============================================

Draft model: Qwen2-0.5B-Instruct (fast, 24 layers, 0.5B params)
Target model: Llama-3-8B-Instruct-4bit (slow, 32 layers, 8B params)

Algorithm:
1. Draft model generates N tokens greedily (cheap)
2. Target model runs ONE forward pass on all N draft tokens (parallel verification)
3. Accept tokens sequentially until target disagrees with draft
4. On disagreement, use target's token instead
5. Repeat from step 1

Why it works: The draft model is ~16x cheaper per token. If it guesses correctly
70% of the time, you get ~2-3x speedup because you're amortizing the expensive
target model call over multiple accepted tokens.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask
import json
import time

from mlv_benchmark import BENCHMARK, check_correct


# =============================================================
# Model setup
# =============================================================

def setup_qwen2(model):
    """Setup for Qwen2-0.5B draft model."""
    def get_logits(h):
        h_normed = model.model.norm(h)
        return model.model.embed_tokens.as_linear(h_normed)
    return {
        "embed": model.model.embed_tokens,
        "layers": model.model.layers,
        "norm": model.model.norm,
        "get_logits": get_logits,
        "get_mask": lambda h: nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1]).astype(h.dtype),
        "n_layers": len(model.model.layers),
        "name": "Qwen2-0.5B",
    }


def setup_llama(model):
    """Setup for Llama-3-8B target model."""
    def get_logits(h):
        h_normed = model.model.norm(h)
        return model.lm_head(h_normed)
    return {
        "embed": model.model.embed_tokens,
        "layers": model.model.layers,
        "norm": model.model.norm,
        "get_logits": get_logits,
        "get_mask": lambda h: create_attention_mask(h, None),
        "n_layers": len(model.model.layers),
        "name": "Llama-3-8B",
    }


# =============================================================
# Forward pass (no KV cache — full recomputation)
# =============================================================

def forward(mi, ids):
    """Full forward pass. Returns logits for ALL positions [1, seq_len, vocab]."""
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)

    for layer in mi["layers"]:
        h = layer(h, mask, None)

    logits = mi["get_logits"](h)
    mx.eval(logits)
    return logits[0]  # [seq_len, vocab]


def forward_last(mi, ids):
    """Full forward pass. Returns logits for LAST position only."""
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)

    for layer in mi["layers"]:
        h = layer(h, mask, None)

    logits = mi["get_logits"](h[:, -1:, :])
    mx.eval(logits)
    return logits[0, 0]  # [vocab]


# =============================================================
# Baseline generation (target model only, greedy)
# =============================================================

def generate_baseline(target_mi, tokenizer, prompt_ids, max_tokens=60):
    """Standard autoregressive generation with target model."""
    ids = list(prompt_ids)
    eos_id = tokenizer.eos_token_id
    generated = []

    for _ in range(max_tokens):
        logits = forward_last(target_mi, ids)
        tok = int(mx.argmax(logits).item())
        if tok == eos_id:
            break
        generated.append(tok)
        ids.append(tok)

    return generated


# =============================================================
# Speculative decoding
# =============================================================

def generate_speculative(draft_mi, target_mi, tokenizer, prompt_ids,
                         draft_n=5, max_tokens=60):
    """Speculative decoding: draft with small model, verify with big model.

    draft_n: number of tokens to draft before verification
    """
    ids = list(prompt_ids)
    eos_id = tokenizer.eos_token_id
    generated = []
    total_drafted = 0
    total_accepted = 0
    n_verifications = 0

    while len(generated) < max_tokens:
        # Step 1: Draft N tokens with the small model
        draft_ids = list(ids)
        draft_tokens = []
        for _ in range(draft_n):
            logits = forward_last(draft_mi, draft_ids)
            tok = int(mx.argmax(logits).item())
            if tok == eos_id:
                break
            draft_tokens.append(tok)
            draft_ids.append(tok)

        if len(draft_tokens) == 0:
            # Draft model wants to stop — let target decide
            logits = forward_last(target_mi, ids)
            tok = int(mx.argmax(logits).item())
            if tok == eos_id:
                break
            generated.append(tok)
            ids.append(tok)
            n_verifications += 1
            continue

        total_drafted += len(draft_tokens)

        # Step 2: Verify ALL draft tokens with target model in one pass
        # Build the sequence with all draft tokens appended
        verify_ids = ids + draft_tokens
        target_logits = forward(target_mi, verify_ids)
        # target_logits[i] = logits predicting position i+1
        # So target_logits[len(ids)-1] predicts the first generated token
        # And target_logits[len(ids)+k-1] predicts the (k+1)th generated token
        n_verifications += 1

        # Step 3: Accept tokens until disagreement
        accepted = 0
        for k, draft_tok in enumerate(draft_tokens):
            # Target's prediction for this position
            pos = len(ids) - 1 + k  # position in verify_ids that predicts next token
            target_tok = int(mx.argmax(target_logits[pos]).item())

            if target_tok == draft_tok:
                # Agreement — accept
                accepted += 1
                generated.append(draft_tok)
                ids.append(draft_tok)
                if len(generated) >= max_tokens:
                    break
            else:
                # Disagreement — use target's token, discard rest of draft
                generated.append(target_tok)
                ids.append(target_tok)
                if target_tok == eos_id:
                    generated.pop()  # don't include EOS in output
                    break
                break  # stop accepting draft tokens

        # If all draft tokens accepted, also get the next token from target
        if accepted == len(draft_tokens) and len(generated) < max_tokens:
            # Target's prediction after all draft tokens
            pos = len(ids) - 1
            if pos < len(target_logits):
                next_tok = int(mx.argmax(target_logits[pos]).item())
                if next_tok != eos_id:
                    generated.append(next_tok)
                    ids.append(next_tok)
                else:
                    break

        total_accepted += accepted

        if len(generated) >= max_tokens:
            break
        # Check if we hit EOS
        if generated and generated[-1] == eos_id:
            generated.pop()
            break

    acceptance_rate = total_accepted / max(total_drafted, 1)
    return generated, {
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "acceptance_rate": round(acceptance_rate, 3),
        "n_verifications": n_verifications,
        "tokens_generated": len(generated),
    }


# =============================================================
# Evaluation
# =============================================================

def evaluate(draft_mi, target_mi, tokenizer_target, tokenizer_draft,
             benchmark, draft_n=5):
    """Compare baseline vs speculative decoding."""
    results = []
    t_baseline = 0
    t_spec = 0

    for i, (question, keywords) in enumerate(benchmark):
        # Prepare prompts
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]

        # Baseline with target model
        fmt_target = tokenizer_target.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids_target = tokenizer_target.encode(fmt_target, add_special_tokens=True)

        t0 = time.time()
        base_tokens = generate_baseline(target_mi, tokenizer_target, ids_target)
        t1 = time.time()
        t_baseline += (t1 - t0)
        base_answer = tokenizer_target.decode(base_tokens).strip()
        if "<|eot_id|>" in base_answer:
            base_answer = base_answer.split("<|eot_id|>")[0].strip()
        base_correct = check_correct(base_answer, keywords)

        # Speculative decoding
        # Both models need to use the TARGET tokenizer for consistency
        t0 = time.time()
        spec_tokens, spec_stats = generate_speculative(
            draft_mi, target_mi, tokenizer_target, ids_target, draft_n=draft_n)
        t1 = time.time()
        t_spec += (t1 - t0)
        spec_answer = tokenizer_target.decode(spec_tokens).strip()
        if "<|eot_id|>" in spec_answer:
            spec_answer = spec_answer.split("<|eot_id|>")[0].strip()
        spec_correct = check_correct(spec_answer, keywords)

        results.append({
            "question": question,
            "base_answer": base_answer, "base_correct": base_correct,
            "spec_answer": spec_answer, "spec_correct": spec_correct,
            "spec_stats": spec_stats,
        })

        if (i + 1) % 10 == 0:
            elapsed_b = t_baseline
            elapsed_s = t_spec
            print(f"    {i+1}/{len(benchmark)} done... "
                  f"(baseline: {elapsed_b:.0f}s, spec: {elapsed_s:.0f}s)")

    # Aggregate
    base_total = sum(r["base_correct"] for r in results)
    spec_total = sum(r["spec_correct"] for r in results)
    breaks = sum(1 for r in results if not r["spec_correct"] and r["base_correct"])
    fixes = sum(1 for r in results if r["spec_correct"] and not r["base_correct"])

    total_tokens_base = sum(len(tokenizer_target.encode(
        tokenizer_target.apply_chat_template(
            [{"role": "user", "content": f"Answer briefly: {q}"}],
            tokenize=False, add_generation_prompt=True),
        add_special_tokens=True)) for q, _ in benchmark)

    total_gen_tokens = sum(r["spec_stats"]["tokens_generated"] for r in results)
    avg_acceptance = np.mean([r["spec_stats"]["acceptance_rate"] for r in results])
    avg_verifications = np.mean([r["spec_stats"]["n_verifications"] for r in results])

    speedup = t_baseline / max(t_spec, 0.01)

    return {
        "base_correct": base_total,
        "spec_correct": spec_total,
        "fixes": fixes,
        "breaks": breaks,
        "base_time": round(t_baseline, 1),
        "spec_time": round(t_spec, 1),
        "speedup": round(speedup, 2),
        "avg_acceptance_rate": round(float(avg_acceptance), 3),
        "avg_verifications": round(float(avg_verifications), 1),
        "total_gen_tokens": total_gen_tokens,
        "results": results,
    }


def run():
    print("=" * 70)
    print("Speculative Decoding — Draft: Qwen2-0.5B, Target: Llama-3-8B")
    print("=" * 70)

    # Load both models
    print("  Loading draft model (Qwen2-0.5B)...")
    draft_model, draft_tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    draft_mi = setup_qwen2(draft_model)
    print(f"    {draft_mi['name']}: {draft_mi['n_layers']} layers")

    print("  Loading target model (Llama-3-8B)...")
    target_model, target_tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    target_mi = setup_llama(target_model)
    print(f"    {target_mi['name']}: {target_mi['n_layers']} layers")

    # PROBLEM: Draft and target have DIFFERENT tokenizers!
    # Qwen2 uses one tokenizer, Llama uses another.
    # For speculative decoding to work, both models need the SAME tokenizer.
    # Solution: Use Llama's tokenizer for both, but draft with Qwen's architecture.
    # This won't work because Qwen2's embedding layer expects Qwen2 token IDs.
    #
    # Real solution: Self-speculative decoding (use Llama's early layers as draft)
    # OR use two models with the same tokenizer family.
    #
    # For now: test with self-speculative (Llama early layers as draft)

    print("\n  NOTE: Draft and target use different tokenizers!")
    print("  Switching to SELF-SPECULATIVE: Llama early layers as draft.\n")

    # Self-speculative: use first 16 layers of Llama as the "draft model"
    # and full 32 layers as the "target model"
    # This is valid because both use the same tokenizer and embeddings.

    # Test different draft depths and speculation lengths
    configs = [
        ("SelfSpec depth=16 n=3", 16, 3),
        ("SelfSpec depth=16 n=5", 16, 5),
        ("SelfSpec depth=16 n=8", 16, 8),
        ("SelfSpec depth=20 n=5", 20, 5),
        ("SelfSpec depth=24 n=5", 24, 5),
        ("SelfSpec depth=12 n=5", 12, 5),
    ]

    # Use first 50 questions for faster iteration
    benchmark_subset = BENCHMARK[:50]

    all_results = {}
    for name, draft_depth, draft_n in configs:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"  Draft: Llama layers 0-{draft_depth-1}, Speculation length: {draft_n}")
        print(f"{'='*70}")

        r = evaluate_self_speculative(
            target_mi, target_tokenizer, benchmark_subset,
            draft_depth=draft_depth, draft_n=draft_n)

        print(f"\n  Baseline: {r['base_correct']}/50 in {r['base_time']}s")
        print(f"  SelfSpec: {r['spec_correct']}/50 in {r['spec_time']}s")
        print(f"  Quality: +{r['fixes']}/-{r['breaks']}")
        print(f"  Speedup: {r['speedup']}x")
        print(f"  Acceptance rate: {r['avg_acceptance_rate']*100:.0f}%")
        print(f"  Avg verifications per question: {r['avg_verifications']}")

        if r["breaks"] > 0:
            print(f"\n  Broken ({r['breaks']}):")
            for res in r["results"]:
                if not res["spec_correct"] and res["base_correct"]:
                    print(f"    x {res['question'][:55]}")

        all_results[name] = {k: v for k, v in r.items() if k != "results"}

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY — Self-Speculative Decoding on Llama-3-8B (50 questions)")
    print(f"{'='*70}\n")
    print(f"  {'Config':30} {'Acc':>5} {'Speed':>6} {'Accept':>7} {'Brk':>4}")
    print("  " + "-" * 55)
    for name, r in all_results.items():
        acc = f"{r['spec_correct']}/50"
        spd = f"{r['speedup']}x"
        accept = f"{r['avg_acceptance_rate']*100:.0f}%"
        brk = str(r['breaks'])
        star = " ★" if r['speedup'] >= 1.3 and r['breaks'] == 0 else ""
        print(f"  {name:30} {acc:>5} {spd:>6} {accept:>7} {brk:>4}{star}")

    with open("speculative_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved speculative_results.json")


# =============================================================
# Self-Speculative Decoding
# =============================================================

def forward_partial(mi, ids, n_layers):
    """Forward pass through only the first n_layers. Returns logits."""
    ids_mx = mx.array([ids])
    h = mi["embed"](ids_mx)
    mask = mi["get_mask"](h)

    for i, layer in enumerate(mi["layers"]):
        if i >= n_layers:
            break
        h = layer(h, mask, None)

    logits = mi["get_logits"](h[:, -1:, :])
    mx.eval(logits)
    return logits[0, 0]  # [vocab]


def generate_self_speculative(mi, tokenizer, prompt_ids,
                               draft_depth, draft_n=5, max_tokens=60):
    """Self-speculative: draft with early layers, verify with full model.

    draft_depth: number of layers for drafting (e.g., 16 out of 32)
    draft_n: tokens to draft before verification
    """
    ids = list(prompt_ids)
    eos_id = tokenizer.eos_token_id
    generated = []
    total_drafted = 0
    total_accepted = 0
    n_verifications = 0
    n_layers = mi["n_layers"]

    while len(generated) < max_tokens:
        # Step 1: Draft N tokens using only first `draft_depth` layers
        draft_ids = list(ids)
        draft_tokens = []
        for _ in range(draft_n):
            logits = forward_partial(mi, draft_ids, draft_depth)
            tok = int(mx.argmax(logits).item())
            if tok == eos_id:
                break
            draft_tokens.append(tok)
            draft_ids.append(tok)

        if len(draft_tokens) == 0:
            # Draft wants to stop — use full model for final decision
            logits = forward_last(mi, ids)
            tok = int(mx.argmax(logits).item())
            if tok == eos_id:
                break
            generated.append(tok)
            ids.append(tok)
            n_verifications += 1
            continue

        total_drafted += len(draft_tokens)

        # Step 2: Verify with full model (all layers) in one pass
        verify_ids = ids + draft_tokens
        target_logits = forward(mi, verify_ids)
        n_verifications += 1

        # Step 3: Accept until disagreement
        accepted = 0
        for k, draft_tok in enumerate(draft_tokens):
            pos = len(ids) - 1 + k
            target_tok = int(mx.argmax(target_logits[pos]).item())

            if target_tok == draft_tok:
                accepted += 1
                generated.append(draft_tok)
                ids.append(draft_tok)
                if len(generated) >= max_tokens:
                    break
            else:
                generated.append(target_tok)
                ids.append(target_tok)
                if target_tok == eos_id:
                    generated.pop()
                    return generated, {
                        "total_drafted": total_drafted,
                        "total_accepted": total_accepted + accepted,
                        "acceptance_rate": round((total_accepted + accepted) / max(total_drafted, 1), 3),
                        "n_verifications": n_verifications,
                        "tokens_generated": len(generated),
                    }
                break

        total_accepted += accepted

        # If all accepted, get bonus token
        if accepted == len(draft_tokens) and len(generated) < max_tokens:
            pos = len(ids) - 1
            if pos < len(target_logits):
                next_tok = int(mx.argmax(target_logits[pos]).item())
                if next_tok != eos_id:
                    generated.append(next_tok)
                    ids.append(next_tok)
                else:
                    break

    acceptance_rate = total_accepted / max(total_drafted, 1)
    return generated, {
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "acceptance_rate": round(acceptance_rate, 3),
        "n_verifications": n_verifications,
        "tokens_generated": len(generated),
    }


def evaluate_self_speculative(mi, tokenizer, benchmark,
                               draft_depth, draft_n=5):
    """Evaluate self-speculative decoding."""
    results = []
    t_baseline = 0
    t_spec = 0

    for i, (question, keywords) in enumerate(benchmark):
        messages = [{"role": "user", "content": f"Answer briefly: {question}"}]
        fmt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(fmt, add_special_tokens=True)

        # Baseline
        t0 = time.time()
        base_tokens = generate_baseline(mi, tokenizer, ids)
        t1 = time.time()
        t_baseline += (t1 - t0)
        base_answer = tokenizer.decode(base_tokens).strip()
        if "<|eot_id|>" in base_answer:
            base_answer = base_answer.split("<|eot_id|>")[0].strip()
        base_correct = check_correct(base_answer, keywords)

        # Self-speculative
        t0 = time.time()
        spec_tokens, spec_stats = generate_self_speculative(
            mi, tokenizer, ids, draft_depth=draft_depth, draft_n=draft_n)
        t1 = time.time()
        t_spec += (t1 - t0)
        spec_answer = tokenizer.decode(spec_tokens).strip()
        if "<|eot_id|>" in spec_answer:
            spec_answer = spec_answer.split("<|eot_id|>")[0].strip()
        spec_correct = check_correct(spec_answer, keywords)

        results.append({
            "question": question,
            "base_answer": base_answer, "base_correct": base_correct,
            "spec_answer": spec_answer, "spec_correct": spec_correct,
            "spec_stats": spec_stats,
        })

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(benchmark)} done... "
                  f"(base: {t_baseline:.0f}s, spec: {t_spec:.0f}s)")

    base_total = sum(r["base_correct"] for r in results)
    spec_total = sum(r["spec_correct"] for r in results)
    breaks = sum(1 for r in results if not r["spec_correct"] and r["base_correct"])
    fixes = sum(1 for r in results if r["spec_correct"] and not r["base_correct"])

    avg_acceptance = np.mean([r["spec_stats"]["acceptance_rate"] for r in results])
    avg_verifications = np.mean([r["spec_stats"]["n_verifications"] for r in results])

    speedup = t_baseline / max(t_spec, 0.01)

    return {
        "base_correct": base_total,
        "spec_correct": spec_total,
        "fixes": fixes,
        "breaks": breaks,
        "base_time": round(t_baseline, 1),
        "spec_time": round(t_spec, 1),
        "speedup": round(speedup, 2),
        "avg_acceptance_rate": round(float(avg_acceptance), 3),
        "avg_verifications": round(float(avg_verifications), 1),
    }


if __name__ == "__main__":
    run()
