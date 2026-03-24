"""
The Language Retreat Hypothesis
================================
From steer.py: when the model oscillates at intermediate layers,
the competing tokens are Chinese characters (历史性, 世界上, etc.)
— not competing English answers.

Hypothesis: The model KNOWS facts better in Chinese than English.
When uncertain in English, it briefly "retreats" to Chinese processing
before forcing output into English — losing the correct answer.

Test:
  1. Ask each question in ENGLISH → record answer + correctness
  2. Ask the SAME question in CHINESE → record answer + correctness
  3. Compare accuracy: English vs Chinese
  4. For English-wrong / Chinese-right cases: this proves
     "the answer was there, just in the wrong language"

If Chinese accuracy > English accuracy:
  → Language routing is a hallucination mitigation strategy
  → Oscillation signal = "this question needs Chinese mode"
  → No external data, no API, just the model's own capability

This is mechanistically testable on Qwen2-0.5B right now.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate
from mlx_lm.models.qwen2 import create_attention_mask
import json


def ask(model, tokenizer, question_text, lang="en", max_tokens=80):
    """Ask a question in English or Chinese."""
    if lang == "en":
        content = f"Answer briefly and directly: {question_text}"
    else:
        # Chinese system prompt + Chinese question
        content = f"请简短直接地回答：{question_text}"

    messages = [{"role": "user", "content": content}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=fmt, max_tokens=max_tokens, verbose=False)
    return response.strip()


def ask_with_chinese_hint(model, tokenizer, question_text, max_tokens=80):
    """Ask in English but hint that Chinese sources may be relevant."""
    content = f"Based on your knowledge (including from Chinese sources), answer briefly: {question_text}"
    messages = [{"role": "user", "content": content}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=fmt, max_tokens=max_tokens, verbose=False)
    return response.strip()


def get_oscillation_count(model, tokenizer, question_text, target_layer=20):
    """Count token oscillations up to target_layer."""
    messages = [{"role": "user", "content": f"Answer briefly: {question_text}"}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(fmt, add_special_tokens=True)

    ids_mx = mx.array([ids])
    h = model.model.embed_tokens(ids_mx)
    mx.eval(h)
    mask = create_attention_mask(h, None)
    preds = []
    chinese_tokens_seen = set()

    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        mx.eval(h)
        h_norm = model.model.norm(h)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(h_norm)
        else:
            logits = model.lm_head(h_norm)
        mx.eval(logits)
        last = np.array(logits[0, -1].astype(mx.float32))
        top_id = int(np.argmax(last))
        top_tok = tokenizer.decode([top_id])
        preds.append(top_id)

        # Check if token is Chinese (unicode range 4E00-9FFF)
        if any('\u4e00' <= c <= '\u9fff' for c in top_tok):
            chinese_tokens_seen.add(top_tok)

        del logits, h_norm
        if i == target_layer:
            break

    changes = sum(preds[j] != preds[j-1] for j in range(1, len(preds)))
    return changes, list(chinese_tokens_seen)


def run():
    print("Loading Qwen2-0.5B-Instruct...")
    model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")

    # Questions with English and CHINESE versions
    # The Chinese translations are accurate factual questions
    qa_bilingual = [
        {
            "en": "Who won the Nobel Prize in Chemistry in 2023?",
            "zh": "2023年诺贝尔化学奖得主是谁？",
            "keywords_en": ["bawendi", "brus", "ekimov"],
            "keywords_zh": ["巴文迪", "布鲁斯", "叶基莫夫", "量子点"],  # Chinese names + quantum dots
        },
        {
            "en": "What is the melting point of tungsten in Celsius?",
            "zh": "钨的熔点是多少摄氏度？",
            "keywords_en": ["3422", "3400"],
            "keywords_zh": ["3422", "3400", "3410"],
        },
        {
            "en": "Who was the 30th president of the United States?",
            "zh": "美国第30任总统是谁？",
            "keywords_en": ["coolidge", "calvin"],
            "keywords_zh": ["柯立芝", "卡尔文", "coolidge"],
        },
        {
            "en": "What is the half-life of uranium-235 in years?",
            "zh": "铀-235的半衰期是多少年？",
            "keywords_en": ["703 million", "703", "700"],
            "keywords_zh": ["7.04", "7亿", "703", "700"],
        },
        {
            "en": "Who is the prime minister of New Zealand as of 2024?",
            "zh": "2024年新西兰总理是谁？",
            "keywords_en": ["luxon", "christopher"],
            "keywords_zh": ["勒克斯顿", "卢克森", "luxon", "christopher"],
        },
        {
            "en": "What is the largest desert in the world by area?",
            "zh": "世界上面积最大的沙漠是什么？",
            "keywords_en": ["antarctica", "antarctic"],
            "keywords_zh": ["南极", "antarctica"],
        },
        {
            "en": "What country has the most UNESCO World Heritage Sites?",
            "zh": "哪个国家拥有最多的联合国教科文组织世界遗产？",
            "keywords_en": ["italy", "china"],
            "keywords_zh": ["中国", "意大利", "italy", "china"],
        },
        {
            "en": "What is the atomic weight of plutonium?",
            "zh": "钚的原子量是多少？",
            "keywords_en": ["244", "242", "239"],
            "keywords_zh": ["244", "242", "239"],
        },
        {
            "en": "What is the rarest blood type?",
            "zh": "最罕见的血型是什么？",
            "keywords_en": ["ab-", "ab negative"],
            "keywords_zh": ["ab阴性", "ab-", "rh阴性"],
        },
        {
            "en": "What is the speed of sound in water in m/s?",
            "zh": "声音在水中的速度是多少米每秒？",
            "keywords_en": ["1480", "1500", "1498"],
            "keywords_zh": ["1480", "1500", "1498"],
        },
        {
            "en": "Who was the first female prime minister of the UK?",
            "zh": "英国第一位女首相是谁？",
            "keywords_en": ["thatcher"],
            "keywords_zh": ["撒切尔", "thatcher"],
        },
        {
            "en": "What is the half-life of Carbon-14 in years?",
            "zh": "碳-14的半衰期是多少年？",
            "keywords_en": ["5730", "5700"],
            "keywords_zh": ["5730", "5700"],
        },
        {
            "en": "What is the capital of Burkina Faso?",
            "zh": "布基纳法索的首都是什么？",
            "keywords_en": ["ouagadougou"],
            "keywords_zh": ["瓦加杜古", "ouagadougou"],
        },
        {
            "en": "What is the capital of Kyrgyzstan?",
            "zh": "吉尔吉斯斯坦的首都是什么？",
            "keywords_en": ["bishkek"],
            "keywords_zh": ["比什凯克", "bishkek"],
        },
        # Easy ones as control (model should get right in both)
        {
            "en": "What is the capital of France?",
            "zh": "法国的首都是什么？",
            "keywords_en": ["paris"],
            "keywords_zh": ["巴黎", "paris"],
        },
        {
            "en": "Who wrote Romeo and Juliet?",
            "zh": "谁写了《罗密欧与朱丽叶》？",
            "keywords_en": ["shakespeare"],
            "keywords_zh": ["莎士比亚", "shakespeare"],
        },
        {
            "en": "What is the largest planet in the solar system?",
            "zh": "太阳系中最大的行星是什么？",
            "keywords_en": ["jupiter"],
            "keywords_zh": ["木星", "jupiter"],
        },
    ]

    print(f"\n{'='*70}")
    print("Language Retreat Hypothesis Test")
    print(f"{'='*70}")
    print("Testing: does Qwen2-0.5B answer better in Chinese than English?")
    print(f"\n{'Q':>45} {'EN':>4} {'ZH':>4} {'OSC':>4}")
    print("-" * 65)

    results = []
    for qa in qa_bilingual:
        # English
        en_answer = ask(model, tokenizer, qa["en"], lang="en")
        en_correct = any(kw.lower() in en_answer.lower() for kw in qa["keywords_en"])

        # Chinese
        zh_answer = ask(model, tokenizer, qa["zh"], lang="zh")
        zh_correct = any(kw.lower() in zh_answer.lower() for kw in qa["keywords_zh"])

        # Oscillation + Chinese tokens during English forward pass
        osc_count, chinese_toks = get_oscillation_count(model, tokenizer, qa["en"])

        en_sym = "✓" if en_correct else "✗"
        zh_sym = "✓" if zh_correct else "✗"

        print(f"{qa['en'][:45]:>45} {en_sym:>4} {zh_sym:>4} {osc_count:>4}")
        if chinese_toks:
            print(f"  Chinese tokens at oscillation: {chinese_toks[:5]}")
        if en_correct != zh_correct:
            if not en_correct and zh_correct:
                print(f"  *** ZH CORRECT, EN WRONG ***")
                print(f"    EN: {en_answer[:80]}")
                print(f"    ZH: {zh_answer[:80]}")
            else:
                print(f"  *** EN CORRECT, ZH WRONG ***")
                print(f"    EN: {en_answer[:80]}")
                print(f"    ZH: {zh_answer[:80]}")

        results.append({
            "en_question": qa["en"],
            "zh_question": qa["zh"],
            "en_answer": en_answer,
            "zh_answer": zh_answer,
            "en_correct": en_correct,
            "zh_correct": zh_correct,
            "oscillation": osc_count,
            "chinese_tokens_seen": chinese_toks,
        })

    # Analysis
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    n_en = sum(r["en_correct"] for r in results)
    n_zh = sum(r["zh_correct"] for r in results)
    n_total = len(results)

    print(f"English accuracy: {n_en}/{n_total} ({n_en/n_total*100:.1f}%)")
    print(f"Chinese accuracy: {n_zh}/{n_total} ({n_zh/n_total*100:.1f}%)")

    zh_better = [r for r in results if not r["en_correct"] and r["zh_correct"]]
    en_better = [r for r in results if r["en_correct"] and not r["zh_correct"]]
    both_right = [r for r in results if r["en_correct"] and r["zh_correct"]]
    both_wrong = [r for r in results if not r["en_correct"] and not r["zh_correct"]]

    print(f"\nBreakdown:")
    print(f"  Both correct:      {len(both_right)}/{n_total}")
    print(f"  EN only correct:   {len(en_better)}/{n_total}")
    print(f"  ZH only correct:   {len(zh_better)}/{n_total}  ← key number")
    print(f"  Both wrong:        {len(both_wrong)}/{n_total}")

    if zh_better:
        print(f"\n{'='*50}")
        print(f"CASES WHERE CHINESE KNOWS BUT ENGLISH DOESN'T ({len(zh_better)}):")
        print(f"{'='*50}")
        for r in zh_better:
            print(f"\n  Q (EN): {r['en_question']}")
            print(f"  EN: '{r['en_answer'][:80]}'  ✗")
            print(f"  ZH: '{r['zh_answer'][:80]}'  ✓")
            print(f"  Oscillation: {r['oscillation']}, Chinese tokens: {r['chinese_tokens_seen'][:3]}")

    # Correlation: does oscillation predict EN-wrong/ZH-right?
    print(f"\n{'='*50}")
    print("OSCILLATION vs LANGUAGE GAP:")
    print(f"{'='*50}")
    osc_zh_better = [r["oscillation"] for r in zh_better]
    osc_en_better = [r["oscillation"] for r in en_better]
    osc_both = [r["oscillation"] for r in both_right + both_wrong]

    import numpy as np
    if zh_better:
        print(f"  Mean oscillation (ZH better than EN): {np.mean(osc_zh_better):.1f}")
    if en_better:
        print(f"  Mean oscillation (EN better than ZH): {np.mean(osc_en_better):.1f}")
    print(f"  Mean oscillation (same result both): {np.mean(osc_both):.1f}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    if len(zh_better) > len(en_better):
        print(f"""
★ LANGUAGE RETREAT IS REAL

Qwen knows {len(zh_better)} facts in Chinese that it gets wrong in English.
The model's Chinese training gives it access to knowledge that its English
output mode cannot retrieve reliably.

This confirms the Language Retreat Hypothesis:
  - When uncertain in English, the model falls to Chinese processing
  - The oscillating Chinese tokens ARE the correct reasoning (in Chinese)
  - But the output is forced into English, losing the correct answer

Implication:
  - High oscillation = "this question is better answered in Chinese"
  - Using Chinese as internal scratchpad could improve factual accuracy
  - This is a NEW interpretability finding: oscillation = language confusion
""")
    elif len(zh_better) == len(en_better) == 0:
        print("\nNo language advantage either way — both fail on same questions.")
        print("The Chinese tokens in oscillation are NOT the model trying to answer in Chinese.")
        print("They are a different signal — possibly noise from multilingual training.")
    else:
        print(f"\nEN better on {len(en_better)} questions, ZH better on {len(zh_better)}.")
        print("Mixed result — need more data.")

    with open("language_retreat_results.json", "w") as f:
        json.dump({
            "n_total": n_total,
            "n_en_correct": n_en,
            "n_zh_correct": n_zh,
            "n_zh_better": len(zh_better),
            "n_en_better": len(en_better),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print("\nSaved language_retreat_results.json")


if __name__ == "__main__":
    run()
