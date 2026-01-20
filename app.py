import gradio as gr
import re
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# =============================
# Config
# =============================
DetectorFactory.seed = 0

LANG_MAP = {
    "Auto detect": "auto",
    "Vietnamese": "vi",
    "German": "de",
    "French": "fr",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh"
}

POSITIVE = {
    # General positive
    "good", "great", "excellent", "amazing", "awesome", "perfect",
    "nice", "love", "like", "enjoy", "satisfied",

    # Quality / experience
    "comfortable", "clean", "cozy", "quiet", "spacious",
    "modern", "beautiful", "convenient", "reliable",

    # Service / people
    "friendly", "helpful", "professional", "polite", "attentive",

    # Food / restaurant / hotel
    "tasty", "delicious", "fresh", "yummy",

    # Value
    "affordable", "reasonable", "worth", "value",

    # Location / travel
    "central", "conveniently", "accessible",

    # Stay / usage
    "pleasant", "smooth", "easy", "fast", "efficient"
}

NEGATIVE = {
    # General negative
    "bad", "terrible", "awful", "poor", "worst", "hate",

    # Quality / condition
    "dirty", "old", "outdated", "broken", "damaged",
    "uncomfortable", "noisy", "crowded", "small",

    # Service
    "rude", "unhelpful", "slow", "careless", "unprofessional",

    # Food
    "tasteless", "cold", "stale", "bland",

    # Value / price
    "expensive", "overpriced", "costly",

    # Technical / facilities
    "slow", "unstable", "laggy", "disconnect",

    # Experience
    "disappointed", "frustrating", "annoying", "problematic"
}

NEGATIONS = {
    "not", "no", "never",
    "dont", "don't",
    "doesnt", "doesn't",
    "didnt", "didn't",
    "cannot", "can't",
    "hardly", "rarely"
}


# =============================
# Utils
# =============================
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate(text, src, tgt):
    try:
        if src in ["auto", "unknown", "zh", "zh-cn", "zh-tw"]:
            return GoogleTranslator(target=tgt).translate(text)
        else:
            return GoogleTranslator(source=src, target=tgt).translate(text)
    except Exception:
        return "[Translation error]"


# =============================
# Sentiment analysis (window=3)
# =============================
def sentiment_window3(text_en):
    clean = re.sub(r"[^a-z\s']", "", text_en.lower())
    tokens = clean.split()

    score = 0
    pos_cnt = 0
    neg_cnt = 0
    highlights = {}
    used_indexes = set()   # 👈 THÊM

    i = 0
    while i < len(tokens):
        word = tokens[i]
        norm = word.replace("'", "")

        # Negation handling
        if norm in NEGATIONS:
            for j in range(1, 4):
                if i + j >= len(tokens):
                    break

                nxt = tokens[i + j].replace("'", "")
                phrase = " ".join(tokens[i:i + j + 1])

                if nxt in POSITIVE:
                    score -= 1
                    neg_cnt += 1
                    highlights[phrase] = "neg"

                    # 👇 ĐÁNH DẤU TOKEN ĐÃ DÙNG
                    for k in range(i, i + j + 1):
                        used_indexes.add(k)

                    i += j + 1
                    break

                if nxt in NEGATIVE:
                    score += 1
                    pos_cnt += 1
                    highlights[phrase] = "pos"

                    for k in range(i, i + j + 1):
                        used_indexes.add(k)

                    i += j + 1
                    break
            else:
                i += 1
            continue

        # Normal sentiment (CHỈ NẾU CHƯA BỊ CONSUME)
        if i not in used_indexes:
            if norm in POSITIVE:
                score += 1
                pos_cnt += 1
                highlights[word] = "pos"

            elif norm in NEGATIVE:
                score -= 1
                neg_cnt += 1
                highlights[word] = "neg"

        i += 1

    sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
    return sentiment, score, pos_cnt, neg_cnt, highlights


# =============================
# Highlight English text
# =============================
def render_highlight_en(text, highlights):
    html = text
    spans = []  # lưu (start, end, label)

    # 1️⃣ Tìm tất cả match, phrase dài ưu tiên trước
    for phrase in sorted(highlights, key=len, reverse=True):
        label = highlights[phrase]
        for m in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
            s, e = m.span()

            # ❌ nếu span này nằm TRONG span đã có → bỏ
            if any(s >= ps and e <= pe for ps, pe, _ in spans):
                continue

            spans.append((s, e, label))
            break  # mỗi phrase chỉ highlight 1 lần

    # 2️⃣ Render từ trái sang phải
    spans.sort(key=lambda x: x[0])

    result = ""
    last = 0
    for s, e, label in spans:
        color = "#c8f7c5" if label == "pos" else "#f7c5c5"
        result += text[last:s]
        result += (
            f"<span style='background:{color};padding:4px 6px;border-radius:6px'>"
            f"{text[s:e]}"
            "</span>"
        )
        last = e

    result += text[last:]

    return f"<div style='font-size:16px;line-height:1.8'>{result}</div>"


# =============================
# Sentiment card (TOP)
# =============================
def sentiment_card(sentiment, score):
    if sentiment == "Positive":
        bg = "#d1fae5"   # xanh nhạt
        icon = "😊"
        symbol = "✔"
    elif sentiment == "Negative":
        bg = "#fee2e2"   # đỏ nhạt
        icon = "😠"
        symbol = "✖"
    else:
        bg = "#e5e7eb"   # xám
        icon = "😐"
        symbol = "•"

    return f"""
    <div style="
        background:{bg};
        border-radius:20px;
        padding:24px;
        margin-bottom:20px;
        text-align:center;
        color:#000;
        opacity:1 !important;
    ">
        <div style="font-size:46px; line-height:1;">
            {icon}
        </div>

        <div style="
            font-size:32px;
            font-weight:800;
            margin-top:6px;
            color:#000;
        ">
            {symbol} {sentiment}
        </div>

        <div style="
            font-size:18px;
            margin-top:10px;
            color:#000;
        ">
            Score: {score}
        </div>
    </div>
    """



# =============================
# Chart (compact)
# =============================
def sentiment_chart(pos, neg):
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.bar(["Summary"], [pos], width=0.4, label="Positive")
    ax.bar(["Summary"], [neg], bottom=[pos], width=0.4, label="Negative")

    ax.set_title("Sentiment Signals Overview")
    ax.set_ylabel("Signal Count (rule-based)")
    ax.legend()
    ax.set_ylim(0, max(pos + neg + 1, 3))

    return fig

# =============================
# Backend
# =============================
def analyze(text, lang_choice):
    lang = detect_language(text) if LANG_MAP[lang_choice] == "auto" else LANG_MAP[lang_choice]
    en = translate(text, lang, "en")

    sentiment, score, pos, neg, highlights = sentiment_window3(en)

    return (
        lang.upper(),
        en,
        sentiment_card(sentiment, score),
        sentiment_chart(pos, neg),
        render_highlight_en(en, highlights),
    )

# =============================
# UI
# =============================
with gr.Blocks(title="Explainable Multilingual Sentiment Analyzer") as demo:
    gr.HTML("""
    <style>
        .gradio-container { max-width: 100% !important; }
    </style>
    """)

    gr.Markdown("""
    <h1 style='text-align:center'>🌍 Explainable Multilingual Sentiment Analyzer</h1>
    <p style='text-align:center;color:gray'>
    Rule-based • Negation-aware • Interpretable NLP
    </p>
    """)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Nhập comment", lines=5)
            lang_select = gr.Dropdown(
                choices=list(LANG_MAP.keys()),
                value="Auto detect",
                label="Ngôn ngữ đầu vào"
            )
            btn = gr.Button("Phân tích", variant="primary")

        with gr.Column():
            out_lang = gr.Textbox(label="Ngôn ngữ")
            out_trans = gr.Textbox(label="English translation", lines=4)

    gr.Markdown("## 🧠 Sentiment Summary")
    sentiment_html = gr.HTML()

    gr.Markdown("## 🇬🇧 Highlight sentiment (English)")
    highlight_en = gr.HTML()

    gr.Markdown("## 📊 Sentiment Overview")
    chart = gr.Plot()



    btn.click(
        analyze,
        [input_text, lang_select],
        [out_lang, out_trans, sentiment_html, chart, highlight_en]
    )

    gr.Examples(
        examples=[
            ["この商品はデザインがとても良くて使いやすいし、品質も良いと思いますが、バッテリーの持ちはあまり良くなく、価格も少し高いです。ただし、全体的には満足しています。", "Auto detect"],
            ["I don't like this product but the quality is good and the price is not very good", "Auto detect"],
            ["Tôi thích sản phẩm này nhưng pin không tốt lắm", "Auto detect"]
        ],
        inputs=[input_text, lang_select]
    )

if __name__ == "__main__":
    demo.launch()
