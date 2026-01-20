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

POSITIVE = {"good", "great", "like", "love", "excellent", "amazing", "nice"}
NEGATIVE = {"bad", "terrible", "hate", "awful", "poor", "worse", "worst"}
NEGATIONS = {
    "not", "dont", "don't", "doesnt", "doesn't",
    "didnt", "didn't", "never", "no"
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
    return GoogleTranslator(source=src, target=tgt).translate(text)

# =============================
# Sentiment analysis (EN, window=3)
# =============================
def sentiment_window3(text_en):
    clean = re.sub(r"[^a-z\s']", "", text_en.lower())
    tokens = clean.split()

    score = 0
    pos_cnt = 0
    neg_cnt = 0
    highlights = {}

    i = 0
    while i < len(tokens):
        word = tokens[i]
        norm = word.replace("'", "")

        # Negation handling
        if norm in NEGATIONS:
            flipped = False
            for j in range(1, 4):
                if i + j >= len(tokens):
                    break

                nxt = tokens[i + j]
                norm_nxt = nxt.replace("'", "")
                phrase = " ".join(tokens[i:i + j + 1])

                if norm_nxt in POSITIVE:
                    score -= 1
                    neg_cnt += 1
                    highlights[phrase] = "neg"
                    i += j + 1
                    flipped = True
                    break

                if norm_nxt in NEGATIVE:
                    score += 1
                    pos_cnt += 1
                    highlights[phrase] = "pos"
                    i += j + 1
                    flipped = True
                    break

            if not flipped:
                i += 1
            continue

        # Normal sentiment
        if norm in POSITIVE:
            score += 1
            pos_cnt += 1
            highlights[word] = "pos"

        elif norm in NEGATIVE:
            score -= 1
            neg_cnt += 1
            highlights[word] = "neg"

        i += 1

    if score > 0:
        sentiment = "Positive"
    elif score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, score, pos_cnt, neg_cnt, highlights

# =============================
# Highlight EN (no overlap)
# =============================
def render_highlight_en(text, highlights):
    html = text
    occupied = []

    for phrase in sorted(highlights.keys(), key=len, reverse=True):
        label = highlights[phrase]
        color = "#c8f7c5" if label == "pos" else "#f7c5c5"

        pattern = re.escape(phrase)
        for match in re.finditer(pattern, html, flags=re.IGNORECASE):
            start, end = match.span()

            if any(not (end <= s or start >= e) for s, e in occupied):
                continue

            html = (
                html[:start]
                + f"<span style='background:{color};padding:2px 6px;border-radius:6px'>"
                + html[start:end]
                + "</span>"
                + html[end:]
            )
            occupied.append((start, end))
            break

    return f"<div style='line-height:1.8;font-size:16px'>{html}</div>"

# =============================
# Sentiment card (FULL HEIGHT)
# =============================
def sentiment_card(sentiment, score):
    if sentiment == "Positive":
        bg, icon = "#c8f7c5", "😊"
    elif sentiment == "Negative":
        bg, icon = "#f7c5c5", "😠"
    else:
        bg, icon = "#f0f0f0", "😐"

    return f"""
    <div style="
        background:{bg};
        height:100%;
        min-height:420px;
        border-radius:22px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
    ">
        <div style="font-size:52px">{icon}</div>
        <div style="font-size:30px;font-weight:700;margin-top:10px">{sentiment}</div>
        <div style="font-size:18px;margin-top:14px">Score: {score}</div>
    </div>
    """

# =============================
# FULL SIZE STACKED BAR
# =============================
def sentiment_stacked_bar(pos, neg):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(["Summary"], [pos], width=0.5, label="Positive")
    ax.bar(["Summary"], [neg], bottom=[pos], width=0.5, label="Negative")

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Sentiment Breakdown", fontsize=14)
    ax.legend()

    ax.set_ylim(0, max(pos + neg + 1, 3))
    return fig

# =============================
# Backend
# =============================
def analyze(text, lang_choice):
    if not text or text.strip() == "":
        return "N/A", "N/A", "", "", None

    lang = detect_language(text) if LANG_MAP[lang_choice] == "auto" else LANG_MAP[lang_choice]
    translated = translate(text, lang, "en")

    sentiment, score, pos_cnt, neg_cnt, highlights = sentiment_window3(translated)

    return (
        lang.upper(),
        translated,
        sentiment_card(sentiment, score),
        render_highlight_en(translated, highlights),
        sentiment_stacked_bar(pos_cnt, neg_cnt)
    )

# =============================
# UI
# =============================
with gr.Blocks(title="AI Multilingual Comment Analyzer") as demo:
    gr.HTML("""
    <style>
    .gradio-container {max-width: 100% !important;}
    </style>
    """)

    gr.Markdown("""
    <h1 style='text-align:center'>🌍 AI Multilingual Comment Analyzer</h1>
    <p style='text-align:center;color:gray'>
    Detect → Translate → Explainable Sentiment (negation window = 3)
    </p>
    """)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Nhập comment",
                lines=5,
                placeholder="Nhập comment ở bất kỳ ngôn ngữ nào..."
            )
            lang_select = gr.Dropdown(
                choices=list(LANG_MAP.keys()),
                value="Auto detect",
                label="Ngôn ngữ đầu vào"
            )
            btn = gr.Button("Phân tích", variant="primary")

        with gr.Column():
            out_lang = gr.Textbox(label="Ngôn ngữ sử dụng", interactive=False)
            out_trans = gr.Textbox(label="Bản dịch tiếng Anh", lines=4, interactive=False)

    gr.Markdown("## 📊 Sentiment Overview")

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            chart = gr.Plot(container=True)
        with gr.Column(scale=1):
            sentiment_html = gr.HTML()

    gr.Markdown("## 🇬🇧 Highlight sentiment trên bản dịch tiếng Anh")
    highlight_en = gr.HTML()

    btn.click(
        analyze,
        inputs=[input_text, lang_select],
        outputs=[out_lang, out_trans, sentiment_html, highlight_en, chart]
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
