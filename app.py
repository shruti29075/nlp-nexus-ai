import random
import re

import gradio as gr
from datasets import load_dataset
from transformers import pipeline


# Use compact, reliable models that run well on Hugging Face Spaces.
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
spam_model = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
)
sms_dataset = load_dataset("ucirvine/sms_spam")

SPAM_PATTERNS = [
    r"\bwin\b",
    r"\bwon\b",
    r"\bprize\b",
    r"\bfree\b",
    r"\bclaim\b",
    r"\burgent\b",
    r"\bcash\b",
    r"\breward\b",
    r"\bselected\b",
    r"\breply\b",
    r"\bcall now\b",
    r"\bclick\b",
    r"\bcongratulations\b",
]

CUSTOM_CSS = """
.app-shell {
    max-width: 1060px;
    margin: 0 auto;
}
.hero {
    background: linear-gradient(135deg, #0f172a, #1e293b 45%, #14532d);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
}
.hero h1 {
    margin: 0 0 6px 0;
    font-size: 34px;
}
.hero p {
    margin: 0;
    color: #cbd5e1;
}
.pill-row {
    margin-top: 12px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.pill {
    border: 1px solid #475569;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 12px;
    color: #e2e8f0;
}
"""


def _decode_spam_label(raw_label: str) -> str:
    raw = str(raw_label)
    normalized = raw.upper().strip()

    # Try to decode LABEL_X using the model's own config first.
    if normalized.startswith("LABEL_"):
        try:
            label_id = int(normalized.split("_")[-1])
            decoded = str(spam_model.model.config.id2label.get(label_id, raw))
            decoded_lower = decoded.lower()
            if "spam" in decoded_lower:
                return "Spam"
            if "ham" in decoded_lower or "not" in decoded_lower:
                return "Not Spam"
        except Exception:
            pass

    raw_lower = raw.lower()
    if "spam" in raw_lower:
        return "Spam"
    if "ham" in raw_lower or "not" in raw_lower:
        return "Not Spam"

    # Known label convention for this model family.
    if normalized == "LABEL_1":
        return "Spam"
    if normalized == "LABEL_0":
        return "Not Spam"

    return raw


def _is_rule_based_spam(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in SPAM_PATTERNS)


def _rule_hits(text: str) -> list[str]:
    lowered = text.lower()
    hits = []
    for pattern in SPAM_PATTERNS:
        token = pattern.replace(r"\b", "")
        if re.search(pattern, lowered):
            hits.append(token)
    # Keep output compact and deterministic.
    return sorted(set(hits))


def analyze_sentiment(text: str) -> str:
    text = text.strip()
    if not text:
        return "Please enter text."

    pred = sentiment_model(text)[0]
    return f"{pred['label']} (confidence: {pred['score']:.2f})"


def detect_spam(text: str) -> str:
    text = text.strip()
    if not text:
        return "Please enter a message."

    if _is_rule_based_spam(text):
        return "Spam (confidence: 1.00, rule-based)"

    pred = spam_model(text)[0]
    label = _decode_spam_label(pred["label"])
    return f"{label} (confidence: {pred['score']:.2f})"


def detect_spam_with_reason(text: str) -> tuple[str, str]:
    text = text.strip()
    if not text:
        return "Please enter a message.", "No input provided."

    hits = _rule_hits(text)
    if hits:
        reason = f"Rule-based trigger matched: {', '.join(hits)}"
        return "Spam (confidence: 1.00, rule-based)", reason

    pred = spam_model(text)[0]
    label = _decode_spam_label(pred["label"])
    reason = f"Model output label: {pred['label']} | confidence: {pred['score']:.2f}"
    return f"{label} (confidence: {pred['score']:.2f})", reason


def show_random_sms_sample() -> str:
    row = random.choice(sms_dataset["train"])
    return f"label={row['label']} | sms={row['sms']}"


def show_random_samples_table(sample_count: int):
    count = max(1, min(8, int(sample_count)))
    rows = random.sample(list(sms_dataset["train"]), count)
    table = []
    for row in rows:
        label_name = "Ham" if row["label"] == 0 else "Spam"
        table.append([label_name, row["sms"]])
    return table


def get_label_distribution() -> str:
    counts = {0: 0, 1: 0}
    for row in sms_dataset["train"]:
        counts[row["label"]] += 1
    total = counts[0] + counts[1]
    ham_pct = (counts[0] / total) * 100
    spam_pct = (counts[1] / total) * 100
    return (
        f"Ham (0): {counts[0]} ({ham_pct:.1f}%)\\n"
        f"Spam (1): {counts[1]} ({spam_pct:.1f}%)"
    )


with gr.Blocks(title="NLP Nexus AI", css=CUSTOM_CSS) as app:
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(
            """
            <div class='hero'>
              <h1>NLP Nexus AI</h1>
              <p>A compact AI lab for sentiment, spam intelligence, and live dataset exploration.</p>
              <div class='pill-row'>
                <span class='pill'>Hugging Face Models</span>
                <span class='pill'>SMS Spam Dataset</span>
                <span class='pill'>Gradio Space</span>
              </div>
            </div>
            """
        )

        with gr.Tab("Sentiment"):
            sentiment_input = gr.Textbox(
                label="Input text",
                placeholder="Type a review, message, or sentence",
                lines=4,
            )
            gr.Examples(
                examples=[
                    ["I absolutely loved this app. Clean UI and fast response!"],
                    ["This experience was frustrating and disappointing."],
                ],
                inputs=sentiment_input,
                label="Quick examples",
            )
            sentiment_output = gr.Textbox(label="Prediction")
            sentiment_btn = gr.Button("Analyze Sentiment", variant="primary")
            sentiment_btn.click(analyze_sentiment, inputs=sentiment_input, outputs=sentiment_output)

        with gr.Tab("Spam Detection"):
            spam_input = gr.Textbox(
                label="SMS text",
                placeholder="Example: Congratulations! You won a free ticket...",
                lines=4,
            )
            gr.Examples(
                examples=[
                    ["Congratulations! You have won a 1000 cash prize. Reply WIN to claim now."],
                    ["Hi, I will reach by 7 pm. Please keep dinner ready."],
                ],
                inputs=spam_input,
                label="Quick examples",
            )
            spam_output = gr.Textbox(label="Prediction")
            spam_reason = gr.Textbox(label="Why this result?")
            spam_btn = gr.Button("Detect Spam", variant="primary")
            spam_btn.click(detect_spam_with_reason, inputs=spam_input, outputs=[spam_output, spam_reason])

        with gr.Tab("Dataset Explorer"):
            sample_output = gr.Textbox(label="Random training sample", lines=4)
            sample_btn = gr.Button("Show Random Sample", variant="secondary")
            sample_btn.click(show_random_sms_sample, outputs=sample_output)

            distribution_output = gr.Textbox(label="Label distribution", lines=3)
            distribution_btn = gr.Button("Show Distribution", variant="secondary")
            distribution_btn.click(get_label_distribution, outputs=distribution_output)

            gr.Markdown("### Sample Table")
            sample_count = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Number of random samples")
            sample_table = gr.Dataframe(headers=["Label", "SMS"], datatype=["str", "str"], wrap=True)
            sample_table_btn = gr.Button("Generate Sample Table")
            sample_table_btn.click(show_random_samples_table, inputs=sample_count, outputs=sample_table)


if __name__ == "__main__":
    app.launch()