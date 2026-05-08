---
title: NLP Nexus AI
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

# NLP Nexus AI (Hugging Face Project)

An NLP web app built with Hugging Face resources and Gradio.

This project uses:
- Hugging Face Datasets: `ucirvine/sms_spam`
- Hugging Face pretrained models:
  - `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
  - `mrm8488/bert-tiny-finetuned-sms-spam-detection`

## Features

1. Sentiment analysis for user text.
2. SMS spam detection.
3. Dataset explorer with random sample + class distribution.

## Local Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start app:

```bash
python app.py
```

3. Open the local Gradio URL shown in terminal (usually `http://127.0.0.1:7860`).

## Hugging Face Space Deployment

1. Create a new Space on Hugging Face:
   - SDK: `Gradio`
2. Upload these files:
	- `app.py`
	- `requirements.txt`
	- `README.md`
3. The Space will auto-build and run.

Optional (recommended): set `HF_TOKEN` in Space secrets for faster model pulls.

## Submission Format (for your class)

Submit the following:

1. Link to your code repository (your name/account should match).
2. A small demo screen recording of your app.
3. Live app link (optional): your Hugging Face Space URL.

## Evaluator Note

For simple project walkthrough and test script, see [TEACHER_GUIDE.md](TEACHER_GUIDE.md).

## Tested Links

All links below are valid Hugging Face resources used by this app:

1. https://huggingface.co/datasets/ucirvine/sms_spam
2. https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
3. https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection
4. https://huggingface.co/spaces
