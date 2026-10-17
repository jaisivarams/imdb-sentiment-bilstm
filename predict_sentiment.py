"""
predict_sentiment.py - Classify a movie review as POSITIVE or NEGATIVE

Run train_sentiment.py first. It creates the files this script needs:
    sentiment_bilstm.keras, vocab.json, sentiment_config.json
(sentiment_utils.py must be in the same folder too.)

HOW TO USE
  1) Interactive (it asks you to type a review):
         python predict_sentiment.py
  2) One review from the command line:
         python predict_sentiment.py --text "Great acting and a moving story"
  3) Many reviews from a text file, one review per line:
         python predict_sentiment.py --file my_reviews.txt

The model was trained on English movie reviews. Use English text.
"""
import os
import sys
import json
import argparse

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.models import load_model

from sentiment_utils import clean, encode

for needed in ("sentiment_bilstm.keras", "vocab.json", "sentiment_config.json"):
    if not os.path.exists(needed):
        sys.exit(f"Missing {needed}. Run train_sentiment.py first (in this same folder).")

with open("vocab.json") as f:
    vocab = json.load(f)
with open("sentiment_config.json") as f:
    config = json.load(f)
MAX_LEN = config["max_len"]

print("Loading model...")
model = load_model("sentiment_bilstm.keras")
print("Model loaded.")


def predict_many(reviews):
    """Return a list of (label, confidence, note) for each review."""
    cleaned = [clean(r) for r in reviews]
    X = np.array([encode(c, vocab, MAX_LEN) for c in cleaned], dtype="int32")
    probs = model.predict(X, verbose=0).ravel()
    results = []
    for text, p in zip(cleaned, probs):
        p = float(p)
        label = "positive" if p > 0.5 else "negative"
        confidence = max(p, 1 - p)
        known = sum(1 for w in text.split() if w in vocab)
        if known < 3:
            note = "too few known words - result is unreliable"
        elif confidence < 0.65:
            note = "low confidence - the review may be mixed or unclear"
        else:
            note = ""
        results.append((label, confidence, note))
    return results


def show(review, result):
    label, confidence, note = result
    line = f"  -> {label.upper()}  (confidence {confidence:.1%})"
    print(line + (f"   [{note}]" if note else ""))


def interactive():
    print("\nType or paste a movie review and press Enter. Type 'q' to quit.")
    while True:
        try:
            text = input("\nReview> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            continue
        show(text, predict_many([text])[0])


def from_file(path):
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        reviews = [line.strip() for line in f if line.strip()]
    if not reviews:
        sys.exit("The file has no reviews.")
    results = predict_many(reviews)
    for review, result in zip(reviews, results):
        snippet = review if len(review) <= 70 else review[:67] + "..."
        print(f"\n{snippet}")
        show(review, result)
    positives = sum(1 for r in results if r[0] == "positive")
    print(f"\n{len(results)} reviews: {positives} positive, {len(results) - positives} negative")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify movie reviews as positive or negative.")
    parser.add_argument("--text", help="a single review to classify")
    parser.add_argument("--file", help="a text file with one review per line")
    args = parser.parse_args()

    if args.text:
        print(f"\n{args.text}")
        show(args.text, predict_many([args.text])[0])
    elif args.file:
        from_file(args.file)
    else:
        interactive()
