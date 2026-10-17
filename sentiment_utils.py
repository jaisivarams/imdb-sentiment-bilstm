"""
sentiment_utils.py - text processing shared by train_sentiment.py and predict_sentiment.py

Keeping these functions in ONE file guarantees that training and prediction
clean and encode text in exactly the same way.
"""
import re
from collections import Counter

PAD, OOV = 0, 1   # index 0 = padding, index 1 = unknown word


def clean(text):
    """Lowercase, remove HTML tags, keep letters, apostrophes and the symbols ! ?"""
    text = re.sub(r"<.*?>", " ", text)             # HTML tags such as <br />
    text = text.lower()
    text = re.sub(r"[^a-z\s'!?]", " ", text)       # drop digits and other punctuation
    text = re.sub(r"([!?])", r" \1 ", text)        # make ! and ? separate tokens
    return re.sub(r"\s+", " ", text).strip()


def build_vocab(texts, max_words):
    """Map the (max_words - 2) most frequent words to indexes 2, 3, 4, ..."""
    counts = Counter(word for text in texts for word in text.split())
    common = [word for word, _ in counts.most_common(max_words - 2)]
    return {word: i + 2 for i, word in enumerate(common)}


def encode(text, vocab, max_len):
    """
    Turn cleaned text into exactly max_len word indexes.
    - Long reviews: keep the LAST max_len words (the verdict is usually at the end).
    - Short reviews: pad with zeros at the end (the model ignores these via masking).
    """
    ids = [vocab.get(word, OOV) for word in text.split()][-max_len:]
    return ids + [PAD] * (max_len - len(ids))
