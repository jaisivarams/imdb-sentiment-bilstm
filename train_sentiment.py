"""
train_sentiment.py - IMDB movie review sentiment classifier (Bidirectional LSTM)

TASK
    Read a movie review and predict POSITIVE or NEGATIVE.

WHAT THIS SCRIPT DOES
    1. Loads IMDB_Dataset.csv and removes duplicate reviews
    2. Splits it: 80% train (10% of that for validation) and 20% test
    3. Trains a simple baseline (TF-IDF + Logistic Regression) for comparison
    4. Trains a Bidirectional LSTM
    5. Prints both results on the unseen test set
    6. Saves the model, vocabulary, config and plots

RUN:   python train_sentiment.py
       (needs sentiment_utils.py in the same folder)
OUTPUT FILES:
    sentiment_bilstm.keras   trained model
    vocab.json               word -> number mapping
    sentiment_config.json    settings predict_sentiment.py needs
    training_curves.png      accuracy and loss per epoch
    confusion_matrix.png     test-set confusion matrix
"""
import os
import json

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots as files, no pop-up windows
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sentiment_utils import clean, build_vocab, encode

# ----------------------------- Settings -----------------------------
CSV_FILE = "IMDB_Dataset.csv"
MAX_WORDS = 20000    # vocabulary size
MAX_LEN = 256        # words kept per review
EMBED_DIM = 100
LSTM_UNITS = 32
EPOCHS = 8
BATCH_SIZE = 64
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# ----------------------------- Data -----------------------------
print("Loading data...")
df = pd.read_csv(CSV_FILE)
before = len(df)
df = df.drop_duplicates(subset="review").reset_index(drop=True)
print(f"Reviews: {before} -> {len(df)} after removing {before - len(df)} duplicates "
      f"(duplicates in both train and test would inflate the score)")

df["clean"] = df["review"].apply(clean)
df["label"] = (df["sentiment"] == "positive").astype(int)

train_val, test = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=SEED)
train, val = train_test_split(train_val, test_size=0.10, stratify=train_val["label"], random_state=SEED)
print(f"Train: {len(train)}  Validation: {len(val)}  Test: {len(test)}")

# ----------------------------- Baseline -----------------------------
print("\nTraining baseline (TF-IDF + Logistic Regression)...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=200000, sublinear_tf=True)
baseline = LogisticRegression(C=4, max_iter=1000)
baseline.fit(tfidf.fit_transform(train["clean"]), train["label"])
baseline_acc = accuracy_score(test["label"], baseline.predict(tfidf.transform(test["clean"])))
print(f"Baseline test accuracy: {baseline_acc:.4f}")

# ----------------------------- Encode for the LSTM -----------------------------
vocab = build_vocab(train["clean"], MAX_WORDS)   # built from TRAINING reviews only


def to_array(frame):
    return np.array([encode(t, vocab, MAX_LEN) for t in frame["clean"]], dtype="int32")


X_train, X_val, X_test = to_array(train), to_array(val), to_array(test)
y_train, y_val, y_test = train["label"].values, val["label"].values, test["label"].values

# ----------------------------- Model -----------------------------
model = models.Sequential([
    layers.Input(shape=(MAX_LEN,)),
    layers.Embedding(MAX_WORDS, EMBED_DIM, mask_zero=True),   # mask_zero: ignore the padding
    layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
    layers.Dropout(0.5),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

print("\nTraining the BiLSTM (stops automatically when validation loss stops improving)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ],
)

# ----------------------------- Evaluation -----------------------------
prob = model.predict(X_test, batch_size=256, verbose=0).ravel()
pred = (prob > 0.5).astype(int)
lstm_acc = accuracy_score(y_test, pred)

print("\n=========== TEST RESULTS (reviews the models never saw) ===========")
print(f"{'Model':<34}{'Accuracy':>10}")
print(f"{'TF-IDF + Logistic Regression':<34}{baseline_acc:>10.4f}")
print(f"{'Bidirectional LSTM':<34}{lstm_acc:>10.4f}")
print("\nBiLSTM details:")
print(classification_report(y_test, pred, target_names=["negative", "positive"], digits=4))
print("Confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_test, pred))

# ----------------------------- Plots -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.close()

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], ["negative", "positive"]); plt.yticks([0, 1], ["negative", "positive"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix (test set)")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# ----------------------------- Save -----------------------------
model.save("sentiment_bilstm.keras")
with open("vocab.json", "w") as f:
    json.dump(vocab, f)
with open("sentiment_config.json", "w") as f:
    json.dump({"max_len": MAX_LEN, "max_words": MAX_WORDS,
               "test_accuracy": float(lstm_acc), "baseline_accuracy": float(baseline_acc)}, f, indent=2)

print("\nSaved: sentiment_bilstm.keras, vocab.json, sentiment_config.json, "
      "training_curves.png, confusion_matrix.png")
