# train_fakenews_clean.py

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# -----------------------------
# LOAD DATA (Fake + True)
# -----------------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0   # fake
true["label"] = 1   # real

df = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

# Combine text
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

print("Dataset size:", len(df))
print(df["label"].value_counts())

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# -----------------------------
# TF-IDF (CLEAN VERSION)
# -----------------------------
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------------
# MODEL (STRONG BASELINE)
# -----------------------------
model = LogisticRegression(
    max_iter=4000,
    class_weight={0: 1.0, 1: 1.5},  # boost real news slightly
    solver="saga"
)

print("Training model...")
model.fit(X_train_tfidf, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# -----------------------------
# SAVE MODEL
# -----------------------------
artifact = {
    "tfidf": tfidf,
    "model": model,
    "label_mapping": {0: "fake", 1: "real"}
}

Path("models").mkdir(exist_ok=True)

with open("models/fakenews_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\n✅ Model saved to models/fakenews_model.pkl")