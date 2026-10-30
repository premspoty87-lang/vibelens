"""
model.py
Handles TF-IDF vectorization, model training, evaluation, and comparison.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_tfidf(df):
    """
    Convert clean_review text into TF-IDF vectors.
    Returns the TF-IDF matrix, vectorizer, and feature names.
    """
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["clean_review"])
    y = df["sentiment"]
    return X, y, tfidf


def split_data(X, y):
    """
    Split data into 80% training and 20% testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_naive_bayes(X_train, y_train):
    """
    Train Multinomial Naive Bayes model.
    """
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    """
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    return lr


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model and print metrics.
    Returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n===== {model_name} Evaluation =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    return metrics, y_pred


def compare_models(metrics_list):
    """
    Create a comparison DataFrame and save to CSV.
    Highlights the model with highest accuracy.
    """
    df = pd.DataFrame(metrics_list)

    # Find best model
    best_idx = df["Accuracy"].idxmax()
    best_model = df.loc[best_idx, "Model"]

    print("\n===== Model Comparison =====\n")
    print(df.to_string(index=False))
    print(f"\nBest Model: {best_model}")

    # Save to CSV
    df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("Saved: output/model_comparison.csv")

    return df
