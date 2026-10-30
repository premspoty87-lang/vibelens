"""
sarcasm_model.py
Handles sarcasm detection: loading data, training, evaluation, and prediction.
Completely independent of the Flipkart sentiment model.
"""

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

# Download required NLTK data (only runs once)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Global stores for trained objects (never retrained during prediction)
_sarcasm_model = None
_sarcasm_vectorizer = None


def _load_sarcasm_data():
    """
    Load the sarcasm dataset from data/sarcastic_flipkart_reviews_10000.csv.
    Returns the DataFrame.
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", "data",
                             "sarcastic_flipkart_reviews_10000.csv")
    df = pd.read_csv(data_path)
    print("[Sarcasm] Dataset loaded successfully!")
    return df


def _explore_sarcasm_data(df):
    """
    Display basic EDA for the sarcasm dataset:
    shape, columns, missing values.
    """
    print("\n===== Sarcasm Dataset - Exploratory Data Analysis =====\n")

    # Dataset shape
    print("Dataset Shape:", df.shape)
    print("Rows   :", df.shape[0])
    print("Columns:", df.shape[1])

    # Column names
    print("\n--- Columns ---")
    print(df.columns.tolist())

    # Missing values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Rating distribution
    print("\n--- Rating Distribution ---")
    print(df["rating"].value_counts().sort_index())


def _create_sarcasm_labels(df):
    """
    Create sarcasm column from rating.
    Rating <= 2 -> Sarcastic
    Rating >= 3 -> Not Sarcastic
    """
    def get_sarcasm_label(rating):
        if rating <= 2:
            return "Sarcastic"
        else:
            return "Not Sarcastic"

    df["sarcasm"] = df["rating"].apply(get_sarcasm_label)
    print("[Sarcasm] Sarcasm labels created!")
    return df


def clean_sarcasm_text(text):
    """
    Clean a single review text for sarcasm detection:
    - Convert to lowercase
    - Remove URLs
    - Remove punctuation
    - Remove numbers
    - Remove extra spaces
    - Tokenize
    - Remove stopwords
    """
    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def _preprocess_sarcasm_data(df):
    """
    Apply clean_sarcasm_text to all reviews and create clean_review column.
    """
    df["clean_review"] = df["review"].apply(clean_sarcasm_text)
    print("[Sarcasm] Reviews cleaned successfully!")
    return df


def _build_sarcasm_tfidf(df):
    """
    Convert clean_review text into TF-IDF vectors.
    Returns the TF-IDF matrix, vectorizer, and labels.
    """
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["clean_review"])
    y = df["sarcasm"]
    return X, y, tfidf


def _split_sarcasm_data(X, y):
    """
    Split data into 80% training and 20% testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def _train_sarcasm_lr(X_train, y_train):
    """
    Train Logistic Regression model for sarcasm detection.
    """
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    return lr


def _evaluate_sarcasm_model(model, X_test, y_test):
    """
    Evaluate the sarcasm model and print metrics.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n===== Sarcasm Model Evaluation =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return y_pred


def train_sarcasm_model():
    """
    Complete sarcasm model training pipeline.
    Called once from predictor.py during first prediction.
    Responsibilities:
    - Load dataset
    - Preprocess
    - TF-IDF
    - Train Logistic Regression
    - Evaluate
    - Store model and vectorizer globally
    """
    global _sarcasm_model, _sarcasm_vectorizer

    # ---------------------------
    # 1. Load Dataset
    # ---------------------------
    df = _load_sarcasm_data()

    # ---------------------------
    # 2. Exploratory Data Analysis
    # ---------------------------
    _explore_sarcasm_data(df)

    # ---------------------------
    # 3. Create Sarcasm Labels
    # ---------------------------
    df = _create_sarcasm_labels(df)

    # ---------------------------
    # 4. Clean Reviews
    # ---------------------------
    df = _preprocess_sarcasm_data(df)

    # ---------------------------
    # 5. Feature Engineering (TF-IDF)
    # ---------------------------
    X, y, tfidf = _build_sarcasm_tfidf(df)

    # ---------------------------
    # 6. Train/Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = _split_sarcasm_data(X, y)

    # ---------------------------
    # 7. Train Logistic Regression
    # ---------------------------
    lr = _train_sarcasm_lr(X_train, y_train)

    # ---------------------------
    # 8. Evaluate Model
    # ---------------------------
    _evaluate_sarcasm_model(lr, X_test, y_test)

    # ---------------------------
    # 9. Store trained objects globally
    # ---------------------------
    _sarcasm_model = lr
    _sarcasm_vectorizer = tfidf

    print("\n[Sarcasm] Model trained and stored successfully!\n")


def predict_sarcasm(review):
    """
    Predict sarcasm for a single raw review.
    - Accept raw review text
    - Clean review using the same preprocessing as training
    - Transform using the already fitted TF-IDF vectorizer (never fit_transform)
    - Predict using the already trained Logistic Regression model
    - Return prediction and confidence score

    Returns:
        prediction: "Sarcastic" or "Not Sarcastic"
        confidence: confidence score as a percentage (0-100)
    """
    # Clean the review text
    cleaned = clean_sarcasm_text(review)

    # Transform using the already fitted vectorizer
    vector = _sarcasm_vectorizer.transform([cleaned])

    # Predict class and probabilities
    prediction = _sarcasm_model.predict(vector)[0]
    probabilities = _sarcasm_model.predict_proba(vector)[0]

    # Get confidence for the predicted class
    class_index = list(_sarcasm_model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100

    return prediction, confidence
