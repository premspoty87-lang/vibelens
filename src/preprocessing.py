"""
preprocessing.py
Handles sentiment creation and text cleaning for review data.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (only runs once)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


def create_sentiment(df):
    """
    Create sentiment column from rating.
    Rating 4 or 5 -> Positive
    Rating 3      -> Neutral
    Rating 1 or 2 -> Negative
    """
    def get_sentiment(rating):
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"

    df["sentiment"] = df["rating"].apply(get_sentiment)
    print("Sentiment column created!")
    return df


def clean_text(text):
    """
    Clean a single review text:
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


def preprocess_reviews(df):
    """
    Apply clean_text to all reviews and create clean_review column.
    """
    df["clean_review"] = df["review"].apply(clean_text)
    print("Reviews cleaned successfully!")
    return df
