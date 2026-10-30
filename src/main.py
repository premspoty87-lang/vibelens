"""
main.py
Main entry point for Flipkart Sentiment Analysis project.
Runs the complete pipeline from data loading to interactive prediction.
"""

import os
import sys
from collections import Counter

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_data, explore_data
from src.preprocessing import create_sentiment, preprocess_reviews
from src.visualization import (plot_sentiment_distribution, plot_wordcloud,
                               plot_confusion_matrix)
from src.model import (build_tfidf, split_data, train_naive_bayes,
                       train_logistic_regression, evaluate_model, compare_models)
from src.predictor import interactive_predictor


def show_analysis(df):
    """
    Display analytical results: sentiment percentages and top words.
    """
    total = len(df)

    # Sentiment percentages
    pos_count = len(df[df["sentiment"] == "Positive"])
    neg_count = len(df[df["sentiment"] == "Negative"])
    neu_count = len(df[df["sentiment"] == "Neutral"])

    pos_pct = (pos_count / total) * 100
    neg_pct = (neg_count / total) * 100
    neu_pct = (neu_count / total) * 100

    print("\n===== Analytical Results =====\n")
    print(f"Positive Reviews : {pos_count} ({pos_pct:.1f}%)")
    print(f"Negative Reviews : {neg_count} ({neg_pct:.1f}%)")
    print(f"Neutral Reviews  : {neu_count} ({neu_pct:.1f}%)")

    # Top 20 words in Positive reviews
    pos_words = " ".join(df[df["sentiment"] == "Positive"]["clean_review"]).split()
    pos_common = Counter(pos_words).most_common(20)
    print("\nTop 20 Words in Positive Reviews:")
    for word, count in pos_common:
        print(f"  {word}: {count}")

    # Top 20 words in Negative reviews
    neg_words = " ".join(df[df["sentiment"] == "Negative"]["clean_review"]).split()
    neg_common = Counter(neg_words).most_common(20)
    print("\nTop 20 Words in Negative Reviews:")
    for word, count in neg_common:
        print(f"  {word}: {count}")


def main():
    """
    Complete pipeline:
    Load -> EDA -> Clean -> TF-IDF -> Train -> Evaluate -> Compare -> Predict
    """
    # ---------------------------
    # 1. Load Dataset
    # ---------------------------
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "flipkart_data.csv")
    df = load_data(data_path)

    # ---------------------------
    # 2. Exploratory Data Analysis
    # ---------------------------
    explore_data(df)

    # ---------------------------
    # 3. Create Sentiment Labels
    # ---------------------------
    df = create_sentiment(df)

    # ---------------------------
    # 4. Clean Reviews
    # ---------------------------
    df = preprocess_reviews(df)

    # ---------------------------
    # 5. Visualizations
    # ---------------------------
    plot_sentiment_distribution(df)
    plot_wordcloud(df, "Positive", "positive_wordcloud.png")
    plot_wordcloud(df, "Negative", "negative_wordcloud.png")

    # ---------------------------
    # 6. Feature Engineering (TF-IDF)
    # ---------------------------
    X, y, tfidf = build_tfidf(df)

    # ---------------------------
    # 7. Train/Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ---------------------------
    # 8. Train Naive Bayes
    # ---------------------------
    nb = train_naive_bayes(X_train, y_train)
    nb_metrics, nb_pred = evaluate_model(nb, X_test, y_test, "Naive Bayes")
    plot_confusion_matrix(y_test, nb_pred, "Naive Bayes", "confusion_matrix_nb.png")

    # ---------------------------
    # 9. Train Logistic Regression
    # ---------------------------
    lr = train_logistic_regression(X_train, y_train)
    lr_metrics, lr_pred = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", "confusion_matrix_lr.png")

    # ---------------------------
    # 10. Model Comparison
    # ---------------------------
    compare_models([nb_metrics, lr_metrics])

    # ---------------------------
    # 11. Analytical Results
    # ---------------------------
    show_analysis(df)

    # ---------------------------
    # 12. Interactive Prediction
    # ---------------------------
    interactive_predictor(tfidf, lr)


if __name__ == "__main__":
    main()
