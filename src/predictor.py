"""
predictor.py
Interactive sentiment predictor that uses the trained model.
Also handles sarcasm detection for each review.
"""

from src.preprocessing import clean_text
from src.sarcasm_model import train_sarcasm_model, predict_sarcasm


def predict_sentiment(tfidf, model, review):
    """
    Predict sentiment for a single review.
    Uses the same TF-IDF vectorizer (transform only, not fit_transform).
    """
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]


def interactive_predictor(tfidf, model):
    """
    Interactive loop: ask user for reviews, predict sentiment and sarcasm,
    and repeat until user enters N.
    """
    # ---------------------------
    # Train sarcasm model once at startup
    # ---------------------------
    train_sarcasm_model()

    print("\n=================================")
    print("Flipkart Review Sentiment Checker")
    print("=================================")

    while True:
        review = input("\nEnter your review:\n")

        # ---------------------------
        # Predict sentiment
        # ---------------------------
        sentiment = predict_sentiment(tfidf, model, review)

        # ---------------------------
        # Predict sarcasm
        # ---------------------------
        sarcasm_label, sarcasm_confidence = predict_sarcasm(review)

        # ---------------------------
        # Display results as labels
        # ---------------------------
        print(f"\nPredicted Sentiment : {sentiment}")
        print(f"Sarcasm Detected    : {sarcasm_label} (Confidence: {sarcasm_confidence:.1f}%)")

        choice = input("\nWould you like to test another review? (Y/N): ").strip().upper()
        if choice != "Y":
            print("Thank you for using Sentiment Checker!")
            break
