"""
visualization.py
Generates and saves all charts: sentiment distribution, word clouds, confusion matrices.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# Ensure output folder exists
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_sentiment_distribution(df):
    """
    Bar chart showing count of Positive, Neutral, Negative reviews.
    Saves to output/sentiment_distribution.png
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x="sentiment", data=df, palette="viridis",
                  order=["Positive", "Neutral", "Negative"])
    plt.title("Sentiment Distribution", fontsize=14)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sentiment_distribution.png"))
    plt.show()
    print("Saved: output/sentiment_distribution.png")


def plot_wordcloud(df, sentiment, filename):
    """
    Generate word cloud for a specific sentiment.
    Saves to output/<filename>.png
    """
    text = " ".join(df[df["sentiment"] == sentiment]["clean_review"].astype(str))

    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap="viridis", max_words=100).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{sentiment} Reviews - Word Cloud", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.show()
    print(f"Saved: output/{filename}")


def plot_confusion_matrix(y_test, y_pred, model_name, filename):
    """
    Plot and save confusion matrix for a model.
    Saves to output/<filename>.png
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred, labels=["Positive", "Neutral", "Negative"])

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Positive", "Neutral", "Negative"],
                yticklabels=["Positive", "Neutral", "Negative"])
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.show()
    print(f"Saved: output/{filename}")
