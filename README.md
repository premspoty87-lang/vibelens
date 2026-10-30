# Flipkart Reviews Sentiment Analysis

An end-to-end machine learning project that analyzes Flipkart product reviews to determine sentiment polarity and detect sarcasm in user-generated text.

---

## About

This project provides an automated pipeline for classifying customer reviews into **Positive**, **Neutral**, and **Negative** sentiments. It also incorporates a dedicated **sarcasm detection** module that flags potentially sarcastic reviews, helping improve the reliability of sentiment predictions.

Built with a modular architecture, the system is designed for clarity, maintainability, and ease of extension.

---

## What This Project Includes

- **Sentiment Classification** — Categorizes reviews into Positive, Neutral, or Negative using trained ML models
- **Sarcasm Detection** — Identifies sarcastic reviews with confidence scores using Logistic Regression
- **Model Comparison** — Evaluates Naive Bayes against Logistic Regression to select the best-performing model
- **Interactive Prediction** — Allows users to input custom reviews and receive instant predictions
- **Text Preprocessing Pipeline** — Handles lowercasing, URL removal, punctuation stripping, tokenization, and stopword removal
- **TF-IDF Vectorization** — Converts raw text into meaningful numerical features for model training
- **Visualization Suite** — Generates word clouds, sentiment distribution charts, and confusion matrices

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn (Logistic Regression, Naive Bayes, TF-IDF) |
| NLP | NLTK (tokenization, stopwords) |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Version Control | Git, GitHub |

---

## Project Structure

```
vibelens/
│
├── data/
│   ├── flipkart_data.csv
│   └── sarcastic_flipkart_reviews_10000.csv
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── model.py
│   ├── sarcasm_model.py
│   ├── predictor.py
│   └── main.py
│
├── output/
│   ├── sentiment_distribution.png
│   ├── positive_wordcloud.png
│   ├── negative_wordcloud.png
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_lr.png
│   └── model_comparison.csv
│
├── app.py
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.12 or higher
- pip

### Installation

```bash
git clone https://github.com/premspoty87-lang/vibelens.git
cd vibelens
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell / CMD)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

The required NLTK data (`punkt_tab` and `stopwords`) downloads automatically the first time the pipeline runs.

### Running the Project

```bash
python src/main.py
```

This executes the full pipeline — data loading, exploratory analysis, preprocessing, model training, evaluation, and launches the interactive prediction interface.

---

## Web Prototype

A zero-dependency web dashboard is included for demos and presentations. It reuses the trained pipeline and adds a modern UI for analyzing reviews.

```bash
python app.py
```

Then open **http://127.0.0.1:5000**.

The models train in the background on first launch; the status pill in the header shows progress. Once ready, the dashboard provides:

- **Review analyzer** — sentiment, per-class probabilities, sarcasm likelihood, and confidence scores
- **Dataset insights** — KPI cards, sentiment distribution, top words, word clouds, and confusion matrices
- **Model comparison** — Naive Bayes vs. Logistic Regression accuracy, precision, recall, and F1
- **Session history** — recent analyses kept locally in the browser

Opening `frontend/index.html` directly (without the server) runs an offline **demo mode** with sample data, so the interface can be previewed anywhere.

### API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/status` | Model readiness state |
| `GET` | `/api/overview` | Dataset stats, model metrics, chart artifacts |
| `POST` | `/api/predict` | Analyze a review: `{ "review": "..." }` |
| `GET` | `/output/<file>` | Serve generated charts |

---

## How It Works

The system follows a structured workflow:

1. **Data Loading** — Reads Flipkart review datasets from CSV files
2. **Exploratory Analysis** — Displays dataset shape, missing values, and rating distribution
3. **Label Creation** — Derives sentiment and sarcasm labels from review ratings
4. **Text Cleaning** — Applies a unified preprocessing pipeline to all reviews
5. **Feature Engineering** — Transforms cleaned text into TF-IDF vectors
6. **Model Training** — Trains Naive Bayes and Logistic Regression for sentiment; Logistic Regression for sarcasm
7. **Evaluation** — Measures performance using accuracy, precision, recall, and F1 score
8. **Interactive Prediction** — Accepts user input and returns sentiment and sarcasm predictions

---

## Sample Output

```
=================================
Flipkart Review Sentiment Checker
=================================

Enter your review:
This phone is amazing.

Predicted Sentiment : Positive
Sarcasm Detected    : Not Sarcastic (Confidence: 85.2%)

Would you like to test another review? (Y/N):
```

---

## License

This project is for educational and academic purposes.
