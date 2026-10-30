"""
app.py
VibeLens prototype web app.

Serves the frontend in ./frontend and exposes a small JSON API backed by the
existing machine learning pipeline in ./src. Uses only the Python standard
library, so no extra dependencies are required.

Run:
    python app.py

Then open:
    http://127.0.0.1:5000
"""

import json
import mimetypes
import os
import sys
import threading
import traceback
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from src.data_loader import load_data
from src.model import (build_tfidf, split_data, train_logistic_regression,
                       train_naive_bayes)
from src.preprocessing import clean_text, create_sentiment, preprocess_reviews
from src.sarcasm_model import (_build_sarcasm_tfidf, _create_sarcasm_labels,
                               _load_sarcasm_data, _preprocess_sarcasm_data,
                               _split_sarcasm_data, _train_sarcasm_lr,
                               clean_sarcasm_text)

FRONTEND_DIR = os.path.join(ROOT, "frontend")
OUTPUT_DIR = os.path.join(ROOT, "output")
DATA_DIR = os.path.join(ROOT, "data")

MAX_REVIEW_LENGTH = 5000
SENTIMENTS = ("Positive", "Neutral", "Negative")


def _top_words(df, sentiment, limit=12):
    """Return the most frequent clean words for a sentiment label."""
    text = " ".join(df.loc[df["sentiment"] == sentiment, "clean_review"].astype(str))
    counter = Counter(text.split())
    return [[word, int(count)] for word, count in counter.most_common(limit)]


def _metric_block(name, kind, y_true, y_pred):
    """Build a serializable metrics dictionary."""
    return {
        "name": name,
        "kind": kind,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted",
                                                zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted",
                                           zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted",
                                   zero_division=0)), 4),
    }


class VibeLensEngine:
    """Trains the models once and serves predictions from memory."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ready = False
        self._state = "idle"
        self._error = None
        self._tfidf = None
        self._sentiment_model = None
        self._sarcasm_tfidf = None
        self._sarcasm_model = None
        self._stats = {}
        self._models = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def status(self):
        return {
            "ready": self._ready,
            "state": self._state,
            "error": self._error,
        }

    def warmup(self):
        try:
            self.ensure_ready()
        except Exception:
            pass

    def ensure_ready(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._state = "training"
            self._error = None
            try:
                self._train()
                self._ready = True
                self._state = "ready"
                print("[VibeLens] Models ready.", flush=True)
            except Exception as exc:
                self._state = "error"
                self._error = str(exc)
                traceback.print_exc()
                raise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train(self):
        print("[VibeLens] Training sentiment models...", flush=True)
        df = load_data(os.path.join(DATA_DIR, "flipkart_data.csv"))
        df = create_sentiment(df)
        df = preprocess_reviews(df)

        total = int(len(df))
        counts = {label: int((df["sentiment"] == label).sum()) for label in SENTIMENTS}
        percentages = {
            label: round((counts[label] / total) * 100, 1) if total else 0.0
            for label in SENTIMENTS
        }
        average_rating = round(float(df["rating"].mean()), 2)

        X, y, tfidf = build_tfidf(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        nb = train_naive_bayes(X_train, y_train)
        lr = train_logistic_regression(X_train, y_train)

        self._tfidf = tfidf
        self._sentiment_model = lr

        models = [
            _metric_block("Naive Bayes", "Sentiment", y_test, nb.predict(X_test)),
            _metric_block("Logistic Regression", "Sentiment", y_test, lr.predict(X_test)),
        ]

        print("[VibeLens] Training sarcasm model...", flush=True)
        sdf = _load_sarcasm_data()
        sdf = _create_sarcasm_labels(sdf)
        sdf = _preprocess_sarcasm_data(sdf)
        Xs, ys, sarcasm_tfidf = _build_sarcasm_tfidf(sdf)
        Xs_train, Xs_test, ys_train, ys_test = _split_sarcasm_data(Xs, ys)
        sarcasm_lr = _train_sarcasm_lr(Xs_train, ys_train)

        self._sarcasm_tfidf = sarcasm_tfidf
        self._sarcasm_model = sarcasm_lr

        models.append(
            _metric_block("Sarcasm (Logistic Regression)", "Sarcasm",
                          ys_test, sarcasm_lr.predict(Xs_test))
        )

        sarcastic = int((sdf["sarcasm"] == "Sarcastic").sum())

        self._stats = {
            "totalReviews": total,
            "averageRating": average_rating,
            "sentimentCounts": counts,
            "sentimentPercentages": percentages,
            "topWords": {label: _top_words(df, label) for label in SENTIMENTS},
            "sarcasmDataset": {
                "total": int(len(sdf)),
                "sarcastic": sarcastic,
                "notSarcastic": int(len(sdf)) - sarcastic,
            },
        }
        self._models = models

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, review):
        self.ensure_ready()

        cleaned = clean_text(review)
        vector = self._tfidf.transform([cleaned])
        probabilities = self._sentiment_model.predict_proba(vector)[0]
        classes = [str(c) for c in self._sentiment_model.classes_]
        sentiment_scores = {
            label: round(float(prob) * 100, 1)
            for label, prob in zip(classes, probabilities)
        }
        sentiment = max(sentiment_scores, key=sentiment_scores.get)

        sarcasm_cleaned = clean_sarcasm_text(review)
        sarcasm_vector = self._sarcasm_tfidf.transform([sarcasm_cleaned])
        sarcasm_probs = self._sarcasm_model.predict_proba(sarcasm_vector)[0]
        sarcasm_classes = [str(c) for c in self._sarcasm_model.classes_]
        sarcasm_scores = {
            label: round(float(prob) * 100, 1)
            for label, prob in zip(sarcasm_classes, sarcasm_probs)
        }
        sarcasm = max(sarcasm_scores, key=sarcasm_scores.get)

        warnings = []
        if sarcasm == "Sarcastic" and sentiment == "Positive":
            warnings.append(
                "The review reads positive on the surface but is flagged as "
                "sarcastic. Treat the positive score with caution."
            )
        elif sarcasm == "Sarcastic":
            warnings.append(
                "Sarcastic tone detected. Surface-level sentiment may not reflect "
                "the reviewer's true opinion."
            )
        if sentiment_scores[sentiment] < 55:
            warnings.append(
                "Sentiment confidence is low, which usually means the review is "
                "mixed or ambiguous."
            )

        return {
            "review": review,
            "cleaned": cleaned,
            "sentiment": sentiment,
            "sentimentConfidence": sentiment_scores[sentiment],
            "sentimentScores": {label: sentiment_scores.get(label, 0.0)
                                for label in SENTIMENTS},
            "sarcasm": sarcasm,
            "sarcasmConfidence": sarcasm_scores[sarcasm],
            "sarcasmScores": sarcasm_scores,
            "sarcasmLikelihood": sarcasm_scores.get("Sarcastic", 0.0),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Overview payload
    # ------------------------------------------------------------------
    def overview(self):
        self.ensure_ready()
        return {
            "ready": True,
            "stats": self._stats,
            "models": self._models,
            "artifacts": [
                "sentiment_distribution.png",
                "positive_wordcloud.png",
                "negative_wordcloud.png",
                "confusion_matrix_nb.png",
                "confusion_matrix_lr.png",
            ],
        }


ENGINE = VibeLensEngine()


class VibeLensHandler(BaseHTTPRequestHandler):
    server_version = "VibeLens/1.0"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _send_bytes(self, body, status=200, content_type="application/octet-stream",
                    cache="no-cache"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, status, "application/json; charset=utf-8")

    def _error(self, message, status=400):
        self._json({"error": message}, status)

    def _serve_static(self, relative_path):
        target = os.path.normpath(os.path.join(FRONTEND_DIR, relative_path))
        base = os.path.normpath(FRONTEND_DIR)
        if not (target == base or target.startswith(base + os.sep)):
            self._error("Forbidden", 403)
            return
        if not os.path.isfile(target):
            self._error("Not found", 404)
            return
        content_type = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
            "application/javascript", "application/x-javascript",
        ):
            content_type += "; charset=utf-8"
        with open(target, "rb") as handle:
            self._send_bytes(handle.read(), 200, content_type)

    def _serve_output(self, relative_path):
        target = os.path.normpath(os.path.join(OUTPUT_DIR, relative_path))
        base = os.path.normpath(OUTPUT_DIR)
        if not (target == base or target.startswith(base + os.sep)):
            self._error("Forbidden", 403)
            return
        if not os.path.isfile(target):
            self._error("Not found", 404)
            return
        content_type = mimetypes.guess_type(target)[0] or "application/octet-stream"
        with open(target, "rb") as handle:
            self._send_bytes(handle.read(), 200, content_type, "public, max-age=300")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path == "/api/status":
            self._json(ENGINE.status())
            return

        if path == "/api/overview":
            try:
                self._json(ENGINE.overview())
            except Exception as exc:
                self._error("Model unavailable: %s" % exc, 503)
            return

        if path.startswith("/output/"):
            self._serve_output(path[len("/output/"):])
            return

        if path in ("/", "/index.html"):
            self._serve_static("index.html")
            return

        self._serve_static(path.lstrip("/"))

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if path != "/api/predict":
            self._error("Not found", 404)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error("Invalid content length", 400)
            return

        if length > 200_000:
            self._error("Payload too large", 413)
            return

        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error("Invalid JSON body", 400)
            return

        review = str(payload.get("review", "")).strip()
        if len(review) < 2:
            self._error("Please enter a review with at least 2 characters.", 422)
            return
        if len(review) > MAX_REVIEW_LENGTH:
            self._error("Review is too long. Keep it under %d characters."
                        % MAX_REVIEW_LENGTH, 422)
            return

        try:
            self._json(ENGINE.predict(review))
        except Exception as exc:
            traceback.print_exc()
            self._error("Prediction failed: %s" % exc, 500)

    def log_message(self, fmt, *args):
        sys.stderr.write("[http] %s - %s\n" % (self.address_string(), fmt % args))


def main():
    host = os.environ.get("VIBELENS_HOST", "127.0.0.1")
    port = int(os.environ.get("VIBELENS_PORT", "5000"))

    print("=" * 46)
    print(" VibeLens prototype")
    print("=" * 46)
    print(" Frontend : http://%s:%d" % (host, port))
    print(" Models   : training in the background")
    print(" Stop     : Ctrl + C")
    print("=" * 46, flush=True)

    threading.Thread(target=ENGINE.warmup, daemon=True).start()

    server = ThreadingHTTPServer((host, port), VibeLensHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[VibeLens] Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
