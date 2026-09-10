"""
Baseline 2: TF-IDF + Logistic Regression Classifier.
Uses character and word n-grams with multinomial Logistic Regression.
"""
from typing import List, Dict, Any
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfLogisticClassifier:
    """TF-IDF + Logistic Regression Intent Classifier."""

    def __init__(self, c: float = 1.0, max_features: int = 5000):
        self.c = c
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=self.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.clf = LogisticRegression(
            C=self.c,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", self.clf),
        ])
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]):
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["classifier"].classes_)
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        probs = self.pipeline.predict_proba(texts)
        results = []
        for p in probs:
            results.append({self.classes_[i]: float(p[i]) for i in range(len(self.classes_))})
        return results

    def predict_one(self, text: str) -> Dict[str, Any]:
        if not text:
            return {
                "intent": self.classes_[0] if self.classes_ else "unknown",
                "confidence": 0.0,
                "probabilities": {},
                "model_name": "tfidf_logistic",
            }
        probs = self.pipeline.predict_proba([text])[0]
        max_idx = int(np.argmax(probs))
        best_intent = self.classes_[max_idx]
        confidence = float(probs[max_idx])
        return {
            "intent": best_intent,
            "confidence": round(confidence, 4),
            "probabilities": {self.classes_[i]: round(float(probs[i]), 4) for i in range(len(self.classes_))},
            "model_name": "tfidf_logistic",
        }

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "TfidfLogisticClassifier":
        with open(filepath, "rb") as f:
            return pickle.load(f)
