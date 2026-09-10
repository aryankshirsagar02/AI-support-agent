"""
Baseline 1: Majority Class Intent Classifier.
Always predicts the most frequent intent observed in the training dataset.
"""
from typing import List, Dict, Any, Tuple
from collections import Counter
import pickle


class MajorityClassifier:
    """Trivial baseline that always predicts the majority class."""

    def __init__(self):
        self.majority_intent: str = "order_tracking"
        self.confidence: float = 0.5
        self.classes_: List[str] = []
        self.class_priors_: Dict[str, float] = {}

    def fit(self, texts: List[str], labels: List[str]):
        counts = Counter(labels)
        total = sum(counts.values()) if counts else 1
        most_common = counts.most_common(1)
        if most_common:
            self.majority_intent = most_common[0][0]
            self.confidence = most_common[0][1] / total
        self.classes_ = sorted(list(counts.keys()))
        self.class_priors_ = {c: counts[c] / total for c in self.classes_}
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_intent for _ in texts]

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        return [{c: self.class_priors_.get(c, 0.0) for c in self.classes_} for _ in texts]

    def predict_one(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "confidence": round(self.confidence, 4),
            "probabilities": {c: round(self.class_priors_.get(c, 0.0), 4) for c in self.classes_},
            "model_name": "majority_baseline",
        }

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "MajorityClassifier":
        with open(filepath, "rb") as f:
            return pickle.load(f)
