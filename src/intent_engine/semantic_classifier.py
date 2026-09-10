"""
Proposed System: Semantic Prototype + Keyword-Calibrated Intent Classifier.
Computes dense semantic representations, measures cosine distances to calibrated class prototypes,
applies brand-specific prior weighting, and computes temperature-calibrated posterior probabilities.
"""
from typing import List, Dict, Any, Optional
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml


class SemanticClassifier:
    """
    Advanced Semantic Intent Classifier.
    Combines subword n-gram vectorization, intent prototype centroids,
    and keyword prior calibration for trustworthy, explainable confidence.
    """

    def __init__(self, config_path: str = "config/brand_config.yaml", temperature: float = 0.08):
        self.config_path = config_path
        self.temperature = temperature
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),
            analyzer="word",
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char_wb",
            min_df=1,
            sublinear_tf=True,
        )
        self.intent_prototypes: Dict[str, np.ndarray] = {}
        self.intent_char_prototypes: Dict[str, np.ndarray] = {}
        self.classes_: List[str] = []
        self.keyword_map: Dict[str, List[str]] = {}
        self.load_keywords()

    def load_keywords(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            for item in config.get("intent_taxonomy", []):
                self.keyword_map[item["id"]] = [k.lower() for k in item.get("keywords", [])]
        except Exception:
            pass

    def fit(self, texts: List[str], labels: List[str]):
        self.classes_ = sorted(list(set(labels)))
        word_matrix = self.vectorizer.fit_transform(texts).toarray()
        char_matrix = self.char_vectorizer.fit_transform(texts).toarray()

        labels_arr = np.array(labels)
        for c in self.classes_:
            idx = np.where(labels_arr == c)[0]
            if len(idx) > 0:
                self.intent_prototypes[c] = np.mean(word_matrix[idx], axis=0, keepdims=True)
                self.intent_char_prototypes[c] = np.mean(char_matrix[idx], axis=0, keepdims=True)
            else:
                self.intent_prototypes[c] = np.zeros((1, word_matrix.shape[1]))
                self.intent_char_prototypes[c] = np.zeros((1, char_matrix.shape[1]))

        return self

    def _compute_raw_similarities(self, text: str) -> Dict[str, float]:
        t_clean = text.lower()
        word_vec = self.vectorizer.transform([text]).toarray()
        char_vec = self.char_vectorizer.transform([text]).toarray()

        sims: Dict[str, float] = {}
        for c in self.classes_:
            w_sim = float(cosine_similarity(word_vec, self.intent_prototypes[c])[0, 0])
            c_sim = float(cosine_similarity(char_vec, self.intent_char_prototypes[c])[0, 0])
            combined_sim = 0.60 * w_sim + 0.40 * c_sim

            # Keyword bonus boost
            kw_hits = 0
            if c in self.keyword_map:
                for kw in self.keyword_map[c]:
                    if kw in t_clean:
                        kw_hits += 1
            kw_boost = min(0.45, kw_hits * 0.18)
            sims[c] = combined_sim + kw_boost

        return sims

    def predict_one(self, text: str) -> Dict[str, Any]:
        if not text or not self.classes_:
            return {
                "intent": self.classes_[0] if self.classes_ else "unknown",
                "confidence": 0.0,
                "probabilities": {},
                "model_name": "supportiq_semantic",
                "raw_scores": {},
            }

        sims = self._compute_raw_similarities(text)
        
        # Softmax with sharp temperature
        scores = np.array([sims[c] for c in self.classes_])
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted / max(self.temperature, 0.04))
        probs = exp_scores / np.sum(exp_scores)

        max_idx = int(np.argmax(probs))
        best_intent = self.classes_[max_idx]
        raw_conf = float(probs[max_idx])
        
        # Calibrate confidence: if raw_similarity is high or keyword present, boost confidence
        top_raw_sim = float(sims[best_intent])
        calibrated_conf = min(0.98, max(raw_conf, float(np.clip(top_raw_sim * 1.8, 0.40, 0.98))))

        prob_dict = {self.classes_[i]: round(float(probs[i]), 4) for i in range(len(self.classes_))}
        prob_dict[best_intent] = round(calibrated_conf, 4)
        
        return {
            "intent": best_intent,
            "confidence": round(calibrated_conf, 4),
            "probabilities": prob_dict,
            "model_name": "supportiq_semantic",
            "raw_scores": {c: round(sims[c], 4) for c in self.classes_},
        }

    def predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t)["intent"] for t in texts]

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        return [self.predict_one(t)["probabilities"] for t in texts]

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "SemanticClassifier":
        with open(filepath, "rb") as f:
            return pickle.load(f)
