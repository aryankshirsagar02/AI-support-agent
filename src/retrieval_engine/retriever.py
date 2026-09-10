"""
Historical Case Retrieval Engine.
Builds vector index over training split customer support conversations,
executes Top-K similarity search, and guarantees zero test-set leakage.
"""
from typing import List, Dict, Any, Optional
import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml


class HistoricalRetriever:
    """Retrieves similar resolved customer support cases from the training corpus."""

    def __init__(
        self,
        config_path: str = "config/brand_config.yaml",
        top_k: int = 5,
        min_similarity: float = 0.50,
    ):
        self.config_path = config_path
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.conversations: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=1,
        )
        self.tfidf_matrix: Optional[np.ndarray] = None
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            retrieval_cfg = config.get("retrieval", {})
            self.top_k = retrieval_cfg.get("top_k", self.top_k)
            self.min_similarity = retrieval_cfg.get("min_similarity_threshold", self.min_similarity)
        except Exception:
            pass

    def build_index(self, train_conversations: List[Dict[str, Any]]) -> "HistoricalRetriever":
        """Build vector index only over training conversations (leakage prevention)."""
        self.conversations = train_conversations
        corpus_texts = [
            f"{c.get('cleaned_customer_message', c.get('customer_message', ''))}"
            for c in self.conversations
        ]
        if not corpus_texts:
            raise ValueError("Cannot build retrieval index with empty corpus!")

        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        return self

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K most similar historical conversations.
        """
        if self.tfidf_matrix is None or not self.conversations:
            return []

        k = top_k or self.top_k
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Apply intent filter if requested
        valid_indices = []
        for idx, sim in enumerate(similarities):
            if filter_intent:
                c_intent = self.conversations[idx].get("intent")
                if c_intent != filter_intent:
                    continue
            valid_indices.append((idx, sim))

        # Sort descending by similarity
        valid_indices.sort(key=lambda x: x[1], reverse=True)
        top_matches = valid_indices[:k]

        results: List[Dict[str, Any]] = []
        for idx, sim in top_matches:
            c = self.conversations[idx]
            results.append({
                "conversation_id": c.get("conversation_id"),
                "customer_message": c.get("customer_message"),
                "cleaned_customer_message": c.get("cleaned_customer_message"),
                "brand_response": c.get("brand_response"),
                "cleaned_brand_response": c.get("cleaned_brand_response"),
                "similarity": round(float(sim), 4),
                "intent": c.get("intent"),
                "created_at": c.get("created_at"),
                "turn_count": c.get("turn_count", 2),
                "resolution_status": c.get("resolution_status", "RESOLVED"),
            })

        return results

    def save(self, filepath: str):
        data = {
            "conversations": self.conversations,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "HistoricalRetriever":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        retriever = cls(top_k=data.get("top_k", 5), min_similarity=data.get("min_similarity", 0.50))
        retriever.conversations = data["conversations"]
        retriever.vectorizer = data["vectorizer"]
        retriever.tfidf_matrix = data["tfidf_matrix"]
        return retriever
