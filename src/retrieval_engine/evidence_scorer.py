"""
Evidence Quality Scoring Engine.
Evaluates the strength, relevance, density, and consistency of retrieved historical support cases.
"""
from typing import List, Dict, Any, Optional
import numpy as np


class EvidenceScorer:
    """Computes comprehensive Evidence Quality score (0.0 to 1.0)."""

    def __init__(
        self,
        min_similarity: float = 0.20,
        high_similarity_threshold: float = 0.45,
        min_evidence_count: int = 1,
    ):
        self.min_similarity = min_similarity
        self.high_similarity_threshold = high_similarity_threshold
        self.min_evidence_count = min_evidence_count

    def score_evidence(
        self,
        retrieved_cases: List[Dict[str, Any]],
        predicted_intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate retrieved cases and return evidence quality metrics.
        Scales sparse n-gram cosine similarities appropriately for short customer messages.
        """
        if not retrieved_cases:
            return {
                "evidence_quality": 0.0,
                "evidence_count": 0,
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
                "intent_concordance": 0.0,
                "high_quality_count": 0,
                "assessment": "NO_EVIDENCE",
                "is_sufficient": False,
            }

        sims = [c.get("similarity", 0.0) for c in retrieved_cases]
        raw_max_sim = float(np.max(sims))
        
        # Scale raw cosine similarity for short-text information retrieval
        # A raw cosine of 0.35 in short sparse text indicates strong semantic overlap
        scaled_max = float(np.clip(raw_max_sim * 2.2, 0.0, 1.0))
        top3_sims = sims[:min(3, len(sims))]
        scaled_mean_top3 = float(np.clip(np.mean(top3_sims) * 2.0, 0.0, 1.0))

        # Count cases with strong similarity
        high_quality_matches = [s for s in sims if s >= self.high_similarity_threshold or (s * 2.2 >= 0.70)]
        valid_matches = [s for s in sims if s >= self.min_similarity or (s * 2.2 >= 0.40)]
        high_quality_count = len(high_quality_matches)
        evidence_count = len(valid_matches)

        # Intent concordance: fraction of retrieved cases agreeing with predicted intent
        if predicted_intent:
            agreed = [c for c in retrieved_cases if c.get("intent") == predicted_intent]
            intent_concordance = len(agreed) / max(len(retrieved_cases), 1)
        else:
            intent_concordance = 0.8  # neutral fallback

        # Normalized density factor
        density_factor = min(1.0, high_quality_count / 2.0)

        # Composite Evidence Quality Formula
        raw_quality = (
            0.40 * scaled_max +
            0.25 * scaled_mean_top3 +
            0.20 * density_factor +
            0.15 * intent_concordance
        )
        
        evidence_quality = round(float(np.clip(raw_quality, 0.0, 1.0)), 4)

        is_sufficient = (
            evidence_quality >= 0.55 and
            evidence_count >= self.min_evidence_count and
            raw_max_sim >= self.min_similarity
        )

        if evidence_quality >= 0.80:
            assessment = "STRONG_EVIDENCE"
        elif evidence_quality >= 0.65:
            assessment = "MODERATE_EVIDENCE"
        elif evidence_quality >= 0.45:
            assessment = "WEAK_EVIDENCE"
        else:
            assessment = "INSUFFICIENT_EVIDENCE"

        return {
            "evidence_quality": evidence_quality,
            "evidence_count": evidence_count,
            "max_similarity": round(raw_max_sim, 4),
            "mean_similarity": round(float(np.mean(top3_sims)), 4),
            "intent_concordance": round(intent_concordance, 4),
            "high_quality_count": high_quality_count,
            "assessment": assessment,
            "is_sufficient": is_sufficient,
        }
