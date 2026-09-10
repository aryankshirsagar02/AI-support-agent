"""
Trust Scoring Engine for SupportIQ AI.
Computes an evidence-grounded multi-factor Trust Score (0 to 100).
"""
from typing import Dict, Any, Optional
import yaml


class TrustScorer:
    """
    Computes transparent multi-factor Trust Score.
    
    Formula:
        Trust Score = 100 * (
            w_intent * Intent_Confidence
          + w_evidence * Evidence_Quality
          + w_consistency * Historical_Consistency
          + w_safety * Safety_Score
        )
    """

    def __init__(self, config_path: str = "config/brand_config.yaml"):
        self.config_path = config_path
        self.w_intent = 0.40
        self.w_evidence = 0.35
        self.w_consistency = 0.15
        self.w_safety = 0.10
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            weights = config.get("trust_engine", {}).get("weights", {})
            self.w_intent = weights.get("intent_confidence", self.w_intent)
            self.w_evidence = weights.get("evidence_quality", self.w_evidence)
            self.w_consistency = weights.get("historical_consistency", self.w_consistency)
            self.w_safety = weights.get("safety_score", self.w_safety)
        except Exception:
            pass

    def compute_trust_score(
        self,
        intent_confidence: float,
        evidence_quality: float,
        historical_consistency: float = 0.90,
        safety_score: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compute trust score and sub-component contributions.
        """
        # Bound inputs between 0.0 and 1.0
        c_intent = max(0.0, min(1.0, float(intent_confidence)))
        c_evidence = max(0.0, min(1.0, float(evidence_quality)))
        c_consistency = max(0.0, min(1.0, float(historical_consistency)))
        c_safety = max(0.0, min(1.0, float(safety_score)))

        raw_score = (
            self.w_intent * c_intent +
            self.w_evidence * c_evidence +
            self.w_consistency * c_consistency +
            self.w_safety * c_safety
        )

        trust_score_100 = round(raw_score * 100.0, 1)
        trust_score_int = int(round(trust_score_100))

        if trust_score_100 >= 80.0:
            level = "HIGH"
        elif trust_score_100 >= 60.0:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "trust_score": trust_score_int,
            "trust_score_exact": trust_score_100,
            "trust_level": level,
            "breakdown": {
                "intent_confidence_contrib": round(self.w_intent * c_intent * 100, 1),
                "evidence_quality_contrib": round(self.w_evidence * c_evidence * 100, 1),
                "historical_consistency_contrib": round(self.w_consistency * c_consistency * 100, 1),
                "safety_score_contrib": round(self.w_safety * c_safety * 100, 1),
            },
            "weights": {
                "intent_confidence": self.w_intent,
                "evidence_quality": self.w_evidence,
                "historical_consistency": self.w_consistency,
                "safety_score": self.w_safety,
            }
        }
